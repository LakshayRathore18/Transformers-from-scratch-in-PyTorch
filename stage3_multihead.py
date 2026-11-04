"""
STAGE 3: Multi-Head Attention (from scratch, built on Stage 2)
=============================================================

WHY THIS STAGE EXISTS
----------------------
A single attention computation forces every position to blend information
from other positions using ONE shared notion of "similarity" (one Q/K/V
projection). That's limiting: a sentence has many kinds of relationships at
once, e.g. syntactic (subject-verb agreement), positional (nearby words),
coreference (pronoun -> noun it refers to), and one shared projection has to
compromise across all of them.

Multi-head attention runs several independent, smaller attention operations
("heads") in parallel, each with its own learned Q/K/V projections, so each
head is free to specialize in a different kind of relationship. Their outputs
are concatenated and passed through one more learned linear layer to mix the
heads back together into a single representation.

HOW THE SPLITTING WORKS
------------------------
Instead of literally building `num_heads` separate small linear layers, the
standard (and equivalent, but much faster) trick is:
1. Project the full input once to d_model with one big linear layer for Q,
   one for K, one for V (each: d_model -> d_model).
2. Reshape that (batch, seq_len, d_model) tensor into
   (batch, seq_len, num_heads, d_k) where d_k = d_model / num_heads, then
   transpose to (batch, num_heads, seq_len, d_k) so that head becomes a
   batch-like dimension attention already broadcasts over in Stage 2.
3. Run scaled_dot_product_attention once, batched across all heads at once.
4. Transpose back and reshape (concatenate) heads into (batch, seq_len,
   d_model), then apply one final learned output projection.

This is mathematically identical to running num_heads separate attention
modules and concatenating their outputs, but is one batched matmul rather
than a Python loop, way faster on GPU/CPU alike.
"""

import torch
import torch.nn as nn

from stage2_attention import scaled_dot_product_attention, build_causal_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimensionality per head

        # Four learned linear projections: one each for Q, K, V (input ->
        # d_model), and one final output projection (concatenated heads ->
        # d_model) to let the model recombine information across heads.
        # bias=True (default) mirrors torch.nn.MultiheadAttention's default.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # transpose so num_heads acts like a batch dim for the attention matmuls
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2)  # (batch, seq_len, num_heads, d_k)
        # .contiguous() needed before .view() because transpose() only changes
        # strides/metadata, not the underlying memory layout, and view()
        # requires contiguous memory.
        return x.contiguous().view(batch_size, seq_len, num_heads * d_k)

    def forward(self, query, key, value, mask: torch.Tensor = None):
        """
        query/key/value: (batch, seq_len, d_model). For self-attention these
        are all the same tensor; for cross-attention (not used in this
        decoder-only project, but supported for generality) they can differ.
        mask: broadcastable to (batch, num_heads, seq_len_q, seq_len_k).
        """
        # --- Project inputs into Q, K, V spaces, then split into heads. ---
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        # --- Run the Stage 2 attention primitive, batched across all heads
        # at once (num_heads folds into the leading batch dims). ---
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )

        # --- Merge heads back into a single d_model-wide vector per position,
        # then mix across heads with the final learned output projection. ---
        combined = self._combine_heads(attn_output)
        output = self.w_o(combined)

        return output, attn_weights


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # SANITY CHECK
    # -----------------------------------------------------------------------
    # torch.nn.MultiheadAttention is PyTorch's reference implementation. To
    # compare fairly we must:
    #   1. Match its internal parameterization: it stores one fused
    #      in_proj_weight (3*d_model, d_model) for Q/K/V instead of three
    #      separate Linear layers, so we copy our separate weights into that
    #      fused layout (and vice versa for out_proj).
    #   2. Feed it the SAME input and the SAME mask semantics. Careful: torch's
    #      `attn_mask` uses the OPPOSITE boolean convention from ours -- True
    #      means "block this position" for torch, whereas True means "allow"
    #      in our Stage 2 code. We convert explicitly to avoid a silent bug.
    #   3. Use batch_first=True so shapes line up as (batch, seq, d_model)
    #      like the rest of this project, instead of torch's legacy
    #      (seq, batch, d_model) default.
    # -----------------------------------------------------------------------

    torch.manual_seed(0)

    d_model = 32
    num_heads = 4
    batch_size = 2
    seq_len = 5

    ours = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    ours.eval()  # disable dropout for a deterministic comparison

    torch_mha = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=num_heads, dropout=0.0, batch_first=True
    )
    torch_mha.eval()

    # --- Copy weights so both modules compute the literal same function. ---
    with torch.no_grad():
        # torch's in_proj_weight/bias stacks [W_q; W_k; W_v] along dim 0.
        torch_mha.in_proj_weight.copy_(
            torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight], dim=0)
        )
        torch_mha.in_proj_bias.copy_(
            torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias], dim=0)
        )
        torch_mha.out_proj.weight.copy_(ours.w_o.weight)
        torch_mha.out_proj.bias.copy_(ours.w_o.bias)

    x = torch.randn(batch_size, seq_len, d_model)

    # --- Test 1: no mask ---
    our_out, _ = ours(x, x, x, mask=None)
    torch_out, _ = torch_mha(x, x, x, attn_mask=None, need_weights=False)

    max_diff = (our_out - torch_out).abs().max().item()
    print(f"[No mask] max abs difference vs torch.nn.MultiheadAttention: {max_diff:.2e}")
    assert torch.allclose(our_out, torch_out, atol=1e-4), "Mismatch vs torch MultiheadAttention (no mask)!"

    # --- Test 2: causal mask, being careful about the inverted boolean convention. ---
    our_causal_mask = build_causal_mask(seq_len)  # True = allowed (our convention)
    # torch expects a 2D (seq_len, seq_len) additive/bool mask where True = BLOCKED.
    torch_causal_mask = ~our_causal_mask.squeeze(0).squeeze(0)  # invert + drop the (1,1) dims

    our_out_causal, _ = ours(x, x, x, mask=our_causal_mask)
    torch_out_causal, _ = torch_mha(x, x, x, attn_mask=torch_causal_mask, need_weights=False)

    max_diff_causal = (our_out_causal - torch_out_causal).abs().max().item()
    print(f"[Causal mask] max abs difference vs torch.nn.MultiheadAttention: {max_diff_causal:.2e}")
    assert torch.allclose(our_out_causal, torch_out_causal, atol=1e-4), \
        "Mismatch vs torch MultiheadAttention (causal mask)!"

    # --- Test 3: shape check for a batch/head configuration not equal to the
    # ones above, to make sure the reshape logic isn't accidentally hardcoded. ---
    d_model2, num_heads2, batch2, seq2 = 64, 8, 3, 7
    mha2 = MultiHeadAttention(d_model2, num_heads2)
    x2 = torch.randn(batch2, seq2, d_model2)
    out2, weights2 = mha2(x2, x2, x2)
    assert out2.shape == (batch2, seq2, d_model2), "Output shape wrong for a different config!"
    assert weights2.shape == (batch2, num_heads2, seq2, seq2), "Attention weight shape wrong!"
    print(f"[Shape check] different config (d_model={d_model2}, heads={num_heads2}) -> "
          f"output shape {tuple(out2.shape)}, attn weights shape {tuple(weights2.shape)}")

    print("\n✅ STAGE 3 SANITY CHECK PASSED: matches torch.nn.MultiheadAttention exactly "
          "(masked and unmasked), shapes verified for a second configuration.")
