"""
STAGE 2: Scaled Dot-Product Attention (from scratch)
=============================================================

WHY THIS STAGE EXISTS
----------------------
This is the mathematical heart of the Transformer. The core idea: for every
position in a sequence, decide how much to "look at" every other position,
and produce an output that's a weighted blend of those other positions'
values. That's it, that's attention. Everything else in the Transformer
(multi-head splitting, blocks, stacking) is just plumbing around this
one operation.

THE FORMULA
-----------
    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + mask ) @ V

Where:
- Q (query): "what am I looking for?" -- one vector per position.
- K (key):   "what do I contain, as a thing others can match against?"
- V (value): "what do I actually want to hand over if someone attends to me?"

Q, K, V are all derived from the same input via separate learned linear
projections (that happens one layer up, in stage3; here we implement the
attention math itself given Q, K, V tensors).

WHY divide by sqrt(d_k):
As d_k (the dimensionality of each query/key vector) grows, the dot product
Q·K sums over more terms, so its variance grows roughly linearly with d_k
(assuming components are roughly independent, zero-mean, unit-variance).
Large-magnitude logits push softmax into a very "peaked" or saturated regime,
where gradients vanish almost everywhere except the max. Dividing by sqrt(d_k)
rescales the dot product back down to roughly unit variance regardless of
d_k, keeping softmax in a well-behaved regime and gradients healthy.

WHY causal masking:
For decoder-style (autoregressive) generation, position i must not be
allowed to attend to positions j > i - otherwise the model could "cheat" by
peeking at future tokens it's supposed to be predicting. We implement this
by setting the attention logits for disallowed (i, j) pairs to -infinity
*before* the softmax, so that after softmax those positions get exactly zero
weight (since exp(-inf) = 0).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
):
    """
    Args:
        query: (..., seq_len_q, d_k)
        key:   (..., seq_len_k, d_k)
        value: (..., seq_len_k, d_v)
        mask:  optional bool tensor broadcastable to (..., seq_len_q, seq_len_k).
               True/1 = "allowed to attend", False/0 = "blocked".
        dropout: optional nn.Dropout module applied to attention weights
                 (standard Transformer regularization, applied AFTER softmax
                 so it randomly zeroes some attention connections per step).

    Returns:
        output: (..., seq_len_q, d_v) -- the attended values
        attn_weights: (..., seq_len_q, seq_len_k) -- for inspection/debugging
    """
    d_k = query.size(-1)

    # --- Step 1: raw similarity scores between every query and every key. ---
    # (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    # key.transpose(-2, -1) swaps the last two dims, so this is a batched matmul
    # over however many leading (batch, head, ...) dimensions we have.
    scores = torch.matmul(query, key.transpose(-2, -1))

    # --- Step 2: scale by sqrt(d_k) to control variance (see module docstring). ---
    scores = scores / math.sqrt(d_k)

    # --- Step 3: apply mask (if any) by setting disallowed positions to -inf
    # BEFORE softmax, so softmax assigns them ~0 probability. We use a very
    # large negative number rather than literal -inf to avoid NaN issues if an
    # entire row were ever fully masked (defensive, though shouldn't happen
    # here since every query position can at least attend to itself).
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-1e9"))

    # --- Step 4: softmax over the key dimension turns scores into a proper
    # probability distribution (non-negative, sums to 1) that each query uses
    # to weight the values. dim=-1 because that's the seq_len_k axis. ---
    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # --- Step 5: weighted sum of the value vectors using the attention
    # probabilities as weights. ---
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


def build_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    Builds a (1, 1, seq_len, seq_len) boolean mask where entry (i, j) is True
    iff j <= i (query position i is allowed to see key position j).

    WHY lower-triangular: position i should see itself and everything before
    it (j = 0..i), but nothing after it (j > i). torch.tril keeps the lower
    triangle (including diagonal) of a matrix and zeros out the rest, which
    is exactly this "can see the past and present, not the future" pattern.

    The extra leading (1, 1) dims are so this broadcasts cleanly against a
    (batch, num_heads, seq_len, seq_len) attention score tensor without
    needing to manually expand it.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # SANITY CHECK
    # -----------------------------------------------------------------------
    # We compare our from-scratch implementation directly against PyTorch's
    # built-in torch.nn.functional.scaled_dot_product_attention (SDPA), which
    # is the reference, highly-optimized implementation of the exact same
    # math. If our numbers match closely, our formula/implementation is
    # correct.
    # -----------------------------------------------------------------------

    torch.manual_seed(0)

    batch_size = 2
    num_heads = 1  # attention math here doesn't care about heads yet (stage 3 does)
    seq_len = 6
    d_k = 8

    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_k)

    # --- Test 1: no mask (bidirectional / encoder-style attention) ---
    our_out, our_weights = scaled_dot_product_attention(q, k, v, mask=None)
    torch_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)

    max_diff_no_mask = (our_out - torch_out).abs().max().item()
    print(f"[No mask] max abs difference vs torch SDPA: {max_diff_no_mask:.2e}")
    assert torch.allclose(our_out, torch_out, atol=1e-5), "Mismatch vs torch SDPA (no mask)!"

    # --- Test 2: causal mask (decoder-style attention) ---
    causal_mask = build_causal_mask(seq_len)  # (1, 1, seq_len, seq_len), bool
    our_out_causal, our_weights_causal = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    # torch's built-in causal flag implements the identical lower-triangular masking
    torch_out_causal = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

    max_diff_causal = (our_out_causal - torch_out_causal).abs().max().item()
    print(f"[Causal mask] max abs difference vs torch SDPA: {max_diff_causal:.2e}")
    assert torch.allclose(our_out_causal, torch_out_causal, atol=1e-5), "Mismatch vs torch SDPA (causal)!"

    # --- Test 3: verify the causal mask actually zeroes out future attention
    # weights, i.e. attn_weights[..., i, j] == 0 whenever j > i. This confirms
    # the masking is doing its job, not just that shapes happen to match. ---
    upper_triangle_weights = torch.triu(our_weights_causal[0, 0], diagonal=1)
    max_future_weight = upper_triangle_weights.abs().max().item()
    print(f"[Causal mask] max attention weight leaked to future positions: {max_future_weight:.2e}")
    assert max_future_weight < 1e-8, "Causal mask is leaking attention to future positions!"

    # --- Test 4: sanity check that attention weights form a valid probability
    # distribution over keys (rows sum to 1, all non-negative). ---
    row_sums = our_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Attention rows don't sum to 1!"
    assert (our_weights >= 0).all(), "Negative attention weight found!"
    print("Attention weight rows sum to 1 and are non-negative: confirmed valid probability distributions.")

    print("\n✅ STAGE 2 SANITY CHECK PASSED: matches torch's built-in SDPA (masked and unmasked), "
          "causal masking verified to block future positions.")
