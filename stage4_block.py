"""
STAGE 4: Full Transformer Block
=============================================================

WHY THIS STAGE EXISTS
----------------------
Multi-head attention alone (Stage 3) only mixes information ACROSS positions.
It has no way to do any further nonlinear "thinking" about what it just
gathered, per position. A Transformer block combines:

    1. Multi-head self-attention (mixes information across positions)
    2. A position-wise feed-forward network, FFN (adds per-position nonlinear
       processing capacity)

...each wrapped with a residual connection and a layer normalization. Stacking
many of these blocks (Stage 5) is what gives the model depth.

NOTE ON ARCHITECTURE CHOICE: this project builds a DECODER-only Transformer
(like GPT), since our end goal (Stage 6) is autoregressive character-level
text generation. So "encoder block" here specifically means a block with
CAUSAL (masked) self-attention -- each position can only attend to itself and
earlier positions. A true bidirectional encoder block (like BERT) would use
this exact same code with mask=None. We plumb the mask through as a parameter
precisely so this block is reusable for either case.

WHY RESIDUAL CONNECTIONS (x + Sublayer(x)):
Deep networks suffer from vanishing gradients: each layer squashes gradient
signal a bit more, and after enough layers, backprop that reaches early
layers is tiny/uninformative. A residual (skip) connection gives gradients a
direct, unimpeded path straight back to earlier layers (d(output)/d(x)
includes an identity term, gradient=1, regardless of what the sublayer
does). This is what makes it practical to stack many blocks (Stage 5) without
training becoming unstable.

WHY LAYER NORMALIZATION:
Layer norm rescales each individual position's feature vector to have zero
mean/unit variance (with learned scale/shift afterward), independently per
sample and per position. This stabilizes the distribution of activations
flowing between layers, which speeds up and stabilizes training a lot,
independent of batch size (unlike batch norm, which normalizes across the
batch dimension and behaves badly with small/variable batch sizes or
sequence tasks).

WHY Pre-LN (LayerNorm BEFORE the sublayer) instead of Post-LN (the original
paper's choice, LayerNorm AFTER the residual add):
    Post-LN: x = LayerNorm(x + Sublayer(x))
    Pre-LN:  x = x + Sublayer(LayerNorm(x))
Pre-LN keeps an unbroken, un-normalized residual "highway" from input to
output of the whole stack, whereas Post-LN's normalization sits directly in
that residual path and can distort/rescale gradients as they flow through it.
In practice, Pre-LN trains much more stably for deeper stacks and doesn't
require a learning-rate warm-up to avoid divergence, which is why virtually
all modern LLMs (GPT-2 onward) use Pre-LN. We use Pre-LN here for that
practical reason, and mention it explicitly since interviewers often ask
about this exact design choice.

WHY THE FEED-FORWARD NETWORK EXPANDS THEN CONTRACTS (d_model -> d_ff -> d_model):
The FFN is applied independently to each position (hence "position-wise"),
and is just: Linear -> activation -> Linear. Expanding to a wider hidden
dimension (typically d_ff = 4 * d_model) before contracting back gives the
network more representational capacity to compute a richer nonlinear
transform of each position's vector, without changing the residual stream's
width, before it has to be added back via the residual connection.
"""

import torch
import torch.nn as nn

from stage3_multihead import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network: applied identically and independently
    to every position in the sequence (there's no mixing across positions
    here at all, that's what attention already did).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # GELU is the modern default (GPT-2/GPT-3/BERT use it): unlike ReLU's
        # hard cutoff at 0, GELU is smooth and weights inputs by their
        # magnitude probabilistically, which empirically trains better for
        # Transformers. We still support ReLU since the original 2017 paper
        # used it, and it's worth knowing both for an interview.
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)  # regularize the wide hidden representation
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    One Transformer block: Pre-LN self-attention sublayer + Pre-LN FFN sublayer,
    each with its own residual connection.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

        # Two separate LayerNorms: one gating into attention, one into FFN.
        # Each needs its own learned scale/shift parameters, since the two
        # sublayers operate on differently-distributed activations.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout applied directly to each sublayer's OUTPUT before adding it
        # back to the residual stream. This is standard "residual dropout":
        # it regularizes what each sublayer is allowed to contribute, without
        # ever touching the residual path itself (which must stay a clean,
        # dropout-free gradient highway, per the Pre-LN rationale above).
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # --- Sublayer 1: self-attention, Pre-LN style ---
        # Normalize first, THEN attend, THEN add back to the un-normalized
        # residual stream `x`. This is the defining trait of Pre-LN.
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        x = x + self.dropout1(attn_out)

        # --- Sublayer 2: feed-forward, Pre-LN style ---
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_out)

        return x


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # SANITY CHECK
    # -----------------------------------------------------------------------
    # There's no single "torch.nn.TransformerBlock" with an identical Pre-LN
    # decoder-only recipe to diff against line-for-line (torch's
    # TransformerEncoderLayer defaults to Post-LN and has a slightly
    # different sublayer arrangement). So for this stage we verify:
    #   1. Shapes are preserved and correct through the whole block.
    #   2. The residual connection is doing real work (output != a plain
    #      pass-through, but also output stays "close" to input in a
    #      well-behaved way at initialization).
    #   3. The causal mask still prevents future-position leakage all the way
    #      through a full block, not just through raw attention (Stage 2/3
    #      already proved the attention primitive itself is correct against
    #      PyTorch, so here we confirm that correctness survives being
    #      wrapped in the rest of the block's plumbing).
    #   4. Gradients flow back to the input (proves the residual path isn't
    #      accidentally disconnected).
    # -----------------------------------------------------------------------

    torch.manual_seed(0)

    d_model = 32
    num_heads = 4
    d_ff = 128
    batch_size = 2
    seq_len = 6

    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.0)
    block.eval()

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    from stage2_attention import build_causal_mask
    causal_mask = build_causal_mask(seq_len)

    out = block(x, mask=causal_mask)
    print(f"Block output shape: {tuple(out.shape)} (expected: {(batch_size, seq_len, d_model)})")
    assert out.shape == (batch_size, seq_len, d_model), "Transformer block output shape mismatch!"

    # --- Check the residual connection is actually contributing (output
    # should not be identical to a fresh random tensor, nor identical to the
    # input, it should be a genuinely transformed-but-related version of x). ---
    assert not torch.allclose(out, x), "Output is suspiciously identical to input; residual/sublayers may be no-ops!"

    # --- Gradient flow check: backprop a dummy scalar loss and confirm the
    # input received a non-zero gradient everywhere. If residual connections
    # were broken, some paths could end up with zero gradient. ---
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient reached the input -- residual path may be broken!"
    assert (x.grad.abs().sum() > 0), "Gradient at input is all zero -- something is disconnected!"
    print(f"Gradient reached input successfully, mean abs grad = {x.grad.abs().mean().item():.6f}")

    # --- Re-verify causal masking survives the full block by checking that
    # changing a FUTURE token's input value doesn't change an EARLIER
    # position's output. This is a black-box behavioral test: if position 0's
    # output changes when we perturb position 5's input, that means
    # information leaked backward through time somewhere in this block. ---
    block.eval()
    with torch.no_grad():
        x_a = torch.randn(1, seq_len, d_model)
        x_b = x_a.clone()
        x_b[0, -1, :] += 100.0  # drastically perturb only the LAST (future) position

        out_a = block(x_a, mask=causal_mask)
        out_b = block(x_b, mask=causal_mask)

        # All positions except the last one should be completely unaffected.
        earlier_positions_diff = (out_a[0, :-1, :] - out_b[0, :-1, :]).abs().max().item()
        print(f"Max change in earlier positions' outputs after perturbing the future position: {earlier_positions_diff:.2e}")
        assert earlier_positions_diff < 1e-5, "Causal mask leaking information from future positions through the block!"

    print("\n✅ STAGE 4 SANITY CHECK PASSED: shapes correct, residual/gradient flow verified, "
          "causal masking confirmed to hold end-to-end through the full block.")
