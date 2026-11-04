"""
STAGE 1: Token Embeddings + Sinusoidal Positional Encoding
=============================================================

WHY THIS STAGE EXISTS
----------------------
A Transformer has no built-in notion of word identity or word order - unlike an
RNN, it doesn't process tokens sequentially, and unlike a CNN, it has no fixed
receptive field tied to position. So before any attention happens, we need to:

1. Turn integer token IDs into dense vectors (embeddings) that the model can
   actually do math on and learn semantic structure over.
2. Inject information about *where* each token sits in the sequence, because
   self-attention treats a sequence as an unordered set of vectors otherwise
   (attention is permutation-equivariant: shuffle the inputs, and you shuffle
   the outputs in the same way, with no penalty). Positional encoding is what
   breaks that symmetry and gives the model a sense of order.

We use the original "Attention Is All You Need" sinusoidal scheme (not a
learned positional embedding table) because:
- It requires no extra trainable parameters.
- It can, in principle, generalize to sequence lengths not seen during
  training extrapolating the sine/cosine pattern.
- It's a nice concrete formula we can unit-test against by hand.
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Thin wrapper around nn.Embedding, but we scale the output by sqrt(d_model).

    WHY the sqrt(d_model) scaling: in the original paper, embeddings and the
    positional encodings are summed together. Positional encodings live in a
    fixed range of about [-1, 1] regardless of d_model. If d_model is large,
    randomly-initialized embedding vectors are relatively small on the same
    scale, so the positional signal would numerically dominate the token
    identity signal. Scaling embeddings up by sqrt(d_model) balances the two
    contributions so neither drowns out the other early in training.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        # nn.Embedding is just a learnable lookup table: shape (vocab_size, d_model).
        # We are allowed to use this built-in because it's a lookup table, not the
        # actual Transformer math we are here to implement from scratch.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len) of integers -> (batch, seq_len, d_model)
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding:

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    where `pos` is the position index in the sequence (0, 1, 2, ...) and `i`
    indexes pairs of dimensions in the embedding vector.

    WHY sine AND cosine, and WHY different frequencies per dimension:
    - Each pair of dimensions (2i, 2i+1) oscillates at its own frequency,
      controlled by 10000^(2i/d_model). Low i -> high frequency (changes fast
      across positions), high i -> low frequency (changes slowly). This gives
      the model a mixture of "clocks" ticking at different rates, similar to
      binary encoding of numbers using bits of different significance.
    - Using sin/cos pairs (rather than, say, just sin) means that PE(pos+k)
      can be expressed as a *linear function* of PE(pos) for any fixed offset
      k. That linear relationship is what theoretically allows the model to
      learn to attend by *relative* position, not just absolute position.

    We precompute the full table up to `max_len` once (it's not trainable),
    and just slice out however many positions we need at forward time.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # pe[pos, dim] will hold the final positional encoding table.
        pe = torch.zeros(max_len, d_model)

        # Column vector of positions: shape (max_len, 1) -> broadcasts against
        # the per-dimension frequency vector below.
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # div_term holds 1 / 10000^(2i/d_model) for i = 0, 1, ..., d_model/2 - 1.
        # We compute it in log-space (exp(-x * log(10000))) purely for numerical
        # stability, this is mathematically identical to 10000^(-2i/d_model)
        # but avoids overflow issues with directly raising 10000 to a power.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        # Even dimensions get sine, odd dimensions get cosine, of the same
        # underlying angle (position * div_term). This is what makes each
        # (sin, cos) pair behave like a single rotating "clock hand".
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: (1, max_len, d_model) so it broadcasts over
        # any batch size when we add it to token embeddings.
        pe = pe.unsqueeze(0)

        # register_buffer: this tensor is part of the module's state (moves
        # with .to(device), gets saved in state_dict) but is NOT a learnable
        # parameter, i.e. no gradient, no optimizer update. That's correct
        # here because the formula is fixed, not learned.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model). We only need the first seq_len rows
        # of our precomputed table, then broadcast-add across the batch.
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # SANITY CHECK
    # -----------------------------------------------------------------------
    # Two things to prove here:
    #   (a) Shapes flow through correctly end-to-end (embedding -> +pos enc).
    #   (b) Our positional encoding numerically matches the closed-form
    #       formula from the paper, computed independently (not just by
    #       re-running our own code, but by re-deriving values by hand for a
    #       few (pos, dim) entries).
    # -----------------------------------------------------------------------

    torch.manual_seed(0)

    vocab_size = 100
    d_model = 16
    batch_size = 4
    seq_len = 10

    token_emb = TokenEmbedding(vocab_size, d_model)
    pos_enc = PositionalEncoding(d_model, max_len=50)

    dummy_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    embedded = token_emb(dummy_tokens)
    print(f"Token embedding output shape: {embedded.shape} (expected: {(batch_size, seq_len, d_model)})")
    assert embedded.shape == (batch_size, seq_len, d_model), "Embedding shape mismatch!"

    encoded = pos_enc(embedded)
    print(f"After positional encoding shape: {encoded.shape} (expected: {(batch_size, seq_len, d_model)})")
    assert encoded.shape == (batch_size, seq_len, d_model), "Positional encoding shape mismatch!"

    # --- Check (b): manually recompute PE(pos, dim) for a handful of spot
    # checks using the raw formula, independent of the class's internals. ---
    print("\nVerifying positional encoding formula by hand for spot-check (pos, dim) pairs:")
    max_abs_error = 0.0
    test_cases = [(0, 0), (0, 1), (1, 0), (1, 1), (5, 4), (5, 5), (9, 14), (9, 15)]
    for pos, dim in test_cases:
        i = dim // 2  # pair index
        angle = pos / (10000 ** (2 * i / d_model))
        expected = math.sin(angle) if dim % 2 == 0 else math.cos(angle)
        actual = pos_enc.pe[0, pos, dim].item()
        error = abs(expected - actual)
        max_abs_error = max(max_abs_error, error)
        print(f"  pos={pos:2d} dim={dim:2d} | formula={expected: .6f} | module={actual: .6f} | abs_error={error:.2e}")

    assert max_abs_error < 1e-5, f"Positional encoding formula mismatch! max error={max_abs_error}"

    # --- Extra structural check: sin^2 + cos^2 == 1 for every (pos, i) pair,
    # since sin and cos at a given pair index share the same angle. This is a
    # cheap way to catch an indexing bug (e.g. mismatched angles between the
    # even/odd slots) without hand-deriving every single value. ---
    even = pos_enc.pe[0, :, 0::2]
    odd = pos_enc.pe[0, :, 1::2]
    identity = even ** 2 + odd ** 2
    assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5), \
        "sin^2 + cos^2 != 1 somewhere -- angle mismatch between sin/cos slots!"
    print("\nStructural check passed: sin^2(angle) + cos^2(angle) == 1 for all (pos, freq-pair) entries.")

    print("\n✅ STAGE 1 SANITY CHECK PASSED: shapes correct, formula matches by hand, structure verified.")
