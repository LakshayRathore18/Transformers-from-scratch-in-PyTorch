"""
STAGE 5: Full Transformer Model (decoder-only, GPT-style)
=============================================================

WHY THIS STAGE EXISTS
----------------------
This assembles everything from Stages 1-4 into a complete, trainable
language model:

    token ids -> [embeddings + positional encoding] (Stage 1)
              -> [N stacked Transformer blocks]      (Stage 4, using Stage 2/3)
              -> [final LayerNorm]
              -> [output projection to vocab logits]
              -> softmax (implicitly, via cross-entropy loss at training time)

WHY WEIGHT TYING (sharing weights between input embedding and output projection):
The input token embedding table maps token_id -> d_model vector. The output
projection maps d_model vector -> a score per vocab entry, i.e. essentially
the TRANSPOSE of the same kind of mapping. Both matrices are secretly trying
to learn "what does this token mean, as a point in d_model-dimensional
space" -- one direction turns an id into a vector, the other turns a vector
back into scores over ids. Tying them (using literally the same weight
matrix, transposed, for both) is motivated by:
  1. Fewer parameters: saves vocab_size * d_model parameters, which for a
     large vocab can be a substantial fraction of the whole model.
  2. Regularization / inductive bias: forces the embedding and the "un-
     embedding" to live in a mutually consistent space, which empirically
     improves language modeling quality (used in GPT-2, and originally
     proposed in "Using the Output Embedding to Improve Language Models").

WHY CONTEXT-WINDOW CROPPING DURING GENERATION:
Our positional encoding table (Stage 1) is precomputed only up to `max_len`
positions. If we ever feed the model a sequence longer than the model's
configured `max_seq_len` (a.k.a. context window), Stage 1 would either error
out (index out of range) or worse, silently break if max_len happens to be
larger than max_seq_len. During autoregressive generation, the sequence
grows by one token every step, so eventually it WILL exceed max_seq_len if
we generate for long enough. The fix: before each forward pass during
generation, crop the running sequence to only the last `max_seq_len` tokens.
This is exactly what GPT-style models do in practice: they have a fixed
context window, and once you exceed it, only the most recent tokens are kept
as context, older ones just fall out of view.
"""

import torch
import torch.nn as nn

from stage1_embeddings import TokenEmbedding, PositionalEncoding
from stage4_block import TransformerBlock
from stage2_attention import build_causal_mask


class TransformerLM(nn.Module):
    """
    A minimal GPT-style decoder-only Transformer language model.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len  # the model's context window
        self.d_model = d_model

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.embed_dropout = nn.Dropout(dropout)

        # A stack of identical Transformer blocks. nn.ModuleList (not a plain
        # python list!) is required so PyTorch tracks each block's parameters
        # for autograd/optimizer/`.to(device)` purposes.
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final LayerNorm after the last block. Standard in Pre-LN
        # architectures: since every block only normalizes its own sublayer
        # INPUTS (Pre-LN), the very last residual stream itself is never
        # normalized by any block, so we normalize it once here before the
        # output projection.
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection: d_model -> vocab_size logits.
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # --- WEIGHT TYING ---
        # nn.Embedding.weight has shape (vocab_size, d_model). nn.Linear.weight
        # for a (d_model -> vocab_size) layer, by PyTorch convention, is
        # stored with shape (out_features, in_features) = (vocab_size,
        # d_model) too -- i.e. the SAME shape, not a transpose we need to
        # manage ourselves. So we can literally point both modules at the
        # exact same underlying Parameter tensor. From now on, any gradient
        # update to one updates the other, because they ARE the same tensor.
        self.output_projection.weight = self.token_embedding.embedding.weight

        self._init_weights()

    def _init_weights(self):
        # Standard small-Transformer initialization: normal(0, 0.02) for
        # linear/embedding weights. This isn't the focus of the "from
        # scratch" exercise (attention/blocks are), but sensible init matters
        # a lot in practice for training stability, so we set it explicitly
        # rather than relying on PyTorch's default (which is fine, but this
        # is the convention used in GPT-2 and is worth naming in an
        # interview).
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len) integer token ids, seq_len <= max_seq_len.
        Returns logits: (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = token_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"Input sequence length {seq_len} exceeds model's max_seq_len={self.max_seq_len}. "
            f"Crop the input before calling forward() (see generate() for how)."
        )

        # --- Stage 1: embed tokens + add positional information. ---
        x = self.token_embedding(token_ids)
        x = self.positional_encoding(x)
        x = self.embed_dropout(x)

        # --- Build the causal mask ONCE per forward pass (same for every
        # block and every head, since it only depends on sequence length). ---
        causal_mask = build_causal_mask(seq_len, device=token_ids.device)

        # --- Stage 4: pass through the stack of Transformer blocks. ---
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        # --- Final norm + project to vocabulary logits. ---
        x = self.final_norm(x)
        logits = self.output_projection(x)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation: repeatedly predict the next token and
        append it, one token at a time.

        Args:
            token_ids: (batch, seq_len) starting context ("prompt").
            max_new_tokens: how many new tokens to generate.
            temperature: >1 = more random, <1 = more confident/deterministic,
                         dividing logits before softmax reshapes the
                         probability distribution's sharpness.
            top_k: if set, only sample from the top_k highest-probability
                   tokens at each step (a common trick to avoid sampling
                   very low-probability "junk" tokens from the long tail).

        Returns:
            token_ids extended with max_new_tokens new tokens:
            (batch, seq_len + max_new_tokens)
        """
        self.eval()  # disable dropout for generation

        for _ in range(max_new_tokens):
            # --- CONTEXT WINDOW CROPPING ---
            # This is the critical piece mentioned in the module docstring:
            # our positional encoding table only has entries for positions
            # 0..max_seq_len-1. If token_ids has grown longer than
            # max_seq_len (which WILL happen for a long enough generation),
            # we must only feed the model its last max_seq_len tokens, or
            # Stage 1's positional encoding slicing (`self.pe[:, :seq_len,
            # :]`) would try to read past the end of the precomputed table.
            cropped = token_ids[:, -self.max_seq_len:]

            logits = self(cropped)  # (batch, seq_len_cropped, vocab_size)

            # We only care about the prediction for the NEXT token, which
            # comes from the last position's logits (since this is causal /
            # autoregressive: position i's logits predict token i+1).
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

            # --- temperature scaling ---
            next_token_logits = next_token_logits / max(temperature, 1e-8)

            # --- optional top-k filtering ---
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, top_k)
                # anything below the k-th largest logit gets sent to -inf so
                # softmax assigns it ~0 probability, i.e. it can't be sampled.
                min_kept_value = values[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < min_kept_value, float("-inf")
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            token_ids = torch.cat([token_ids, next_token], dim=1)

        return token_ids


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # SANITY CHECK
    # -----------------------------------------------------------------------
    # There's no single built-in "torch.nn.TransformerLM" to diff against
    # (that's the whole point, we're assembling our own building blocks), so
    # this stage's sanity check focuses on proving the specific NEW claims
    # this stage makes:
    #   1. Forward pass produces correctly-shaped logits.
    #   2. Weight tying is REAL (same tensor, not just same values) and
    #      gradients through the tied weight are correctly shared.
    #   3. generate() produces the requested number of new tokens.
    #   4. generate() correctly crops context so it NEVER overflows
    #      max_seq_len, verified by generating well past max_seq_len and
    #      confirming no crash/index error occurs, plus explicitly checking
    #      the forward pass is only ever called with <= max_seq_len tokens.
    # -----------------------------------------------------------------------

    torch.manual_seed(0)

    vocab_size = 50
    d_model = 32
    num_heads = 4
    num_layers = 2
    d_ff = 64
    max_seq_len = 16  # deliberately SMALL so we can easily exceed it below
    batch_size = 2

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    model.eval()

    # --- Check 1: forward pass shape ---
    seq_len = 10
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)
    print(f"Forward pass logits shape: {tuple(logits.shape)} (expected: {(batch_size, seq_len, vocab_size)})")
    assert logits.shape == (batch_size, seq_len, vocab_size), "Logits shape mismatch!"

    # --- Check 2: weight tying is a real shared tensor, not coincidentally
    # equal values. `is` checks Python/tensor object identity. ---
    same_object = model.output_projection.weight is model.token_embedding.embedding.weight
    print(f"Output projection and token embedding share the exact same weight tensor: {same_object}")
    assert same_object, "Weight tying is broken -- these should be the literal same Parameter!"

    # Also verify a gradient step updates BOTH "views" identically, since
    # they're really one tensor.
    dummy_loss = logits.sum()
    dummy_loss.backward()
    grad_via_embedding = model.token_embedding.embedding.weight.grad.clone()
    grad_via_output_proj = model.output_projection.weight.grad.clone()
    assert torch.equal(grad_via_embedding, grad_via_output_proj), "Tied weights received different gradients!"
    print("Gradients accumulated identically on both 'views' of the tied weight tensor, as expected.")
    model.zero_grad()

    # --- Check 3 & 4: generation produces correct token count AND survives
    # generating well past max_seq_len without any shape/index errors, by
    # relying on context-window cropping. ---
    prompt = torch.randint(0, vocab_size, (1, 5))  # start shorter than max_seq_len
    max_new_tokens = 40  # 5 + 40 = 45 total tokens, FAR beyond max_seq_len=16
    generated = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0, top_k=10)

    expected_total_len = prompt.size(1) + max_new_tokens
    print(f"Generated sequence length: {generated.size(1)} (expected: {expected_total_len}), "
          f"despite max_seq_len={max_seq_len}")
    assert generated.size(1) == expected_total_len, "Generated sequence has wrong length!"

    # Explicitly monkey-patch forward to assert every internal call during
    # generation respects the context window, proving the cropping logic
    # (not luck) is what prevented a crash above.
    original_forward = TransformerLM.forward
    max_seen_len = {"value": 0}

    def spying_forward(self, token_ids):
        max_seen_len["value"] = max(max_seen_len["value"], token_ids.size(1))
        assert token_ids.size(1) <= self.max_seq_len, "forward() was called with a sequence exceeding max_seq_len!"
        return original_forward(self, token_ids)

    TransformerLM.forward = spying_forward
    _ = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0, top_k=10)
    TransformerLM.forward = original_forward  # restore

    print(f"Largest sequence length actually passed into forward() during generation: {max_seen_len['value']} "
          f"(model's max_seq_len is {max_seq_len} -- cropping confirmed to always kick in on time)")
    assert max_seen_len["value"] <= max_seq_len, "Context window cropping failed to prevent an overflow!"

    print("\n✅ STAGE 5 SANITY CHECK PASSED: forward shapes correct, weight tying verified (identity + shared "
          "gradients), generation produces correct token counts and never overflows the context window.")
