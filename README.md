# Transformer From Scratch (PyTorch)

A decoder-only (GPT-style) Transformer built up **one component at a time**,
where every custom component is directly checked against PyTorch's own
built-in equivalent before moving on. Built as a portfolio project to
demonstrate a genuine, from-first-principles understanding of the
architecture behind modern LLMs — not just the ability to call `nn.Transformer`.

The final model is trained on character-level **tiny Shakespeare** and
demonstrably learns: validation loss drops from ~4.17 (random guessing) to
~2.03, and generated text visibly evolves from random symbols into
Shakespeare-formatted dialogue (character names, capitalization, line breaks).

---

## Why this project is structured the way it is

Most "Transformer from scratch" tutorials show you code and ask you to trust
it. This project instead proves correctness at every stage: each stage's
`if __name__ == "__main__":` block runs a **sanity check that numerically
compares the custom implementation against PyTorch's reference
implementation** (or, where no direct reference exists, against a
mathematically-derived expected result). Every file is runnable standalone
and prints a clear ✅ pass/fail at the end.

This structure mirrors how I'd want to explain the project in an interview:
build up complexity in verifiable, isolated steps, rather than one large
opaque model class.

---

## File-by-file breakdown

| File | What it builds | Sanity check |
|---|---|---|
| `stage1_embeddings.py` | Token embedding (scaled by √d_model) + sinusoidal positional encoding | Shapes verified; positional encoding values hand-checked against the closed-form `sin`/`cos` formula from "Attention Is All You Need"; sin²+cos²=1 structural check |
| `stage2_attention.py` | Scaled dot-product attention: `softmax(QKᵀ/√d_k + mask) @ V`, plus causal mask construction | Output compared against `torch.nn.functional.scaled_dot_product_attention`, with and without causal masking (max abs diff ~1e-7); verifies causal mask fully blocks future positions; verifies attention weights are a valid probability distribution |
| `stage3_multihead.py` | Multi-head attention: splits Q/K/V into heads, runs Stage 2 attention per head (batched), concatenates, projects | Weights copied into `torch.nn.MultiheadAttention` for an apples-to-apples comparison; matches to ~1e-7, masked and unmasked; shape check on a second, different configuration |
| `stage4_block.py` | Full Pre-LN Transformer block: causal self-attention + residual + LayerNorm, feed-forward (GELU) + residual + LayerNorm | Shape checks; verifies gradients flow through the residual path; verifies causal masking still holds after passing through the *entire* block (perturbing a future token doesn't change earlier outputs) |
| `stage5_transformer.py` | Full model: embeddings → stack of N blocks → final norm → output projection, with **weight tying** between the input embedding and output projection, and a `generate()` method | Verifies weight tying is a literal shared tensor (not just equal values) with shared gradients; verifies `generate()` produces the correct token count; verifies context-window cropping prevents any forward pass from ever exceeding `max_seq_len`, even after generating far beyond it |
| `stage6_train.py` | Character-level training loop on tiny Shakespeare: custom char tokenizer, train/val split, AdamW + gradient clipping, periodic loss logging + sample generation, loss curve plot | Confirms final validation loss drops well below the random-guessing baseline `ln(vocab_size)`, i.e. proves the model is genuinely learning, not just running |

Supporting files:
- `requirements.txt` — the two dependencies needed (`torch`, `matplotlib`)
- `tinyshakespeare.txt` — auto-downloaded by `stage6_train.py` on first run (not needed to be present beforehand)

---

## Setup instructions

```bash
# 1. Clone / download this project, then create a virtual environment
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run each stage in order — every file is self-contained and runnable directly.
python stage1_embeddings.py
python stage2_attention.py
python stage3_multihead.py
python stage4_block.py
python stage5_transformer.py

# 4. Run training (downloads tiny Shakespeare automatically on first run,
#    trains ~5-8 minutes on CPU with the default settings)
python stage6_train.py
```

Each of stages 1–5 prints its own sanity check and exits with a `✅` message
on success (or an `AssertionError` with a clear message on failure). Stage 6
trains the full model, prints train/val loss and a text sample every
`eval_interval` steps, then saves:
- `loss_curve.png` — training/validation loss over time
- `final_sample.txt` — a longer generated sample after training
- `transformer_lm.pt` — the trained model's weights

If you have a GPU available, `stage6_train.py` will automatically use it
(`device = "cuda" if torch.cuda.is_available() else "cpu"`) — you can also
bump up `block_size`, `d_model`, `num_layers`, and `max_steps` for
meaningfully better results.

---

## Sample output (from an actual training run, CPU, ~6 minutes, 1500 steps)

**Loss:** dropped from `4.12` (step 1) → `2.03` (step 1500) validation loss,
against a random-guessing baseline of `ln(65) ≈ 4.17` for this 65-character
vocabulary.

**Step 1 (untrained, random weights):**
```
z,TTryeWvqIWIKuSZ'!NTcrnp:j
n:
.!:$?too!Mff! T!itW,W!tbtXX?$$tm!jXjjIl
```

**Step 300:**
```
HIRCANESE:
Any ll mele hagess, mertttth ikis the bethend sond oungend shon whand:
Thaner hingd hind chome ofond setde hougid m hee n oous,
```

**Step 1500 (final), prompted with `"ROMEO:"`:**
```
ROMEO:
A he thy nott bed for destreath, shall for nin beas,
For he sher me hoth the is dond as mie;
Whit knis as o tat shrow that eseres shir
An emecie grouceter and whe say, that o
With with prest with seres
Whereing and be my harde to sprows
Whoth hon dons that he me to go bere sherer.

KING ENWERNCE:
A my some the he have lik you bucher me, proed,
Whath my grascce you reve his grive his kind!
```

Note what the model *has* learned in just 1500 steps on a single CPU core:
correct dialogue formatting (`NAME:` followed by a newline), consistent
capitalization at line starts, plausible English letter/syllable
statistics, and reasonable word lengths and punctuation placement. It has
**not** yet learned real grammar or semantics — that requires far more
compute/data/training time, which is expected and discussed below.

---

## What I'd improve next

This project deliberately prioritizes **correctness and clarity of the core
Transformer mechanics** over state-of-the-art results, given CPU/time
constraints. Concrete next steps to push quality further, roughly in order
of expected impact:

1. **Scale up training compute**: more steps, a larger `d_model`/`num_layers`,
   and a longer `block_size` (context window) — on a GPU this becomes cheap
   and would likely produce genuinely grammatical text within an hour.
2. **Learning rate schedule**: add a warmup + cosine decay schedule instead
   of a flat learning rate; this is standard in production Transformer
   training and usually improves both stability and final loss.
3. **Subword tokenization (BPE)** instead of character-level: would let the
   same context window cover far more actual text, and let the model learn
   word-level structure "for free" instead of from raw characters.
4. **KV-caching for generation**: currently `generate()` recomputes the full
   forward pass at every new token; caching past keys/values would make
   generation much faster, especially for longer sequences.
5. **Mixed precision / `torch.compile`**: would meaningfully speed up
   training throughput on GPU with minimal code changes.
6. **RoPE or ALiBi positional encoding**: modern LLMs mostly favor these
   over sinusoidal encodings — one experiment would be a Stage 1b comparing
   the sinusoidal encoding here against a RoPE implementation head-to-head.
7. **Proper hyperparameter sweep** (learning rate, dropout, batch size)
   rather than the manually-picked values used here.

---

## What this project demonstrates (for interview conversations)

- Understanding of *why* each architectural choice exists (scaling by
  √d_k, residual connections, Pre-LN vs Post-LN, weight tying, causal
  masking), not just how to implement it — see the extensive docstrings
  in each stage file.
- Ability to verify a from-scratch implementation numerically against a
  trusted reference implementation, a good habit for any ML engineering
  role.
- Practical training-loop engineering: train/val split, gradient clipping,
  periodic evaluation, and a documented, defensible check that the model is
  actually learning (loss vs. random baseline) rather than just "the code
  ran without crashing."
- Awareness of the model's current limitations and a concrete, prioritized
  plan for what to do next — knowing what to improve is as important as
  what's already been built.
