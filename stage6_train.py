"""
STAGE 6: Training Loop -- Character-Level Language Modeling on Tiny Shakespeare
================================================================================

WHY THIS STAGE EXISTS
----------------------
Stages 1-5 built a mathematically correct Transformer, verified piece by
piece against PyTorch's own reference implementations. But "correct" isn't
the same as "useful" -- this stage proves the model can actually LEARN: loss
should go down over training, and generated text should visibly improve from
random noise to something with recognizable structure (words, spacing,
some grammar), even though we only trained briefly on a CPU/small setup.

WHY CHARACTER-LEVEL (not word/subword tokenization):
- No tokenizer library needed at all: the "vocabulary" is just the set of
  unique characters that appear in the text (~65 for Shakespeare's English).
  This keeps the whole project self-contained and easy to reason about.
- It's a harder task per-token than word-level (the model has to learn
  spelling, spacing, and punctuation from scratch), which makes visible
  learning progress an even more convincing demonstration of correctness.
- It matches the classic minGPT/char-rnn style pedagogical setup, which is
  well understood and easy to sanity-check by eye.

WHY THIS DATASET (tiny Shakespeare):
It's the standard, widely-used ~1MB toy dataset for exactly this kind of
demo (originally popularized by Andrej Karpathy's char-rnn and minGPT
projects). Small enough to train meaningfully in minutes on a CPU, large
enough to show real structure emerging.

TRAINING OBJECTIVE:
Standard autoregressive / "next-token prediction" language modeling: given
characters x_0 ... x_{t-1}, predict x_t. We implement this by taking a chunk
of text, using it as the input, and using the SAME chunk *shifted left by
one character* as the target -- i.e. at every position, the label is
"whatever character actually comes next". Loss = cross-entropy between
predicted next-character distribution and the true next character, averaged
over all positions and the batch.
"""

import os
import math
import time
import urllib.request

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless backend: we save plots to file, no display needed
import matplotlib.pyplot as plt

from stage5_transformer import TransformerLM


# -----------------------------------------------------------------------------
# 1. DATA LOADING
# -----------------------------------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "tinyshakespeare.txt"


def load_text() -> str:
    """Downloads tiny Shakespeare if not already present locally, then reads it."""
    if not os.path.exists(DATA_PATH):
        print(f"Downloading tiny Shakespeare dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return text


class CharTokenizer:
    """
    The simplest possible tokenizer: one integer ID per unique character.
    WHY build this ourselves instead of using a library: at the character
    level, "tokenization" IS just "list unique symbols and assign each an
    index" -- there's no subword merging algorithm (like BPE) to implement,
    so a from-scratch tokenizer here is both correct and genuinely simple,
    not a shortcut around real complexity.
    """

    def __init__(self, text: str):
        # sorted() ensures a reproducible, deterministic character -> id
        # mapping across runs (set() order in Python isn't guaranteed stable).
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list:
        return [self.char_to_id[ch] for ch in s]

    def decode(self, ids) -> str:
        return "".join(self.id_to_char[int(i)] for i in ids)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device):
    """
    Samples a random batch of (input, target) sequence pairs from `data`.

    `data` is the ENTIRE tokenized corpus as one long 1D tensor of ids.
    For each example in the batch we pick a random starting index, then:
        input  = data[start       : start + block_size]
        target = data[start + 1   : start + block_size + 1]
    i.e. target is input shifted left by exactly one character -- "predict
    the next character at every position" as described in the module
    docstring above.
    """
    max_start = len(data) - block_size - 1
    start_indices = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in start_indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in start_indices])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, device, eval_iters=20):
    """Averages loss over several random batches for a more stable estimate
    than a single batch (single-batch loss is noisy due to random sampling)."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        logits = model(x)
        loss = compute_loss(logits, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  (batch, seq_len, vocab_size)
    targets: (batch, seq_len) integer ids

    nn.functional.cross_entropy expects (N, C) and (N,) shapes (it flattens
    class-dim second), so we flatten batch and seq_len together into one
    dimension: every position in every sequence is treated as one
    independent classification example ("what's the next character here?").
    """
    batch_size, seq_len, vocab_size = logits.shape
    return nn.functional.cross_entropy(
        logits.view(batch_size * seq_len, vocab_size),
        targets.view(batch_size * seq_len),
    )


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt: str = "\n", max_new_tokens: int = 200):
    """Generates and decodes a sample of text from the current model state,
    used to visibly show learning progress every N training steps."""
    model.eval()
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.8, top_k=20)
    model.train()
    return tokenizer.decode(generated_ids[0].tolist())


# -----------------------------------------------------------------------------
# 2. MAIN TRAINING SCRIPT
# -----------------------------------------------------------------------------

def main():
    torch.manual_seed(1337)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Hyperparameters ---
    # Deliberately small so this trains in a few minutes on CPU. This is a
    # PORTFOLIO / LEARNING project, not a from-scratch GPT-2 replication --
    # the point is demonstrating correctness and visible learning, not
    # state-of-the-art perplexity.
    # NOTE ON THESE NUMBERS: tuned to train in a few minutes on a single CPU
    # core for this demo. If you have a GPU, feel free to scale block_size,
    # d_model, and max_steps up significantly for noticeably better samples.
    block_size = 256       # more context per example — GPU makes the O(n²) attention cost cheap
    batch_size = 64         # 1070 has room for this comfortably at these sizes
    d_model = 384
    num_heads = 6           # 384 / 6 = 64 dims per head, a clean, standard head size
    num_layers = 6
    d_ff = 4 * d_model      # 1536
    dropout = 0.1
    learning_rate = 3e-4
    max_steps = 3000
    eval_interval = 300
    eval_iters = 20

    # --- Load and tokenize data ---
    text = load_text()
    print(f"Loaded dataset: {len(text):,} characters total.")

    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters.")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # --- Train/val split (90/10). The val set lets us check the model isn't
    # just memorizing the exact training sequences (overfitting), by seeing
    # if loss on UNSEEN text also goes down. ---
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Train chars: {len(train_data):,} | Val chars: {len(val_data):,}")

    # --- Build model ---
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=block_size,
        dropout=dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Training loop bookkeeping, for the loss curve plot ---
    train_losses_x, train_losses_y = [], []
    val_losses_x, val_losses_y = [], []

    start_time = time.time()
    for step in range(1, max_steps + 1):
        x, y = get_batch(train_data, batch_size, block_size, device)

        logits = model(x)
        loss = compute_loss(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping: prevents rare large/exploding gradients (common
        # early in Transformer training) from taking a destructively large
        # optimizer step. Standard practice, cheap insurance.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses_x.append(step)
        train_losses_y.append(loss.item())

        if step % eval_interval == 0 or step == 1:
            val_loss = estimate_loss(model, val_data, batch_size, block_size, device, eval_iters)
            train_loss_avg = estimate_loss(model, train_data, batch_size, block_size, device, eval_iters)
            val_losses_x.append(step)
            val_losses_y.append(val_loss)

            elapsed = time.time() - start_time
            print(f"\nstep {step:5d} | train loss {train_loss_avg:.4f} | val loss {val_loss:.4f} "
                  f"| elapsed {elapsed:.1f}s")

            sample = generate_sample(model, tokenizer, device, prompt="\n", max_new_tokens=200)
            print(f"--- sample generation at step {step} ---\n{sample}\n{'-'*60}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s ({max_steps} steps).")

    # --- Save a plot of the loss curve, proving the model actually learned
    # (loss trending down over time), not just that the code ran. ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses_x, train_losses_y, alpha=0.3, label="train loss (per-step, noisy)")
    plt.plot(val_losses_x, val_losses_y, marker="o", label="val loss (averaged, every eval_interval)")
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss (nats/char)")
    plt.title("Character-level Transformer LM training curve (tiny Shakespeare)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("Saved loss curve plot to loss_curve.png")

    # --- Final longer sample, to showcase the end result. ---
    final_sample = generate_sample(model, tokenizer, device, prompt="ROMEO:", max_new_tokens=400)
    print(f"\n=== FINAL SAMPLE GENERATION (prompt='ROMEO:') ===\n{final_sample}\n")

    with open("final_sample.txt", "w", encoding="utf-8") as f:
        f.write(final_sample)
    print("Saved final generated sample to final_sample.txt")

    # --- Save model checkpoint for reuse/inspection. ---
    torch.save(model.state_dict(), "transformer_lm.pt")
    print("Saved model weights to transformer_lm.pt")

    # -----------------------------------------------------------------------
    # SANITY / SUCCESS CHECK for this stage
    # -----------------------------------------------------------------------
    # Rather than comparing to a PyTorch built-in (there's no "built-in
    # training loop" to diff against), the meaningful correctness check for
    # a TRAINING stage is: did the loss actually go down substantially, and
    # is it well below the "random guessing" baseline?
    #
    # For a vocab of V characters, a completely untrained/random model would
    # assign roughly uniform probability 1/V to every next character, giving
    # an expected cross-entropy loss of ln(V). If our trained model's final
    # loss is meaningfully below ln(V), that's concrete quantitative proof of
    # real learning, not just "the code executed."
    # -----------------------------------------------------------------------
    random_baseline_loss = math.log(tokenizer.vocab_size)
    final_val_loss = val_losses_y[-1]
    print(f"\nRandom-guessing baseline loss (ln(vocab_size)): {random_baseline_loss:.4f}")
    print(f"Final validation loss achieved: {final_val_loss:.4f}")
    assert final_val_loss < random_baseline_loss * 0.7, (
        "Validation loss did not drop meaningfully below the random-guessing baseline -- "
        "something is likely wrong with training (check learning rate, data loading, etc.)."
    )
    print("✅ STAGE 6 SUCCESS CHECK PASSED: validation loss dropped well below the random-guessing "
          "baseline, confirming the model is genuinely learning character-level structure.")


if __name__ == "__main__":
    main()
