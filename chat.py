import torch
from stage5_transformer import TransformerLM
from stage6_train import CharTokenizer, load_text

device = "cuda" if torch.cuda.is_available() else "cpu"

text = load_text()
tokenizer = CharTokenizer(text)

model = TransformerLM(
    vocab_size=tokenizer.vocab_size,
    d_model=384,
    num_heads=6,
    num_layers=6,
    d_ff=1536,
    max_seq_len=256,
)

model.load_state_dict(torch.load("transformer_lm.pt", map_location=device))
model.to(device)
model.eval()

while True:
    prompt = input("Prompt: ")

    context = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )

    output = model.generate(
        context,
        max_new_tokens=300,
        temperature=0.8,
        top_k=20
    )

    print(tokenizer.decode(output[0].tolist()))