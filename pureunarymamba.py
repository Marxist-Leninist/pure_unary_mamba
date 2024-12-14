import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data():
    if not os.path.isfile("train_data_small.txt"):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        # Combine all train text into one file
        text = "\n".join(ds["train"]["text"])
        text = text.strip()
        with open("train_data_small.txt", "w", encoding="utf-8") as f:
            f.write(text)
    with open("train_data_small.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text

text = prepare_data()
tokens = text.split()

# Count frequencies
freq = Counter(tokens)
# Take top 300 tokens
top_n = 300
most_common = freq.most_common(top_n)
selected_tokens = set([w for w,_ in most_common])

# Filter tokens to only these top tokens, replace others with <unk>
filtered = [t if t in selected_tokens else "<unk>" for t in tokens]
if "<unk>" not in selected_tokens:
    # Ensure <unk> is in vocab
    selected_tokens.add("<unk>")
    selected_tokens = list(selected_tokens)
else:
    selected_tokens = list(selected_tokens)

vocab = selected_tokens
vocab_to_id = {w: i for i,w in enumerate(vocab)}
id_to_vocab = {i: w for i,w in enumerate(vocab)}

data_ids = [vocab_to_id[t if t in vocab_to_id else "<unk>"] for t in filtered]

vocab_size = len(vocab)
print("Reduced Vocab size:", vocab_size)

class UnaryEmbedding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
    def forward(self, x):
        B,L = x.size()
        out = torch.zeros(B,L,self.vocab_size, device=x.device)
        out.scatter_(2, x.unsqueeze(2), 1.0)
        return out

class SelectiveSSM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_g = nn.Linear(d_model, 1)
        self.W_out = nn.Linear(d_model, d_model)
    def forward(self, x):
        B,L,D = x.size()
        h = torch.zeros(B,D, device=x.device)
        outs = []
        for t in range(L):
            xt = x[:,t,:]
            g = torch.sigmoid(self.W_g(xt))
            g = g.expand(-1,D)
            h = (1-g)*h + g*xt
            out = self.W_out(h)
            outs.append(out.unsqueeze(1))
        return torch.cat(outs, dim=1)

class LocalConv(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
    def forward(self, x):
        # x: (B,L,D) -> (B,D,L)
        y = x.transpose(1,2)
        y = self.conv(y)
        y = y.transpose(1,2)
        return y

class GatedMLP(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.fc = nn.Linear(d_model, expand*d_model)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x):
        gate, val = self.fc(x).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        y = gate*val
        y = self.out(y)
        return y

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_conv=4, expand=2):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.conv = LocalConv(d_model)
        self.ssm = SelectiveSSM(d_model)
        self.mlp = GatedMLP(d_model, expand=expand)
    def forward(self, x):
        y = self.ln(x)
        y = self.conv(y)
        y = self.ssm(y)
        y = self.mlp(y)
        return x + y

class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers=4):
        super().__init__()
        self.embed = UnaryEmbedding(vocab_size)
        self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        e = self.embed(x)
        y = e
        for b in self.blocks:
            y = b(y)
        y = self.ln_f(y)
        logits = self.head(y)
        return logits

def get_batch(data_ids, block_size=32, batch_size=16):
    ix = torch.randint(len(data_ids)-block_size, (batch_size,))
    x = torch.stack([torch.tensor(data_ids[i:i+block_size]) for i in ix]).to(device)
    y = torch.stack([torch.tensor(data_ids[i+1:i+block_size+1]) for i in ix]).to(device)
    return x, y

def main():
    d_model = vocab_size # still equals vocab_size but now vocab_size is small (~300)
    model = MambaLM(vocab_size, d_model=d_model, n_layers=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    block_size = 32
    batch_size = 16
    steps = 500

    model.train()
    for step in range(steps):
        x,y = get_batch(data_ids, block_size, batch_size)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}: loss {loss.item():.4f}")

    torch.save(model.state_dict(), "pure_unary_mamba.pth")
    print("Model saved to pure_unary_mamba.pth")

if __name__ == "__main__":
    main()
