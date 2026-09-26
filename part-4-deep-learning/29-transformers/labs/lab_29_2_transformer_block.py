# %% [markdown]
# # Lab 29.2: A transformer, from the block up
#
# 1. A pre-norm decoder block and a small GPT, from scratch (attention from lab 29.1's formula, via SDPA for speed).
# 2. Parameter count: check the 12 d^2 per block formula, and where the parameters of a small model actually are.
# 3. Train it on characters from this course, against lab 28.1's RNN, for the same wall-clock budget.
# 4. Pre-norm vs post-norm at depth, with and without warmup.
# 5. Induction heads: in-context copying needs two attention layers; one can't do it.

# %%
import math
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(292)
torch.set_num_threads(4)

# %% [markdown]
# ## 1. The block

# %%
class Attention(nn.Module):
    def __init__(self, d, h, dropout=0.0):
        super().__init__()
        self.h, self.qkv, self.proj, self.dropout = h, nn.Linear(d, 3 * d), nn.Linear(d, d), dropout

    def forward(self, x):
        B, n, d = x.shape
        q, k, v = (t.view(B, n, self.h, d // self.h).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, n, d))


class Block(nn.Module):
    def __init__(self, d, h, pre_norm=True, mlp=True, dropout=0.0):
        super().__init__()
        self.pre_norm = pre_norm
        self.ln1, self.attn = nn.LayerNorm(d), Attention(d, h, dropout)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout)) if mlp else None
        self.ln2 = nn.LayerNorm(d) if mlp else None

    def forward(self, x):
        if self.pre_norm:                                   # x + f(LN(x)): the residual stream is never normalized
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x)) if self.mlp is not None else x
        x = self.ln1(x + self.attn(x))                      # post-norm, as in the 2017 paper: LN(x + f(x))
        return self.ln2(x + self.mlp(x)) if self.mlp is not None else x


class GPT(nn.Module):
    def __init__(self, V, ctx, d=128, h=4, L=4, pre_norm=True, mlp=True, dropout=0.0, tie=False):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, h, pre_norm, mlp, dropout) for _ in range(L))
        self.ln_f = nn.LayerNorm(d) if pre_norm else nn.Identity()    # pre-norm needs a final norm; post-norm ends normalized
        self.head = nn.Linear(d, V, bias=False)
        self.apply(self._init)
        if tie:                                                       # weight tying: the output matrix is the embedding
            self.head.weight = self.tok.weight
        for name, p in self.named_parameters():                       # GPT-2: residual projections scaled by 1/sqrt(2L)
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * L))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# %% [markdown]
# ## 2. Counting parameters
#
# Per block: attention 4 d^2 (Q, K, V, output), MLP 8 d^2 (d -> 4d -> d), plus biases and LayerNorms: about 12 d^2.

# %%
for d, L in ((128, 4), (768, 12)):
    m = GPT(V=50257 if d == 768 else 100, ctx=1024 if d == 768 else 128, d=d, h=4 if d == 128 else 12, L=L, tie=True)
    per_block = sum(p.numel() for p in m.blocks[0].parameters())
    emb = m.tok.weight.numel() + m.pos.weight.numel()
    total = sum(p.numel() for p in m.parameters())                    # tied weights counted once
    print(f"d = {d:3d}, {L:2d} blocks: per block {per_block:,} (12 d^2 = {12 * d * d:,}); blocks {L * per_block:,}; "
          f"embeddings {emb:,}; total {total:,}")
    assert abs(per_block - 12 * d * d) / (12 * d * d) < 0.01
print("the second line is GPT-2 small's shape (with tied embeddings, as GPT-2 has): 124M parameters, 39M in the embeddings.")

# %% [markdown]
# ## 3. Characters from this course, against the RNN
#
# Same data as lab 28.1 (Parts I to III), same 100-character contexts. The RNN gets the same wall-clock budget.

# %%
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "part-5-language-models" / "_shared"))
from corpus import lesson_files, read_lesson  # noqa: E402  (the frozen course text, tools/snapshot_course.py)
files = lesson_files(("part-1-foundations", "part-2-data-engineering", "part-3-classical-ml"))
text = "\n".join(read_lesson(p) for p in files)
chars = sorted(set(text)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
split = int(0.95 * len(data)); train_d, val_d = data[:split], data[split:]
CTX = 100


def batch(d_, bs=64, g=None):
    i = torch.randint(0, len(d_) - CTX - 1, (bs,), generator=g)
    return torch.stack([d_[j:j + CTX] for j in i]), torch.stack([d_[j + 1:j + CTX + 1] for j in i])


@torch.no_grad()
def val_loss(m, n=20):
    g = torch.Generator().manual_seed(0)
    m.eval()
    out = float(np.mean([F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item() for x, y in (batch(val_d, g=g) for _ in range(n))]))
    m.train()
    return out


class CharRNN(nn.Module):                                              # lab 28.1's model
    def __init__(self, h=256):
        super().__init__()
        self.emb, self.rnn, self.out = nn.Embedding(V, 64), nn.RNN(64, h, batch_first=True), nn.Linear(h, V)

    def forward(self, x):
        return self.out(self.rnn(self.emb(x))[0])


def train_for(m, seconds, lr, warmup=100, clip=1.0):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    t0, step, curve = time.time(), 0, []
    while time.time() - t0 < seconds:
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1.0, (step + 1) / warmup)
        x, y = batch(train_d)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step(); step += 1
    return step


BUDGET = 30
results = {}
for name, make, lr in (("transformer (4 blocks, d=64)", lambda: GPT(V, CTX, d=64, h=4, L=4), 3e-3), ("RNN (lab 28.1)", CharRNN, 2e-3)):
    torch.manual_seed(0)
    m = make()
    steps = train_for(m, BUDGET, lr)
    results[name] = val_loss(m)
    print(f"{name:30s} {sum(p.numel() for p in m.parameters()):8,} params, {steps:4d} steps in {BUDGET}s, "
          f"val loss {results[name]:.3f} (perplexity {math.exp(results[name]):.1f})")
print("on a CPU, for 30 seconds, at this size, the RNN wins: it's cheaper per step and gets twice as many. Transformers")
print("win when the hardware is parallel, the context is long and the budget is large (35.4), which is every setting that")
print("matters for language models and not this one. Don't choose architectures from a 30-second benchmark, including mine.")
assert max(results.values()) < 3.0                                # both well below the unigram level (3.4)

# %% [markdown]
# ## 4. Pre-norm vs post-norm
#
# Eight blocks, 200 steps, a learning rate of 1e-2 (high, but pre-norm tolerates it); post-norm with and without a
# 100-step warmup. The lesson has the 3e-3 story, which is less clean and more interesting.

# %%
def train_steps(m, steps, lr, warmup):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(1)
    losses = []
    for step in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1.0, (step + 1) / warmup) if warmup else lr
        i = torch.randint(0, len(train_d) - 65, (32,), generator=g)
        x = torch.stack([train_d[j:j + 64] for j in i]); y = torch.stack([train_d[j + 1:j + 65] for j in i])
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); losses.append(loss.item())
    return [float(np.mean(losses[k - 30:k])) for k in (100, 200)]


print("\n8 blocks, d = 64, lr 1e-2: training loss (mean of 30 steps) at steps 100 and 200. Unigram level is about 3.4.")
norm_res = {}
t0 = time.time()
for pre, warmup in ((True, 0), (False, 0), (False, 100)):
    torch.manual_seed(0)
    m = GPT(V, 64, d=64, h=4, L=8, pre_norm=pre)
    norm_res[(pre, warmup)] = train_steps(m, 200, 1e-2, warmup)
    print(f"  {'pre-norm ' if pre else 'post-norm'}  warmup {warmup:3d}: " + "  ".join(f"{v:.3f}" for v in norm_res[(pre, warmup)]))
print(f"({time.time() - t0:.0f}s) post-norm never leaves the unigram plateau, and a short warmup doesn't save it. Pre-norm")
print("trains: the residual stream is never normalized, so every block's gradient has a direct path to the loss.")
assert norm_res[(False, 0)][-1] > 3.3 and norm_res[(False, 100)][-1] > 3.3
assert norm_res[(True, 0)][-1] < 3.15

# %% [markdown]
# ## 5. Induction heads
#
# Sequences of random tokens (vocabulary 64, length 64) in which a random segment of 8 to 20 tokens appears twice, at
# random positions. Inside the second copy the next token is predictable, but only by finding the previous occurrence of
# the current token and reading what came after it. That takes two steps: one head copies "the previous token" into
# each position, and a head in the next layer searches for it. Attention-only models, no MLPs.

# %%
def repeat_batch(n, g, V_=64, N=64):
    """A random segment of 8 to 24 tokens, repeated until the sequence is full. After the first copy, every token is
    predictable, but only from context: the segment is new in every sequence, and its length varies."""
    x = torch.empty(n, N, dtype=torch.long)
    predictable = torch.zeros(n, N, dtype=torch.bool)
    for i in range(n):
        L = int(torch.randint(8, 25, (1,), generator=g))
        x[i] = torch.randint(0, V_, (L,), generator=g).repeat(N // L + 1)[:N]
        predictable[i, L:] = True
    return x, predictable


def train_induction(layers, steps=500, seed=0, tie=False):
    torch.manual_seed(seed)
    m = GPT(64, 64, d=64, h=4, L=layers, mlp=False, tie=tie)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 3e-3 * min(1.0, (step + 1) / 100)
        x, _ = repeat_batch(64, g)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, 64), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    x, predictable = repeat_batch(1000, torch.Generator().manual_seed(99))
    with torch.no_grad():
        m.eval()
        hit = m(x[:, :-1]).argmax(-1) == x[:, 1:]
    return m, hit[predictable[:, 1:]].float().mean().item(), hit[~predictable[:, 1:]].float().mean().item()


print("\nnext-token accuracy after 500 steps (chance 1/64 = 0.016)")
t0 = time.time()
ind = {}
for layers, tie in ((1, False), (2, False), (2, True)):
    _, pred_acc, first_acc = train_induction(layers, tie=tie)
    ind[(layers, tie)] = (pred_acc, first_acc)
    print(f"  {layers} attention layer{'s' if layers > 1 else ' '}{', tied embeddings' if tie else '                '}: "
          f"repeated tokens {pred_acc:.3f}, first copy {first_acc:.3f}")
print(f"({time.time() - t0:.0f}s)")
print("the first copy is random: nobody beats chance there. Repeats: two layers copy (a previous-token head feeding an")
print("induction head); one layer can only hedge over the offsets it has seen. With tied embeddings the direct path from")
print("token to logit is E E^T, which pushes every token's own logit up, and an attention-only model has no MLP to undo it.")
assert ind[(2, False)][0] > 0.85 and ind[(1, False)][0] < 0.35 and ind[(2, True)][0] < 0.5
assert ind[(2, False)][1] < 0.05

print("\nAll checks passed.")
