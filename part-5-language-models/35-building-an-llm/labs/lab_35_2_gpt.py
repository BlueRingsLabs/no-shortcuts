# %% [markdown]
# # Lab 35.2: A GPT from scratch, trained on this course
#
# The whole pipeline, small enough for a CPU:
# 1. Tokenize with lab 35.1's BPE and store the tokens as a flat binary file, read back with np.memmap.
# 2. The model (29.2's pre-norm GPT), its parameter count, FLOPs per token, and the data budget it would want.
# 3. Sanity check before training: the initial loss should be ln(vocabulary size).
# 4. Training: AdamW with decay groups, warmup + cosine, gradient clipping, gradient accumulation.
# 5. Validation in bits per byte, and a checkpoint that resumes bit-identically.
# 6. Sampling with temperature and top-k.

# %%
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from bpe import BPE  # noqa: E402
from corpus import lesson_files, read_lesson  # noqa: E402

torch.manual_seed(352)
torch.set_num_threads(4)

# %% [markdown]
# ## 1. Tokens on disk

# %%
files = lesson_files()
train_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 != 4)
val_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 == 4)
t0 = time.time()
tok = BPE.train(train_text, n_merges=1000)
V = tok.vocab_size
tmp = Path(tempfile.mkdtemp())
for name, text in (("train", train_text), ("val", val_text)):
    np.array(tok.encode(text), dtype=np.uint16).tofile(tmp / f"{name}.bin")          # uint16: vocabulary < 65,536
train_ids = np.memmap(tmp / "train.bin", dtype=np.uint16, mode="r")
val_ids = np.memmap(tmp / "val.bin", dtype=np.uint16, mode="r")
val_bytes_per_token = len(val_text.encode("utf-8")) / len(val_ids)
print(f"tokenizer: {V} tokens; train {len(train_ids):,} tokens, val {len(val_ids):,} tokens "
      f"({val_bytes_per_token:.2f} bytes per token), in {time.time() - t0:.0f}s")
print(f"token files on disk: {(tmp / 'train.bin').stat().st_size / 1e6:.2f} MB; memory-mapped, so a 10 TB corpus reads the same way")

CTX = 128


def get_batch(ids, bs, g):
    i = torch.randint(0, len(ids) - CTX - 1, (bs,), generator=g).numpy()
    x = torch.from_numpy(np.stack([ids[j:j + CTX] for j in i]).astype(np.int64))
    y = torch.from_numpy(np.stack([ids[j + 1:j + CTX + 1] for j in i]).astype(np.int64))
    return x, y


# %% [markdown]
# ## 2. The model

# %%
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, n, d = x.shape
        q, k, v = (t.view(B, n, self.h, d // self.h).transpose(1, 2) for t in self.qkv(self.ln1(x)).chunk(3, -1))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, n, d))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, V, ctx, d=128, h=4, L=4):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight                                            # tied (29.2)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * L) if name.endswith(("proj.weight", "mlp.2.weight")) else 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


model = GPT(V, CTX)
N = sum(p.numel() for p in model.parameters())
N_nonemb = N - model.tok.weight.numel() - model.pos.weight.numel()
print(f"\nparameters: {N:,} ({N_nonemb:,} outside the embeddings)")
print(f"training FLOPs per token ~ 6 N = {6 * N / 1e6:.1f} MFLOPs (+ attention)")
print(f"Chinchilla (35.4) would want ~20 tokens per parameter: {20 * N / 1e6:.0f}M tokens. We have {len(train_ids) / 1e6:.2f}M.")
print("so this model will see its data many times and overfit: the small-data regime, on purpose, where it's visible.")

# %% [markdown]
# ## 3. The initial loss

# %%
@torch.no_grad()
def estimate_loss(m, ids, n_batches=20, bs=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    m.eval()
    losses = [F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item() for x, y in (get_batch(ids, bs, g) for _ in range(n_batches))]
    m.train()
    return float(np.mean(losses))


init_loss = estimate_loss(model, val_ids, 5)
print(f"\ninitial loss {init_loss:.3f}; ln(V) = {math.log(V):.3f}. If these disagree, the initialization or the loss is wrong (26.5).")
assert abs(init_loss - math.log(V)) < 0.1

# %% [markdown]
# ## 4. Training

# %%
def make_optimizer(m, lr):
    decay = [p for n, p in m.named_parameters() if p.dim() >= 2]                    # matrices and embeddings
    no_decay = [p for n, p in m.named_parameters() if p.dim() < 2]                   # biases, LayerNorm gains
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


def lr_at(step, max_steps, peak=3e-3, warmup=50, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def train(m, opt, steps, start=0, max_steps=None, micro_bs=16, accum=2, g=None, log_every=None, history=None):
    max_steps = max_steps or steps
    for step in range(start, start + steps):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, max_steps)
        for _ in range(accum):                                                         # gradient accumulation: batch 32 in two halves
            x, y = get_batch(train_ids, micro_bs, g)
            loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)) / accum
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if log_every and (step + 1) % log_every == 0:
            tr, va = estimate_loss(m, train_ids), estimate_loss(m, val_ids)
            history.append((step + 1, tr, va))
            print(f"  step {step + 1:5d}  lr {lr_at(step, max_steps):.2e}  grad norm {norm:.2f}  train {tr:.3f}  val {va:.3f}  "
                  f"val bits/byte {va / math.log(2) / val_bytes_per_token:.3f}  ({time.time() - t0:.0f}s)")


STEPS = 1000
g = torch.Generator().manual_seed(0)
opt = make_optimizer(model, 3e-3)
history = []
t0 = time.time()
train(model, opt, STEPS, g=g, log_every=250, history=history)
final_val = history[-1][2]
best_val = min(h[2] for h in history)
print(f"best validation loss {best_val:.3f} (perplexity {math.exp(best_val):.1f} per token, "
      f"{best_val / math.log(2) / val_bytes_per_token:.3f} bits per byte)")
gap = history[-1][2] - history[-1][1]
print(f"final train/val gap {gap:.3f} nats: the model is memorizing the {len(train_ids) / 1e3:.0f}k training tokens it has seen "
      f"~{STEPS * 32 * CTX / len(train_ids):.0f} times each")
assert best_val < init_loss - 2.5 and gap > 0

# %% [markdown]
# ## 5. Checkpoint and resume
#
# Save everything a restart needs: model, optimizer (Adam's moments), the step, and the RNG state of the data sampler.
# Then check that "10 steps, save, load into fresh objects, 10 more" equals "20 steps" exactly.

# %%
def fresh():
    torch.manual_seed(1)
    m = GPT(V, CTX)
    return m, make_optimizer(m, 3e-3)


m_a, o_a = fresh()
train(m_a, o_a, 20, max_steps=20, g=torch.Generator().manual_seed(5))

m_b, o_b = fresh()
g_b = torch.Generator().manual_seed(5)
train(m_b, o_b, 10, max_steps=20, g=g_b)
torch.save({"model": m_b.state_dict(), "opt": o_b.state_dict(), "step": 10, "rng": g_b.get_state()}, tmp / "ckpt.pt")
del m_b, o_b
ck = torch.load(tmp / "ckpt.pt", weights_only=True)                                    # weights_only: never unpickle code
m_c, o_c = fresh()
m_c.load_state_dict(ck["model"]); o_c.load_state_dict(ck["opt"])
g_c = torch.Generator(); g_c.set_state(ck["rng"])
train(m_c, o_c, 10, start=ck["step"], max_steps=20, g=g_c)
same = all(torch.equal(p, q) for p, q in zip(m_a.state_dict().values(), m_c.state_dict().values()))
print(f"\n20 steps straight vs 10 + checkpoint + 10: parameters bit-identical: {same}")
assert same

# %% [markdown]
# ## 6. Sampling

# %%
@torch.no_grad()
def generate(m, prompt, n=120, temperature=0.8, top_k=40, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    m.eval()
    idx = torch.tensor([tok.encode(prompt)])
    for _ in range(n):
        logits = m(idx[:, -CTX:])[0, -1] / temperature
        if top_k:
            kth = logits.topk(top_k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=g_)
        idx = torch.cat([idx, nxt[None]], 1)
    return tok.decode(idx[0].tolist())


for prompt in ("## Things that will bite you\n\n- **", "The model", "Gradient descent"):
    print(f"\n--- {prompt!r} ---\n{generate(model, prompt)}")
print("\nit has learned the course's formatting, its vocabulary and some of its opinions. Not its reasoning: that took")
print("the other 11 orders of magnitude of compute.")

print("\nAll checks passed.")
