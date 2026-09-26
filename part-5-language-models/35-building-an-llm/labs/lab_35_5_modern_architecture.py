# %% [markdown]
# # Lab 35.5: The modern LLM block
#
# 1. The pieces: RMSNorm, RoPE (from 29.3), SwiGLU, grouped-query attention, no biases.
# 2. Parameter counts: a Llama-style block vs a GPT-2 block of the same width.
# 3. Training both on this course at matched parameter counts and budget.
# 4. Mixture of experts: a top-2 routed MLP, total vs active parameters, and expert collapse with and without the
#    load-balancing loss.

# %%
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from bpe import BPE  # noqa: E402
from corpus import lesson_files, read_lesson  # noqa: E402

torch.manual_seed(355)
torch.set_num_threads(4)

# %% [markdown]
# ## 1. The pieces

# %%
class RMSNorm(nn.Module):
    """x / rms(x) * g: LayerNorm without the mean subtraction and without a bias. Cheaper, and works as well."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


def rope(x, base=10000.0):
    """Rotate pairs of dimensions by position-dependent angles (29.3). x: (B, heads, n, head_dim)."""
    n, hd = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    ang = torch.arange(n, dtype=torch.float32)[:, None] * theta
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class SwiGLU(nn.Module):
    """W2 (silu(W1 x) * W3 x), hidden size ~ 8/3 d so the parameter count matches a 4d GELU MLP."""
    def __init__(self, d, hidden):
        super().__init__()
        self.w1, self.w3, self.w2 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False), nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GQA(nn.Module):
    def __init__(self, d, n_heads, n_kv):
        super().__init__()
        self.h, self.kv, self.hd = n_heads, n_kv, d // n_heads
        self.wq, self.wk, self.wv = (nn.Linear(d, self.hd * n, bias=False) for n in (n_heads, n_kv, n_kv))
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, n, _ = x.shape
        q = self.wq(x).view(B, n, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, n, self.kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, n, self.kv, self.hd).transpose(1, 2)
        q, k = rope(q), rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)   # each KV head serves h/kv query heads
        return self.wo(o.transpose(1, 2).reshape(B, n, -1))


class LlamaBlock(nn.Module):
    def __init__(self, d, n_heads=4, n_kv=2, mlp=None):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = GQA(d, n_heads, n_kv)
        self.mlp = mlp if mlp is not None else SwiGLU(d, int(8 * d / 3 / 16 + 0.5) * 16)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class GPT2Block(nn.Module):
    def __init__(self, d, n_heads=4, use_rope=False):
        super().__init__()
        self.h, self.ln1, self.ln2, self.use_rope = n_heads, nn.LayerNorm(d), nn.LayerNorm(d), use_rope
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, n, d = x.shape
        q, k, v = (t.view(B, n, self.h, d // self.h).transpose(1, 2) for t in self.qkv(self.ln1(x)).chunk(3, -1))
        if self.use_rope:
            q, k = rope(q), rope(k)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, n, d))
        return x + self.mlp(self.ln2(x))


x = torch.randn(2, 10, 64)
r = RMSNorm(64)(x)
assert torch.allclose(r.pow(2).mean(-1), torch.ones(2, 10), atol=1e-3)
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"one block at d = 512: GPT-2 {count(GPT2Block(512)):,} parameters; Llama-style (GQA 8 heads / 2 KV heads) "
      f"{count(LlamaBlock(512, 8, 2)):,}")
print("SwiGLU's three matrices at 8/3 d cost what GELU's two at 4d do; GQA shrinks K and V; no biases anywhere.")

# %% [markdown]
# ## 3. GPT-2 block vs Llama-style block, same budget

# %%
files = lesson_files()
train_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 != 4)
val_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 == 4)
tok = BPE.train(train_text, n_merges=1000)
V = tok.vocab_size
tr_ids, va_ids = torch.tensor(tok.encode(train_text)), torch.tensor(tok.encode(val_text))
CTX = 64


def batch(ids, bs, g):
    i = torch.randint(0, len(ids) - CTX - 1, (bs,), generator=g)
    return torch.stack([ids[j:j + CTX] for j in i]), torch.stack([ids[j + 1:j + CTX + 1] for j in i])


class LM(nn.Module):
    def __init__(self, make_block, d=128, L=4, learned_pos=True):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(CTX, d) if learned_pos else None                  # the Llama block uses RoPE instead
        self.blocks = nn.ModuleList(make_block() for _ in range(L))
        self.norm = nn.LayerNorm(d) if learned_pos else RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.aux = torch.tensor(0.0)

    def forward(self, idx):
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1]))
        aux = 0.0
        for b in self.blocks:
            x = b(x)
            aux = aux + getattr(b.mlp, "aux_loss", 0.0) if hasattr(b, "mlp") else aux
        self.aux = aux
        return self.head(self.norm(x))


def train_lm(model, steps=400, aux_weight=0.0, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 3e-3 * min(1, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        x, y = batch(tr_ids, 32, g)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)) + aux_weight * model.aux
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    ge = torch.Generator().manual_seed(1)
    with torch.no_grad():
        val = np.mean([F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item() for x, y in (batch(va_ids, 32, ge) for _ in range(20))])
    model.train()
    return val


t0 = time.time()
torch.manual_seed(0)
gpt2 = LM(lambda: GPT2Block(128))
torch.manual_seed(0)
gpt2_rope = LM(lambda: GPT2Block(128, use_rope=True), learned_pos=False)
torch.manual_seed(0)
llama = LM(lambda: LlamaBlock(128, 4, 2), learned_pos=False)
res = {"GPT-2 block (LayerNorm, learned positions, GELU, MHA)": (count(gpt2), train_lm(gpt2)),
       "GPT-2 block with RoPE instead of learned positions": (count(gpt2_rope), train_lm(gpt2_rope)),
       "Llama-style block (RMSNorm, RoPE, SwiGLU, GQA 4/2)": (count(llama), train_lm(llama))}
print(f"\n400 steps each, same data and schedule ({time.time() - t0:.0f}s):")
for k, (n, v) in res.items():
    print(f"  {k:52s} {n:9,} params   validation loss {v:.3f}")
v = [x[1] for x in res.values()]
print(f"swapping in RoPE alone gains {v[0] - v[1]:.3f} nats; the full Llama-style block gains {v[0] - v[2]:.3f}. At this tiny scale and")
print("short budget, learning 64 position embeddings from scratch is a real handicap, and a relative scheme is a head start.")
print(f"The other changes (RMSNorm, SwiGLU, GQA, no biases) are a wash here ({v[1] - v[2]:+.3f}) with 7% fewer parameters. At scale,")
print("published ablations find each worth a little, and GQA pays at inference (37.2). None of them is why one model is")
print("good and another isn't: that's data and scale.")
assert v[2] < v[0] and v[1] < v[0]

# %% [markdown]
# ## 4. Mixture of experts
#
# Replace the MLP with E experts (each a small SwiGLU) and a router that sends each token to its top 2. Total
# parameters grow with E; the computation per token doesn't. Without help, the router tends to favor a few experts,
# which then get better and are chosen more: collapse. The Switch/GShard load-balancing loss E * sum_i f_i * P_i
# (fraction of tokens routed to expert i times its mean router probability) pushes toward uniform use.

# %%
class MoE(nn.Module):
    def __init__(self, d, n_experts=8, top_k=2, hidden=96):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d, hidden) for _ in range(n_experts))
        self.k, self.E = top_k, n_experts
        self.aux_loss, self.load = torch.tensor(0.0), torch.zeros(n_experts)

    def forward(self, x):
        B, n, d = x.shape
        flat = x.reshape(-1, d)
        probs = self.router(flat).softmax(-1)
        topv, topi = probs.topk(self.k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e in range(self.E):
            rows, slot = (topi == e).nonzero(as_tuple=True)
            if len(rows):
                out[rows] += topv[rows, slot, None] * self.experts[e](flat[rows])
        frac = F.one_hot(topi, self.E).float().sum(1).mean(0) / self.k                  # f_i: share of routing slots
        self.aux_loss = self.E * (frac * probs.mean(0)).sum()                         # 1.0 when perfectly balanced
        self.load = frac.detach()
        return out.view(B, n, d)


def moe_lm():
    return LM(lambda: LlamaBlock(128, 4, 2, mlp=MoE(128)), learned_pos=False)


t0 = time.time()
moe_res = {}
for w in (0.0, 0.01):
    torch.manual_seed(0)
    m = moe_lm()
    val = train_lm(m, steps=300, aux_weight=w)
    loads = torch.stack([b.mlp.load for b in m.blocks])
    moe_res[w] = (val, loads)
total = count(moe_lm())
expert = count(SwiGLU(128, 96))
active = total - 4 * 8 * expert + 4 * 2 * expert
print(f"\nMoE, 8 experts, top-2 ({time.time() - t0:.0f}s): {total:,} total parameters, ~{active:,} active per token")
for w, (val, loads) in moe_res.items():
    busiest = loads.max(1).values.mean().item()
    idle = (loads < 0.02).float().sum(1).mean().item()
    print(f"  load-balancing weight {w:<5}: validation loss {val:.3f}; busiest expert gets {busiest:.0%} of routing slots "
          f"(uniform = 12.5%); experts with < 2%: {idle:.1f} per layer")
print("  layer 1 load with balancing:", " ".join(f"{x:.2f}" for x in moe_res[0.01][1][0].tolist()))
print("no dramatic collapse in this short run: with 8 experts and top-2 routing the router stays fairly even, and the")
print("balancing loss evens it further. Collapse bites harder with more experts, top-1 routing and long training, and the")
print("loss is cheap insurance. MoE buys more parameters at the same compute per token, if the router spreads the work.")
print("The price: memory for all experts, all-to-all communication across GPUs (33.2), and a routing problem with its own loss.")
assert moe_res[0.01][1].max(1).values.mean() < moe_res[0.0][1].max(1).values.mean()

print("\nAll checks passed.")
