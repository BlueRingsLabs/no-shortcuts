# %% [markdown]
# # Lab 29.3: Positions and efficient attention
#
# 1. Sinusoidal encodings: a shift in position is a rotation.
# 2. RoPE: rotate queries and keys, and the score depends only on the distance between them.
# 3. Length extrapolation: train at 64 tokens, test at 64, 128 and 256, with learned absolute positions, sinusoidal,
#    RoPE, ALiBi and no positions at all.
# 4. FlashAttention's core trick: tiled attention with an online softmax, exact, without the n x n matrix.
# 5. Linear attention is an RNN.
# 6. KV-cache arithmetic: multi-head vs grouped-query vs multi-query attention.

# %%
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(293)
torch.set_num_threads(4)

# %% [markdown]
# ## 1. Sinusoidal encodings

# %%
def sinusoidal(n, d, base=10000.0):
    pos = torch.arange(n, dtype=torch.float64)[:, None]
    freq = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)       # d/2 frequencies, from 1 down to ~1/base
    pe = torch.zeros(n, d, dtype=torch.float64)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe, freq


pe, freq = sinusoidal(200, 16)
k = 7                                                                        # shift by 7 positions
R = torch.zeros(16, 16, dtype=torch.float64)                                 # block-diagonal rotation, one 2x2 block per frequency
for i, w in enumerate(freq):
    c, s = math.cos(k * w), math.sin(k * w)
    R[2 * i:2 * i + 2, 2 * i:2 * i + 2] = torch.tensor([[c, s], [-s, c]], dtype=torch.float64)
err = (pe[k:] - pe[:-k] @ R.T).abs().max().item()
print(f"PE(p + {k}) = R_{k} PE(p) for every p, with R_{k} a rotation that doesn't depend on p: max error {err:.1e}")
assert err < 1e-12
dots = pe[50] @ pe.T
print("similarity of position 50's encoding with its neighbors:", np.round(dots[[45, 48, 50, 52, 55, 80, 150]].numpy(), 2))

# %% [markdown]
# ## 2. RoPE
#
# Rotate each pair of dimensions of q (at position m) and k (at position n) by angles m * theta_i and n * theta_i.
# Their dot product then depends on m - n only: rotations compose, R(m)^T R(n) = R(n - m).

# %%
def rope(x, pos, base=10000.0):
    """x (..., n, d) with d even; rotates pairs (x_2i, x_2i+1) by pos * theta_i."""
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)
    ang = pos[:, None].to(x.dtype) * theta                                   # (n, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = x1 * cos - x2 * sin, x1 * sin + x2 * cos
    return out


q, kk = torch.randn(1, 32, dtype=torch.float64), torch.randn(1, 32, dtype=torch.float64)
scores = {}
for m, n in ((10, 3), (110, 103), (1010, 1003), (3, 10)):
    scores[(m, n)] = (rope(q, torch.tensor([m])) @ rope(kk, torch.tensor([n])).T).item()
print(f"\nRoPE scores for the same q and k: at (10, 3) {scores[(10, 3)]:.6f}, (110, 103) {scores[(110, 103)]:.6f}, "
      f"(1010, 1003) {scores[(1010, 1003)]:.6f}, (3, 10) {scores[(3, 10)]:.6f}")
print("same distance, same score, wherever it is. Opposite direction, different score: RoPE knows before from after.")
assert abs(scores[(10, 3)] - scores[(1010, 1003)]) < 1e-9 and abs(scores[(10, 3)] - scores[(3, 10)]) > 1e-3
assert torch.allclose(rope(q, torch.tensor([5])).norm(), q.norm())          # a rotation: no change in length

# %% [markdown]
# ## 3. Length extrapolation
#
# Lab 29.2's induction task (a random segment repeated until the sequence is full), which needs a previous-token head:
# attention by *relative* position. Two-layer attention-only models, trained on length 64, tested on longer sequences.

# %%
def repeat_batch(n, N, g, V=64):
    x = torch.empty(n, N, dtype=torch.long)
    pred = torch.zeros(n, N, dtype=torch.bool)
    for i in range(n):
        L = int(torch.randint(8, 25, (1,), generator=g))
        x[i] = torch.randint(0, V, (L,), generator=g).repeat(N // L + 1)[:N]
        pred[i, L:] = True
    return x, pred


class PosAttention(nn.Module):
    def __init__(self, d, h, kind):
        super().__init__()
        self.h, self.kind, self.qkv, self.proj = h, kind, nn.Linear(d, 3 * d), nn.Linear(d, d)
        if kind == "alibi":                                                  # head-specific slopes 2^(-8/h), 2^(-16/h), ...
            self.register_buffer("slopes", torch.tensor([2 ** (-8 * (i + 1) / h) for i in range(h)]))

    def forward(self, x):
        B, n, d = x.shape
        q, k, v = (t.view(B, n, self.h, d // self.h).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))
        pos = torch.arange(n)
        if self.kind == "rope":
            q, k = rope(q, pos), rope(k, pos)
        causal = torch.ones(n, n, dtype=torch.bool).tril()
        bias = torch.zeros(n, n).masked_fill(~causal, float("-inf"))
        if self.kind == "alibi":                                            # -slope * distance, added to the scores
            bias = bias - self.slopes[:, None, None] * (pos[:, None] - pos[None, :]).clamp_min(0)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(o.transpose(1, 2).reshape(B, n, d))


class PosModel(nn.Module):
    def __init__(self, kind, V=64, d=64, h=4, L=2, max_len=64):
        super().__init__()
        self.kind, self.tok = kind, nn.Embedding(V, d)
        if kind == "learned":
            self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList(PosAttention(d, h, kind) for _ in range(L))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(L))
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)

    def forward(self, idx):
        n = idx.shape[1]
        x = self.tok(idx)
        if self.kind == "learned":
            x = x + self.pos(torch.arange(n))                                # fails beyond max_len: there is no row for it
        elif self.kind == "sinusoidal":
            x = x + sinusoidal(n, x.shape[-1])[0].float()
        for ln, att in zip(self.norms, self.layers):
            x = x + att(ln(x))
        return self.head(self.ln_f(x))


def train_pos(kind, steps=600, seed=0):
    torch.manual_seed(seed)
    m = PosModel(kind)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 3e-3 * min(1.0, (step + 1) / 100)
        x, _ = repeat_batch(64, 64, g)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, 64), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def acc_at(m, N):
    x, pred = repeat_batch(300, N + 1, torch.Generator().manual_seed(7))
    try:
        hit = m.eval()(x[:, :-1]).argmax(-1) == x[:, 1:]
    except IndexError:
        return float("nan")
    return hit[pred[:, 1:]].float().mean().item()


print("\naccuracy on the repeated tokens, trained at length 64 (nan: the model can't even run at that length)")
print("  positions      64      128     256")
ext = {}
t0 = time.time()
for kind in ("learned", "sinusoidal", "rope", "alibi", "none"):
    m = train_pos(kind)
    ext[kind] = [acc_at(m, N) for N in (64, 128, 256)]
    print(f"  {kind:10s} " + "  ".join(f"{a:6.3f}" for a in ext[kind]))
print(f"({time.time() - t0:.0f}s)")
print("learned absolute positions have no row for position 64: the model can't run. Sinusoidal runs and decays. RoPE holds")
print("at twice the length and slips at four times. ALiBi's accuracy rises (more repeats to copy from), since its bias")
print("only knows distance. No positions: the causal mask lets a model infer some order, not the exact previous token.")
assert all(ext[k][0] > 0.85 for k in ("learned", "sinusoidal", "rope", "alibi"))
assert math.isnan(ext["learned"][1]) and ext["none"][0] < 0.6
assert ext["rope"][1] > 0.9 and ext["alibi"][2] > 0.9 and ext["sinusoidal"][2] < ext["sinusoidal"][0]

# %% [markdown]
# ## 4. Tiled attention with an online softmax
#
# softmax(s) v needs the max and the sum over the whole row, but they can be accumulated block by block: keep a running
# max m, a running denominator l and a running output o; when a new block raises the max, rescale what you have.
# FlashAttention does exactly this in on-chip memory, so the n x n matrix never exists in GPU memory.

# %%
def tiled_attention(q, k, v, block=64):
    n, d = q.shape
    out = torch.empty_like(q)
    for i in range(0, n, block):                                             # a block of queries
        qi = q[i:i + block]
        m_i = torch.full((len(qi), 1), float("-inf"), dtype=q.dtype)
        l_i = torch.zeros(len(qi), 1, dtype=q.dtype)
        o_i = torch.zeros(len(qi), v.shape[1], dtype=q.dtype)
        for j in range(0, i + block, block):                                 # causal: only key blocks up to the diagonal
            s = qi @ k[j:j + block].T / math.sqrt(d)
            qpos, kpos = torch.arange(i, i + len(qi))[:, None], torch.arange(j, j + len(s[0]))[None]
            s = s.masked_fill(kpos > qpos, float("-inf"))
            m_new = torch.maximum(m_i, s.max(1, keepdim=True).values)
            p = torch.exp(s - m_new)
            scale = torch.exp(m_i - m_new)                                   # rescale the old partial sums to the new max
            l_i = l_i * scale + p.sum(1, keepdim=True)
            o_i = o_i * scale + p @ v[j:j + block]
            m_i = m_new
        out[i:i + block] = o_i / l_i
    return out


n, d = 1024, 64
q, kk, v = (torch.randn(n, d, dtype=torch.float64) for _ in range(3))
ref = F.scaled_dot_product_attention(q[None], kk[None], v[None], is_causal=True)[0]
tiled = tiled_attention(q, kk, v)
print(f"\ntiled attention vs the full computation, n = {n}: max difference {(tiled - ref).abs().max():.1e} (exact, not an approximation)")
print(f"largest intermediate: full scores {n * n:,} numbers, tiled {64 * 64:,} per block")
assert (tiled - ref).abs().max() < 1e-10

# %% [markdown]
# ## 5. Linear attention is an RNN
#
# Replace exp(q.k) with phi(q).phi(k) for a feature map phi (here elu + 1). Then the causal output at step t is
# phi(q_t)^T S_t / phi(q_t)^T z_t with S_t = sum_{j<=t} phi(k_j) v_j^T and z_t = sum_{j<=t} phi(k_j): a fixed-size
# state updated once per token. Linear time, constant memory per step, and a different model: it is not softmax attention.

# %%
phi = lambda x: F.elu(x) + 1


def linear_attention_parallel(q, k, v):
    A = (phi(q) @ phi(k).T).tril()                                           # n x n, masked; O(n^2) like softmax attention
    return (A @ v) / A.sum(1, keepdim=True)


def linear_attention_recurrent(q, k, v):
    S, z, out = torch.zeros(q.shape[1], v.shape[1], dtype=q.dtype), torch.zeros(q.shape[1], dtype=q.dtype), []
    for t in range(len(q)):
        S = S + torch.outer(phi(k[t]), v[t]); z = z + phi(k[t])              # the state: d x d_v, whatever t is
        out.append(phi(q[t]) @ S / (phi(q[t]) @ z))
    return torch.stack(out)


qs, ks, vs = (torch.randn(200, 16, dtype=torch.float64) for _ in range(3))
diff = (linear_attention_parallel(qs, ks, vs) - linear_attention_recurrent(qs, ks, vs)).abs().max().item()
print(f"\nlinear attention, parallel form vs recurrent form: max difference {diff:.1e}")
assert diff < 1e-10
print("the state is 16 x 16 numbers whether the sequence has 200 tokens or 2 million: that's the appeal, and the limit,")
print("since everything the model remembers has to fit in it. Softmax attention keeps every key and value (the KV cache).")

# %% [markdown]
# ## 6. KV-cache arithmetic
#
# At generation time each layer keeps K and V for every previous token. Bytes = 2 (K and V) x layers x kv_heads x
# head_dim x tokens x bytes per number. Numbers for a Llama-2-70B-like shape: 80 layers, 64 query heads of dim 128.

# %%
layers, heads, hd, bytes_ = 80, 64, 128, 2
print("\nKV cache for one sequence of 32,768 tokens in bf16:")
for name, kv_heads in (("multi-head (64 KV heads)", 64), ("grouped-query (8 KV heads)", 8), ("multi-query (1 KV head)", 1)):
    gb = 2 * layers * kv_heads * hd * 32768 * bytes_ / 1e9
    print(f"  {name:28s} {gb:6.1f} GB")
print("the weights of a 70B model in bf16 are 140 GB. With full multi-head attention, a few long sequences would need")
print("more memory than the weights. That's why GQA is standard (35.5, 37.2).")

# %% [markdown]
# GQA in code: each group of query heads shares one K/V head. F.scaled_dot_product_attention does it with enable_gqa;
# written out, it's repeat_interleave.

# %%
B, n, hq, hkv, hd_ = 2, 10, 8, 2, 16
q_, k_, v_ = torch.randn(B, hq, n, hd_), torch.randn(B, hkv, n, hd_), torch.randn(B, hkv, n, hd_)
mine = F.scaled_dot_product_attention(q_, k_.repeat_interleave(hq // hkv, 1), v_.repeat_interleave(hq // hkv, 1), is_causal=True)
ref = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True, enable_gqa=True)
assert torch.allclose(mine, ref, atol=1e-6)
print("GQA via repeat_interleave matches enable_gqa=True")

print("\nAll checks passed.")
