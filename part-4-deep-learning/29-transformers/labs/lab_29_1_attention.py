# %% [markdown]
# # Lab 29.1: Attention, written out and broken on purpose
#
# 1. Scaled dot-product attention from scratch, with padding and causal masks, matched against F.scaled_dot_product_attention.
# 2. Multi-head attention from scratch, matched against nn.MultiheadAttention.
# 3. Why sqrt(d_k): softmax entropy and gradient size as the head dimension grows.
# 4. Mask mistakes: masking after the softmax, and rows with nothing to attend to.
# 5. Permutation equivariance.
# 6. Nadaraya-Watson kernel regression is attention with a fixed similarity.
# 7. A causal mask that leaks one position: a beautiful training loss and a useless model.

# %%
import math
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(291)
torch.set_num_threads(4)


def attention(q, k, v, allowed=None):
    """q (..., m, d_k), k (..., n, d_k), v (..., n, d_v); allowed: bool (..., m, n), True = may attend."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))      # before the softmax
    w = scores.softmax(-1)
    return w @ v, w


# %% [markdown]
# ## 1. The line, and its masks

# %%
B, n, dk, dv = 2, 6, 8, 5
q, k, v = torch.randn(B, n, dk), torch.randn(B, n, dk), torch.randn(B, n, dv)
out, w = attention(q, k, v)
assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-6)
assert torch.allclose(w.sum(-1), torch.ones(B, n))

causal = torch.ones(n, n, dtype=torch.bool).tril()
print("causal mask (True = may attend), exercise 3 with n = 6:\n", causal.int())
out_c, w_c = attention(q, k, v, causal)
assert torch.allclose(out_c, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-6)
assert (w_c.triu(1) == 0).all()

lens = torch.tensor([6, 4])                                        # the second sequence has 2 padding positions
key_ok = torch.arange(n)[None] < lens[:, None]                      # (B, n)
pad_allowed = key_ok[:, None, :].expand(B, n, n)
out_p, w_p = attention(q, k, v, pad_allowed)
assert torch.allclose(out_p, F.scaled_dot_product_attention(q, k, v, attn_mask=pad_allowed), atol=1e-6)
assert (w_p[1, :, 4:] == 0).all()
print("hand-written attention matches F.scaled_dot_product_attention: no mask, causal, padding")

# exercise 1
q1, K1, V1 = torch.tensor([[1.0, 0]]), torch.tensor([[1.0, 0], [0, 1], [-1, 0]]), torch.tensor([[1.0], [2], [3]])
o1, w1 = attention(q1, K1, V1)
print(f"exercise 1: weights {np.round(w1[0].numpy(), 3)}, output {o1.item():.3f}")

# %% [markdown]
# ## 2. Multi-head attention

# %%
class MHA(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, allowed=None):
        B, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        split = lambda t: t.view(B, n, self.h, d // self.h).transpose(1, 2)   # (B, h, n, d/h): heads become a batch dim
        o, w = attention(split(q), split(k), split(v), allowed)
        return self.proj(o.transpose(1, 2).reshape(B, n, d)), w


d, h = 32, 4
mine = MHA(d, h)
ref = nn.MultiheadAttention(d, h, batch_first=True)
with torch.no_grad():                                                # nn.MultiheadAttention stores W_Q, W_K, W_V stacked the same way
    ref.in_proj_weight.copy_(mine.qkv.weight); ref.in_proj_bias.copy_(mine.qkv.bias)
    ref.out_proj.weight.copy_(mine.proj.weight); ref.out_proj.bias.copy_(mine.proj.bias)
x = torch.randn(3, 10, d)
o_mine, w_mine = mine(x)
o_ref, w_ref = ref(x, x, x, need_weights=True, average_attn_weights=False)
assert torch.allclose(o_mine, o_ref, atol=1e-5) and torch.allclose(w_mine, w_ref, atol=1e-5)
key_padding = torch.zeros(3, 10, dtype=torch.bool); key_padding[0, 7:] = True    # nn.MultiheadAttention: True = IGNORE
o_ref_p, _ = ref(x, x, x, key_padding_mask=key_padding)
o_mine_p, _ = mine(x, (~key_padding)[:, None, None, :])                          # ours: True = may attend
assert torch.allclose(o_mine_p, o_ref_p, atol=1e-5)
print("multi-head attention matches nn.MultiheadAttention, including padding (with the opposite boolean convention)")

# %% [markdown]
# ## 3. Why divide by sqrt(d_k)
#
# Random queries and keys with unit-variance components; 64 keys per query. Entropy of the attention weights (log 64 = 4.16
# is uniform, 0 is one-hot), the largest weight, and the gradient that gets through the softmax to the scores (median
# over queries: the typical query, not the lucky one sitting on a near-tie).

# %%
print("\n d_k   scaled?   entropy   max weight   median |dL/dscores|")
stats = {}
for dk_ in (16, 64, 256, 1024):
    for scaled in (True, False):
        g = torch.Generator().manual_seed(dk_)
        qq = torch.randn(512, 1, dk_, generator=g).requires_grad_()
        kk, vv = torch.randn(512, 64, dk_, generator=g), torch.randn(512, 64, 16, generator=g)
        s = qq @ kk.transpose(-2, -1) / (math.sqrt(dk_) if scaled else 1.0)
        s.retain_grad()
        w_ = s.softmax(-1)
        (w_ @ vv).square().sum().backward()
        ent = -(w_ * w_.clamp_min(1e-30).log()).sum(-1).mean().item()
        stats[(dk_, scaled)] = (ent, w_.max(-1).values.mean().item(), s.grad.norm(dim=-1).median().item())
        print(f"{dk_:5d}   {str(scaled):7s}   {ent:7.3f}   {stats[(dk_, scaled)][1]:10.3f}   {stats[(dk_, scaled)][2]:12.2e}")
print("scaled, nothing depends on d_k. Unscaled, the softmax turns one-hot as d_k grows. Its gradient first gets larger")
print("(bigger logits, sharper softmax) and then collapses once the softmax saturates: at d_k = 1024 the typical query")
print("gets about 40 times less than the scaled version. Either way, the learning rate you'd need depends on the head size.")
assert abs(stats[(16, True)][0] - stats[(1024, True)][0]) < 0.1
assert stats[(1024, False)][1] > 0.9 and stats[(1024, False)][0] < 0.3
assert stats[(1024, False)][2] < stats[(1024, True)][2] / 10

# %% [markdown]
# ## 4. Mask mistakes

# %%
w_after = torch.randn(1, n, n).softmax(-1) * causal                # mask applied AFTER the softmax
print(f"\nmasking after the softmax: row sums {np.round(w_after[0].sum(-1).numpy(), 3)} (they should all be 1)")
assert not torch.allclose(w_after.sum(-1), torch.ones(1, n))

nothing = torch.zeros(1, 2, n, dtype=torch.bool); nothing[0, 0, :3] = True       # row 1 may attend to nothing
o_nan, w_nan = attention(q[:1, :2], k[:1], v[:1], nothing)
print(f"a query with nothing to attend to: weights {w_nan[0, 1, :3].tolist()} -> output contains NaN: {torch.isnan(o_nan).any().item()}")
print("softmax of all -inf is 0/0. It happens with left padding and packed sequences; one NaN row poisons the whole batch's gradient.")
assert torch.isnan(o_nan[0, 1]).all() and not torch.isnan(o_nan[0, 0]).any()

# %% [markdown]
# ## 5. Permutation equivariance

# %%
perm = torch.randperm(10)
o_perm, _ = mine(x[:, perm])
assert torch.allclose(o_perm, o_mine[:, perm], atol=1e-5)
print("\nshuffle the tokens and the outputs shuffle the same way: self-attention sees a set, not a sequence (29.3 fixes that)")

# %% [markdown]
# ## 6. Nadaraya-Watson is attention
#
# Kernel regression with a Gaussian kernel: prediction at x = sum_j K(x, x_j) y_j / sum_j K(x, x_j).
# exp(-(x - x_j)^2 / (2 b^2)) = exp(x x_j / b^2) * (terms in x alone, which cancel) * exp(-x_j^2 / (2 b^2)). So it's a
# softmax over q.k with q = x / b, k = x_j / b, plus a per-key bias -x_j^2 / (2 b^2). Here it's written as attention
# on squared distances directly.

# %%
g = torch.Generator().manual_seed(0)
xs = torch.rand(200, generator=g) * 6
ys = torch.sin(xs) + 0.2 * torch.randn(200, generator=g)
xq = torch.linspace(0, 6, 50)
bw = 0.3
K_ = torch.exp(-(xq[:, None] - xs[None]) ** 2 / (2 * bw ** 2))
nw = (K_ @ ys) / K_.sum(1)                                           # the textbook formula
att = (-(xq[:, None] - xs[None]) ** 2 / (2 * bw ** 2)).softmax(-1) @ ys   # scores = negative squared distance
qk = ((xq[:, None] * xs[None]) / bw ** 2 - xs[None] ** 2 / (2 * bw ** 2)).softmax(-1) @ ys   # dot product + key bias
assert torch.allclose(nw, att, atol=1e-5) and torch.allclose(nw, qk, atol=1e-4)
print(f"\nNadaraya-Watson = softmax attention (max difference {max((nw - att).abs().max(), (nw - qk).abs().max()):.1e}); "
      f"RMSE against sin(x): {((nw - torch.sin(xq)) ** 2).mean().sqrt():.3f}")
print("a transformer learns the queries, keys and values; the kernel smoother fixes them. Same machine.")

# %% [markdown]
# ## 7. A leaky causal mask
#
# A one-layer attention language model on characters from this course (Part III's lessons). Two copies, identical except
# for the mask: one causal, one that also lets position i see position i + 1, the token it's supposed to predict.

# %%
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "part-5-language-models" / "_shared"))
from corpus import lesson_files, read_lesson  # noqa: E402  (the frozen course text, tools/snapshot_course.py)
text = "\n".join(read_lesson(p) for p in lesson_files(("part-3-classical-ml",)))
chars = sorted(set(text)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
split = int(0.9 * len(data)); train_d, val_d = data[:split], data[split:]
CTX = 64


class TinyAttnLM(nn.Module):
    def __init__(self, leak, d=128, h=4):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(CTX, d)    # learned positions (29.3)
        self.attn, self.ln1, self.ln2 = MHA(d, h), nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.out = nn.Linear(d, V)
        self.leak = leak

    def forward(self, idx):
        n = idx.shape[1]
        allowed = torch.ones(n, n, dtype=torch.bool).tril(1 if self.leak else 0)   # tril(1): one position too many
        x = self.emb(idx) + self.pos(torch.arange(n))
        x = x + self.attn(self.ln1(x), allowed)[0]
        x = x + self.mlp(self.ln2(x))
        return self.out(x)


def batch(d_, bs=64, g=None):
    i = torch.randint(0, len(d_) - CTX - 1, (bs,), generator=g)
    return torch.stack([d_[j:j + CTX] for j in i]), torch.stack([d_[j + 1:j + CTX + 1] for j in i])


def train_lm(leak, steps=600):
    torch.manual_seed(0)
    m = TinyAttnLM(leak)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for _ in range(steps):
        xb, yb = batch(train_d)
        loss = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def losses(m):
    """Validation loss at every position (where the leak is available), and at the last position of a window cut to
    32 characters: there is no position i + 1 to peek at, exactly the situation at generation time."""
    g = torch.Generator().manual_seed(1)
    xb, yb = batch(val_d, bs=512, g=g)
    m.eval()
    all_pos = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1)).item()
    last = F.cross_entropy(m(xb[:, :32])[:, -1], yb[:, 31]).item()
    return all_pos, last


t0 = time.time()
honest, leaky = train_lm(False), train_lm(True)
(h_all, h_last), (l_all, l_last) = losses(honest), losses(leaky)
print(f"\ntrained both in {time.time() - t0:.0f}s. Validation cross-entropy (nats per character):")
print(f"  causal mask:  all positions {h_all:.3f}   at the last position (no future to peek at) {h_last:.3f}")
print(f"  leaky mask:   all positions {l_all:.3f}   at the last position (no future to peek at) {l_last:.3f}")
print(f"  unigram baseline: {-(torch.bincount(train_d, minlength=V).double().add(1).div(len(train_d) + V)).log()[val_d].mean():.3f}")

# the test from exercise 6: change a future token, see whether earlier outputs move
xb, _ = batch(val_d, bs=8, g=torch.Generator().manual_seed(2))
xb2 = xb.clone(); xb2[:, 40] = (xb2[:, 40] + 1) % V
with torch.no_grad():
    moved_h = (honest(xb)[:, :40] - honest(xb2)[:, :40]).abs().max().item()
    moved_l = (leaky(xb)[:, :40] - leaky(xb2)[:, :40]).abs().max().item()
print(f"change token 40, largest change in the outputs at positions 0-39: causal {moved_h:.1e}, leaky {moved_l:.2f}")


@torch.no_grad()
def generate(m, prompt, n_new=120):
    idx = torch.tensor([[stoi[c] for c in prompt]])
    for _ in range(n_new):
        nxt = m(idx[:, -CTX:])[0, -1].argmax()
        idx = torch.cat([idx, nxt.view(1, 1)], 1)
    return "".join(chars[i] for i in idx[0])


print("\ngreedy generation, causal:\n  " + repr(generate(honest, "The model ")))
print("greedy generation, leaky:\n  " + repr(generate(leaky, "The model ")))
print("the leaky model learned to copy the answer it could see. At generation time there's nothing to copy.")
assert l_all < h_all - 0.5                                           # the leak looks like a breakthrough
assert l_last > h_last + 0.3                                         # and is one when it matters
assert moved_h < 1e-5 and moved_l > 0.1

print("\nAll checks passed.")
