# %% [markdown]
# # Lab 35.4: Scaling laws and compute budgets
#
# 1. Chinchilla's fitted loss L(N, D) = E + A / N^alpha + B / D^beta: the compute-optimal model for a budget.
# 2. Famous models against the frontier: undertrained, compute-optimal, and deliberately "overtrained".
# 3. When inference counts too: the optimum moves to smaller models trained longer.
# 4. A real (tiny) scaling experiment on this course, a power-law fit, and how far to trust its extrapolation.
# 5. "Emergence" from a smooth curve: per-token accuracy vs exact match on a 10-token answer.

# %%
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import curve_fit, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from bpe import BPE  # noqa: E402
from corpus import lesson_files, read_lesson  # noqa: E402

torch.manual_seed(354)
torch.set_num_threads(4)

# %% [markdown]
# ## 1. The Chinchilla loss surface
#
# Hoffmann et al. (2022), approach 3, fitted on their runs (loss in nats per token, their data and tokenizer):
# E = 1.69 (irreducible), A = 406.4, alpha = 0.34, B = 410.7, beta = 0.28. Training compute C ~ 6 N D.
# (Besiroglu et al., 2024, refit the same data and got somewhat different constants; the shape is what matters here.)

# %%
E, A, ALPHA, B, BETA = 1.69, 406.4, 0.34, 410.7, 0.28
loss = lambda N, D: E + A / N ** ALPHA + B / D ** BETA


def optimal(C):
    res = minimize_scalar(lambda logN: loss(math.exp(logN), C / (6 * math.exp(logN))), bounds=(math.log(1e6), math.log(1e13)), method="bounded")
    N = math.exp(res.x)
    return N, C / (6 * N), res.fun


print("compute (FLOPs)   optimal params   optimal tokens   tokens/param   predicted loss")
ratios = []
for C in (1e19, 1e21, 1e23, 1e25):
    N, D, L = optimal(C)
    ratios.append(D / N)
    print(f"{C:15.0e}   {N:14.3g}   {D:14.3g}   {D / N:12.1f}   {L:.3f}")
print("both N and D grow roughly as the square root of compute. But with these constants the ratio climbs from 32 to")
print("122 tokens per parameter, because alpha > beta. Chinchilla's other two methods gave a roughly constant ~20, which")
print("is the number everyone quotes; Besiroglu et al. (2024) found the published approach-3 fit was flawed, and their")
print("refit agrees with ~20. Scaling-law constants are estimates with error bars, from one data mix and one tokenizer.")
assert all(a_ < b_ for a_, b_ in zip(ratios, ratios[1:])) and 20 < ratios[0] < 40

# %% [markdown]
# ## 2. Real models against the frontier

# %%
models = [("GPT-3 (175B, 300B tokens)", 175e9, 300e9), ("Chinchilla (70B, 1.4T)", 70e9, 1.4e12),
          ("Llama 2 7B (2T)", 7e9, 2e12), ("Llama 3 8B (15T)", 8e9, 15e12)]
print("\nmodel                          compute     tokens/param   predicted loss   optimal loss at same compute   optimal N")
for name, N, D in models:
    C = 6 * N * D
    No, Do, Lo = optimal(C)
    print(f"{name:30s} {C:9.2e}   {D / N:12.0f}   {loss(N, D):14.3f}   {Lo:28.3f}   {No:9.3g}")
print("GPT-3 was far too big for its data: Chinchilla, a quarter of the size on 4.7x the tokens, beat it. Llama 3 8B is")
print("the opposite: nearly 2,000 tokens per parameter, a worse use of training compute, on purpose. A model is trained")
print("once and served billions of times, and a smaller model is cheaper to serve.")
g3 = models[0]
assert loss(g3[1], g3[2]) > optimal(6 * g3[1] * g3[2])[2] + 0.02

# %% [markdown]
# ## 3. Counting inference
#
# Total cost = training 6 N D + inference 2 N D_inf (a forward pass per generated or processed token). For a target loss,
# find the (N, D) that reaches it at minimum total cost.

# %%
def cheapest(target, D_inf):
    best = None
    for N in np.logspace(8, 11.5, 400):
        rest = target - E - A / N ** ALPHA
        if rest <= 0:
            continue
        D = (B / rest) ** (1 / BETA)
        cost = 6 * N * D + 2 * N * D_inf
        if best is None or cost < best[0]:
            best = (cost, N, D)
    return best


target = loss(7e9, 2e12)                                                               # the quality of Llama 2 7B
print(f"\nreach a loss of {target:.3f} at minimum total cost:")
print("   lifetime inference tokens   params    training tokens   tokens/param")
inf_rows = []
for D_inf in (0, 1e12, 1e13, 1e14):
    cost, N, D = cheapest(target, D_inf)
    inf_rows.append(N)
    print(f"   {D_inf:25.0e}   {N:7.3g}   {D:15.3g}   {D / N:12.0f}")
print("the more the model will be used, the smaller and longer-trained it should be (Sardana et al., 2023).")
assert inf_rows[-1] < inf_rows[0]

# %% [markdown]
# ## 4. A tiny scaling experiment
#
# Four GPTs of increasing width, each trained for the same number of steps on this course (so bigger models also get
# more compute). Fit L(N) = E + A / N^alpha to the four validation losses, then ask what the fit predicts for a model
# 30x larger, and how much that prediction moves under bootstrap resampling of the evaluation batches.

# %%
files = lesson_files()
train_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 != 4)
val_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 == 4)
tok = BPE.train(train_text, n_merges=500)
V = tok.vocab_size
tr_ids = torch.tensor(tok.encode(train_text))
va_ids = torch.tensor(tok.encode(val_text))
CTX = 64


def batch(ids, bs, g):
    i = torch.randint(0, len(ids) - CTX - 1, (bs,), generator=g)
    return torch.stack([ids[j:j + CTX] for j in i]), torch.stack([ids[j + 1:j + CTX + 1] for j in i])


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B_, n, d = x.shape
        q, k, v = (t.view(B_, n, self.h, d // self.h).transpose(1, 2) for t in self.qkv(self.ln1(x)).chunk(3, -1))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B_, n, d))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, d, L=2, h=4):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def run(d, steps=500):
    torch.manual_seed(0)
    m = GPT(d)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.1)
    g = torch.Generator().manual_seed(0)
    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 3e-3 * min(1, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        x, y = batch(tr_ids, 32, g)
        l_ = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); l_.backward(); opt.step()
    m.eval()
    ge = torch.Generator().manual_seed(1)
    with torch.no_grad():
        per_batch = [F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item() for x, y in (batch(va_ids, 64, ge) for _ in range(30))]
    n_nonemb = sum(p.numel() for n, p in m.named_parameters() if "blocks" in n)
    return n_nonemb, np.array(per_batch)


t0 = time.time()
runs = [run(d) for d in (16, 32, 64, 128)]
Ns = np.array([r[0] for r in runs], dtype=float)
print(f"\ntrained 4 models in {time.time() - t0:.0f}s")
for (n, pb) in runs:
    print(f"  non-embedding params {n:8,}   validation loss {pb.mean():.3f}")
power = lambda N, E_, A_, a_: E_ + A_ * N ** (-a_)


def fit(losses):
    p, _ = curve_fit(power, Ns, losses, p0=(3.0, 30.0, 0.3), bounds=([0, 0, 0.01], [10, 1e6, 2]), maxfev=20000)
    return p


p = fit(np.array([r[1].mean() for r in runs]))
N_big = Ns[-1] * 30
print(f"fit: L(N) = {p[0]:.2f} + {p[1]:.1f} N^-{p[2]:.2f};  prediction for {N_big:,.0f} params: {power(N_big, *p):.3f}")
rs = np.random.default_rng(0)
preds, irreducible = [], []
for _ in range(200):                                                                  # resample the evaluation batches
    losses = np.array([r[1][rs.integers(0, 30, 30)].mean() for r in runs])
    try:
        pb = fit(losses)
        preds.append(power(N_big, *pb)); irreducible.append(pb[0])
    except RuntimeError:
        pass
lo, hi = np.percentile(preds, [5, 95])
print(f"bootstrap 90% interval for that prediction: {lo:.3f} to {hi:.3f}; for the 'irreducible' loss E: "
      f"{np.percentile(irreducible, 5):.2f} to {np.percentile(irreducible, 95):.2f}")
print("four points over one order of magnitude pin down the trend nearby and not much else; E, which dominates any")
print("long extrapolation, is the least constrained parameter. Real scaling studies use dozens of runs over several")
print("orders of magnitude, and still get the constants wrong enough to argue about (Besiroglu et al., 2024).")
losses_mean = [r[1].mean() for r in runs]
assert all(a > b for a, b in zip(losses_mean, losses_mean[1:]))
assert hi - lo > 0

# %% [markdown]
# ## 5. Emergence, or a metric with a cliff
#
# Suppose per-token accuracy p improves smoothly with log compute. A task scored by exact match on a 10-token answer
# succeeds with probability p^10 (if errors are independent). Plot both against compute.

# %%
logC = np.linspace(18, 26, 9)
p_tok = 1 / (1 + np.exp(-(logC - 22) * 0.9))                                             # a smooth S-curve
print("\nlog10 compute   per-token accuracy   exact match, 10 tokens")
for c, p_ in zip(logC, p_tok):
    print(f"{c:13.0f}   {p_:18.3f}   {p_ ** 10:22.3f}   " + "#" * int(40 * p_ ** 10))
print("the same smooth improvement looks like a sudden ability 'emerging' at 10^23 under a harsh metric. Schaeffer et")
print("al. (2023) showed many reported emergent abilities behave this way; some real discontinuities may remain, but")
print("the metric deserves suspicion first.")
jumps = np.diff(p_tok ** 10)
assert jumps.argmax() > np.diff(p_tok).argmax()

print("\nAll checks passed.")
