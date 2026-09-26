# %% [markdown]
# # Lab 07.2: Making floating point misbehave, then fixing it
#
# 1. Representability: 0.1 + 0.2, the float32 counter that stops counting, epsilons per dtype.
# 2. NaN is contagious.
# 3. Catastrophic cancellation: the naive variance goes negative; Welford doesn't.
# 4. Overflow/underflow: logsumexp, log-softmax, stable softplus and BCE-with-logits.
# 5. Summation: naive float32 vs pairwise vs Kahan vs float64, and non-associativity.
# 6. float16 vs bfloat16: range vs precision (with torch).

# %%
import math

import numpy as np
import torch
from scipy.special import logsumexp

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Representability

# %%
assert 0.1 + 0.2 != 0.3 and math.isclose(0.1 + 0.2, 0.3)
big = np.float32(2**24)
assert big + np.float32(1) == big, "float32 can't represent 2^24 + 1"
print("float32 spacing at 2^24:", np.spacing(np.float32(2**24)), "   at 1.7e9:", np.spacing(np.float32(1.7e9)))
for dt in (np.float64, np.float32, np.float16):
    fi = np.finfo(dt)
    print(f"{dt.__name__:8s} eps={fi.eps:.2e}  max={fi.max:.2e}  smallest normal={fi.tiny:.2e}")
assert np.finfo(np.float16).max == 65504

# %% [markdown]
# ## 2. NaN

# %%
with np.errstate(invalid="ignore", divide="ignore"):
    x = np.array([1.0, 2.0, np.nan, 4.0])
    assert np.isnan(x.sum()) and np.isnan(x.mean()) and np.nanmean(x) == 7 / 3
    assert np.isnan(np.float64(0) / 0) and np.isnan(np.inf - np.inf) and np.nan != np.nan
w = np.ones(3)
grad = np.array([0.1, np.nan, 0.2])
w -= 0.01 * grad
assert np.isnan(w[1]), "one NaN gradient, and that weight is gone for good"
try:
    with np.errstate(invalid="raise"):
        np.float64(0) / np.float64(0)
    raise RuntimeError("should have raised")
except FloatingPointError:
    print("np.errstate(invalid='raise') pinpoints the first NaN")

# %% [markdown]
# ## 3. Cancellation: variance of data with a large mean

# %%
t = (1.7e9 + rng.uniform(0, 5, 100_000)).astype(np.float64)    # timestamps with a few seconds of spread
true_var = np.var(t - t[0])                                      # shift first: exact enough

naive = np.mean(t**2) - np.mean(t) ** 2                          # E[x^2] - E[x]^2 in float64
t32 = t.astype(np.float32)
naive32 = float(np.mean(t32.astype(np.float64) ** 2) - np.mean(t32.astype(np.float64)) ** 2)


def welford(xs):
    n, mean, m2 = 0, 0.0, 0.0
    for x in xs:
        n += 1
        d = x - mean
        mean += d / n
        m2 += d * (x - mean)
    return m2 / n


w_var = welford(t[:20_000])
print(f"true var {true_var:.4f}   naive float64 {naive:.1f}   Welford {w_var:.4f}")
print(f"float32 can't even store these timestamps: spacing {np.spacing(np.float32(1.7e9))} s -> var {naive32:.1f}")
assert abs(naive - true_var) > 10 * true_var or naive < 0, "the naive formula is garbage here"
assert np.isclose(w_var, np.var(t[:20_000] - t[0]), rtol=1e-6)

# %% [markdown]
# ## 4. Overflow and underflow

# %%
def lse_naive(x):
    with np.errstate(over="ignore", divide="ignore"):
        return np.log(np.sum(np.exp(x)))


for x in (np.array([1000.0, 1001.0]), np.array([-1000.0, -1001.0])):
    print(f"x={x}: naive {lse_naive(x)}, logsumexp {logsumexp(x):.4f}")
    assert not np.isfinite(lse_naive(x)) and np.isfinite(logsumexp(x))
assert np.isclose(logsumexp([1000.0, 1001.0]), 1001 + np.log1p(np.exp(-1)))

logits = np.array([0.0, -800.0, 5.0])
with np.errstate(divide="ignore", over="ignore"):
    p = np.exp(logits) / np.exp(logits).sum()
    log_p_naive = np.log(p)
log_p_stable = logits - logsumexp(logits)
print("log(softmax) naive:", log_p_naive, "  log-softmax stable:", np.round(log_p_stable, 3))
assert np.isneginf(log_p_naive[1]) and np.isfinite(log_p_stable).all()


def softplus_stable(z):
    return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))


z = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
assert np.allclose(softplus_stable(z), np.logaddexp(0, z))

# BCE: sigmoid-then-log vs with-logits
y = np.array([1.0, 0.0])
zz = np.array([-40.0, 40.0])                      # confidently wrong
with np.errstate(divide="ignore"):
    s = 1 / (1 + np.exp(-zz))
    bce_naive = -(y * np.log(s) + (1 - y) * np.log(1 - s))
bce_logits = y * softplus_stable(-zz) + (1 - y) * softplus_stable(zz)
print(f"BCE on confident mistakes: naive {bce_naive}, with logits {bce_logits}")
assert np.isinf(bce_naive).any() and np.allclose(bce_logits, 40, atol=1e-6)
assert np.log1p(1e-17) == 1e-17 and np.log(1 + 1e-17) == 0.0

# %% [markdown]
# ## 5. Summation

# %%
vals = rng.uniform(0, 1, 1_000_000).astype(np.float32)
exact = math.fsum(vals.astype(np.float64))


def naive_sum(v):
    s = np.float32(0)
    for x in v:
        s = np.float32(s + x)
    return s


def kahan_sum(v):
    s, c = np.float32(0), np.float32(0)
    for x in v:
        y = np.float32(x - c)
        t = np.float32(s + y)
        c = np.float32((t - s) - y)       # what got lost in s + y
        s = t
    return s


sub = vals[:300_000]
ex_sub = math.fsum(sub.astype(np.float64))
errs = {
    "naive float32 loop": abs(float(naive_sum(sub)) - ex_sub),
    "numpy float32 (pairwise)": abs(float(np.sum(sub)) - ex_sub),
    "kahan float32": abs(float(kahan_sum(sub)) - ex_sub),
    "float64": abs(float(np.sum(sub.astype(np.float64))) - ex_sub),
}
for k_, e in errs.items():
    print(f"{k_:26s} abs error {e:.4f}")
assert errs["naive float32 loop"] > 10 * errs["numpy float32 (pairwise)"]
assert errs["kahan float32"] < 1.0 and errs["float64"] < 1e-6

a, b, c = 1e16, -1e16, 1.0
assert (a + b) + c == 1.0 and a + (b + c) == 0.0, "addition is not associative"
print("(1e16 + -1e16) + 1 =", (a + b) + c, "   1e16 + (-1e16 + 1) =", a + (b + c))

# %% [markdown]
# ## 6. float16 vs bfloat16

# %%
x = torch.tensor([70000.0, 1e-6, 1.0 + 1 / 256])
f16, bf16 = x.to(torch.float16), x.to(torch.bfloat16)
print("float32 :", x.tolist())
print("float16 :", f16.float().tolist(), "  <- overflow to inf, 1e-6 survives as a subnormal")
print("bfloat16:", bf16.float().tolist(), "  <- no overflow, but 1 + 1/256 rounds to 1")
assert torch.isinf(f16[0]) and torch.isfinite(bf16[0])
assert bf16[2].item() == 1.0 and f16[2].item() != 1.0

small_grad = torch.tensor([1e-8])
assert small_grad.half().item() == 0.0, "float16 underflows small gradients..."
assert (small_grad * 1024).half().item() > 0, "...unless you scale the loss first (GradScaler)"

# %%
print("\nAll checks passed.")
