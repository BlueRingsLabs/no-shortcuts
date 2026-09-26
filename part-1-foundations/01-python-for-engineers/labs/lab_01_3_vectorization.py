# %% [markdown]
# # Lab 01.3: Thinking in arrays
#
# 1. Pairwise distances: loop vs broadcasting vs the matmul trick. Same answer, very different speed.
# 2. The (n,) vs (n, 1) bug, reproduced, so you recognize it when it happens to you.
# 3. Views and copies, verified with `np.shares_memory`.
# 4. Softmax that doesn't overflow, one-hot in one line, covariance by hand.

# %%
import time

import numpy as np

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Pairwise squared distances, three ways

# %%
def pairwise_loop(A, B):
    n, m = len(A), len(B)
    D = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            D[i, j] = np.sum((A[i] - B[j]) ** 2)
    return D


def pairwise_broadcast(A, B):
    # (n, 1, d) - (1, m, d) -> (n, m, d): simple, but materializes n*m*d numbers
    diff = A[:, None, :] - B[None, :, :]
    return np.sum(diff**2, axis=-1)


def pairwise_matmul(A, B):
    sq_a = np.sum(A * A, axis=1)[:, None]
    sq_b = np.sum(B * B, axis=1)[None, :]
    return np.maximum(sq_a - 2.0 * (A @ B.T) + sq_b, 0.0)


A = rng.normal(size=(300, 64))
B = rng.normal(size=(200, 64))

def best_time(fn, *args, repeat=5):
    """Best of several runs, after one untimed warm-up call. The first call of a function that uses BLAS also pays for
    starting the library's thread pool, and a single timing measures whatever else the machine was doing. The minimum is
    the least noisy estimate of what the code itself costs."""
    result = fn(*args)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return result, best


timings = {}
results = {}
for fn in (pairwise_loop, pairwise_broadcast, pairwise_matmul):
    results[fn.__name__], timings[fn.__name__] = best_time(fn, A, B)
    print(f"{fn.__name__:20s} {timings[fn.__name__] * 1000:8.2f} ms")

np.testing.assert_allclose(results["pairwise_broadcast"], results["pairwise_loop"], rtol=1e-10)
np.testing.assert_allclose(results["pairwise_matmul"], results["pairwise_loop"], rtol=1e-8, atol=1e-8)
assert timings["pairwise_matmul"] < timings["pairwise_loop"] / 10, "vectorized should be >10x faster"
print(f"speedup matmul vs loop: {timings['pairwise_loop'] / timings['pairwise_matmul']:.0f}x")

# %% [markdown]
# The price of the fast formula: for nearly identical points it subtracts large, nearly equal numbers.

# %%
p = rng.normal(size=(1, 64)) * 1e4          # a big vector...
q = p + 1e-6                                # ...and an almost identical one
exact = np.sum((p - q) ** 2)
fast = (np.sum(p * p) - 2 * (p @ q.T) + np.sum(q * q)).item()
print(f"exact {exact:.3e}   matmul-trick {fast:.3e}   <- catastrophic cancellation (see 07.2)")
assert abs(fast - exact) > 0.5 * exact, "the trick should lose most (here: all) of the precision"

# %% [markdown]
# ## 2. The broadcasting bug

# %%
y_true = rng.normal(size=100)
y_pred_ok = y_true + rng.normal(scale=0.1, size=100)   # shape (100,)
y_pred_bad = y_pred_ok.reshape(-1, 1)                  # shape (100, 1), like many model outputs

mse_ok = np.mean((y_true - y_pred_ok) ** 2)
mse_bad = np.mean((y_true - y_pred_bad) ** 2)          # (100,) - (100, 1) -> (100, 100)!
print(f"correct MSE {mse_ok:.4f}   buggy 'MSE' {mse_bad:.4f}")
assert (y_true - y_pred_bad).shape == (100, 100)
assert mse_bad > 10 * mse_ok        # looks like a number, means nothing


def mse(y_true, y_pred):
    assert y_true.shape == y_pred.shape, f"shape mismatch {y_true.shape} vs {y_pred.shape}"
    return float(np.mean((y_true - y_pred) ** 2))


try:
    mse(y_true, y_pred_bad)
    raise RuntimeError("the shape assertion should have fired")
except AssertionError as e:
    print("caught:", e)

# %% [markdown]
# ## 3. Views and copies

# %%
a = np.arange(12.0).reshape(3, 4)
assert a.strides == (32, 8)
assert np.shares_memory(a, a.T) and a.T.strides == (8, 32)
assert np.shares_memory(a, a[:, ::2])            # basic slice: view
assert not np.shares_memory(a, a[[0, 2]])        # fancy index: copy
assert not np.shares_memory(a, a[a > 5])         # boolean mask: copy

row = a[0]
row[:] = -1
assert (a[0] == -1).all(), "modifying a view modifies the original"
sel = a[[1, 2]]
sel[:] = 99
assert (a[1:] != 99).all(), "modifying a copy leaves the original alone"
print("views and copies behave as advertised")

# %% [markdown]
# ## 4. Small vectorized building blocks

# %%
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


s = softmax(np.array([[1000.0, 1001.0, 1002.0], [0.0, 0.0, 0.0]]))
assert np.all(np.isfinite(s)) and np.allclose(s.sum(axis=1), 1.0)
assert np.allclose(s[1], 1 / 3)
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp([1000.0, 1001.0]) / np.exp([1000.0, 1001.0]).sum()
assert np.isnan(naive).all(), "the naive softmax overflows to inf/inf = nan"

labels = np.array([2, 0, 1, 2])
onehot = np.eye(3)[labels]
assert onehot.shape == (4, 3) and (onehot.argmax(axis=1) == labels).all()
assert np.array_equal(onehot, (labels[:, None] == np.arange(3)).astype(float))

X = rng.normal(size=(500, 4)) @ rng.normal(size=(4, 4))
Xc = X - X.mean(axis=0)
C = Xc.T @ Xc / (len(X) - 1)
np.testing.assert_allclose(C, np.cov(X, rowvar=False))

# einsum reproduces matmul and row-wise dot products
np.testing.assert_allclose(np.einsum("ij,jk->ik", A, B.T), A @ B.T)
np.testing.assert_allclose(np.einsum("ij,ij->i", A[:200], B), np.sum(A[:200] * B, axis=1))
print("softmax, one-hot, covariance, einsum: ok")

# %% [markdown]
# ## Try this
#
# - Make A and B 3000 x 3000 x 64 and run only `pairwise_broadcast`. Watch memory: it builds a 3000x3000x64
#   float64 array, about 4.6 GB. The matmul version needs 72 MB for the output. Same math, different memory.
# - Remove `np.maximum(..., 0.0)` from `pairwise_matmul`, take a square root, and count the NaNs for points
#   compared with themselves (`pairwise_matmul(A, A)`).

# %%
print("\nAll checks passed.")
