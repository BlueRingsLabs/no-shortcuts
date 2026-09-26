# %% [markdown]
# # Lab 02.1: Vectors, projections and the weirdness of high dimensions
#
# 1. Dot products, projections and norms, checked against their geometric meaning.
# 2. Gram-Schmidt from scratch vs `np.linalg.qr`, and how classical Gram-Schmidt loses orthogonality.
# 3. Two high-dimensional facts: random vectors are nearly orthogonal, and distances concentrate.
# 4. Cosine vs Euclidean ranking on normalized embeddings.

# %%
import numpy as np

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Projections

# %%
def proj(x, v):
    """Vector projection of x onto the line spanned by v."""
    return (v @ x) / (v @ v) * v


x = np.array([3.0, 4.0])
p1, p2 = proj(x, np.array([1.0, 0.0])), proj(x, np.array([1.0, 1.0]))
assert np.allclose(p1, [3, 0]) and np.allclose(p2, [3.5, 3.5])
assert abs((x - p2) @ np.array([1.0, 1.0])) < 1e-12, "the residual is orthogonal to the line"

# Pythagoras: ||x||^2 = ||proj||^2 + ||residual||^2
assert np.isclose(x @ x, p2 @ p2 + (x - p2) @ (x - p2))

v = np.array([1.0, -2.0, 3.0])
l1, l2, linf = np.linalg.norm(v, 1), np.linalg.norm(v), np.linalg.norm(v, np.inf)
assert (l1, linf) == (6.0, 3.0) and np.isclose(l2, np.sqrt(14))
assert linf <= l2 <= l1

# Cauchy-Schwarz, on 1000 random pairs
A, B = rng.normal(size=(1000, 10)), rng.normal(size=(1000, 10))
assert np.all(np.abs(np.sum(A * B, axis=1)) <= np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
print("projections, norms, Cauchy-Schwarz: ok")

# %% [markdown]
# ## 2. Gram-Schmidt vs QR
#
# We build a nasty, nearly dependent set of vectors (a Hilbert-like matrix) and measure how far from orthonormal
# the result is, via ||Q Q^T - I||.

# %%
def gram_schmidt(V):
    Q = []
    for v in V:
        w = v - sum((q @ v) * q for q in Q)
        n = np.linalg.norm(w)
        if n < 1e-14:
            raise ValueError("numerically dependent")
        Q.append(w / n)
    return np.array(Q)


easy = rng.normal(size=(5, 5))
Q = gram_schmidt(easy)
assert np.allclose(Q @ Q.T, np.eye(5), atol=1e-12), "fine on well-conditioned input"

k = 10
hilbert = 1.0 / (np.arange(k)[:, None] + np.arange(k)[None, :] + 1.0)   # famously ill-conditioned
Q_gs = gram_schmidt(hilbert)
Q_qr, _ = np.linalg.qr(hilbert.T)            # qr orthonormalizes columns, so pass the transpose
err_gs = np.linalg.norm(Q_gs @ Q_gs.T - np.eye(k))
err_qr = np.linalg.norm(Q_qr.T @ Q_qr - np.eye(k))
print(f"loss of orthogonality: classical Gram-Schmidt {err_gs:.1e}   QR (Householder) {err_qr:.1e}")
assert err_gs > 1e-3 and err_qr < 1e-12, "QR should be orders of magnitude more stable"

# %% [markdown]
# ## 3. High dimensions
#
# (a) Cosine similarity between random Gaussian vectors, in 2 and 1000 dimensions.
# (b) Ratio between the farthest and the nearest neighbour distance from a random query, as d grows.

# %%
def cosine_rows(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


for d in (2, 1000):
    c = cosine_rows(rng.normal(size=(5000, d)), rng.normal(size=(5000, d)))
    print(f"d={d:5d}: |cos| 95th percentile = {np.quantile(np.abs(c), 0.95):.3f}")
c1000 = cosine_rows(rng.normal(size=(5000, 1000)), rng.normal(size=(5000, 1000)))
assert np.quantile(np.abs(c1000), 0.95) < 0.1, "random vectors in 1000-d are nearly orthogonal"

ratios = {}
for d in (2, 10, 100, 1000):
    pts = rng.uniform(size=(2000, d))
    q = rng.uniform(size=d)
    dist = np.linalg.norm(pts - q, axis=1)
    ratios[d] = dist.max() / dist.min()
    print(f"d={d:5d}: farthest / nearest = {ratios[d]:8.2f}")
assert ratios[2] > 20 and ratios[1000] < 1.5, "distances concentrate as d grows"

# %% [markdown]
# ## 4. Cosine and Euclidean agree on unit vectors

# %%
emb = rng.normal(size=(500, 64))
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
query = rng.normal(size=64)
query /= np.linalg.norm(query)
by_cos = np.argsort(-(emb @ query))
by_euc = np.argsort(np.linalg.norm(emb - query, axis=1))
assert np.array_equal(by_cos, by_euc)
assert np.allclose(np.linalg.norm(emb - query, axis=1) ** 2, 2 - 2 * (emb @ query))

# Without normalization, they disagree: longer vectors win dot-product rankings.
raw = emb * rng.uniform(0.5, 3.0, size=(500, 1))
agree = np.mean(np.argsort(-(raw @ query))[:10] == np.argsort(np.linalg.norm(raw - query, axis=1))[:10])
print(f"top-10 agreement on unnormalized vectors: {agree:.0%}")
assert agree < 1.0

# %%
print("\nAll checks passed.")
