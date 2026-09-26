# %% [markdown]
# # Lab 02.2: Matrices as functions
#
# 1. Columns are where the basis vectors land; composition is multiplication (and it doesn't commute).
# 2. Stacked linear layers collapse into one.
# 3. Rank, null space, rank-nullity, low-rank storage.
# 4. solve vs inv, and what the condition number does to your digits.

# %%
import time

import numpy as np

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Columns, composition, non-commutativity

# %%
R = np.array([[0.0, -1.0], [1.0, 0.0]])     # rotate 90 degrees
S = np.array([[2.0, 0.0], [0.0, 0.5]])      # stretch x, squash y
e1, e2 = np.eye(2)
assert np.allclose(R @ e1, R[:, 0]) and np.allclose(R @ e2, R[:, 1])   # columns = images of basis vectors
x = np.array([1.0, 1.0])
assert np.allclose((S @ R) @ x, S @ (R @ x))                           # composition: first R, then S
assert not np.allclose(S @ R, R @ S)                                    # order matters
print("S@R =\n", S @ R, "\nR@S =\n", R @ S)

# Column view of A @ x: a weighted sum of columns
A = rng.normal(size=(4, 3))
x = rng.normal(size=3)
assert np.allclose(A @ x, sum(x[j] * A[:, j] for j in range(3)))

# Associativity: same answer, different cost
n = 800
A, B, v = rng.normal(size=(n, n)), rng.normal(size=(n, n)), rng.normal(size=n)
t0 = time.perf_counter(); r1 = (A @ B) @ v; t_ab = time.perf_counter() - t0
t0 = time.perf_counter(); r2 = A @ (B @ v); t_bv = time.perf_counter() - t0
assert np.allclose(r1, r2)
print(f"(AB)v {t_ab * 1000:.1f} ms   A(Bv) {t_bv * 1000:.2f} ms")
assert t_bv < t_ab

# %% [markdown]
# ## 2. A deep linear network is a shallow linear network

# %%
Ws = [rng.normal(size=(32, 32)) / np.sqrt(32) for _ in range(10)]
x = rng.normal(size=32)
h = x
for W in Ws:
    h = W @ h                       # ten "layers", no activation
W_total = np.linalg.multi_dot(Ws[::-1])
assert np.allclose(h, W_total @ x), "ten linear layers == one matrix"

def relu_net(z):
    for W in Ws:
        z = np.maximum(W @ z, 0.0)
    return z


# With a nonlinearity there is no single matrix that does this for every x: it isn't even additive.
x2 = rng.normal(size=32)
assert not np.allclose(relu_net(x + x2), relu_net(x) + relu_net(x2)), "a ReLU network is not linear"
print("linear stack collapses; ReLU stack doesn't")

# %% [markdown]
# ## 3. Rank and null space

# %%
X = rng.normal(size=(100, 3))
X = np.column_stack([X, X[:, 0] + 2 * X[:, 1]])     # 4th feature is a combination of the first two
assert np.linalg.matrix_rank(X) == 3
# A null-space direction: the combination that the design matrix can't "see"
null_dir = np.array([1.0, 2.0, 0.0, -1.0])
assert np.allclose(X @ null_dir, 0)
w = rng.normal(size=4)
assert np.allclose(X @ w, X @ (w + 7.3 * null_dir)), "moving weights along the null space changes nothing"
print("rank-deficient design: infinitely many weight vectors give identical predictions")

# Rank-nullity via SVD: singular values that are ~0 count the null space dimension
s = np.linalg.svd(X, compute_uv=False)
null_dim = int(np.sum(s < 1e-10 * s.max()))
assert np.linalg.matrix_rank(X) + null_dim == X.shape[1]

# Low-rank storage: d x d matrix of rank r as U @ V.T
d, r = 1000, 8
U, V = rng.normal(size=(d, r)), rng.normal(size=(d, r))
M = U @ V.T
assert np.linalg.matrix_rank(M) == r
print(f"dense: {M.size:,} numbers; factored: {U.size + V.size:,} numbers ({M.size / (U.size + V.size):.0f}x smaller)")

# %% [markdown]
# ## 4. solve vs inv, and conditioning
#
# We build matrices with a chosen condition number (kappa) by picking singular values explicitly, then measure
# the relative error of the solution. Rule of thumb: you lose about log10(kappa) digits.

# %%
def matrix_with_condition(n, kappa, rng):
    Q1, _ = np.linalg.qr(rng.normal(size=(n, n)))
    Q2, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.logspace(0, -np.log10(kappa), n)          # singular values from 1 down to 1/kappa
    return Q1 @ np.diag(s) @ Q2.T


n = 200
print(f"{'kappa':>8s} {'cond':>10s} {'err solve':>10s} {'err inv':>10s}")
errs = {}
for kappa in (1e2, 1e6, 1e10, 1e14):
    A = matrix_with_condition(n, kappa, rng)
    x_true = rng.normal(size=n)
    b = A @ x_true
    e_solve = np.linalg.norm(np.linalg.solve(A, b) - x_true) / np.linalg.norm(x_true)
    e_inv = np.linalg.norm(np.linalg.inv(A) @ b - x_true) / np.linalg.norm(x_true)
    errs[kappa] = e_solve
    print(f"{kappa:8.0e} {np.linalg.cond(A):10.2e} {e_solve:10.1e} {e_inv:10.1e}")

assert errs[1e2] < 1e-12 and errs[1e14] > 1e-6, "error grows with the condition number"

# Determinant is not conditioning
assert np.isclose(np.linalg.det(1e-5 * np.eye(2)), 1e-10) and np.isclose(np.linalg.cond(1e-5 * np.eye(2)), 1)
assert np.isclose(np.linalg.det(np.diag([1e5, 1e-5])), 1) and np.linalg.cond(np.diag([1e5, 1e-5])) > 1e9

# Orthogonal matrices preserve length and have condition number 1
Q, _ = np.linalg.qr(rng.normal(size=(50, 50)))
x = rng.normal(size=50)
assert np.isclose(np.linalg.norm(Q @ x), np.linalg.norm(x)) and np.isclose(np.linalg.cond(Q), 1)

# X^T X is PSD, and its condition number is the square of X's
X = rng.normal(size=(500, 20)) @ np.diag(np.logspace(0, -3, 20))
assert np.all(np.linalg.eigvalsh(X.T @ X) > -1e-10)
assert np.isclose(np.linalg.cond(X.T @ X), np.linalg.cond(X) ** 2, rtol=1e-3)
print(f"cond(X) = {np.linalg.cond(X):.1e},  cond(X^T X) = {np.linalg.cond(X.T @ X):.1e}  <- squared (see 02.3)")

# %%
print("\nAll checks passed.")
