# %% [markdown]
# # Lab 02.4: Eigenvectors and the SVD
#
# 1. Power iteration vs `eigh`, and how the eigenvalue gap controls convergence speed.
# 2. The spectral theorem, checked numerically on a covariance matrix.
# 3. SVD: rotate-stretch-rotate, connection with A^T A, pseudo-inverse.
# 4. Eckart-Young on a real "image" (the scikit-learn digits, stacked into a matrix), and denoising.
# 5. Randomized SVD, from scratch, vs the exact one.

# %%
import numpy as np
from sklearn.datasets import load_digits

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Power iteration

# %%
def power_iteration(A, iters=1000, tol=1e-10, rng=rng):
    v = rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    for t in range(1, iters + 1):
        w = A @ v
        w /= np.linalg.norm(w)
        if np.linalg.norm(w - v) < tol or np.linalg.norm(w + v) < tol:
            return w @ A @ w, w, t
        v = w
    return v @ A @ v, v, iters


def sym_with_eigs(eigs, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(len(eigs), len(eigs))))
    return Q @ np.diag(eigs) @ Q.T


for gap in (0.5, 0.99):
    A = sym_with_eigs([10.0, 10.0 * gap] + list(rng.uniform(0, 1, 48)), rng)
    lam, v, steps = power_iteration(A, iters=20000)
    lam_ref, Q = np.linalg.eigh(A)
    assert np.isclose(lam, lam_ref[-1], rtol=1e-6)
    assert np.isclose(abs(v @ Q[:, -1]), 1.0, atol=1e-4)     # same direction up to sign
    print(f"lambda2/lambda1 = {gap}: converged in {steps} iterations")
    if gap == 0.5:
        fast = steps
    else:
        slow = steps
assert slow > 10 * fast, "a small eigengap means slow convergence"

# %% [markdown]
# ## 2. Spectral theorem on a covariance matrix

# %%
true_cov = np.array([[3.0, 1.2, 0.0], [1.2, 1.0, 0.3], [0.0, 0.3, 0.5]])
X = rng.multivariate_normal(np.zeros(3), true_cov, size=5000)
C = np.cov(X, rowvar=False)
lam, Q = np.linalg.eigh(C)
assert np.allclose(Q.T @ Q, np.eye(3)), "eigenvectors are orthonormal"
assert np.allclose(Q @ np.diag(lam) @ Q.T, C), "C = Q Lambda Q^T"
assert np.allclose(sum(l * np.outer(q, q) for l, q in zip(lam, Q.T)), C), "sum of rank-1 pieces"
assert np.all(lam > 0), "a covariance matrix is PSD"
# Variance of the data projected on each eigenvector equals the eigenvalue
proj_var = np.var((X - X.mean(0)) @ Q, axis=0, ddof=1)
assert np.allclose(proj_var, lam)
print("eigenvalues of the sample covariance:", np.round(lam[::-1], 3))

# %% [markdown]
# ## 3. SVD basics

# %%
A = rng.normal(size=(6, 4))
U, s, Vt = np.linalg.svd(A, full_matrices=False)
assert np.allclose(U @ np.diag(s) @ Vt, A)
assert np.all(np.diff(s) <= 0), "singular values come sorted descending"
for i in range(4):
    assert np.allclose(A @ Vt[i], s[i] * U[:, i]), "A v_i = sigma_i u_i"
assert np.allclose(np.sort(np.linalg.eigvalsh(A.T @ A))[::-1], s**2), "sigma^2 = eigenvalues of A^T A"
assert np.isclose(np.linalg.norm(A, 2), s[0])
assert np.isclose(np.linalg.cond(A), s[0] / s[-1])

# The unit circle maps to an ellipse with semi-axes sigma_i: check max/min stretch over many directions
A2 = rng.normal(size=(2, 2))
theta = np.linspace(0, 2 * np.pi, 10000)
circle = np.stack([np.cos(theta), np.sin(theta)])
stretch = np.linalg.norm(A2 @ circle, axis=0)
s2 = np.linalg.svd(A2, compute_uv=False)
assert np.isclose(stretch.max(), s2[0], rtol=1e-4) and np.isclose(stretch.min(), s2[1], rtol=1e-4)

# Pseudo-inverse from the SVD
A_pinv = Vt.T @ np.diag(1 / s) @ U.T
assert np.allclose(A_pinv, np.linalg.pinv(A))
print("SVD identities: ok")

# %% [markdown]
# ## 4. Eckart-Young: low-rank approximation of real data
#
# The 1797 handwritten digits (8x8 pixels each) as a 1797 x 64 matrix.

# %%
D = load_digits().data.astype(float)
Dc = D - D.mean(axis=0)
U, s, Vt = np.linalg.svd(Dc, full_matrices=False)
total = np.sum(s**2)
for k in (2, 5, 10, 20, 40):
    Dk = U[:, :k] * s[:k] @ Vt[:k]
    err = np.linalg.norm(Dc - Dk, "fro") ** 2
    assert np.isclose(err, np.sum(s[k:] ** 2)), "Frobenius error = sum of discarded sigma^2"
    assert np.isclose(np.linalg.norm(Dc - Dk, 2), s[k]), "spectral error = next singular value"
    print(f"k={k:2d}: keeps {1 - err / total:6.1%} of the variance")

# No random rank-10 matrix beats the truncated SVD
k = 10
best = np.linalg.norm(Dc - U[:, :k] * s[:k] @ Vt[:k], "fro")
for _ in range(20):
    B = rng.normal(size=(Dc.shape[0], k)) @ rng.normal(size=(k, Dc.shape[1]))
    coef, *_ = np.linalg.lstsq(B.T, Dc.T, rcond=None)     # best combination of B's rows... still rank <= 10
    assert np.linalg.norm(Dc - (B.T @ coef).T, "fro") >= best - 1e-6

# Denoising: add noise, truncate, compare with the clean data
noisy = Dc + rng.normal(scale=4.0, size=Dc.shape)
Un, sn, Vtn = np.linalg.svd(noisy, full_matrices=False)
den = Un[:, :15] * sn[:15] @ Vtn[:15]
e_noisy = np.linalg.norm(noisy - Dc) / np.linalg.norm(Dc)
e_den = np.linalg.norm(den - Dc) / np.linalg.norm(Dc)
print(f"relative error vs clean: noisy {e_noisy:.3f}, rank-15 truncation {e_den:.3f}")
assert e_den < 0.8 * e_noisy

# %% [markdown]
# ## 5. Randomized SVD in ten lines

# %%
def randomized_svd(A, k, oversample=10, n_iter=4, rng=rng):
    Omega = rng.normal(size=(A.shape[1], k + oversample))
    Y = A @ Omega
    for _ in range(n_iter):                    # power iterations sharpen the spectrum
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)                     # orthonormal basis for the dominant range of A
    Ub, s, Vt = np.linalg.svd(Q.T @ A, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


Ur, sr, Vtr = randomized_svd(Dc, 10)
assert np.allclose(sr, s[:10], rtol=1e-3), "top singular values recovered"
print("randomized top-5:", np.round(sr[:5], 2), "\nexact top-5:     ", np.round(s[:5], 2))

# %%
print("\nAll checks passed.")
