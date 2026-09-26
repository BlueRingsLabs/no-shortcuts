# %% [markdown]
# # Lab 21.3: PCA, derived and misused
#
# 1. PCA two ways (eigendecomposition, SVD) vs scikit-learn; the top axis beats every random direction.
# 2. Reconstruction error = sum of discarded eigenvalues.
# 3. Scaling decides the first component.
# 4. How many components on the digits, and what they look like reconstructed.
# 5. PCR throws away a predictive low-variance direction; ridge and PLS don't.
# 6. Whitening.
# 7. Kernel PCA untangles circles.
# 8. Random projections preserve distances (Johnson-Lindenstrauss).

# %%
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_digits, make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

rng = np.random.default_rng(213)

# %% [markdown]
# ## 1. Two routes to the same axes

# %%
n, d = 1000, 5
latent = rng.normal(size=(n, 2))
X = latent @ rng.normal(size=(2, d)) + 0.3 * rng.normal(size=(n, d))
Xc = X - X.mean(0)
S = Xc.T @ Xc / n
evals, evecs = np.linalg.eigh(S)
evals, evecs = evals[::-1], evecs[:, ::-1]                      # descending
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
sk = PCA().fit(X)
align = lambda A, B: np.abs(np.sum(A * B, axis=0))              # |cosine| per column (signs are arbitrary)
assert np.allclose(evals, s**2 / n) and np.allclose(align(evecs, Vt.T), 1) and np.allclose(align(evecs, sk.components_.T), 1)
print("eigendecomposition of S, SVD of X, and scikit-learn: same axes (up to sign), lambda_j = sigma_j^2 / n")
print("explained variance ratio:", np.round(evals / evals.sum(), 3))

random_dirs = rng.normal(size=(d, 20_000)); random_dirs /= np.linalg.norm(random_dirs, axis=0)
proj_var = np.einsum("ij,ik,kj->j", random_dirs, S, random_dirs)
print(f"variance along PC1 {evals[0]:.3f}; best of 20,000 random unit directions {proj_var.max():.3f}")
assert proj_var.max() <= evals[0] + 1e-12

# %% [markdown]
# ## 2. Reconstruction error

# %%
for k in range(1, d + 1):
    W = evecs[:, :k]
    err = np.sum((Xc - Xc @ W @ W.T) ** 2)
    assert np.isclose(err, n * evals[k:].sum())
    # and no other k-dimensional projection does better (spot check with random orthonormal bases)
    Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    assert np.sum((Xc - Xc @ Q @ Q.T) ** 2) >= err - 1e-9
    print(f"k={k}: reconstruction error {err:9.2f} = n * (sum of discarded eigenvalues)")

# %% [markdown]
# ## 3. Scaling

# %%
height_m = rng.normal(1.72, 0.09, n)
weight_g = 1000 * (22 * height_m**2 + rng.normal(0, 8, n))     # grams: huge numbers
resting_hr = 70 - 30 * (height_m - 1.72) + rng.normal(0, 8, n)
Xs = np.column_stack([height_m, weight_g, resting_hr])
pc1_raw = PCA(1).fit(Xs).components_[0]
pc1_std = PCA(1).fit(StandardScaler().fit_transform(Xs)).components_[0]
print(f"PC1 loadings, raw units:    {np.round(np.abs(pc1_raw), 4)}  (weight in grams wins by units alone)")
print(f"PC1 loadings, standardized: {np.round(np.abs(pc1_std), 3)}")
assert np.abs(pc1_raw)[1] > 0.999 and np.abs(pc1_std).min() > 0.3

# %% [markdown]
# ## 4. Digits: how many components?

# %%
digits = load_digits()
Xd = digits.data
pca_d = PCA().fit(Xd)
cum = np.cumsum(pca_d.explained_variance_ratio_)
for target in (0.5, 0.8, 0.9, 0.95, 0.99):
    print(f"{target:.0%} of variance: {np.searchsorted(cum, target) + 1} of 64 components")
cv = KFold(5, shuffle=True, random_state=0)
for k in (5, 10, 20, 40, 64):
    acc = cross_val_score(make_pipeline(PCA(k), LogisticRegression(max_iter=5000)), Xd, digits.target, cv=cv).mean()
    rec = PCA(k).fit(Xd); err = np.mean((Xd - rec.inverse_transform(rec.transform(Xd))) ** 2)
    print(f"k={k:2d}: CV accuracy {acc:.3f}, reconstruction MSE {err:.2f}")

# %% [markdown]
# ## 5. PCR throws away the signal
#
# Two nearly identical sensors (a huge shared component) whose small *difference* is what predicts y, plus nuisance features.

# %%
n5 = 1000
common = rng.normal(0, 10, n5)
diff = rng.normal(0, 0.2, n5)
nuisance = rng.normal(0, 3, size=(n5, 4))
X5 = np.column_stack([common + diff / 2, common - diff / 2, nuisance])
y5 = 5 * diff + rng.normal(0, 0.2, n5)
p5 = PCA().fit(X5)
print("explained variance ratio:", np.round(p5.explained_variance_ratio_, 4), "(the predictive direction is last)")
res = {}
for k in (1, 3, 5, 6):
    res[f"PCR k={k}"] = cross_val_score(make_pipeline(PCA(k), LinearRegression()), X5, y5, cv=cv, scoring="r2").mean()
res["ridge (CV)"] = cross_val_score(RidgeCV(alphas=np.logspace(-3, 3, 13)), X5, y5, cv=cv, scoring="r2").mean()
for k in (1, 2, 3, 4):
    res[f"PLS k={k}"] = cross_val_score(PLSRegression(k, scale=False), X5, y5, cv=cv, scoring="r2").mean()
for k_, v in res.items():
    print(f"{k_:16s} R^2 {v:.3f}")
print("PCR needs every component; PLS finds the signal sooner but not immediately (covariance with y still scales with the")
print("direction's tiny variance, and noisy covariances with the nuisance features compete); ridge just works.")
assert res["PCR k=5"] < 0.1 and res["PCR k=6"] > 0.9 and res["ridge (CV)"] > 0.9
assert res["PLS k=1"] < 0.1 and res["PLS k=4"] > 0.9

# %% [markdown]
# ## 6. Whitening

# %%
Zw = PCA(3, whiten=True).fit_transform(X)
print("covariance after whitening (3 components):\n", np.round(np.cov(Zw.T), 3))
assert np.allclose(np.cov(Zw.T), np.eye(3), atol=0.01)

# %% [markdown]
# ## 7. Kernel PCA

# %%
Xk, yk = make_circles(n_samples=800, factor=0.4, noise=0.05, random_state=0)
lin = cross_val_score(LogisticRegression(), PCA(2).fit_transform(Xk), yk, cv=cv).mean()
kp = KernelPCA(3, kernel="rbf", gamma=2.0).fit_transform(Xk)
by_k = [cross_val_score(LogisticRegression(), kp[:, :k], yk, cv=cv).mean() for k in (1, 2, 3)]
third_only = cross_val_score(LogisticRegression(), kp[:, [2]], yk, cv=cv).mean()
print(f"linear classifier on linear PCA (both components): {lin:.3f}")
print(f"on RBF kernel PCA, first 1/2/3 components: {np.round(by_k, 3)}; the 3rd component alone: {third_only:.3f}")
print("the top two kernel components describe the angle around the rings (more variance); the radius, which separates them, is third.")
print("'most variance' and 'most useful' are different questions, in kernel space too.")
assert lin < 0.65 and by_k[2] > 0.95 and third_only > 0.95 and by_k[1] < 0.7

# %% [markdown]
# ## 8. Random projections

# %%
Xh = rng.normal(size=(500, 10_000))
for k in (50, 200, 1000):
    Z = GaussianRandomProjection(k, random_state=0).fit_transform(Xh)
    i, j = rng.integers(0, 500, (2, 3000)); keep = i != j
    ratio = np.linalg.norm(Z[i[keep]] - Z[j[keep]], axis=1) / np.linalg.norm(Xh[i[keep]] - Xh[j[keep]], axis=1)
    print(f"10,000 -> {k:4d} dims: distance ratio {ratio.mean():.3f}, 1st-99th percentile [{np.percentile(ratio, 1):.3f}, {np.percentile(ratio, 99):.3f}]")
assert np.percentile(ratio, 1) > 0.9 and np.percentile(ratio, 99) < 1.1

print("\nAll checks passed.")
