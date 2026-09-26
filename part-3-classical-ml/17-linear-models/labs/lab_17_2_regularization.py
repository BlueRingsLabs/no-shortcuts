# %% [markdown]
# # Lab 17.2: Regularization from the inside
#
# 1. Ridge: closed form vs scikit-learn, the SVD shrinkage factors, effective degrees of freedom.
# 2. Lasso by coordinate descent (soft-thresholding), checked against scikit-learn, and lambda_max.
# 3. The lasso path: features enter one at a time.
# 4. p > n with a sparse truth: OLS is not identified, ridge barely helps, the lasso finds the support.
# 5. Correlated features: lasso picks one (unstably), elastic net keeps the group.
# 6. Units decide selection when you forget to standardize.
# 7. Choosing lambda by cross-validation, and the one-standard-error rule.

# %%
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LassoCV, Ridge, lasso_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(172)


def standardize(X):
    return (X - X.mean(0)) / X.std(0)

# %% [markdown]
# ## 1. Ridge

# %%
n, p = 80, 6
Z = rng.normal(size=(n, p))
Z[:, 1] = Z[:, 0] + 0.05 * rng.normal(size=n)          # a near-duplicate: one tiny singular value
X = standardize(Z)
beta_true = np.array([1.0, 1.0, 0.5, 0.0, 0.0, -0.8])
y = X @ beta_true + rng.normal(0, 1.0, n)
y = y - y.mean()

lam = 10.0
beta_ridge = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
sk = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
assert np.allclose(beta_ridge, sk)

U, s, Vt = np.linalg.svd(X, full_matrices=False)
beta_ols = Vt.T @ ((U.T @ y) / s)
shrink = s**2 / (s**2 + lam)
beta_svd = Vt.T @ (shrink * (U.T @ y) / s)
assert np.allclose(beta_svd, beta_ridge)
print("singular values:", np.round(s, 2))
print("ridge shrinkage per direction:", np.round(shrink, 3))
assert shrink[-1] < 0.1 < 0.9 < shrink[0], "the ill-determined direction is shrunk hard, the strong one barely"
print("OLS coefs  :", np.round(beta_ols, 2))
print("ridge coefs:", np.round(beta_ridge, 2), "(the near-duplicate pair is tamed)")

for l in (0.0, 1.0, 10.0, 100.0, 1e4):
    print(f"lambda={l:>7}: effective df = {np.sum(s**2 / (s**2 + l)):.2f}")
assert np.isclose(np.sum(s**2 / s**2), p)

# ridge never zeroes anything
betas = np.array([np.linalg.solve(X.T @ X + l * np.eye(p), X.T @ y) for l in np.logspace(-2, 4, 30)])
assert (np.abs(betas) > 0).all()

# %% [markdown]
# ## 2. Lasso by coordinate descent
#
# Objective (scikit-learn's scaling): (1/2n)||y - Xb||^2 + lam * ||b||_1, with standardized columns so (1/n) x_j.x_j = 1.

# %%
def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso_cd(X, y, lam, beta=None, tol=1e-10, max_iter=10_000):
    n, p = X.shape
    beta = np.zeros(p) if beta is None else beta.copy()
    col_sq = (X**2).sum(0) / n
    r = y - X @ beta
    for _ in range(max_iter):
        max_step = 0.0
        for j in range(p):
            old = beta[j]
            z = X[:, j] @ r / n + col_sq[j] * old          # correlation with the partial residual
            beta[j] = soft(z, lam) / col_sq[j]
            if beta[j] != old:
                r -= X[:, j] * (beta[j] - old)
                max_step = max(max_step, abs(beta[j] - old))
        if max_step < tol:
            break
    return beta


lam = 0.25
mine = lasso_cd(X, y, lam)
theirs = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=100_000).fit(X, y).coef_
print("my lasso    :", np.round(mine, 4))
print("sklearn     :", np.round(theirs, 4))
assert np.allclose(mine, theirs, atol=1e-6)
assert (mine == 0).sum() >= 2, "exact zeros, not small numbers"
assert (mine[:2] == 0).sum() == 1, "from the near-duplicate pair (0, 1) the lasso keeps one and drops the other"

# KKT conditions: |x_j.r|/n <= lam for zeros, = lam * sign(b_j) for the rest
grad = X.T @ (y - X @ mine) / n
active = mine != 0
assert np.allclose(grad[active], lam * np.sign(mine[active]), atol=1e-6)
assert (np.abs(grad[~active]) <= lam + 1e-9).all()
print("KKT conditions hold")

lam_max = np.max(np.abs(X.T @ y)) / n
assert (lasso_cd(X, y, lam_max * 1.0001) == 0).all() and (lasso_cd(X, y, lam_max * 0.99) != 0).any()
print(f"lambda_max = {lam_max:.3f}: above it everything is zero")

# %% [markdown]
# ## 3. The lasso path

# %%
lams = lam_max * np.logspace(0, -3, 60)
path, beta = [], np.zeros(p)
for l in lams:
    beta = lasso_cd(X, y, l, beta)                           # warm start
    path.append(beta.copy())
path = np.array(path)
entry = [int(np.argmax(path[:, j] != 0)) if (path[:, j] != 0).any() else None for j in range(p)]
order = sorted(range(p), key=lambda j: entry[j] if entry[j] is not None else 10**9)
print("order of entry (feature index):", order, " true coefficients:", beta_true)
_, sk_path, _ = lasso_path(X, y, alphas=lams, tol=1e-12, max_iter=100_000)
assert np.allclose(path, sk_path.T, atol=1e-4)
assert set(order[:2]) <= {0, 1, 5}, "strong features enter first"

# %% [markdown]
# ## 4. p > n with a sparse truth

# %%
n4, p4, k = 60, 200, 5
X4 = standardize(rng.normal(size=(n4, p4)))
b4 = np.zeros(p4); support = rng.choice(p4, k, replace=False); b4[support] = rng.choice([-2.0, 2.0], k)
y4 = X4 @ b4 + rng.normal(0, 0.5, n4)
X4t = standardize(rng.normal(size=(2000, p4))); y4t = X4t @ b4

print("rank of X4:", np.linalg.matrix_rank(X4), "for", p4, "coefficients: OLS is not identified")
mn = np.linalg.pinv(X4) @ y4                                 # the minimum-norm interpolator
r4 = Ridge(alpha=5.0).fit(X4, y4)
l4 = LassoCV(cv=5, random_state=0).fit(X4, y4)
mse = lambda pred: np.mean((pred - y4t) ** 2)
print(f"test MSE: min-norm OLS {mse(X4t @ mn):.2f}, ridge {mse(r4.predict(X4t)):.2f}, lasso {mse(l4.predict(X4t)):.3f} "
      f"(signal variance {np.var(y4t):.1f})")
found = set(np.flatnonzero(np.abs(l4.coef_) > 0.1))
print(f"lasso kept {np.sum(l4.coef_ != 0)} features; true support recovered: {set(support) <= found}")
assert set(support) <= found and mse(l4.predict(X4t)) < 0.1 * mse(r4.predict(X4t))

# %% [markdown]
# ## 5. Correlated groups: lasso vs elastic net

# %%
n5 = 200
latent = rng.normal(size=n5)
group = np.column_stack([latent + 0.1 * rng.normal(size=n5) for _ in range(4)])   # 4 noisy copies of one signal
noise_feats = rng.normal(size=(n5, 6))
X5 = standardize(np.column_stack([group, noise_feats]))
y5 = 3 * latent + rng.normal(0, 1, n5)

picks = []
for b in range(50):
    idx = rng.integers(0, n5, n5)
    c = Lasso(alpha=0.1).fit(X5[idx], y5[idx]).coef_[:4]
    picks.append(int(np.argmax(np.abs(c))))
lasso_c = Lasso(alpha=0.1).fit(X5, y5).coef_[:4]
enet_c = ElasticNet(alpha=0.1, l1_ratio=0.2).fit(X5, y5).coef_[:4]
print("lasso on the group:      ", np.round(lasso_c, 2))
print("elastic net on the group:", np.round(enet_c, 2))
print(f"across 50 bootstraps the lasso's favorite group member was: {np.bincount(picks, minlength=4)}")
assert np.std(enet_c) < np.std(lasso_c), "elastic net spreads the weight across the group"
assert (np.bincount(picks, minlength=4) > 0).sum() >= 2, "and the lasso's pick changes with the sample"

# %% [markdown]
# ## 6. Units decide the selection if you let them

# %%
n6 = 500
age_years = rng.uniform(20, 65, n6)
income_k = rng.lognormal(3.8, 0.5, n6)                       # thousands
y6 = 0.8 * (age_years - 42) / 13 + 0.8 * (income_k - income_k.mean()) / income_k.std() + rng.normal(0, 1, n6)
raw = np.column_stack([age_years * 365.25, income_k / 1000])  # age in days, income in millions
c_raw = Lasso(alpha=0.05).fit(raw, y6).coef_
c_std = make_pipeline(StandardScaler(), Lasso(alpha=0.05)).fit(raw, y6)[-1].coef_
print(f"raw units (days, millions): {c_raw}  -> income {'dropped' if c_raw[1] == 0 else 'kept'}")
print(f"standardized:               {np.round(c_std, 3)}  -> both kept, similar size (as in the truth)")
assert c_raw[1] == 0 and (c_std != 0).all() and abs(c_std[0] - c_std[1]) < 0.15

# %% [markdown]
# ## 7. Choosing lambda: CV minimum and the one-standard-error rule

# %%
cv = LassoCV(cv=10, alphas=80, random_state=0).fit(X4, y4)
mean_mse, se_mse = cv.mse_path_.mean(1), cv.mse_path_.std(1) / np.sqrt(cv.mse_path_.shape[1])
i_min = int(np.argmin(mean_mse))
ok = np.flatnonzero(mean_mse <= mean_mse[i_min] + se_mse[i_min])
i_1se = ok[np.argmax(cv.alphas_[ok])]                        # the largest lambda within one SE
lam_min, lam_1se = cv.alphas_[i_min], cv.alphas_[i_1se]
nz = lambda a: int((Lasso(alpha=a).fit(X4, y4).coef_ != 0).sum())
print(f"lambda_min {lam_min:.3f} keeps {nz(lam_min)} features; lambda_1se {lam_1se:.3f} keeps {nz(lam_1se)}")
assert lam_1se >= lam_min and nz(lam_1se) <= nz(lam_min)
assert set(support) <= set(np.flatnonzero(Lasso(alpha=lam_1se).fit(X4, y4).coef_)), "the simpler model still has the true support"

print("\nAll checks passed.")
