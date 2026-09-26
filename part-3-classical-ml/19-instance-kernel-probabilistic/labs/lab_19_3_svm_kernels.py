# %% [markdown]
# # Lab 19.3: SVMs and kernels
#
# 1. Linear SVM as hinge loss + L2, by subgradient descent, against scikit-learn.
# 2. The dual, solved with a general-purpose optimizer; support vectors from the KKT conditions.
# 3. The kernel trick, numerically: explicit feature map vs kernel, and what it costs.
# 4. RBF SVM: the C-gamma grid on a nonlinear problem.
# 5. Scaling matters.
# 6. Kernel SVM cost in n vs random Fourier features + a linear model.

# %%
import time
from math import comb

import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_circles, make_classification
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

rng = np.random.default_rng(193)

# %% [markdown]
# ## 1. Hinge loss + L2 by subgradient descent

# %%
X, y01 = make_classification(n_samples=400, n_features=2, n_redundant=0, n_informative=2, class_sep=1.2, flip_y=0.02, random_state=3)
X = StandardScaler().fit_transform(X)
y = 2 * y01 - 1                                                 # labels in {-1, +1}
C = 1.0


def primal(w, b):
    return 0.5 * w @ w + C * np.maximum(0, 1 - y * (X @ w + b)).sum()


w, b = np.zeros(2), 0.0
for t in range(1, 50_001):
    margin = y * (X @ w + b)
    viol = margin < 1
    gw = w - C * (y[viol, None] * X[viol]).sum(0)
    gb = -C * y[viol].sum()
    lr = 1.0 / (t + 100) / len(X) * 10
    w -= lr * gw; b -= lr * gb

sk = SVC(kernel="linear", C=C).fit(X, y)
w_sk, b_sk = sk.coef_[0], sk.intercept_[0]
print(f"subgradient: w={np.round(w, 3)}, b={b:.3f}, objective {primal(w, b):.4f}")
print(f"scikit-learn: w={np.round(w_sk, 3)}, b={b_sk:.3f}, objective {primal(w_sk, b_sk):.4f}")
assert primal(w, b) - primal(w_sk, b_sk) < 0.01 * primal(w_sk, b_sk)
print(f"margin width 2/||w|| = {2 / np.linalg.norm(w_sk):.3f}; support vectors: {len(sk.support_)} of {len(X)}")

# %% [markdown]
# ## 2. The dual
#
# max sum(a) - 1/2 a^T Q a with Q_ij = y_i y_j x_i.x_j, subject to 0 <= a <= C and sum(a y) = 0. Small problem, general-purpose solver.

# %%
m = 80
Xd, yd = X[:m], y[:m]
Q = (yd[:, None] * Xd) @ (yd[:, None] * Xd).T
res = minimize(lambda a: 0.5 * a @ Q @ a - a.sum(), np.zeros(m), jac=lambda a: Q @ a - 1,
               bounds=[(0, C)] * m, constraints=[{"type": "eq", "fun": lambda a: a @ yd, "jac": lambda a: yd}],
               method="SLSQP", options={"maxiter": 1000, "ftol": 1e-12})
alpha = res.x
alpha[alpha < 1e-6] = 0; alpha[alpha > C - 1e-6] = C
w_dual = (alpha * yd) @ Xd
on_margin = (alpha > 0) & (alpha < C)
b_dual = np.mean(yd[on_margin] - Xd[on_margin] @ w_dual)
ref = SVC(kernel="linear", C=C).fit(Xd, yd)
print(f"dual solution: w={np.round(w_dual, 3)}, b={b_dual:.3f}; scikit-learn (libsvm): w={np.round(ref.coef_[0], 3)}, b={ref.intercept_[0]:.3f}")
assert np.allclose(w_dual, ref.coef_[0], atol=1e-3) and abs(b_dual - ref.intercept_[0]) < 1e-2
assert set(np.flatnonzero(alpha > 0)) == set(ref.support_), "support vectors = points with alpha > 0"
fm = yd * (Xd @ w_dual + b_dual)
print(f"alpha=0: {np.sum(alpha == 0)} points, all with y f(x) >= 1: {bool(np.all(fm[alpha == 0] >= 1 - 1e-3))}")
print(f"0<alpha<C: {on_margin.sum()} points, y f(x) = {np.round(fm[on_margin], 3)} (on the margin)")
print(f"alpha=C: {np.sum(alpha == C)} points, all with y f(x) <= 1: {bool(np.all(fm[alpha == C] <= 1 + 1e-3))}")
assert np.allclose(fm[on_margin], 1, atol=1e-2)

# %% [markdown]
# ## 3. The kernel trick, numerically

# %%
x2, z2 = rng.normal(size=2), rng.normal(size=2)
phi = lambda v: np.array([v[0] ** 2, v[1] ** 2, np.sqrt(2) * v[0] * v[1], np.sqrt(2) * v[0], np.sqrt(2) * v[1], 1.0])
assert np.isclose(phi(x2) @ phi(z2), (x2 @ z2 + 1) ** 2)
print("explicit degree-2 map in R^2 (6 features) matches the polynomial kernel")

d, p = 100, 3
print(f"degree <= {p} monomials in {d} variables: {comb(d + p, p):,} explicit features")
A = rng.normal(size=(300, d))
t0 = time.perf_counter(); K = (A @ A.T + 1) ** p; t_kernel = time.perf_counter() - t0
t0 = time.perf_counter(); F = PolynomialFeatures(p).fit_transform(A[:60]); t_explicit = time.perf_counter() - t0
print(f"300x300 Gram matrix via the kernel: {t_kernel * 1000:.1f} ms. Explicit features for just 60 of those rows: "
      f"{F.shape[1]:,} columns, {F.nbytes / 1e6:.0f} MB, {t_explicit * 1000:.0f} ms to build")
assert F.shape[1] == comb(d + p, p) and t_kernel < t_explicit

# RBF: the Gram matrix is PSD (Mercer), and its feature space has no finite dimension
Xk = rng.normal(size=(200, 5))
Kr = np.exp(-0.5 * ((Xk[:, None, :] - Xk[None, :, :]) ** 2).sum(-1))
eig = np.linalg.eigvalsh(Kr)
print(f"RBF Gram matrix: min eigenvalue {eig.min():.2e} (>= 0 up to rounding), rank {np.sum(eig > 1e-10)} of 200")
assert eig.min() > -1e-8 and np.sum(eig > 1e-10) == 200

# %% [markdown]
# ## 4. RBF SVM: the C-gamma grid

# %%
Xc, yc = make_circles(n_samples=600, noise=0.12, factor=0.5, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
Cs, gammas = np.logspace(-2, 3, 6), np.logspace(-2, 3, 6)
grid = np.array([[cross_val_score(SVC(C=c, gamma=g), Xc, yc, cv=cv).mean() for g in gammas] for c in Cs])
print("CV accuracy (rows: C, columns: gamma)")
print("        " + " ".join(f"{g:>7.0e}" for g in gammas))
for c, row in zip(Cs, grid):
    print(f"{c:>7.0e} " + " ".join(f"{v:7.3f}" for v in row))
print(f"linear SVM for reference: {cross_val_score(SVC(kernel='linear'), Xc, yc, cv=cv).mean():.3f}")
i, j = np.unravel_index(grid.argmax(), grid.shape)
print(f"best: C={Cs[i]:.0e}, gamma={gammas[j]:.0e}")
assert grid.max() > 0.9 and cross_val_score(SVC(kernel="linear"), Xc, yc, cv=cv).mean() < 0.65
big = SVC(C=1e3, gamma=1e3).fit(Xc, yc)
print(f"C=1e3, gamma=1e3: training accuracy {big.score(Xc, yc):.3f}, CV accuracy {grid[-1, -1]:.3f} (memorized)")
assert big.score(Xc, yc) > 0.99 and grid[-1, -1] < grid.max() - 0.1

# %% [markdown]
# ## 5. Scaling

# %%
Xs, ys = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=0)
Xs_bad = Xs * np.r_[1000, np.ones(9)]                           # one feature in different units
raw = cross_val_score(SVC(), Xs_bad, ys, cv=cv).mean()
scaled = cross_val_score(make_pipeline(StandardScaler(), SVC()), Xs_bad, ys, cv=cv).mean()
print(f"RBF SVM with one feature x1000: raw {raw:.3f}, standardized {scaled:.3f}")
assert scaled > raw + 0.1

# %% [markdown]
# ## 6. Cost in n: kernel SVM vs random Fourier features

# %%
Xl, yl = make_classification(n_samples=30_000, n_features=20, n_informative=10, class_sep=0.8, flip_y=0.02, random_state=1)
Xl = StandardScaler().fit_transform(Xl)
Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(Xl, yl, test_size=5000, random_state=0)
gamma = 1 / Xl.shape[1]
times = {}
for n in (2000, 4000, 8000):
    t0 = time.perf_counter(); s = SVC(gamma=gamma, C=1.0).fit(Xl_tr[:n], yl_tr[:n]); times[n] = time.perf_counter() - t0
    print(f"kernel SVM, n={n:5d}: fit {times[n]:.2f} s, test accuracy {s.score(Xl_te, yl_te):.3f}, support vectors {len(s.support_)}")
rff = make_pipeline(RBFSampler(gamma=gamma, n_components=2000, random_state=0),
                    SGDClassifier(loss="hinge", alpha=1e-5, max_iter=30, tol=None, random_state=0))
t0 = time.perf_counter(); rff.fit(Xl_tr, yl_tr); t_rff = time.perf_counter() - t0
acc_rff = rff.score(Xl_te, yl_te)
print(f"random Fourier features (2,000) + linear SVM, n={len(Xl_tr)}: fit {t_rff:.2f} s, test accuracy {acc_rff:.3f}")
print(f"kernel SVM time grew {times[8000] / times[2000]:.1f}x for 4x the data; at n={len(Xl_tr)} expect roughly "
      f"{times[8000] * (len(Xl_tr) / 8000) ** 2:.0f} s or more, and memory for the kernel cache grows as n^2")
print("(at this size the random-feature model isn't faster yet; its cost is linear in n, and it keeps being linear at 10^6 rows)")
assert times[8000] / times[2000] > 4, "superlinear in n"
assert acc_rff > s.score(Xl_te, yl_te) - 0.03, "the approximation trained on all the data is competitive"

print("\nAll checks passed.")
