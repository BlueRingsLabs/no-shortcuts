# %% [markdown]
# # Lab 16.3: Checking learning theory numerically
#
# 1. Hoeffding for a fixed classifier: the bound holds (and is conservative).
# 2. Selection bias: the best of many models on one validation set is optimistic.
# 3. Shattering: half-planes shatter 3 points but not the XOR square (VC dimension 3 in R^2).
# 4. The curse of dimensionality: neighborhood size, shell volume, distance concentration.
# 5. Double descent with random ReLU features and minimum-norm least squares.

# %%
from itertools import product

import numpy as np
from sklearn.svm import LinearSVC

rng = np.random.default_rng(163)

# %% [markdown]
# ## 1. Hoeffding for one fixed classifier

# %%
true_err, n, delta = 0.08, 2500, 0.05
eps = np.sqrt(np.log(2 / delta) / (2 * n))
observed = rng.binomial(n, true_err, 100_000) / n
coverage = np.mean(np.abs(observed - true_err) <= eps)
print(f"Hoeffding half-width at n={n}: {eps:.4f}; empirical coverage {coverage:.4f} (guaranteed >= {1 - delta})")
assert coverage >= 1 - delta and coverage > 0.999, "the bound holds, conservatively"
n_needed = int(np.ceil(np.log(2 / delta) / (2 * 0.005**2)))
print(f"examples needed for +-0.5%: {n_needed:,}")

# %% [markdown]
# ## 2. Picking the best of many on the same validation set

# %%
n_val, n_models, reps = 5000, 200, 300
optimism = []
for _ in range(reps):
    # 200 models that are all exactly 85% accurate; their validation scores differ only by noise
    val_acc = rng.binomial(n_val, 0.85, n_models) / n_val
    optimism.append(val_acc.max() - 0.85)
print(f"best of {n_models} equally good models looks {np.mean(optimism):.2%} better than it is (on average)")
assert 0.01 < np.mean(optimism) < 0.03
union_bound = np.sqrt((np.log(n_models) + np.log(2 / delta)) / (2 * n_val))
assert np.mean(optimism) < union_bound, "the union bound is an upper bound (correlated models do better than it)"

# %% [markdown]
# ## 3. Shattering

# %%
def linearly_separable(points, labels):
    if len(set(labels)) < 2:
        return True
    clf = LinearSVC(C=1e6, max_iter=200_000, tol=1e-8).fit(points, labels)
    return bool((clf.predict(points) == labels).all())


three = np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 1.0]])
shattered = all(linearly_separable(three, np.array(lab)) for lab in product([0, 1], repeat=3))
square = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
failures = [lab for lab in product([0, 1], repeat=4) if not linearly_separable(square, np.array(lab))]
print(f"3 points shattered by half-planes: {shattered}; labelings of the square that fail: {failures}")
assert shattered and set(failures) == {(0, 0, 1, 1), (1, 1, 0, 0)}, "only the XOR labelings fail"

# %% [markdown]
# ## 4. The curse of dimensionality

# %%
for d in (1, 2, 10, 100):
    side = 0.01 ** (1 / d)
    shell = 1 - 0.95**d
    print(f"d={d:3d}: sub-cube holding 1% of the data needs side {side:.3f}; {shell:.1%} of a ball's volume is in its outer 5% shell")
assert 0.01 ** (1 / 100) > 0.95 and 1 - 0.95**100 > 0.99

for d in (2, 20, 200):
    pts = rng.uniform(size=(2000, d))
    q = rng.uniform(size=d)
    dist = np.linalg.norm(pts - q, axis=1)
    print(f"d={d:3d}: (farthest - nearest) / nearest = {(dist.max() - dist.min()) / dist.min():.2f}")

# Real data is lower-dimensional: 3 latent factors embedded in 200 features
latent = rng.normal(size=(2000, 3))
embedded = latent @ rng.normal(size=(3, 200)) + 0.05 * rng.normal(size=(2000, 200))
s = np.linalg.svd(embedded - embedded.mean(0), compute_uv=False)
share = (s[:3] ** 2).sum() / (s**2).sum()
print(f"200 features, but the top 3 singular directions explain {share:.1%} of the variance (the manifold hypothesis, in miniature)")
assert share > 0.95

# %% [markdown]
# ## 5. Double descent
#
# Regression with p random ReLU features on n = 60 training points, fitted by minimum-norm least squares (pinv).
# Test error peaks near p = n (the interpolation threshold) and falls again for p >> n. Whether the second descent
# goes *below* the classical sweet spot depends on the problem; here it gets close, with no regularization at all.

# %%
n_train, d_in = 60, 5
w_true = rng.normal(size=d_in)
target = lambda X: np.tanh(X @ w_true)
X_tr = rng.normal(size=(n_train, d_in)); y_tr = target(X_tr) + 0.1 * rng.normal(size=n_train)
X_te = rng.normal(size=(2000, d_in)); y_te = target(X_te)
widths = [5, 20, 40, 55, 60, 65, 80, 150, 400, 2000]
errs = {}
for p in widths:
    e = []
    for rep in range(20):
        W = rng.normal(size=(d_in, p)) / np.sqrt(d_in)
        feats = lambda X: np.maximum(X @ W, 0)
        beta = np.linalg.pinv(feats(X_tr)) @ y_tr          # minimum-norm solution (what GD from zero finds)
        e.append(np.mean((feats(X_te) @ beta - y_te) ** 2))
    errs[p] = np.median(e)
    print(f"p = {p:5d} features: median test MSE {errs[p]:.3f}")
peak = max(errs, key=errs.get)
print(f"peak at p = {peak} (n = {n_train})")
assert 40 <= peak <= 80, "the peak sits at the interpolation threshold p ~ n"
best_classical = min(errs[p] for p in widths if p < n_train)
print(f"best under-parameterized error {best_classical:.3f}; at p=2000 {errs[2000]:.3f}")
assert errs[2000] < errs[peak] / 20, "far past the threshold, the error comes back down (the second descent)"
assert errs[2000] < 1.5 * best_classical, "and lands in the same range as the classical sweet spot, without any explicit regularization"

print("\nAll checks passed.")
