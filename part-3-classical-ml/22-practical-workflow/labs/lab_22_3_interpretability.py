# %% [markdown]
# # Lab 22.3: Explaining models, and the limits of the explanations
#
# Synthetic data with known structure, so every explanation can be checked against the truth:
#   y = 2*x1 + 3*x2*x3 + 0.5*x4 + noise,  x5 = x1 + small noise (a near-copy),  x6 pure noise.
#
# 0. How far is an additive (GAM-like) model from the black box?
# 1. Permutation importance, and what a correlated copy does to it; drop-column importance.
# 2. PDP vs ICE: the interaction the average hides.
# 3. PDP vs ALE when features are correlated.
# 4. Shapley values by brute force = TreeSHAP; efficiency; the linear-model formula.
# 5. Explanations are about the model, not the world: a consequence of the label gets all the credit.
# 6. Instability: correlated features trade places across retrains.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
from itertools import combinations  # noqa: E402
from math import factorial  # noqa: E402

import numpy as np  # noqa: E402
import shap  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import partial_dependence, permutation_importance  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

rng = np.random.default_rng(223)
n = 6000
x1, x2, x3, x4, x6 = (rng.normal(size=n) for _ in range(5))
x5 = x1 + 0.1 * rng.normal(size=n)
X = np.column_stack([x1, x2, x3, x4, x5, x6])
names = ["x1", "x2", "x3", "x4", "x5 (copy of x1)", "x6 (noise)"]
y = 2 * x1 + 3 * x2 * x3 + 0.5 * x4 + 0.5 * rng.normal(size=n)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X_tr, y_tr)
print(f"black box (boosting) test R^2 {model.score(X_te, y_te):.3f}")

# %% [markdown]
# ## 0. An interpretable baseline first

# %%
gam = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, interaction_cst="no_interactions", random_state=0).fit(X_tr, y_tr)
print(f"additive model (sum of one-feature curves, readable feature by feature): test R^2 {gam.score(X_te, y_te):.3f}")
print("here the gap is large because the truth has a strong interaction; on many real tables it's a point or two, and then I'd ship the readable one.")
assert model.score(X_te, y_te) > gam.score(X_te, y_te) + 0.2

# %% [markdown]
# ## 1. Permutation importance

# %%
pi = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=0, scoring="r2")
for nm, m_, s_ in zip(names, pi.importances_mean, pi.importances_std):
    print(f"{nm:16s} permutation importance (R^2 drop) {m_:.3f} ± {s_:.3f}")
both = X_te.copy()
perm = rng.permutation(len(both)); both[:, 0] = both[perm, 0]; both[:, 4] = both[perm, 4]   # permute x1 and x5 together
group_drop = model.score(X_te, y_te) - model.score(both, y_te)
print(f"x1 and x5 permuted together: R^2 drop {group_drop:.3f}, far more than the two individual drops summed "
      f"({pi.importances_mean[0] + pi.importances_mean[4]:.3f})")
assert group_drop > 1.3 * (pi.importances_mean[0] + pi.importances_mean[4])
assert pi.importances_mean[5] < 0.01

drop = {}
for j in (0, 4):
    keep = [k for k in range(6) if k != j]
    m_j = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X_tr[:, keep], y_tr)
    drop[names[j]] = model.score(X_te, y_te) - m_j.score(X_te[:, keep], y_te)
print("drop-column importance (retrain without it):", {k: round(v, 3) for k, v in drop.items()}, "-> each copy is replaceable by the other")

# %% [markdown]
# ## 2. PDP vs ICE: the hidden interaction

# %%
pd_x2 = partial_dependence(model, X_te[:500], [1], kind="both", grid_resolution=20)
avg, ice = pd_x2["average"][0], pd_x2["individual"][0]
slopes = (ice[:, -1] - ice[:, 0]) / (pd_x2["grid_values"][0][-1] - pd_x2["grid_values"][0][0])
print(f"PDP of x2: range {avg.max() - avg.min():.2f} (looks almost flat)")
print(f"ICE slopes for x2 across rows: from {slopes.min():.2f} to {slopes.max():.2f}; correlation of slope with x3: "
      f"{np.corrcoef(slopes, X_te[:500, 2])[0, 1]:.2f}")
assert (avg.max() - avg.min()) < 0.25 * (np.percentile(slopes, 95) - np.percentile(slopes, 5)) * 2
assert np.corrcoef(slopes, X_te[:500, 2])[0, 1] > 0.8

# %% [markdown]
# ## 3. PDP vs ALE with correlated features
#
# x5 is x1 plus a little noise, so rows with x5 far from x1 don't exist. The PDP of x5 sets x5 = v for *every* row and
# asks the model about exactly those rows. ALE only looks at rows whose x5 is actually near v.

# %%
def ale_1d(model, X, j, bins=20):
    edges = np.quantile(X[:, j], np.linspace(0, 1, bins + 1))
    effects = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = X[(X[:, j] >= lo) & (X[:, j] <= hi)]
        if len(rows) == 0:
            effects.append(0.0); continue
        a, b = rows.copy(), rows.copy()
        a[:, j], b[:, j] = lo, hi
        effects.append(np.mean(model.predict(b) - model.predict(a)))
    ale = np.concatenate([[0], np.cumsum(effects)])
    return edges, ale - ale.mean()


grid = np.quantile(X_te[:, 4], np.linspace(0.05, 0.95, 15))
off = np.mean([np.mean(np.abs(v_ - X_te[:, 0]) > 0.5) for v_ in grid])
print(f"PDP of x5: {off:.0%} of the rows it evaluates have |x5 - x1| > 0.5, a combination that never occurs (|x5 - x1| is ~0.1)")

# two models, equally good on real data, that differ only in how they behave off the manifold
m_a = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X_tr, y_tr)
idx = rng.permutation(len(X_tr))
m_b = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=1).fit(X_tr[idx[: len(idx) // 2]], y_tr[idx[: len(idx) // 2]])
pd_a = partial_dependence(m_a, X_te, [4], grid_resolution=15)["average"][0]
pd_b = partial_dependence(m_b, X_te, [4], grid_resolution=15)["average"][0]
_, ale_a = ale_1d(m_a, X_te, 4); _, ale_b = ale_1d(m_b, X_te, 4)
_, ale_a1 = ale_1d(m_a, X_te, 0); _, ale_b1 = ale_1d(m_b, X_te, 0)
print(f"test R^2: model A {m_a.score(X_te, y_te):.3f}, model B {m_b.score(X_te, y_te):.3f}")
print(f"effect range of x5 by PDP: A {np.ptp(pd_a):.2f}, B {np.ptp(pd_b):.2f}; by ALE: A {np.ptp(ale_a):.2f}, B {np.ptp(ale_b):.2f}")
print(f"ALE of the pair (x1 + x5): A {np.ptp(ale_a1) + np.ptp(ale_a):.2f}, B {np.ptp(ale_b1) + np.ptp(ale_b):.2f}")
print("how two copies split the credit is arbitrary for any method; the PDP adds answers from rows that don't exist on top.")
assert off > 0.5

# %% [markdown]
# ## 4. Shapley values: brute force vs TreeSHAP

# %%
small = HistGradientBoostingRegressor(max_iter=100, random_state=0).fit(X_tr[:, :4], y_tr)   # 4 features: 16 coalitions
background = X_tr[:100, :4]                                    # shap silently subsamples backgrounds above 100 rows
x0 = X_te[0, :4]


def v(S):                                                       # interventional value: known features from x0, the rest from background
    Z = background.copy()
    Z[:, list(S)] = x0[list(S)]
    return small.predict(Z).mean()


F = range(4)
phi = np.zeros(4)
for j in F:
    others = [k for k in F if k != j]
    for r in range(len(others) + 1):
        for S in combinations(others, r):
            w = factorial(len(S)) * factorial(4 - len(S) - 1) / factorial(4)
            phi[j] += w * (v(S + (j,)) - v(S))
explainer = shap.TreeExplainer(small, data=background, feature_perturbation="interventional")
tree_phi = explainer.shap_values(x0[None])[0]
print("brute-force Shapley:", np.round(phi, 4))
print("TreeSHAP:           ", np.round(tree_phi, 4), "(tiny differences: the trees' float32 thresholds)")
print(f"efficiency: sum of contributions {phi.sum():.4f} = prediction - average {small.predict(x0[None])[0] - v(()):.4f}")
assert np.allclose(phi, tree_phi, atol=0.02) and np.isclose(phi.sum(), small.predict(x0[None])[0] - v(()), atol=1e-8)

lin = LinearRegression().fit(X_tr[:, :4], y_tr)
lin_phi = shap.LinearExplainer(lin, background).shap_values(x0[None])[0]
formula = lin.coef_ * (x0 - background.mean(0))
assert np.allclose(lin_phi, formula, atol=1e-8)
print("linear model: SHAP = beta_j * (x_j - mean_j), exactly")

# %% [markdown]
# ## 5. A consequence of the label gets the credit

# %%
flag = y + rng.normal(0, 1.0, n)                                # e.g. "escalated to manager", recorded *after* the outcome
X_leaky = np.column_stack([X, flag])
Xl_tr, Xl_te = X_leaky[: len(X_tr)], X_leaky[len(X_tr):]
yl_tr, yl_te = y[: len(X_tr)], y[len(X_tr):]
m_leak = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xl_tr, yl_tr)
sv = shap.TreeExplainer(m_leak).shap_values(Xl_te[:500])
mean_abs = np.abs(sv).mean(0)
print("mean |SHAP|:", {nm: round(float(v_), 3) for nm, v_ in zip(names + ["flag (after the fact)"], mean_abs)})
print("the explanation is correct about the model and useless about the world: the flag doesn't cause anything, and it won't exist at prediction time")
assert mean_abs.argmax() == 6

# %% [markdown]
# ## 6. Instability with correlated features

# %%
top_of_pair = []
for b in range(8):
    idx = rng.integers(0, len(X_tr), len(X_tr))
    mb = HistGradientBoostingRegressor(max_iter=200, random_state=b).fit(X_tr[idx], y_tr[idx])
    s = np.abs(shap.TreeExplainer(mb).shap_values(X_te[:300])).mean(0)
    top_of_pair.append((round(float(s[0]), 2), round(float(s[4]), 2)))
print("mean |SHAP| of (x1, x5) across 8 bootstrap retrains:", top_of_pair)
share = [a / (a + b) for a, b in top_of_pair]
print(f"x1's share of the pair's credit ranges from {min(share):.0%} to {max(share):.0%}")
assert max(share) - min(share) > 0.05

print("\nAll checks passed.")
