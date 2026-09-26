# %% [markdown]
# # Lab 20.2: Bagging and random forests
#
# 1. The variance of an average of correlated predictors, simulated.
# 2. Bagging from scratch: bootstrap, deep trees, average. Against a single tree.
# 3. Bagging can't fix bias: bagged stumps.
# 4. max_features: tree-to-tree correlation vs forest error.
# 5. Out-of-bag: the 36.8% and the OOB score vs cross-validation.
# 6. Error vs number of trees: forests plateau, aggressive boosting turns back up.
# 7. Leaf size, model size and accuracy.

# %%
import pickle

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(202)
X, y = make_friedman1(n_samples=3000, n_features=10, noise=1.0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
mse = lambda p: np.mean((p - y_te) ** 2)

# %% [markdown]
# ## 1. Averaging correlated predictors

# %%
sigma2 = 1.0
for rho in (0.0, 0.3, 0.8):
    for B in (1, 10, 100):
        cov = sigma2 * (rho * np.ones((B, B)) + (1 - rho) * np.eye(B))
        draws = rng.multivariate_normal(np.zeros(B), cov, 20_000).mean(1)
        theory = rho * sigma2 + (1 - rho) * sigma2 / B
        assert abs(draws.var() - theory) < 0.05 * theory + 0.002
    print(f"rho={rho}: variance of the average of 100 = {rho * sigma2 + (1 - rho) * sigma2 / 100:.3f} (floor {rho * sigma2})")
print("simulation matches rho*s^2 + (1-rho)*s^2/B")

# %% [markdown]
# ## 2. Bagging from scratch

# %%
def bag(X, y, B, make_model):
    models = []
    for _ in range(B):
        idx = rng.integers(0, len(X), len(X))
        models.append(make_model().fit(X[idx], y[idx]))
    return models


trees = bag(X_tr, y_tr, 200, lambda: DecisionTreeRegressor())
P = np.array([t.predict(X_te) for t in trees])                  # 200 x n_test
single = mse(P[0])
bagged = mse(P.mean(0))
per_tree_var = P.var(0).mean()
rho_hat = np.mean([np.corrcoef(P[i] - y_te, P[j] - y_te)[0, 1] for i, j in rng.integers(0, 200, (100, 2)) if i != j])
print(f"test MSE: one deep tree {single:.2f}; bag of 200 {bagged:.2f} (noise floor 1.0)")
print(f"average error correlation between two bagged trees: {rho_hat:.2f}")
assert bagged < 0.6 * single

# %% [markdown]
# ## 3. Bagging stumps: no bias reduction

# %%
stumps = bag(X_tr, y_tr, 200, lambda: DecisionTreeRegressor(max_depth=1))
one_stump = mse(stumps[0].predict(X_te))
bag_stumps = mse(np.mean([s.predict(X_te) for s in stumps], 0))
print(f"test MSE: one stump {one_stump:.2f}; 200 bagged stumps {bag_stumps:.2f} (still terrible: the bias stays)")
assert bag_stumps > 0.85 * one_stump and bag_stumps > 3 * bagged

# %% [markdown]
# ## 4. max_features: decorrelation vs per-tree quality

# %%
rows = []
for mf in (1.0, 0.6, 0.3, 0.1):
    rf = RandomForestRegressor(n_estimators=300, max_features=mf, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    Pt = np.array([t.predict(X_te) for t in rf.estimators_])
    E = Pt - y_te
    C = np.corrcoef(E[:60])
    rho = C[np.triu_indices(60, 1)].mean()
    rows.append((mf, rho, np.mean([mse(p) for p in Pt[:60]]), mse(rf.predict(X_te))))
    print(f"max_features={mf:>4}: tree error correlation {rho:.2f}, mean single-tree MSE {rows[-1][2]:.2f}, forest MSE {rows[-1][3]:.2f}")
rows = np.array(rows)
assert rows[0, 1] > rows[2, 1], "fewer features per split -> less correlated trees"
assert rows[0, 2] < rows[2, 2], "... each of which is individually worse"
best_mf = rows[rows[:, 3].argmin(), 0]
print(f"best max_features here: {best_mf} (plain bagging is 1.0)")
et = ExtraTreesRegressor(n_estimators=300, max_features=0.6, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
print(f"extra trees (random thresholds): forest MSE {mse(et.predict(X_te)):.2f}")

# %% [markdown]
# ## 5. Out of bag

# %%
n = 1000
left_out = np.mean([len(np.setdiff1d(np.arange(n), rng.integers(0, n, n))) / n for _ in range(200)])
print(f"fraction of rows left out of a bootstrap sample: {left_out:.4f} (1/e = {np.exp(-1):.4f})")
assert abs(left_out - np.exp(-1)) < 0.005
rf = RandomForestRegressor(n_estimators=300, max_features=0.3, oob_score=True, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
cv_r2 = cross_val_score(RandomForestRegressor(n_estimators=300, max_features=0.3, random_state=0, n_jobs=-1),
                        X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=0)).mean()
test_r2 = rf.score(X_te, y_te)
print(f"R^2: OOB {rf.oob_score_:.3f}, 5-fold CV {cv_r2:.3f}, test {test_r2:.3f}")
assert abs(rf.oob_score_ - test_r2) < 0.03 and abs(rf.oob_score_ - cv_r2) < 0.03

# %% [markdown]
# ## 6. More trees vs more boosting rounds
#
# The forest's test error falls and plateaus. Gradient boosting with a high learning rate and deep trees keeps
# fitting the training data and its test error eventually rises (20.3 explains, and fixes it with early stopping).

# %%
rf_big = RandomForestRegressor(n_estimators=800, max_features=0.3, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
Pt = np.array([t.predict(X_te) for t in rf_big.estimators_])
cum = np.cumsum(Pt, 0) / np.arange(1, 801)[:, None]
rf_curve = np.array([mse(cum[b - 1]) for b in (1, 10, 50, 100, 200, 400, 800)])
gb = GradientBoostingRegressor(n_estimators=800, learning_rate=1.0, max_depth=3, random_state=0).fit(X_tr, y_tr)
gb_curve = np.array([mse(p) for p in gb.staged_predict(X_te)])
print("trees:        " + "  ".join(f"{b:>6d}" for b in (1, 10, 50, 100, 200, 400, 800)))
print("forest MSE:   " + "  ".join(f"{v:6.2f}" for v in rf_curve))
print("boosting MSE: " + "  ".join(f"{gb_curve[b - 1]:6.2f}" for b in (1, 10, 50, 100, 200, 400, 800)))
print(f"boosting: best at round {gb_curve.argmin() + 1} ({gb_curve.min():.2f}), round 800 {gb_curve[-1]:.2f}; training MSE at 800: "
      f"{np.mean((gb.predict(X_tr) - y_tr) ** 2):.4f}")
assert rf_curve[-1] <= rf_curve[3] + 0.02, "the forest doesn't get worse with more trees"
assert gb_curve[-1] > gb_curve.min() * 1.1, "aggressive boosting does"
sane = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=0).fit(X_tr, y_tr)
print(f"for fairness: boosting with a sane learning rate (0.1, 300 rounds) gets {mse(sane.predict(X_te)):.2f}, "
      f"better than the forest's {rf_curve[-1]:.2f}. That's 20.3's story.")

# %% [markdown]
# ## 7. Leaf size, model size, accuracy

# %%
for leaf in (1, 5, 20):
    m = RandomForestRegressor(n_estimators=200, max_features=0.3, min_samples_leaf=leaf, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    size = len(pickle.dumps(m)) / 1e6
    print(f"min_samples_leaf={leaf:2d}: test MSE {mse(m.predict(X_te)):.2f}, pickled size {size:5.1f} MB, "
          f"nodes per tree {np.mean([t.tree_.node_count for t in m.estimators_]):.0f}")
print("25x smaller for a real accuracy cost on this fairly clean data; on noisier data the cost is often near zero. Measure, then decide.")

print("\nAll checks passed.")
