# %% [markdown]
# # Lab 20.3: Boosting from scratch
#
# 1. Gradient boosting for squared loss: fit trees to residuals. Matches scikit-learn exactly.
# 2. Gradient boosting for log loss with Newton leaf values. Matches scikit-learn.
# 3. AdaBoost from scratch, and what label noise does to it vs log-loss boosting.
# 4. Learning rate x rounds, with early stopping.
# 5. The second-order view: leaf weights and split gain by hand, checked against XGBoost's own tree.
# 6. Tree depth = interaction order: stumps can't learn an interaction.
# 7. Quantile loss: prediction intervals that cover.

# %%
import json

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_friedman1
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

rng = np.random.default_rng(203)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# %% [markdown]
# ## 1. Squared loss: trees on residuals

# %%
X, y = make_friedman1(n_samples=2000, noise=1.0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
nu, M, depth = 0.1, 200, 3

F0 = y_tr.mean()
F_tr, F_te = np.full(len(y_tr), F0), np.full(len(y_te), F0)
tie_rng = np.random.RandomState(0)                              # scikit-learn shares one RNG across trees (it only breaks ties)
for m in range(M):
    residual = y_tr - F_tr                                      # the negative gradient of 1/2 (y - F)^2
    h = DecisionTreeRegressor(max_depth=depth, random_state=tie_rng).fit(X_tr, residual)
    F_tr += nu * h.predict(X_tr); F_te += nu * h.predict(X_te)

sk = GradientBoostingRegressor(n_estimators=M, learning_rate=nu, max_depth=depth, random_state=0).fit(X_tr, y_tr)
print(f"test MSE: mine {np.mean((F_te - y_te) ** 2):.4f}, scikit-learn {np.mean((sk.predict(X_te) - y_te) ** 2):.4f}")
assert np.allclose(F_te, sk.predict(X_te), atol=1e-8), "ten lines of code, same model"

# %% [markdown]
# ## 2. Log loss with Newton leaf values

# %%
Xc = rng.normal(size=(3000, 6))
yc = (rng.random(3000) < sigmoid(Xc[:, 0] * Xc[:, 1] + np.sin(2 * Xc[:, 2]) + 0.5 * Xc[:, 3])).astype(int)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.5, random_state=0)


def gb_logloss(X, y, Xt, M=150, nu=0.1, depth=3):
    p0 = y.mean()
    F, Ft = np.full(len(y), np.log(p0 / (1 - p0))), np.full(len(Xt), np.log(p0 / (1 - p0)))
    tie_rng = np.random.RandomState(0)
    for m in range(M):
        p = sigmoid(F)
        tree = DecisionTreeRegressor(max_depth=depth, random_state=tie_rng).fit(X, y - p)
        leaf, leaf_t = tree.apply(X), tree.apply(Xt)
        vals = {}
        for l in np.unique(leaf):                               # replace the tree's leaf means by Newton steps
            idx = leaf == l
            vals[l] = (y[idx] - p[idx]).sum() / max((p[idx] * (1 - p[idx])).sum(), 1e-12)
        F += nu * np.array([vals[l] for l in leaf]); Ft += nu * np.array([vals[l] for l in leaf_t])
    return F, Ft


F_train, Ft = gb_logloss(Xc_tr, yc_tr, Xc_te)
skc = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=0).fit(Xc_tr, yc_tr)
diff_tr = np.abs(F_train - skc.decision_function(Xc_tr)).max()
diff_te = np.abs(sigmoid(Ft) - skc.predict_proba(Xc_te)[:, 1])
print(f"log-loss boosting: accuracy mine {np.mean((Ft > 0) == yc_te):.4f}, scikit-learn {skc.score(Xc_te, yc_te):.4f}")
print(f"training-set log-odds: max difference {diff_tr:.1e}; test probabilities: median difference {np.median(diff_te):.1e}, max {diff_te.max():.3f}")
# The training fits are identical. On test data a few points differ: now and then two splits produce exactly the same
# training partition (a tie), the tie is broken by a random draw, and the two libraries' draws aren't aligned. The
# thresholds differ only where there's no training data, which is exactly where test points can fall.
assert diff_tr < 1e-8 and np.median(diff_te) < 1e-6

# %% [markdown]
# ## 3. AdaBoost, and label noise

# %%
def adaboost(X, y, M=200):                                     # y in {-1, +1}, stumps, discrete SAMME
    w = np.full(len(y), 1 / len(y))
    learners, alphas = [], []
    for m in range(M):
        stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=w)
        pred = stump.predict(X)
        err = w[pred != y].sum() / w.sum()
        if err >= 0.5:
            break
        a = 0.5 * np.log((1 - err) / max(err, 1e-12))
        w *= np.exp(-a * y * pred); w /= w.sum()
        learners.append(stump); alphas.append(a)
    return learners, np.array(alphas), w


ys = 2 * yc - 1
ys_tr, ys_te = 2 * yc_tr - 1, 2 * yc_te - 1
learners, alphas, _ = adaboost(Xc_tr, ys_tr)
F_ada = sum(a * l.predict(Xc_te) for a, l in zip(alphas, learners))
sk_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, learning_rate=1.0, random_state=0).fit(Xc_tr, ys_tr)
print(f"AdaBoost: accuracy mine {np.mean(np.sign(F_ada) == ys_te):.4f}, scikit-learn {sk_ada.score(Xc_te, ys_te):.4f}")
assert abs(np.mean(np.sign(F_ada) == ys_te) - sk_ada.score(Xc_te, ys_te)) < 0.01

# clean labels vs 20% flipped labels (test labels stay clean)
Xn = rng.normal(size=(4000, 5))
yn = (Xn[:, 0] + Xn[:, 1] > 0).astype(int)
Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(Xn, yn, test_size=0.5, random_state=0)
flip = rng.random(len(yn_tr)) < 0.2
yn_noisy = np.where(flip, 1 - yn_tr, yn_tr)
accs = {}
for name, labels in [("clean labels", yn_tr), ("20% flipped", yn_noisy)]:
    lrn, al, w_final = adaboost(Xn_tr, 2 * labels - 1, M=400)
    acc_ada = np.mean(np.sign(sum(a * l.predict(Xn_te) for a, l in zip(al, lrn))) == 2 * yn_te - 1)
    acc_gb = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=1, random_state=0).fit(Xn_tr, labels).score(Xn_te, yn_te)
    note = ""
    if name == "20% flipped":
        note = f"; weight on flipped points {w_final[flip].sum():.0%} (they're {flip.mean():.0%} of the data)"
    print(f"{name:13s}: AdaBoost {acc_ada:.3f}, log-loss boosting {acc_gb:.3f}{note}")
    accs[name] = (acc_ada, acc_gb)
drop_ada = accs["clean labels"][0] - accs["20% flipped"][0]
drop_gb = accs["clean labels"][1] - accs["20% flipped"][1]
print(f"accuracy lost to label noise: AdaBoost {drop_ada:.3f}, log-loss boosting {drop_gb:.3f}")
assert w_final[flip].sum() > 1.5 * flip.mean(), "AdaBoost pours its attention into the mislabeled points"
assert drop_ada > drop_gb

# %% [markdown]
# ## 4. Learning rate, rounds, early stopping

# %%
Xv_tr, Xv_val, yv_tr, yv_val = train_test_split(X_tr, y_tr, test_size=0.3, random_state=1)
results = {}
for lr in (1.0, 0.3, 0.1, 0.03):
    g = GradientBoostingRegressor(n_estimators=3000, learning_rate=lr, max_depth=3, subsample=0.8, random_state=0).fit(Xv_tr, yv_tr)
    val = np.array([np.mean((p - yv_val) ** 2) for p in g.staged_predict(Xv_val)])
    best = int(val.argmin()) + 1
    test_at_best = np.mean((list(g.staged_predict(X_te))[best - 1] - y_te) ** 2)
    results[lr] = (best, val.min(), test_at_best, val[-1])
    print(f"lr={lr:5}: best round {best:5d}, validation MSE {val.min():.3f} (at round 3000: {val[-1]:.3f}), test MSE at the best round {test_at_best:.3f}")
assert results[0.1][2] < results[1.0][2] and results[1.0][3] > results[1.0][1] * 1.1
es = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.1, max_depth=3, subsample=0.8, validation_fraction=0.2,
                               n_iter_no_change=50, random_state=0).fit(X_tr, y_tr)
print(f"built-in early stopping (lr=0.1): stopped at {es.n_estimators_} rounds, test MSE {np.mean((es.predict(X_te) - y_te) ** 2):.3f}")

# %% [markdown]
# ## 5. The second-order view, checked against XGBoost
#
# One boosting round of depth 1 on log loss, starting from base_score = 0.5 (logit 0): g = p - y, h = p(1 - p).

# %%
x1 = rng.normal(size=(500, 1))
y1 = (rng.random(500) < sigmoid(2 * x1[:, 0])).astype(float)
lam, eta = 1.0, 0.3
booster = xgb.train({"objective": "binary:logistic", "max_depth": 1, "eta": eta, "lambda": lam, "base_score": 0.5,
                     "tree_method": "exact", "min_child_weight": 0}, xgb.DMatrix(x1, label=y1), num_boost_round=1)
tree = json.loads(booster.get_dump(dump_format="json")[0])
thr = tree["split_condition"]
leaves = {c["nodeid"]: c["leaf"] for c in tree["children"]}
left_id, right_id = tree["yes"], tree["no"]

p = np.full(500, 0.5)
g, h = p - y1, p * (1 - p)
L = x1[:, 0] < thr
w = lambda m: -g[m].sum() / (h[m].sum() + lam)
print(f"XGBoost split at x < {thr:.4f}")
print(f"left leaf: mine {eta * w(L):.6f}, XGBoost {leaves[left_id]:.6f}; right leaf: mine {eta * w(~L):.6f}, XGBoost {leaves[right_id]:.6f}")
assert np.isclose(eta * w(L), leaves[left_id], atol=1e-5) and np.isclose(eta * w(~L), leaves[right_id], atol=1e-5)

score = lambda m: g[m].sum() ** 2 / (h[m].sum() + lam)
order = np.argsort(x1[:, 0])
xs = x1[order, 0]
gains = [0.5 * (score(x1[:, 0] < t) + score(x1[:, 0] >= t) - score(np.ones(500, bool)))
         for t in (xs[1:] + xs[:-1]) / 2]
best_t = ((xs[1:] + xs[:-1]) / 2)[int(np.argmax(gains))]
print(f"best split by the gain formula: x < {best_t:.4f} (gain {max(gains):.3f})")
assert abs(best_t - thr) < 1e-3 or (x1[:, 0] < best_t).sum() == (x1[:, 0] < thr).sum()

# %% [markdown]
# ## 6. Depth is interaction order

# %%
Xi = rng.uniform(-1, 1, size=(4000, 4))
yi = Xi[:, 0] * Xi[:, 1] + 0.1 * rng.normal(size=4000)         # a pure interaction: no additive part at all
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.5, random_state=0)
for d in (1, 2, 3):
    g_ = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=d, random_state=0).fit(Xi_tr, yi_tr)
    print(f"max_depth={d}: test R^2 {g_.score(Xi_te, yi_te):.3f}")
assert GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=1, random_state=0).fit(Xi_tr, yi_tr).score(Xi_te, yi_te) < 0.1
assert GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=2, random_state=0).fit(Xi_tr, yi_tr).score(Xi_te, yi_te) > 0.8

# %% [markdown]
# ## 7. Quantile loss for intervals

# %%
xq = rng.uniform(0, 10, size=(4000, 1))
yq = np.sin(xq[:, 0]) * 3 + rng.normal(0, 0.3 + 0.2 * xq[:, 0])   # noise grows with x
xq_tr, xq_te, yq_tr, yq_te = train_test_split(xq, yq, test_size=0.5, random_state=0)
lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(xq_tr, yq_tr)
hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(xq_tr, yq_tr)
inside = (yq_te >= lo.predict(xq_te)) & (yq_te <= hi.predict(xq_te))
width_small = np.mean(hi.predict(xq_te[xq_te[:, 0] < 2]) - lo.predict(xq_te[xq_te[:, 0] < 2]))
width_large = np.mean(hi.predict(xq_te[xq_te[:, 0] > 8]) - lo.predict(xq_te[xq_te[:, 0] > 8]))
print(f"90% interval coverage on test data: {inside.mean():.3f}; mean width for x<2 {width_small:.2f}, for x>8 {width_large:.2f}")
# a bit under 90% is typical: each quantile model is fit to the training data, and in-sample quantiles are optimistic.
# Conformal calibration on a held-out set (23.4) fixes the coverage.
assert 0.85 < inside.mean() < 0.95 and width_large > 2 * width_small

print("\nAll checks passed.")
