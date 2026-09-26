# %% [markdown]
# # Lab 20.1: Decision trees from scratch
#
# 1. A CART classifier: Gini, sorted-sweep split search, recursion. Same tree as scikit-learn.
# 2. Depth vs training and test accuracy.
# 3. Cost-complexity pruning, alpha chosen by cross-validation.
# 4. Instability: bootstrap the data, count the different root splits.
# 5. Monotone transforms don't change the tree; diagonal boundaries need staircases.
# 6. No extrapolation.
# 7. Impurity importance vs permutation importance with a planted random ID.

# %%
from collections import Counter

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

rng = np.random.default_rng(201)
bc = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(bc.data, bc.target, test_size=0.3, random_state=0, stratify=bc.target)

# %% [markdown]
# ## 1. CART from scratch

# %%
def gini_from_counts(counts):
    n = counts.sum(-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    return 1 - (p**2).sum(-1)


def best_split(X, y, n_classes, min_leaf):
    n = len(y)
    parent = gini_from_counts(np.bincount(y, minlength=n_classes)[None])[0]
    best = (0.0, None, None)                                    # (gain, feature, threshold)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        xs, ys = X[order, j], y[order]
        left = np.cumsum(np.eye(n_classes, dtype=int)[ys], axis=0)[:-1]   # class counts left of each cut
        right = left[-1] + np.eye(n_classes, dtype=int)[ys[-1]] - left
        n_left = np.arange(1, n)
        impurity = (n_left * gini_from_counts(left) + (n - n_left) * gini_from_counts(right)) / n
        valid = (xs[1:] > xs[:-1]) & (n_left >= min_leaf) & (n - n_left >= min_leaf)
        if not valid.any():
            continue
        i = np.argmin(np.where(valid, impurity, np.inf))
        gain = parent - impurity[i]
        if gain > best[0] + 1e-12:
            best = (gain, j, (xs[i] + xs[i + 1]) / 2)
    return best


def grow(X, y, n_classes, depth, max_depth, min_leaf=1):
    counts = np.bincount(y, minlength=n_classes)
    if depth == max_depth or counts.max() == len(y):
        return {"leaf": counts}
    gain, j, t = best_split(X, y, n_classes, min_leaf)
    if j is None:
        return {"leaf": counts}
    m = X[:, j] <= t
    return {"feature": j, "threshold": t,
            "left": grow(X[m], y[m], n_classes, depth + 1, max_depth, min_leaf),
            "right": grow(X[~m], y[~m], n_classes, depth + 1, max_depth, min_leaf)}


def predict_one(node, x):
    while "leaf" not in node:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["leaf"].argmax()


tree = grow(X_tr, y_tr, 2, 0, max_depth=4)
mine = np.array([predict_one(tree, x) for x in X_te])
sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
print(f"root split: mine feature {tree['feature']} ({bc.feature_names[tree['feature']]}) <= {tree['threshold']:.4f}; "
      f"scikit-learn feature {sk.tree_.feature[0]} <= {sk.tree_.threshold[0]:.4f}")
print(f"test accuracy: mine {np.mean(mine == y_te):.4f}, scikit-learn {sk.score(X_te, y_te):.4f}; agreement {np.mean(mine == sk.predict(X_te)):.3f}")
assert tree["feature"] == sk.tree_.feature[0] and np.isclose(tree["threshold"], sk.tree_.threshold[0])
assert np.mean(mine == sk.predict(X_te)) > 0.97                 # ties between equally good splits can differ deeper down

# %% [markdown]
# ## 2. Depth

# %%
for depth in (1, 2, 3, 5, 8, None):
    m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    print(f"max_depth={str(depth):>4}: leaves {m.get_n_leaves():3d}, train {m.score(X_tr, y_tr):.3f}, test {m.score(X_te, y_te):.3f}")
assert DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr).score(X_tr, y_tr) == 1.0

# %% [markdown]
# ## 3. Cost-complexity pruning

# %%
path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
alphas = path.ccp_alphas[:-1]                                   # the last alpha prunes to the root
cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = np.array([cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=0), X_tr, y_tr, cv=cv).mean() for a in alphas])
best_alpha = alphas[scores.argmax()]
pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=0).fit(X_tr, y_tr)
full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print(f"{len(alphas)} subtrees on the pruning path; CV picks alpha={best_alpha:.4f}")
print(f"full tree: {full.get_n_leaves()} leaves, test {full.score(X_te, y_te):.3f}; pruned: {pruned.get_n_leaves()} leaves, test {pruned.score(X_te, y_te):.3f}")
assert pruned.get_n_leaves() < full.get_n_leaves()

# %% [markdown]
# ## 4. Instability

# %%
roots = Counter()
for b in range(200):
    idx = rng.integers(0, len(X_tr), len(X_tr))
    t = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr[idx], y_tr[idx])
    roots[str(bc.feature_names[t.tree_.feature[0]])] += 1
print(f"root split feature across 200 bootstrap samples: {dict(roots.most_common())}")
assert len(roots) >= 3, "small changes in the data change the whole tree"

preds = np.array([DecisionTreeClassifier(random_state=0).fit(X_tr[i], y_tr[i]).predict(X_te)
                  for i in [rng.integers(0, len(X_tr), len(X_tr)) for _ in range(50)]])
disagree = np.mean(preds.min(0) != preds.max(0))
print(f"fully grown trees on bootstrap samples disagree on {disagree:.0%} of test points (bagging, 20.2, averages this away)")

# %% [markdown]
# ## 5. Monotone transforms; diagonal boundaries

# %%
Xpos = X_tr - X_tr.min(0) + 1
t_raw = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xpos, y_tr)
t_log = DecisionTreeClassifier(max_depth=5, random_state=0).fit(np.log(Xpos), y_tr)
Xpos_te = np.clip(X_te - X_tr.min(0) + 1, 1e-9, None)
assert (t_raw.predict(Xpos_te) == t_log.predict(np.log(Xpos_te))).all()
assert (t_raw.tree_.feature == t_log.tree_.feature).all()
print("log-transforming every feature: identical tree structure and identical predictions")

Xd = rng.uniform(0, 1, (4000, 2)); yd = (Xd[:, 0] > Xd[:, 1]).astype(int)
Xd_te = rng.uniform(0, 1, (4000, 2)); yd_te = (Xd_te[:, 0] > Xd_te[:, 1]).astype(int)
for leaves in (2, 8, 32, 128):
    t = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=0).fit(Xd, yd)
    print(f"diagonal boundary, {leaves:3d} leaves: test accuracy {t.score(Xd_te, yd_te):.3f}")
print(f"logistic regression (one direction): {LogisticRegression(C=100).fit(Xd, yd).score(Xd_te, yd_te):.3f}")
assert DecisionTreeClassifier(max_leaf_nodes=8, random_state=0).fit(Xd, yd).score(Xd_te, yd_te) < 0.97

# %% [markdown]
# ## 6. No extrapolation

# %%
year = np.arange(2000, 2025)[:, None].astype(float)
price = 100 * 1.04 ** (year[:, 0] - 2000) + rng.normal(0, 3, len(year))
future = np.arange(2025, 2031)[:, None].astype(float)
tr_pred = DecisionTreeRegressor(min_samples_leaf=2).fit(year, price).predict(future)
lin_pred = np.exp(LinearRegression().fit(year, np.log(price)).predict(future))
truth = 100 * 1.04 ** (future[:, 0] - 2000)
print("year  truth  tree  log-linear")
for f, t, a, b in zip(future[:, 0], truth, tr_pred, lin_pred):
    print(f"{int(f)}  {t:5.0f}  {a:5.0f}  {b:5.0f}")
assert np.ptp(tr_pred) < 1e-9, "the tree predicts the same constant for every future year"

# %% [markdown]
# ## 7. Importance with a planted random ID
#
# Noisy labels (so a deep tree has noise to memorize), 3 real features, a random unique ID, and a random coin flip.

# %%
n7 = 3000
real = rng.normal(size=(n7, 3))
y7 = ((real @ np.array([1.0, 0.7, 0.4]) + rng.normal(0, 1.0, n7)) > 0).astype(int)
X7 = np.column_stack([real, rng.permutation(n7), rng.integers(0, 2, n7)])
names = ["real_1", "real_2", "real_3", "random_id", "random_coin"]
X7_tr, X7_te, y7_tr, y7_te = train_test_split(X7, y7, test_size=0.5, random_state=0)
deep = DecisionTreeClassifier(random_state=0).fit(X7_tr, y7_tr)
imp_impurity = deep.feature_importances_
imp_perm = permutation_importance(deep, X7_te, y7_te, n_repeats=20, random_state=0).importances_mean
for nm, a, b in zip(names, imp_impurity, imp_perm):
    print(f"{nm:12s} impurity {a:.3f}   permutation (test) {b:+.3f}")
assert imp_impurity[3] > 0.5 * imp_impurity[2] and imp_impurity[3] > 5 * imp_impurity[4], \
    "a pure-noise ID gets impurity importance comparable to a real feature (and far above low-cardinality noise)"
assert imp_perm[3] < 0.01 and imp_perm[2] > imp_perm[3], "held-out permutation importance sees through it"

print("\nAll checks passed.")
