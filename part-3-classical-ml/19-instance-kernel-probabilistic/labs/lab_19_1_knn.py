# %% [markdown]
# # Lab 19.1: k-nearest neighbors, from scratch to approximate search
#
# 1. A vectorized k-NN classifier, checked against scikit-learn.
# 2. Choosing k by cross-validation; the bias-variance curve.
# 3. Scaling: the telecom data with and without it.
# 4. Irrelevant dimensions: accuracy as noise features are added.
# 5. Cover-Hart: 1-NN error vs the Bayes error, on a problem where we know the Bayes error.
# 6. KD-tree vs brute force, in low and high dimensions.
# 7. A tiny IVF index: recall@10 vs the fraction of the data scanned.
# 8. k-NN distance as an anomaly score.

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(191)

# %% [markdown]
# ## 1. k-NN in a few lines

# %%
def sq_dists(Q, X):
    d2 = (Q**2).sum(1)[:, None] - 2 * Q @ X.T + (X**2).sum(1)[None, :]
    return np.maximum(d2, 0.0)                                  # cancellation can make tiny distances negative


def knn_predict(X_tr, y_tr, Q, k):
    idx = np.argpartition(sq_dists(Q, X_tr), k - 1, axis=1)[:, :k]   # O(n) selection instead of a full sort
    votes = y_tr[idx]
    n_classes = y_tr.max() + 1
    counts = np.apply_along_axis(np.bincount, 1, votes, minlength=n_classes)
    return counts.argmax(1)


digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
mine = knn_predict(X_tr, y_tr, X_te, k=3)
sk = KNeighborsClassifier(n_neighbors=3, algorithm="brute").fit(X_tr, y_tr).predict(X_te)
print(f"digits, k=3: my accuracy {np.mean(mine == y_te):.4f}, scikit-learn {np.mean(sk == y_te):.4f}, agreement {np.mean(mine == sk):.4f}")
assert np.mean(mine == sk) > 0.995 and np.mean(mine == y_te) > 0.97    # ties may break differently

# %% [markdown]
# ## 2. k and the bias-variance trade-off

# %%
cv = StratifiedKFold(5, shuffle=True, random_state=0)
Xd, yd = digits.data, digits.target
for k in (1, 3, 5, 11, 31, 101, 301):
    tr_acc = KNeighborsClassifier(n_neighbors=k).fit(Xd, yd).score(Xd, yd)
    cv_acc = cross_val_score(KNeighborsClassifier(n_neighbors=k), Xd, yd, cv=cv).mean()
    print(f"k={k:4d}: training accuracy {tr_acc:.3f}, CV accuracy {cv_acc:.3f}")
assert KNeighborsClassifier(n_neighbors=1).fit(Xd, yd).score(Xd, yd) == 1.0, "1-NN on its own training set: every point is its own neighbor"

# %% [markdown]
# ## 3. Scaling
#
# First the telecom data (the dataset k-NN is traditionally taught on): who decides the neighbors?

# %%
tele = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "telecust_1000.csv")
Xt, yt = tele.drop(columns="custcat").to_numpy(float), tele["custcat"].to_numpy()
inc = tele.columns.get_loc("income")


def income_share(X):
    i, j = rng.integers(0, len(X), (2, 5000))
    diff2 = (X[i] - X[j]) ** 2
    return np.mean(diff2[:, inc] / diff2.sum(1).clip(1e-12))


Xt_std = StandardScaler().fit_transform(Xt)
print(f"share of squared distance contributed by income: raw {income_share(Xt):.0%}, standardized {income_share(Xt_std):.0%}")
raw = max(cross_val_score(KNeighborsClassifier(k), Xt, yt, cv=cv).mean() for k in (5, 15, 31, 51))
scaled = max(cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), Xt, yt, cv=cv).mean() for k in (5, 15, 31, 51))
print(f"best CV accuracy: raw {raw:.3f}, standardized {scaled:.3f} (majority class {np.bincount(yt).max() / len(yt):.3f})")
print("  -> scaling changes who the neighbors are, but here no set of neighbors predicts the category well (see 17.4)")
assert income_share(Xt) > 0.5 > income_share(Xt_std)

# Where it decides everything: the informative feature is small-scale, an irrelevant one is in the thousands
n3 = 3000
signal = rng.uniform(0, 1, n3)
balance = rng.normal(5000, 2000, n3)                          # irrelevant, huge scale
y3 = (signal > 0.5).astype(int)
X3 = np.column_stack([signal, balance])
acc_raw = cross_val_score(KNeighborsClassifier(15), X3, y3, cv=cv).mean()
acc_std = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(15)), X3, y3, cv=cv).mean()
print(f"synthetic: raw {acc_raw:.3f}, standardized {acc_std:.3f}")
assert acc_raw < 0.6 and acc_std > 0.9

# %% [markdown]
# ## 4. Irrelevant dimensions

# %%
n = 2000
X_sig = rng.normal(size=(n, 5))
y_sig = (X_sig[:, 0] + X_sig[:, 1] * X_sig[:, 2] + np.sin(2 * X_sig[:, 3]) > 0).astype(int)
accs = {}
for n_noise in (0, 5, 20, 50, 200):
    Xn = np.column_stack([X_sig, rng.normal(size=(n, n_noise))])
    accs[n_noise] = cross_val_score(KNeighborsClassifier(15), Xn, y_sig, cv=cv).mean()
    print(f"5 signal + {n_noise:3d} noise features: CV accuracy {accs[n_noise]:.3f}")
assert accs[0] - accs[200] > 0.15

# %% [markdown]
# ## 5. Cover and Hart
#
# Two Gaussian classes with known means and identity covariance: the Bayes classifier is a hyperplane and its error is computable.

# %%
d, delta = 2, 1.5
bayes_err = stats.norm.cdf(-delta / 2)                          # distance between means = delta
for n_tr in (100, 1000, 20_000):
    Xa = np.vstack([rng.normal(0, 1, (n_tr // 2, d)), rng.normal(0, 1, (n_tr // 2, d)) + [delta, 0]])
    ya = np.r_[np.zeros(n_tr // 2, int), np.ones(n_tr // 2, int)]
    Xq = np.vstack([rng.normal(0, 1, (10_000, d)), rng.normal(0, 1, (10_000, d)) + [delta, 0]])
    yq = np.r_[np.zeros(10_000, int), np.ones(10_000, int)]
    e1 = 1 - KNeighborsClassifier(1).fit(Xa, ya).score(Xq, yq)
    ek = 1 - KNeighborsClassifier(int(np.sqrt(n_tr))).fit(Xa, ya).score(Xq, yq)
    print(f"n={n_tr:6d}: 1-NN error {e1:.3f}, sqrt(n)-NN error {ek:.3f}; Bayes {bayes_err:.3f}, 2x Bayes {2 * bayes_err:.3f}")
assert e1 < 2 * bayes_err * 1.05 and ek < bayes_err + 0.01, "1-NN within twice Bayes; growing k approaches Bayes"

# %% [markdown]
# ## 6. KD-tree vs brute force

# %%
def query_time(algorithm, X, Q):
    nn = NearestNeighbors(n_neighbors=10, algorithm=algorithm).fit(X)
    t0 = time.perf_counter(); nn.kneighbors(Q); return time.perf_counter() - t0


times = {}
for dim in (3, 100):
    X = rng.normal(size=(20_000, dim)); Q = rng.normal(size=(200, dim))
    times[dim] = (query_time("kd_tree", X, Q), query_time("brute", X, Q))
    print(f"d={dim:3d}: KD-tree {times[dim][0] * 1000:7.1f} ms, brute force {times[dim][1] * 1000:7.1f} ms")
assert times[3][0] < times[3][1], "low dimension: the tree wins"
assert times[100][0] > 0.5 * times[100][1], "high dimension: the tree's advantage is gone"

# %% [markdown]
# ## 7. A tiny IVF index
#
# Cluster the database with k-means; at query time scan only the `nprobe` clusters whose centroids are nearest.
# Data with cluster structure (like real embeddings) in 32 dimensions.

# %%
n_db, dim, n_centers = 40_000, 32, 64
centers = rng.normal(0, 4, size=(n_centers, dim))
db = centers[rng.integers(0, n_centers, n_db)] + rng.normal(size=(n_db, dim))
queries = centers[rng.integers(0, n_centers, 300)] + rng.normal(size=(300, dim))
exact = NearestNeighbors(n_neighbors=10, algorithm="brute").fit(db).kneighbors(queries, return_distance=False)

n_lists = 128
km = KMeans(n_lists, n_init=1, random_state=0).fit(db)
lists = [np.flatnonzero(km.labels_ == c) for c in range(n_lists)]


def ivf_search(q, nprobe, k=10):
    probe = np.argsort(sq_dists(q[None], km.cluster_centers_)[0])[:nprobe]
    cand = np.concatenate([lists[c] for c in probe])
    top = cand[np.argsort(sq_dists(q[None], db[cand])[0])[:k]]
    return top, len(cand)


for nprobe in (1, 4, 16):
    recall, scanned = [], []
    for q, truth in zip(queries, exact):
        found, m = ivf_search(q, nprobe)
        recall.append(len(set(found) & set(truth)) / 10); scanned.append(m / n_db)
    print(f"nprobe={nprobe:2d}: recall@10 {np.mean(recall):.3f}, scanned {np.mean(scanned):.1%} of the database")
    if nprobe == 16:
        assert np.mean(recall) > 0.95 and np.mean(scanned) < 0.3

# %% [markdown]
# ## 8. Distance to the k-th neighbor as an anomaly score

# %%
normal = rng.normal(size=(3000, 4))
anomalies = rng.uniform(-6, 6, size=(60, 4))
anomalies = anomalies[np.linalg.norm(anomalies, axis=1) > 4]
Xan = np.vstack([normal, anomalies]); yan = np.r_[np.zeros(len(normal)), np.ones(len(anomalies))]
nn = NearestNeighbors(n_neighbors=11).fit(Xan)
dist, _ = nn.kneighbors(Xan)
score = dist[:, -1]                                             # 10th neighbor (the first is the point itself)
print(f"k-NN anomaly score: ROC-AUC {roc_auc_score(yan, score):.3f} on {len(anomalies)} planted anomalies")
assert roc_auc_score(yan, score) > 0.95

print("\nAll checks passed.")
