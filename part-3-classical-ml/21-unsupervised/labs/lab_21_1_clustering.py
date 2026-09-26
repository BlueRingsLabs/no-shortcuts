# %% [markdown]
# # Lab 21.1: Clustering, and how to not fool yourself with it
#
# 1. k-means from scratch (Lloyd + k-means++), against scikit-learn; random init getting stuck.
# 2. Choosing k: elbow and silhouette on real clusters, and on pure noise.
# 3. Shapes: k-means vs DBSCAN vs single/Ward linkage on crescents and on a bridge.
# 4. Varying density: DBSCAN's single radius vs HDBSCAN.
# 5. Stability: bootstrap ARI separates structure from noise.
# 6. A customer segmentation on the shop data, with the honest caveats.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "part-2-data-engineering" / "_shared"))
from shop import make_shop  # noqa: E402

rng = np.random.default_rng(211)

# %% [markdown]
# ## 1. k-means from scratch

# %%
def kmeans_pp_init(X, k, rng):
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])     # D^2 sampling
    return np.array(centers)


def lloyd(X, centers, max_iter=300):
    history = []
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
        labels = d2.argmin(1)
        history.append(d2.min(1).sum())
        new = np.array([X[labels == j].mean(0) if np.any(labels == j) else centers[j] for j in range(len(centers))])
        if np.allclose(new, centers):
            break
        centers = new
    return labels, centers, history


X, y_true = make_blobs(n_samples=1500, centers=6, cluster_std=0.6, center_box=(-12, 12), random_state=6)
runs = [lloyd(X, kmeans_pp_init(X, 6, rng)) for _ in range(10)]
for _, _, hist in runs:
    assert all(a >= b - 1e-9 for a, b in zip(hist, hist[1:])), "the objective never increases"
labels, centers, hist = min(runs, key=lambda r: r[2][-1])
sk = KMeans(6, n_init=10, random_state=0).fit(X)
print(f"k-means++ + Lloyd: single runs ended between {min(r[2][-1] for r in runs):.1f} and {max(r[2][-1] for r in runs):.1f}; "
      f"best of 10 {hist[-1]:.1f}; scikit-learn (best of 10) {sk.inertia_:.1f}; ARI vs truth {adjusted_rand_score(y_true, labels):.3f}")
assert abs(hist[-1] - sk.inertia_) / sk.inertia_ < 0.01

rand_J, pp_J = [], []
for s in range(30):
    r = np.random.default_rng(s)
    rand_J.append(lloyd(X, X[r.choice(len(X), 6, replace=False)])[2][-1])
    pp_J.append(lloyd(X, kmeans_pp_init(X, 6, r))[2][-1])
rand_J, pp_J = np.array(rand_J), np.array(pp_J)
best = min(rand_J.min(), pp_J.min())
print(f"30 single runs: random init stuck (>5% above best) {np.mean(rand_J > 1.05 * best):.0%} of the time, "
      f"k-means++ {np.mean(pp_J > 1.05 * best):.0%}")
assert np.mean(pp_J > 1.05 * best) < np.mean(rand_J > 1.05 * best)

# %% [markdown]
# ## 2. Choosing k, on clusters and on noise

# %%
noise = rng.uniform(-10, 10, size=(1500, 2))
for name, data in [("6 real blobs", X), ("uniform noise", noise)]:
    rows = []
    for k in range(2, 10):
        km = KMeans(k, n_init=5, random_state=0).fit(data)
        rows.append((k, km.inertia_, silhouette_score(data, km.labels_)))
    best_k = max(rows, key=lambda r: r[2])
    print(f"{name:14s}: silhouette by k " + " ".join(f"{k}:{s:.2f}" for k, _, s in rows) + f"  -> 'best' k={best_k[0]}")
    if name == "uniform noise":
        noise_sil = best_k[2]
    else:
        assert best_k[0] == 6
print(f"noise gets a best silhouette of {noise_sil:.2f}: k-means always finds *something*")
assert noise_sil > 0.3

# and even with real, well-separated blobs, the silhouette's favorite k is often not the true one
right = 0
for layout in range(10):
    Xl, _ = make_blobs(n_samples=1500, centers=6, cluster_std=0.6, center_box=(-12, 12), random_state=layout)
    sil_l = {k: silhouette_score(Xl, KMeans(k, n_init=5, random_state=0).fit_predict(Xl)) for k in range(2, 10)}
    right += max(sil_l, key=sil_l.get) == 6
print(f"10 random layouts of 6 well-separated blobs: the silhouette picked k=6 in {right} of them (nearby blobs look like one)")
assert right < 10

# %% [markdown]
# ## 3. Shapes

# %%
Xm, ym = make_moons(n_samples=800, noise=0.06, random_state=0)
Xm = StandardScaler().fit_transform(Xm)
results = {
    "k-means": KMeans(2, n_init=10, random_state=0).fit_predict(Xm),
    "Ward": AgglomerativeClustering(2, linkage="ward").fit_predict(Xm),
    "single linkage": AgglomerativeClustering(2, linkage="single").fit_predict(Xm),
    "DBSCAN": DBSCAN(eps=0.3, min_samples=5).fit_predict(Xm),
}
for k, v in results.items():
    print(f"crescents, {k:15s}: ARI {adjusted_rand_score(ym, v):.3f}")
assert adjusted_rand_score(ym, results["k-means"]) < 0.6 and adjusted_rand_score(ym, results["DBSCAN"]) > 0.95

# two blobs joined by a thin bridge of points: single linkage chains through it
A = rng.normal([0, 0], 0.5, size=(300, 2)); B = rng.normal([6, 0], 0.5, size=(300, 2))
bridge = np.column_stack([np.linspace(0.8, 5.2, 25), rng.normal(0, 0.05, 25)])
Xb = np.vstack([A, B, bridge]); yb = np.r_[np.zeros(300), np.ones(300), (bridge[:, 0] > 3).astype(float)]
for link in ("single", "ward"):
    lab = AgglomerativeClustering(2, linkage=link).fit_predict(Xb)
    print(f"bridge, {link:6s} linkage: ARI {adjusted_rand_score(yb, lab):.3f}, cluster sizes {np.bincount(lab).tolist()}")
assert adjusted_rand_score(yb, AgglomerativeClustering(2, linkage="single").fit_predict(Xb)) < 0.5

# %% [markdown]
# ## 4. Different densities
#
# Two tight clusters close together, one loose cluster far away. A radius small enough to separate the tight pair
# dissolves the loose one into "noise"; a radius big enough to hold the loose one together merges the pair.
# (Careful with ARI here: it treats DBSCAN's noise label as a cluster, which can flatter it. Count real clusters.)

# %%
d1 = rng.normal([0, 0], 0.2, size=(300, 2)); d2 = rng.normal([1.6, 0], 0.2, size=(300, 2))
loose = rng.normal([6, 6], 1.5, size=(400, 2))
Xd = np.vstack([d1, d2, loose]); yd = np.r_[np.zeros(300), np.ones(300), 2 * np.ones(400)]


def describe(lab):
    n_clusters, noise_frac = len(set(lab) - {-1}), np.mean(lab == -1)
    return n_clusters, noise_frac


found_three = False
for eps in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
    k, nf = describe(DBSCAN(eps=eps, min_samples=10).fit_predict(Xd))
    found_three |= (k == 3 and nf < 0.1)
    print(f"DBSCAN eps={eps:3}: {k} clusters, {nf:4.0%} labeled noise")
hd = HDBSCAN(min_cluster_size=30, copy=True).fit_predict(Xd)
k, nf = describe(hd)
print(f"HDBSCAN:         {k} clusters, {nf:4.0%} labeled noise, ARI {adjusted_rand_score(yd, hd):.3f}")
assert not found_three and k == 3 and nf < 0.05

# %% [markdown]
# ## 5. Stability: structure survives resampling, noise doesn't

# %%
def stability(data, k, B=20):
    scores = []
    for _ in range(B):
        i1, i2 = rng.integers(0, len(data), len(data)), rng.integers(0, len(data), len(data))
        m1 = KMeans(k, n_init=3, random_state=int(rng.integers(1e6))).fit(data[i1])
        m2 = KMeans(k, n_init=3, random_state=int(rng.integers(1e6))).fit(data[i2])
        scores.append(adjusted_rand_score(m1.predict(data), m2.predict(data)))  # compare the two clusterings on all points
    return np.mean(scores)


s_real, s_noise = stability(X, 6), stability(noise, 6)
print(f"bootstrap stability (ARI between runs): 6 blobs {s_real:.3f}, uniform noise {s_noise:.3f}")
assert s_real > 0.95 and s_noise < s_real - 0.1

# %% [markdown]
# ## 6. Segmenting the shop's customers (RFM)
#
# Recency, frequency, monetary value from the synthetic shop of Part II.

# %%
shop = make_shop(n_customers=4000, seed=21)
orders, items = shop["orders"], shop["order_items"]
revenue = items.assign(rev=items["quantity"] * items["unit_price"]).groupby("order_id")["rev"].sum()
o = orders.merge(revenue.rename("revenue"), left_on="order_id", right_index=True)
ref = o["order_ts"].max()
rfm = o.groupby("customer_id").agg(recency_days=("order_ts", lambda t: (ref - t.max()).days),
                                   frequency=("order_id", "count"), monetary=("revenue", "sum"))
Z = StandardScaler().fit_transform(np.column_stack([rfm["recency_days"], np.log1p(rfm["frequency"]), np.log1p(rfm["monetary"])]))
sil = {k: silhouette_score(Z, KMeans(k, n_init=5, random_state=0).fit_predict(Z)) for k in range(2, 7)}
km = KMeans(4, n_init=10, random_state=0).fit(Z)
profile = rfm.assign(segment=km.labels_).groupby("segment").agg(customers=("frequency", "size"), recency=("recency_days", "median"),
                                                                 orders=("frequency", "median"), spend=("monetary", "median")).round(0)
print(f"{len(rfm):,} customers; silhouette by k: " + " ".join(f"{k}:{v:.2f}" for k, v in sil.items()))
print(profile.to_string())
print(f"stability of the 4 segments: {stability(Z, 4, B=10):.2f}")
print("The segments are useful handles for a campaign; with silhouettes this modest, they're cuts through a continuum, not species.")

print("\nAll checks passed.")
