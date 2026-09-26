# %% [markdown]
# # Lab 21.4: t-SNE and UMAP, measured instead of admired
#
# 1. Swiss roll: PCA vs Isomap vs t-SNE, with neighbor preservation and trustworthiness.
# 2. Cluster sizes: a 10x difference in spread comes out the same size.
# 3. Distances between clusters: the ordering gets scrambled.
# 4. Clusters in pure noise at low perplexity.
# 5. Digits: how much of each point's neighborhood survives, PCA vs t-SNE vs UMAP (if installed).
# 6. UMAP can embed new points; t-SNE can't.

# %%
import warnings

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, trustworthiness
from sklearn.neighbors import NearestNeighbors

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import umap                                             # optional: pip install umap-learn
except ImportError:
    umap = None

rng = np.random.default_rng(214)


def knn_preservation(X_high, X_low, k=10):
    a = NearestNeighbors(n_neighbors=k + 1).fit(X_high).kneighbors(X_high, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(X_low).kneighbors(X_low, return_distance=False)[:, 1:]
    return np.mean([len(set(r) & set(s)) / k for r, s in zip(a, b)])


def tsne(X, perplexity=30, seed=0):
    return TSNE(2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(X)

# %% [markdown]
# ## 1. Swiss roll

# %%
Xs, t = make_swiss_roll(n_samples=1200, noise=0.05, random_state=0)
emb = {"PCA": PCA(2).fit_transform(Xs), "Isomap": Isomap(n_neighbors=10, n_components=2).fit_transform(Xs), "t-SNE": tsne(Xs)}
for name, Y in emb.items():
    corr = abs(np.corrcoef(Y[:, 0], t)[0, 1]) if name != "t-SNE" else float("nan")
    print(f"{name:7s}: 10-NN preserved {knn_preservation(Xs, Y):.3f}, trustworthiness {trustworthiness(Xs, Y, n_neighbors=10):.3f}"
          + (f", |corr(first axis, position along the roll)| {corr:.2f}" if name != "t-SNE" else ""))
assert knn_preservation(Xs, emb["Isomap"]) > knn_preservation(Xs, emb["PCA"]) + 0.2

# %% [markdown]
# ## 2. Cluster sizes mean (almost) nothing

# %%
tight = rng.normal(0, 1, size=(400, 10))
loose = rng.normal(0, 10, size=(400, 10)) + np.r_[80, np.zeros(9)]
Xc = np.vstack([tight, loose])
Y = tsne(Xc)
spread = lambda A: np.sqrt(((A - A.mean(0)) ** 2).sum(1)).mean()
ratio_high = spread(loose) / spread(tight)
ratio_low = spread(Y[400:]) / spread(Y[:400])
print(f"spread ratio (loose / tight): in 10-D {ratio_high:.1f}x, in the t-SNE plot {ratio_low:.1f}x")
assert ratio_high > 8 and ratio_low < 3

# %% [markdown]
# ## 3. Distances between clusters

# %%
centers = np.zeros((4, 20))
centers[1, 0] = 10                                              # B is close to A
centers[2, 0] = 40                                              # C is far
centers[3, 1] = 100                                             # D is very far
Xd = np.vstack([c + rng.normal(size=(200, 20)) for c in centers])
lab = np.repeat(np.arange(4), 200)
Yd = tsne(Xd, perplexity=30, seed=1)
cent_hi = np.array([Xd[lab == k].mean(0) for k in range(4)])
cent_lo = np.array([Yd[lab == k].mean(0) for k in range(4)])
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
d_hi = np.array([np.linalg.norm(cent_hi[i] - cent_hi[j]) for i, j in pairs])
d_lo = np.array([np.linalg.norm(cent_lo[i] - cent_lo[j]) for i, j in pairs])
print("pair      distance in 20-D   in t-SNE")
for (i, j), a, b in zip(pairs, d_hi, d_lo):
    print(f"{'ABCD'[i]}-{'ABCD'[j]}       {a:8.1f}       {b:7.1f}")
print(f"ratio of largest to smallest inter-cluster distance: 20-D {d_hi.max() / d_hi.min():.1f}x, t-SNE {d_lo.max() / d_lo.min():.1f}x")
assert d_hi.max() / d_hi.min() > 3 * (d_lo.max() / d_lo.min())

# %% [markdown]
# ## 4. Clusters in noise

# %%
noise = rng.normal(size=(800, 10))                              # one Gaussian blob: no clusters at all
found = {}
for perp in (2, 30):
    Yn = tsne(noise, perplexity=perp, seed=2)
    found[perp] = len(set(HDBSCAN(min_cluster_size=15, copy=True).fit_predict(Yn)) - {-1})
    print(f"t-SNE of pure Gaussian noise, perplexity {perp:2d}: HDBSCAN finds {found[perp]} 'clusters' in the plot")
found_orig = len(set(HDBSCAN(min_cluster_size=15, copy=True).fit_predict(noise)) - {-1})
print(f"HDBSCAN on the original 10-D noise: {found_orig} clusters")
assert min(found.values()) > found_orig

# %% [markdown]
# ## 5. Digits: neighborhoods preserved

# %%
digits = load_digits()
Xg = PCA(30, random_state=0).fit_transform(digits.data)       # the standard pipeline: PCA to ~30-50 dims first
Xg_tr, Xg_new = Xg[:1500], Xg[1500:]
results = {"PCA (2 components)": PCA(2).fit_transform(Xg_tr), "t-SNE perplexity 30": tsne(Xg_tr, 30)}
if umap is not None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit(Xg_tr)
    results["UMAP"] = um.embedding_
else:
    print("umap-learn not installed; skipping UMAP (pip install umap-learn)")
for name, Y in results.items():
    print(f"{name:20s}: 10-NN preserved {knn_preservation(Xg_tr, Y):.3f}, trustworthiness {trustworthiness(Xg_tr, Y, n_neighbors=10):.3f}")
assert knn_preservation(Xg_tr, results["t-SNE perplexity 30"]) > 2 * knn_preservation(Xg_tr, results["PCA (2 components)"])

# %% [markdown]
# ## 6. New points

# %%
if umap is not None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Y_new = um.transform(Xg_new)
    y_tr, y_new = digits.target[:1500], digits.target[1500:]
    nn = NearestNeighbors(n_neighbors=5).fit(um.embedding_)
    votes = y_tr[nn.kneighbors(Y_new, return_distance=False)]
    acc = np.mean([np.bincount(v).argmax() == t for v, t in zip(votes, y_new)])
    print(f"UMAP.transform placed {len(Y_new)} unseen digits; 5-NN label agreement in the map {acc:.3f}")
    assert acc > 0.8
print("t-SNE has no transform: re-running it with the new points moves every old point too.")
Y_a = tsne(Xg_tr[:600], seed=0); Y_b = tsne(np.vstack([Xg_tr[:600], Xg_new[:100]]), seed=0)[:600]
print(f"old points' positions correlate {abs(np.corrcoef(Y_a[:, 0], Y_b[:, 0])[0, 1]):.2f} across the two runs (first axis)")

print("\nAll checks passed.")
