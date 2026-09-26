# %% [markdown]
# # Lab 41.2: Vector indexes: IVF, PQ and HNSW, from scratch
#
# 20,000 unit vectors in 64 dimensions, clustered the way real embeddings are, and 300 queries. Exact search is the
# reference. The cost of a query is counted in distance computations, not seconds: in Python, interpreter overhead
# would swamp the comparison, while the count is what a C++ library's time is proportional to.
# 1. IVF: k-means cells, probe the nearest few.
# 2. PQ: compress each vector to 8 bytes; search by table lookups; rerank with the originals.
# 3. HNSW: a navigable small-world graph, searched greedily.
# 4. Why it works at all: the same index on structureless random data.
# 5. Filters: what post-filtering does to a selective query.

# %%
import heapq
import math
import time

import numpy as np
from sklearn.cluster import KMeans


def make_data(n, d=64, clusters=200, seed=0, structured=True):
    r = np.random.default_rng(seed)
    if structured:
        # clusters live near low-dimensional subspaces, like real embeddings (low intrinsic dimension)
        centers = r.normal(size=(clusters, d))
        basis = r.normal(size=(clusters, 8, d))
        c = r.integers(0, clusters, n)
        X = centers[c] + np.einsum("nk,nkd->nd", r.normal(size=(n, 8)) * 0.5, basis[c]) + 0.05 * r.normal(size=(n, d))
    else:
        X, c = r.normal(size=(n, d)), np.zeros(n, dtype=int)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), c


N, NQ, K = 20_000, 300, 10
data, labels = make_data(N + NQ)
X, Q = data[:N], data[N:]
Q = Q + 0.02 * np.random.default_rng(1).normal(size=Q.shape).astype(np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
exact = np.argsort(-(Q @ X.T), axis=1)[:, :K]
recall = lambda found, i: len(set(found) & set(exact[i])) / K
print(f"{N:,} vectors x 64 dims ({X.nbytes / 1e6:.1f} MB as float32); exact search: {N:,} distances per query")

# %% [markdown]
# ## 1. IVF

# %%
t0 = time.time()
nlist = 128
km = KMeans(nlist, n_init=1, random_state=0).fit(X)
cent = (km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)).astype(np.float32)
lists = [np.flatnonzero(km.labels_ == j) for j in range(nlist)]
print(f"\nIVF: {nlist} cells, trained in {time.time() - t0:.1f}s; cell sizes {min(map(len, lists))} to {max(map(len, lists))}")


def ivf_search(q, nprobe, k=K):
    cells = np.argsort(-(cent @ q))[:nprobe]
    cand = np.concatenate([lists[j] for j in cells])
    return cand[np.argsort(-(X[cand] @ q))[:k]], nlist + len(cand)


print("  nprobe   recall@10   distances/query")
ivf_res = {}
for nprobe in (1, 2, 4, 8, 16, 32):
    out = [ivf_search(q, nprobe) for q in Q]
    ivf_res[nprobe] = (np.mean([recall(f, i) for i, (f, _) in enumerate(out)]), np.mean([c for _, c in out]))
    print(f"  {nprobe:6d}   {ivf_res[nprobe][0]:9.3f}   {ivf_res[nprobe][1]:15,.0f}")
print("one cell finds 80% of the neighbors: the rest sit across a cell border. Each doubling of nprobe buys a few")
print("points of recall for roughly twice the distance computations.")

# %% [markdown]
# ## 2. Product quantization
#
# Split each vector into 8 sub-vectors of 8 dims; learn 256 centroids per subspace; store each vector as 8 one-byte
# centroid IDs. A query's distance to every code is a sum of 8 table lookups (asymmetric distance computation, Jegou
# et al., 2011).

# %%
t0 = time.time()
M_SUB, KS = 8, 256
ds = X.shape[1] // M_SUB
books = [KMeans(KS, n_init=1, random_state=0).fit(X[:, m * ds:(m + 1) * ds]) for m in range(M_SUB)]
codes = np.stack([b.labels_ for b in books], 1).astype(np.uint8)
print(f"\nPQ: {M_SUB} x {KS} centroids, trained in {time.time() - t0:.1f}s; {codes.nbytes / N:.0f} bytes per vector "
      f"instead of {X.nbytes / N:.0f} ({X.nbytes / codes.nbytes:.0f}x smaller)")


def pq_search(q, k=K, rerank=0):
    tables = np.stack([b.cluster_centers_ @ q[m * ds:(m + 1) * ds] for m, b in enumerate(books)])   # (8, 256) dot products
    approx = tables[np.arange(M_SUB), codes].sum(1)
    if not rerank:
        return np.argsort(-approx)[:k]
    cand = np.argpartition(-approx, rerank)[:rerank]
    return cand[np.argsort(-(X[cand] @ q))[:k]]


print("  method                          recall@10")
pq_res = {}
for name, kw in (("PQ codes only", {}), ("PQ, rerank top 100 exactly", {"rerank": 100}), ("PQ, rerank top 500", {"rerank": 500})):
    pq_res[name] = np.mean([recall(pq_search(q, **kw), i) for i, q in enumerate(Q)])
    print(f"  {name:30s}  {pq_res[name]:9.3f}")
print("compressed distances are coarse: enough to shortlist, not to rank. Production systems combine IVF (which cells)")
print("with PQ (compact codes in each cell) and rerank the shortlist with full vectors, often kept on disk.")

# %% [markdown]
# ## 3. HNSW (Malkov and Yashunin, 2018)
#
# Every vector is a node linked to about M near neighbors; a few nodes also appear on sparser upper layers. Search
# starts at the top, walks greedily toward the query, drops a layer, and at the bottom runs a beam search that keeps
# the ef best candidates.

# %%
class HNSW:
    def __init__(self, X, M=16, ef_construction=64, seed=0):
        self.X, self.M, self.M0, self.efc = X, M, 2 * M, ef_construction
        self.mL = 1 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.graph, self.entry, self.top, self.ndist = [], None, -1, 0
        for i in range(len(X)):
            self._insert(i)

    def _dist(self, q, ids):
        self.ndist += len(ids)
        return 1 - self.X[ids] @ q

    def _search_layer(self, q, entries, ef, level):
        visited = set(entries)
        d = self._dist(q, list(entries))
        cand = list(zip(d, entries)); heapq.heapify(cand)                     # closest first
        best = [(-x, e) for x, e in zip(d, entries)]; heapq.heapify(best)      # farthest first, at most ef
        while cand:
            dc, c = heapq.heappop(cand)
            if dc > -best[0][0]:
                break
            nbrs = [n for n in self.graph[level].get(c, ()) if n not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            for dn, n in zip(self._dist(q, nbrs), nbrs):
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(cand, (dn, n)); heapq.heappush(best, (-dn, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-x, n) for x, n in best)

    def _insert(self, i):
        q = self.X[i]
        level = int(-math.log(1 - self.rng.random()) * self.mL)
        while len(self.graph) <= level:
            self.graph.append({})
        if self.entry is None:
            for lv in range(level + 1):
                self.graph[lv][i] = []
            self.entry, self.top = i, level
            return
        ep = [self.entry]
        for lv in range(self.top, level, -1):
            ep = [self._search_layer(q, ep, 1, lv)[0][1]]
        for lv in range(min(level, self.top), -1, -1):
            W = self._search_layer(q, ep, self.efc, lv)
            mmax = self.M0 if lv == 0 else self.M
            self.graph[lv][i] = [n for _, n in W[:self.M]]
            for n in self.graph[lv][i]:
                nb = self.graph[lv][n]
                nb.append(i)
                if len(nb) > mmax:                                             # keep the closest (simple selection)
                    d = 1 - self.X[nb] @ self.X[n]
                    self.graph[lv][n] = [nb[j] for j in np.argsort(d)[:mmax]]
            ep = [n for _, n in W]
        for lv in range(level + 1):
            self.graph[lv].setdefault(i, [])
        if level > self.top:
            self.entry, self.top = i, level

    def search(self, q, k=K, ef=50):
        self.ndist = 0
        ep = [self.entry]
        for lv in range(self.top, 0, -1):
            ep = [self._search_layer(q, ep, 1, lv)[0][1]]
        W = self._search_layer(q, ep, max(ef, k), 0)
        return [n for _, n in W[:k]], self.ndist


t0 = time.time()
hnsw = HNSW(X)
links = sum(len(v) for v in hnsw.graph[0].values())
print(f"\nHNSW: built in {time.time() - t0:.0f}s (pure Python); {len(hnsw.graph)} layers, "
      f"{[len(g) for g in hnsw.graph]} nodes per layer; {links / N:.1f} links per node at the bottom "
      f"(+{links * 4 / N:.0f} bytes per vector)")
print("  ef   recall@10   distances/query")
hnsw_res = {}
for ef in (10, 20, 40, 80, 160):
    out = [hnsw.search(q, ef=ef) for q in Q]
    hnsw_res[ef] = (np.mean([recall(f, i) for i, (f, _) in enumerate(out)]), np.mean([c for _, c in out]))
    print(f"  {ef:3d}   {hnsw_res[ef][0]:9.3f}   {hnsw_res[ef][1]:15,.0f}")
best_ivf = min((c for r, c in ivf_res.values() if r >= 0.95), default=float("inf"))
best_hnsw = min((c for r, c in hnsw_res.values() if r >= 0.95), default=float("inf"))
print(f"distances per query for recall >= 0.95: exact {N:,}; IVF {best_ivf:,.0f}; HNSW {best_hnsw:,.0f}")
print("HNSW reaches high recall with the fewest distance computations and is the default in most vector databases.")
print("It pays in memory (the links), in build time, and in deletes, which graphs handle badly.")
assert hnsw_res[160][0] > 0.95 and best_hnsw < best_ivf < N

# %% [markdown]
# ## 4. Why approximate search works: structure

# %%
Xr, _ = make_data(5000 + 100, structured=False, seed=2)
Xr_base, Qr = Xr[:5000], Xr[5000:]
ex_r = np.argsort(-(Qr @ Xr_base.T), axis=1)[:, :K]
h_r = HNSW(Xr_base)
h_s = HNSW(X[:5000])
ex_s = np.argsort(-(Q[:100] @ X[:5000].T), axis=1)[:, :K]
rec = lambda h, QQ, ex, ef: np.mean([len(set(h.search(q, ef=ef)[0]) & set(ex[i])) / K for i, q in enumerate(QQ)])
r_s, r_r = rec(h_s, Q[:100], ex_s, 40), rec(h_r, Qr, ex_r, 40)
print(f"\n5,000 vectors, HNSW with ef=40: clustered data recall@10 {r_s:.3f}; uniformly random 64-d data {r_r:.3f}")
print("in 64 truly random dimensions, every point is about equally far from every other (the curse of")
print("dimensionality): the 'nearest' neighbors are barely nearer than the rest, and a graph or a clustering has little")
print("structure to exploit. Real embeddings have low intrinsic dimension, which is what makes indexes work.")
assert r_s > r_r + 0.1

# %% [markdown]
# ## 5. Filtered search
#
# "Nearest neighbors among documents this user may see" or "in this language": a metadata filter. The simple way is to
# search, then filter; with a selective filter, the results mostly disappear.

# %%
attr = np.random.default_rng(3).integers(0, 100, N)                             # e.g. a tenant ID: 1% of the data each
print("\nfilter keeps 1% of the vectors; top-10 among them:")
print("  method                                         results returned   recall@10")
for name, fetch in (("HNSW top 10, then filter", 10), ("HNSW top 200, then filter", 200)):
    got, rec_f = [], []
    for i, q in enumerate(Q[:100]):
        ids, _ = hnsw.search(q, k=fetch, ef=max(fetch, 50))
        keep = [j for j in ids if attr[j] == attr[i % 100]][:K]
        allowed = np.flatnonzero(attr == attr[i % 100])
        truth = allowed[np.argsort(-(X[allowed] @ q))[:K]]
        got.append(len(keep)); rec_f.append(len(set(keep) & set(truth)) / K)
    print(f"  {name:46s} {np.mean(got):16.1f}   {np.mean(rec_f):9.3f}")
print("post-filtering returns almost nothing for selective filters. Fixes: pre-filter and search exactly when the")
print("allowed set is small (here 200 vectors: exact is cheap), filter inside the graph traversal, or partition the index")
print("by the attribute (one index per tenant). For access control, the filter must be applied by the system, never")
print("left to the model (44.1).")

print("\nAll checks passed.")
