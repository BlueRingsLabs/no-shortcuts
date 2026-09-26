# %% [markdown]
# # Lab 23.2: Recommenders on data with a known truth
#
# 3,000 users and 600 items with hidden 8-dimensional tastes; items belong to 12 categories that shape their latent
# vector; popularity is heavily skewed. Each user's last 5 interactions (in time) are the test set. 40 "new" items
# have no training interactions at all.
#
# 1. Popularity and item-item baselines.
# 2. Implicit ALS (Hu, Koren, Volinsky) from scratch.
# 3. A two-tower model in PyTorch with an item category feature, and cold-start recommendations for the new items.
# 4. Full-catalog vs sampled metrics.
# 5. Coverage: who recommends only bestsellers?

# %%
import numpy as np
import torch

rng = np.random.default_rng(232)
torch.manual_seed(0)
n_users, n_items, k_true, n_cat = 3000, 600, 8, 12
cat_centers = rng.normal(0, 1, (n_cat, k_true))
item_cat = rng.integers(0, n_cat, n_items)
V = cat_centers[item_cat] + 0.5 * rng.normal(size=(n_items, k_true))
U = rng.normal(0, 1, (n_users, k_true)) * 0.9
pop = rng.normal(0, 1.2, n_items)                                 # popularity bias (log scale)
new_items = rng.choice(n_items, 40, replace=False)

seqs = []
for u in range(n_users):
    logits = U[u] @ V.T + pop
    g = rng.gumbel(size=n_items)                                  # Gumbel top-k = sampling without replacement from softmax
    seqs.append(np.argsort(-(logits + g))[: rng.integers(15, 45)])   # in time order (first = earliest)
train_pairs, test = [], {}
for u, s in enumerate(seqs):
    hist, fut = s[:-5], s[-5:]
    hist = hist[~np.isin(hist, new_items)]                        # new items: no training interactions anywhere
    train_pairs += [(u, i) for i in hist]
    test[u] = set(fut)
train_pairs = np.array(train_pairs)
R = np.zeros((n_users, n_items), dtype=np.float32)
R[train_pairs[:, 0], train_pairs[:, 1]] = 1
print(f"{len(train_pairs):,} training interactions; density {R.mean():.2%}; "
      f"top 10% of items get {np.sort(R.sum(0))[::-1][: n_items // 10].sum() / R.sum():.0%} of them")

# %% [markdown]
# ## Evaluation helpers: full-catalog ranking, excluding each user's training items

# %%
def evaluate(scores, k=10, users=None):
    users = range(n_users) if users is None else users
    rec_at_k, ndcg, recommended = [], [], set()
    for u in users:
        s = scores(u).copy()
        s[R[u] > 0] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        recommended.update(top.tolist())
        hits = np.array([i in test[u] for i in top])
        rec_at_k.append(hits.sum() / len(test[u]))
        dcg = (hits / np.log2(np.arange(2, k + 2))).sum()
        idcg = (1 / np.log2(np.arange(2, min(k, len(test[u])) + 2))).sum()
        ndcg.append(dcg / idcg)
    return np.mean(rec_at_k), np.mean(ndcg), len(recommended) / n_items


results = {}

# %% [markdown]
# ## 1. Baselines

# %%
popularity = R.sum(0)
results["popularity"] = evaluate(lambda u: popularity.astype(float))
norms = np.linalg.norm(R, axis=0) + 1e-9
S_items = (R.T @ R) / np.outer(norms, norms)                     # item-item cosine
np.fill_diagonal(S_items, 0)
results["item-item cosine"] = evaluate(lambda u: R[u] @ S_items)

# %% [markdown]
# ## 2. Implicit ALS

# %%
def implicit_als(R, k=16, alpha=20.0, lam=5.0, iters=10, seed=0):
    r = np.random.default_rng(seed)
    X = 0.1 * r.normal(size=(R.shape[0], k)); Y = 0.1 * r.normal(size=(R.shape[1], k))
    user_items = [np.flatnonzero(row) for row in R]
    item_users = [np.flatnonzero(col) for col in R.T]

    def solve(F, idx_lists):                                      # one ALS half-step
        FtF = F.T @ F
        out = np.zeros((len(idx_lists), k))
        for a, idx in enumerate(idx_lists):
            Fi = F[idx]
            A = FtF + alpha * Fi.T @ Fi + lam * np.eye(k)         # F^T C F = F^T F + F_i^T (C - I) F_i, C - I = alpha on the history
            b = (1 + alpha) * Fi.sum(0)                            # F^T C p: preference 1, confidence 1 + alpha, only on the history
            out[a] = np.linalg.solve(A, b)
        return out

    for _ in range(iters):
        X = solve(Y, user_items)
        Y = solve(X, item_users)
    return X, Y


Xals, Yals = implicit_als(R)
results["implicit ALS"] = evaluate(lambda u: Xals[u] @ Yals.T)

# %% [markdown]
# ## 3. Two towers, with a content feature for cold start

# %%
cat_t = torch.as_tensor(item_cat)
log_pop = torch.as_tensor(np.log(R.sum(0) + 1), dtype=torch.float32)


def train_two_tower(logq=True, epochs=15, lr=0.005, p_hide_id=0.3, dim=32, seed=0):
    torch.manual_seed(seed)
    user, item_id, item_cat_emb = (torch.nn.Embedding(n, dim) for n in (n_users, n_items, n_cat))
    for e in (user, item_id, item_cat_emb):
        torch.nn.init.normal_(e.weight, 0, 0.1)
    opt = torch.optim.Adam([*user.parameters(), *item_id.parameters(), *item_cat_emb.parameters()], lr=lr)
    pairs = torch.as_tensor(train_pairs)
    for _ in range(epochs):
        perm = torch.randperm(len(pairs))
        for start in range(0, len(pairs), 1024):
            batch = pairs[perm[start:start + 1024]]
            items = batch[:, 1]
            keep_id = (torch.rand(len(batch), 1) > p_hide_id).float()      # sometimes hide the ID, so the category part
            ivec = item_cat_emb(cat_t[items]) + keep_id * item_id(items)     # learns to stand on its own (cold start)
            logits = user(batch[:, 0]) @ ivec.T                             # in-batch negatives
            if logq:
                logits = logits - log_pop[items][None, :]                    # popular items show up as negatives too often: correct it
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch)))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Uv = user.weight.numpy()
        Iv = (item_cat_emb(cat_t) + item_id.weight).numpy()
        Iv[new_items] = item_cat_emb(cat_t[torch.as_tensor(new_items)]).numpy()   # new items: category only, no ID
    return Uv, Iv


Ut_raw, It_raw = train_two_tower(logq=False)
results["two-tower, no logQ"] = evaluate(lambda u: Ut_raw[u] @ It_raw.T)
Ut, It_cold = train_two_tower(logq=True)
results["two-tower + logQ"] = evaluate(lambda u: Ut[u] @ It_cold.T)

for name, (rec, ndcg, cov) in results.items():
    print(f"{name:19s} recall@10 {rec:.3f}  NDCG@10 {ndcg:.3f}  catalog coverage {cov:.0%}")
assert results["implicit ALS"][0] > results["popularity"][0] and results["item-item cosine"][0] > results["popularity"][0]
assert results["two-tower + logQ"][0] > 2 * results["two-tower, no logQ"][0], "the popularity correction matters enormously"

# cold start: how often do new items (zero training interactions) appear in the test set, and who can recommend them?
cold_users = [u for u in range(n_users) if test[u] & set(new_items)]
def cold_hits(score_fn):
    hits = 0
    for u in cold_users:
        s = score_fn(u).copy(); s[R[u] > 0] = -np.inf
        top = set(np.argpartition(-s, 50)[:50].tolist())
        hits += len(top & test[u] & set(new_items))
    return hits
print(f"{len(cold_users)} users interacted with a new item in the test period. New-item hits in their top 50: "
      f"ALS {cold_hits(lambda u: Xals[u] @ Yals.T)}, item-item {cold_hits(lambda u: R[u] @ S_items)}, "
      f"two-tower with category {cold_hits(lambda u: Ut[u] @ It_cold.T)}")
assert cold_hits(lambda u: Ut[u] @ It_cold.T) > max(cold_hits(lambda u: Xals[u] @ Yals.T), cold_hits(lambda u: R[u] @ S_items))

# %% [markdown]
# ## 4. Sampled metrics vs full ranking

# %%
def sampled_hit_rate(score_fn, n_neg=100, k=10):
    hits = []
    for u in range(0, n_users, 3):
        s = score_fn(u)
        candidates = np.flatnonzero((R[u] == 0) & ~np.isin(np.arange(n_items), list(test[u])))
        for pos in test[u]:
            negs = rng.choice(candidates, n_neg, replace=False)
            rank = np.sum(s[negs] > s[pos])
            hits.append(rank < k)
    return np.mean(hits)


samp = {name: sampled_hit_rate(fn) for name, fn in [("popularity", lambda u: popularity.astype(float)),
                                                      ("item-item cosine", lambda u: R[u] @ S_items),
                                                      ("implicit ALS", lambda u: Xals[u] @ Yals.T)]}
print("hit rate@10 against 100 sampled negatives vs recall@10 against the full catalog:")
for name in samp:
    print(f"  {name:17s} sampled {samp[name]:.3f}   full {results[name][0]:.3f}")
order_full = sorted(samp, key=lambda k_: -results[k_][0])
order_samp = sorted(samp, key=lambda k_: -samp[k_])
print(f"ranking of methods: full {order_full}; sampled {order_samp}")
assert all(samp[k_] > 2 * results[k_][0] for k_ in samp), "sampled metrics are far rosier"
gap_full = results["implicit ALS"][0] / results["popularity"][0]
gap_samp = samp["implicit ALS"] / samp["popularity"]
print(f"ALS vs popularity: {gap_full:.1f}x better on the full catalog, {gap_samp:.1f}x on sampled negatives (the metric compresses differences;")
print("Krichene and Rendle show it can also reverse them)")
assert gap_samp < gap_full

print("\nAll checks passed.")
