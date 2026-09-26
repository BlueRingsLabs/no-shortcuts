# %% [markdown]
# # Lab 34.3: Word embeddings
#
# 1. Count-based vectors: a co-occurrence matrix, PPMI weighting, and a truncated SVD.
# 2. word2vec from scratch: skip-gram with negative sampling, frequent-word subsampling, the unigram^0.75 noise.
# 3. Evaluation: related word pairs vs random pairs, and nearest neighbors, for count-based, word2vec and random vectors.
# 4. Averaged embeddings as document features, against TF-IDF (lab 34.2's task).
#
# The corpus is small (about 150,000 words of this course), so expect topical similarity, not analogy arithmetic.

# %%
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from corpus import course_text, labeled_paragraphs, words  # noqa: E402

torch.manual_seed(343)
torch.set_num_threads(4)
rng = np.random.default_rng(343)

toks = words(course_text())
counts = Counter(toks)
vocab = [w for w, c in counts.most_common() if c >= 5]
w2i = {w: i for i, w in enumerate(vocab)}
ids = np.array([w2i[t] for t in toks if t in w2i])
V = len(vocab)
freq = np.array([counts[w] for w in vocab], dtype=np.float64)
print(f"{len(toks):,} tokens; vocabulary (count >= 5): {V:,} words covering {len(ids) / len(toks):.0%} of tokens")

PAIRS = [("train", "test"), ("precision", "recall"), ("mean", "variance"), ("rows", "columns"), ("sql", "query"),
         ("matrix", "vector"), ("tree", "forest"), ("gradient", "descent"), ("kafka", "stream"), ("join", "table"),
         ("probability", "distribution"), ("regression", "linear"), ("parquet", "csv"), ("bias", "variance"),
         ("cluster", "clustering"), ("eigenvalues", "eigenvectors"), ("python", "code"), ("loss", "gradient"),
         ("feature", "features"), ("label", "labels"), ("schema", "table"), ("partition", "partitions"),
         ("normal", "gaussian"), ("boosting", "trees"), ("pipeline", "dbt"), ("likelihood", "posterior")]
PAIRS = [(a, b) for a, b in PAIRS if a in w2i and b in w2i]


def evaluate(E, name):
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    rel = np.mean([E[w2i[a]] @ E[w2i[b]] for a, b in PAIRS])
    r = np.random.default_rng(0)
    rnd = np.mean([E[i] @ E[j] for i, j in r.integers(0, min(V, 3000), (500, 2)) if i != j])
    ranks = []
    for a, b in PAIRS:
        sims = E @ E[w2i[a]]
        sims[w2i[a]] = -np.inf
        ranks.append(int((sims > sims[w2i[b]]).sum()) + 1)
    print(f"  {name:28s} related pairs {rel:.3f}   random pairs {rnd:.3f}   median rank of the partner {int(np.median(ranks)):5d} of {V:,}")
    return rel, rnd, np.median(ranks)


def neighbors(E, word, k=6):
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sims = E @ E[w2i[word]]
    return [vocab[i] for i in np.argsort(-sims)[1:k + 1]]


# %% [markdown]
# ## 1. Counts, PPMI and SVD

# %%
t0 = time.time()
WIN = 4
C = np.zeros((V, V), dtype=np.float32)
for off in range(1, WIN + 1):
    np.add.at(C, (ids[:-off], ids[off:]), 1.0 / off)                          # closer words count more
C = C + C.T
total = C.sum()
pw = C.sum(1, keepdims=True) / total
pc = (C.sum(0, keepdims=True) / total) ** 0.75                                # context distribution smoothing, as in word2vec
pc /= pc.sum()
with np.errstate(divide="ignore"):
    ppmi = np.maximum(np.log((C / total) / (pw * pc)), 0)
U, S, _ = randomized_svd(ppmi, 100, random_state=0)                           # only the top 100 directions
E_svd = U * np.sqrt(S)
print(f"\nco-occurrence matrix {V}x{V}, PPMI, SVD to 100 dimensions in {time.time() - t0:.0f}s")

# %% [markdown]
# ## 2. Skip-gram with negative sampling
#
# For each (center, context) pair within the window, maximize log sigma(u_c . v_w) and, for k negatives drawn from the
# unigram distribution raised to 0.75, log sigma(-u_n . v_w). Frequent words are randomly dropped (subsampling).

# %%
def make_pairs(ids, win, g):
    keep_p = np.minimum(1.0, np.sqrt(1e-3 / (freq / freq.sum())))           # Mikolov's subsampling of frequent words
    kept = ids[g.random(len(ids)) < keep_p[ids]]
    centers, contexts = [], []
    for off in range(1, win + 1):
        centers += [kept[:-off], kept[off:]]
        contexts += [kept[off:], kept[:-off]]
    return np.concatenate(centers), np.concatenate(contexts)


class SGNS(nn.Module):
    def __init__(self, V, d=100):
        super().__init__()
        self.inp, self.out = nn.Embedding(V, d), nn.Embedding(V, d)
        nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d)
        nn.init.zeros_(self.out.weight)

    def forward(self, w, c, neg):
        v = self.inp(w)
        pos = F.logsigmoid((v * self.out(c)).sum(-1))
        negs = F.logsigmoid(-(self.out(neg) @ v[:, :, None]).squeeze(-1)).sum(-1)
        return -(pos + negs).mean()


noise = torch.tensor(freq ** 0.75 / (freq ** 0.75).sum(), dtype=torch.float32)
model = SGNS(V)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
t0 = time.time()
for epoch in range(2):
    cw, cc = make_pairs(ids, WIN, rng)
    perm = rng.permutation(len(cw))
    cw, cc = torch.tensor(cw[perm]), torch.tensor(cc[perm])
    losses = []
    for s in range(0, len(cw), 8192):
        w, c = cw[s:s + 8192], cc[s:s + 8192]
        neg = torch.multinomial(noise, len(w) * 5, replacement=True).view(len(w), 5)
        loss = model(w, c, neg)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"epoch {epoch + 1}: {len(cw):,} pairs, loss {np.mean(losses[-50:]):.3f}  ({time.time() - t0:.0f}s)")
E_w2v = model.inp.weight.detach().numpy()

# %% [markdown]
# ## 3. Evaluation

# %%
print(f"\ncosine similarity, averaged over {len(PAIRS)} related pairs I picked vs 500 random pairs:")
res = {"random vectors": evaluate(rng.normal(size=(V, 100)), "random vectors"),
       "PPMI + SVD": evaluate(E_svd, "PPMI + SVD"),
       "word2vec (SGNS)": evaluate(E_w2v, "word2vec (SGNS)")}
print("\nnearest neighbors (word2vec | PPMI + SVD):")
for w in ("gradient", "sql", "variance", "kafka", "tree", "python"):
    if w in w2i:
        print(f"  {w:10s} {', '.join(neighbors(E_w2v, w, 5)):48s} | {', '.join(neighbors(E_svd, w, 5))}")
print("the neighbors are the words these lessons use together: similarity of topic and usage, from counting alone.")
print(f"note word2vec's random pairs: cosine {res['word2vec (SGNS)'][1]:.2f}, not 0. Its vectors share a common direction (anisotropy), so raw")
print("cosines aren't comparable across methods; ranks are. By rank, the count-based vectors match or beat word2vec on a corpus")
print("this small, which is Levy and Goldberg's (2015) finding: the tricks matter more than the neural network.")
assert res["word2vec (SGNS)"][0] > res["word2vec (SGNS)"][1] + 0.1 and res["PPMI + SVD"][0] > res["PPMI + SVD"][1] + 0.1
assert res["word2vec (SGNS)"][2] < 200 and res["PPMI + SVD"][2] < 50 and res["random vectors"][2] > 500

# %% [markdown]
# ## 4. Averaged embeddings as document features

# %%
data = labeled_paragraphs()
labels = np.array([l for _, l in data])


def doc_vectors(E):
    out = []
    for text, _ in data:
        idx = [w2i[w] for w in words(text) if w in w2i]
        out.append(E[idx].mean(0) if idx else np.zeros(E.shape[1]))
    X = np.array(out)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


cv = StratifiedKFold(5, shuffle=True, random_state=0)
print("\nparagraph classification (lab 34.2's task), 5-fold accuracy:")
doc_res = {}
for name, E in (("mean of word2vec vectors", E_w2v), ("mean of PPMI + SVD vectors", E_svd)):
    doc_res[name] = cross_val_score(LogisticRegression(C=1.0, max_iter=3000), doc_vectors(E), labels, cv=cv).mean()
    print(f"  {name:28s} {doc_res[name]:.3f}")
print("  TF-IDF + logistic regression (lab 34.2)   0.814")
print("averaging throws away which words were there, keeping only a blurry centroid. For topic classification the")
print("bag of words is hard to beat; embeddings earn their keep when words must be compared, not just counted.")
assert max(doc_res.values()) > 0.6

print("\nAll checks passed.")
