# %% [markdown]
# # Lab 41.1: Embeddings and semantic search, over this course
#
# The corpus: every lesson written so far, cut into chunks. Three query sets with known answers:
#   exercises   every lesson's exercise questions (not in the searchable text); the answer is their lesson
#   references  sentences that point to another lesson ("... (37.2)"), pointer removed; the answer is that lesson
#   identifiers code names found in exactly one lesson (`ctc_loss`, `nn.ModuleList`)
# 1. BM25 from scratch, and TF-IDF.
# 2. Dense vectors: LSA, a language model's mean-pooled states (and their anisotropy), and the same model trained
#    contrastively (in-batch negatives, pairs of random crops of the same passage) on Parts I to III.
# 3. Where each fails, and hybrid search with reciprocal rank fusion.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from retrieval import (BM25, chunk, evaluate, exercise_queries, identifier_queries, lesson_ranking, lessons,  # noqa: E402
                       load_or_train_encoder, reference_queries, rrf, tokenize)

torch.set_num_threads(4)
L = lessons()
titles = {les.id: les.title for les in L}
part_of = {les.id: les.part for les in L}
C = chunk(L, "sections", size=120)
texts = [f"{titles[c.lesson_id]}. {c.heading}. {c.text}" for c in C]            # a chunk carries its context
EX = exercise_queries(L)
REF = reference_queries(L)
ID = identifier_queries(L)
print(f"{len(L)} lessons -> {len(C):,} chunks of ~{np.mean([len(c.text.split()) for c in C]):.0f} words")
print(f"queries: {len(EX)} exercises, {len(REF)} cross-references, {len(ID)} identifiers")
print("e.g.", repr(REF[3][0][:110]), "->", REF[3][1], titles[REF[3][1]])

QSETS = {"exercises": [(q, r, None) for q, r in EX], "references": REF, "identifiers": [(q, r, None) for q, r in ID]}


def run(score_fn):
    out = {}
    for name, Q in QSETS.items():
        rankings = [lesson_ranking(score_fn(q), C, exclude=src) for q, _, src in Q]
        out[name] = evaluate(rankings, [r for _, r, _ in Q], k=5)
    return out


results = {}


def show(name, res):
    results[name] = res
    print(f"  {name:34s} " + "   ".join(f"{res[k][0]:.3f} / {res[k][1]:.3f}" for k in QSETS))


header = "  " + " " * 34 + "   ".join(f"{k:>13s}" for k in QSETS)

# %% [markdown]
# ## 1. Sparse: BM25 and TF-IDF

# %%
t0 = time.time()
bm25 = BM25(texts)
print(f"\nBM25 index: {len(bm25.vocab):,} terms, built in {time.time() - t0:.1f}s")
tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
T = tfidf.fit_transform(texts)
print("recall@5 / MRR@10 (lesson level)")
print(header)
show("BM25", run(bm25.scores))
show("TF-IDF cosine", run(lambda q: (T @ tfidf.transform([q]).T).toarray().ravel()))

# %% [markdown]
# ## 2. Dense
#
# LSA: the TF-IDF matrix compressed to 256 dimensions with a truncated SVD (Deerwester et al., 1990): the first
# "embeddings" for search. Then a neural encoder: the course's tiny GPT, its final hidden states averaged over the text.

# %%
svd = TruncatedSVD(256, random_state=0)
Z = svd.fit_transform(T)
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
lsa_q = lambda q: (lambda z: z / (np.linalg.norm(z) + 1e-9))(svd.transform(tfidf.transform([q]))[0])
show("LSA (TF-IDF + SVD, 256 dims)", run(lambda q: Z @ lsa_q(q)))

raw, _ = load_or_train_encoder(verbose=False, trained=False)
E_raw = raw.embed(texts)
rs = np.random.default_rng(0)
pairs = rs.integers(0, len(C), (2000, 2))
cos_raw = np.mean(np.sum(E_raw[pairs[:, 0]] * E_raw[pairs[:, 1]], 1))
show("LM mean-pooled, untrained", run(lambda q: E_raw @ raw.embed([q])[0]))
mu = E_raw.mean(0)
E_c = E_raw - mu; E_c /= np.linalg.norm(E_c, axis=1, keepdims=True)
cos_c = np.mean(np.sum(E_c[pairs[:, 0]] * E_c[pairs[:, 1]], 1))
show("  same, mean-centered", run(lambda q: E_c @ ((lambda v: v / np.linalg.norm(v))(raw.embed([q])[0] - mu))))
print(f"mean cosine between two random chunks: {cos_raw:.3f} raw, {cos_c:.3f} centered. The raw vectors all point the same")
print("way (anisotropy, 34.3): every text looks similar to every other. A language model is not an embedding model.")

t0 = time.time()
enc, _ = load_or_train_encoder(verbose=True)
print(f"(ready in {time.time() - t0:.0f}s)")
E = enc.embed(texts)
cos_t = np.mean(np.sum(E[pairs[:, 0]] * E[pairs[:, 1]], 1))
q_cache = {}


def dense(q):
    if q not in q_cache:
        q_cache[q] = enc.embed([q])[0]
    return E @ q_cache[q]


show("LM trained contrastively", run(dense))
print(f"mean cosine between random chunks after contrastive training: {cos_t:.3f}")
assert cos_raw > 0.5 and cos_t < cos_raw
assert results["LM trained contrastively"]["exercises"][0] > results["LM mean-pooled, untrained"]["exercises"][0] + 0.05

# %% [markdown]
# ## 3. Who fails where, and hybrid search

# %%
seen_parts = lambda Q: [x for x in Q if part_of[x[1]] <= 3]
new_parts = lambda Q: [x for x in Q if part_of[x[1]] >= 4]
print("\nexercise queries, recall@5, by where the answer lives:")
for label, f in (("Parts I-III (encoder trained on this text)", seen_parts), ("Parts IV-V (never seen by the encoder)", new_parts)):
    Q = f(QSETS["exercises"])
    rb = evaluate([lesson_ranking(bm25.scores(q), C) for q, _, _ in Q], [r for _, r, _ in Q])[0]
    rd = evaluate([lesson_ranking(dense(q), C) for q, _, _ in Q], [r for _, r, _ in Q])[0]
    print(f"  {label:44s} BM25 {rb:.3f}   dense {rd:.3f}   ({len(Q)} queries)")

print(header)
show("hybrid: RRF(BM25, trained LM)", run(lambda q: rrf(bm25.scores(q), dense(q))))
show("hybrid: RRF(BM25, LSA)", run(lambda q: rrf(bm25.scores(q), Z @ lsa_q(q))))
for name in ("BM25", "LSA (TF-IDF + SVD, 256 dims)", "LM trained contrastively", "hybrid: RRF(BM25, trained LM)", "hybrid: RRF(BM25, LSA)"):
    print(f"  {name:34s} mean recall@5 over the three sets {np.mean([results[name][k][0] for k in QSETS]):.3f}")

ident = QSETS["identifiers"]
miss = [(q, r) for q, r, _ in ident if r not in lesson_ranking(dense(q), C)[:5] and r in lesson_ranking(bm25.scores(q), C)[:5]]
print(f"\nidentifier queries dense search misses and BM25 finds: {len(miss)} of {len(ident)}, e.g. "
      + ", ".join(f"{q} ({r})" for q, r in miss[:4]))
print("an identifier is a rare string that means exactly itself; an embedding squeezes it through subword tokens into")
print("a vector of 'what this is about'. Product codes, error IDs, function names and people's names need exact match.")
assert results["BM25"]["identifiers"][0] > results["LM trained contrastively"]["identifiers"][0] + 0.2
# assertions on fusion: see the lesson for the numbers this run is expected to give

print("\nAll checks passed.")
