# %% [markdown]
# # Lab 41.4: Advanced retrieval, and evaluating it
#
# Same corpus and queries as 41.1 (this course; exercise questions and cross-reference sentences).
# 1. Metrics: recall@k, MRR and nDCG with graded relevance. Do they agree on which system is best?
# 2. Query rewriting without an LLM: pseudo-relevance feedback (the ancestor of HyDE) and query decomposition.
# 3. Structure: the course's own cross-reference graph, used to expand results.
# 4. Is the difference real? A paired bootstrap on per-query scores.
# 5. Evaluating the generation step: what a lexical faithfulness score catches, and what it misses.

# %%
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from retrieval import (BM25, _sentences, chunk, exercise_queries, lesson_ranking, lessons, load_or_train_encoder,  # noqa: E402
                       reference_queries, rrf, tokenize)

torch.set_num_threads(4)
L = lessons()
titles = {les.id: les.title for les in L}
module_of = {les.id: les.path.parent.name for les in L}
C = chunk(L, "sections", 120)
texts = [f"{titles[c.lesson_id]}. {c.heading}. {c.text}" for c in C]
EX = [(q, r, None) for q, r in exercise_queries(L)]
REF = reference_queries(L)
QS = {"exercises": EX, "references": REF}
bm = BM25(texts)
enc, _ = load_or_train_encoder(verbose=False)
E = enc.embed(texts)
QV = {name: enc.embed([q for q, _, _ in Q]) for name, Q in QS.items()}

# %% [markdown]
# ## 1. Metrics

# %%
def gain(lid, rel):
    """Graded relevance: the lesson itself 2, another lesson of the same module 1, anything else 0."""
    return 2 if lid == rel else 1 if module_of[lid] == module_of[rel] else 0


def ndcg(ranking, rel, k=10):
    dcg = sum(gain(l, rel) / np.log2(i + 2) for i, l in enumerate(ranking[:k]))
    same = sorted([2] + [1] * (sum(module_of[x] == module_of[rel] for x in titles) - 1), reverse=True)[:k]
    return dcg / sum(g / np.log2(i + 2) for i, g in enumerate(same))


def per_query(rankings, Q):
    rel = [r for _, r, _ in Q]
    return {"recall@5": np.array([r in rk[:5] for rk, r in zip(rankings, rel)], float),
            "MRR@10": np.array([1 / (rk.index(r) + 1) if r in rk[:10] else 0 for rk, r in zip(rankings, rel)]),
            "nDCG@10": np.array([ndcg(rk, r) for rk, r in zip(rankings, rel)])}


def run(score_fn, name_q):
    Q = QS[name_q]
    return per_query([lesson_ranking(score_fn(i, q), C, k=10, exclude=src) for i, (q, _, src) in enumerate(Q)], Q)


systems = {
    "BM25": lambda nq: (lambda i, q: bm.scores(q)),
    "dense": lambda nq: (lambda i, q: E @ QV[nq][i]),
    "hybrid (RRF)": lambda nq: (lambda i, q: rrf(bm.scores(q), E @ QV[nq][i])),
}
scores = {}
print(f"{'':30s}" + "".join(f"{nq + ' ' + m:>22s}" for nq in QS for m in ("recall@5", "nDCG@10")))
for s, f in systems.items():
    for nq in QS:
        scores[(s, nq)] = run(f(nq), nq)
    print(f"{s:30s}" + "".join(f"{scores[(s, nq)][m].mean():22.3f}" for nq in QS for m in ("recall@5", "nDCG@10")))
print("nDCG also credits near misses (another lesson of the right module), so a system that 'gets the area right'")
print("scores better on it than on recall. Pick the metric that matches what the generator needs.")

# %% [markdown]
# ## 2. Query rewriting
#
# Pseudo-relevance feedback (Rocchio, 1971; RM3): assume the top few results are relevant, add their most distinctive
# terms to the query, search again. HyDE (Gao et al., 2022) does the same thing with an LLM: write a hypothetical
# answer, embed that. Decomposition: split a multi-sentence question into sentences, search each, fuse.

# %%
def prf_scores(q, k=3, n_terms=10, weight=0.5):
    first = bm.scores(q)
    top = np.argsort(-first)[:k]
    counts = Counter(t for i in top for t in tokenize(texts[i]))
    ranked_terms = sorted(counts, key=lambda t: -counts[t] * bm.idf[bm.vocab[t]])
    expansion = [t for t in ranked_terms if t not in set(tokenize(q))][:n_terms]
    second = bm.scores(" ".join(expansion))
    return first + weight * second * first.max() / (second.max() + 1e-9)


def decomposed(q):
    parts = [s for s in re.split(r"(?<=[.?!])\s+", q) if len(s.split()) >= 4]
    return rrf(*[bm.scores(p) for p in parts]) if len(parts) > 1 else rrf(bm.scores(q))


print(f"\n{'':30s}" + "".join(f"{nq + ' recall@5':>22s}" for nq in QS))
for s, f in (("BM25", lambda i, q: bm.scores(q)), ("BM25 + PRF", lambda i, q: prf_scores(q)),
             ("BM25, decomposed + RRF", lambda i, q: decomposed(q))):
    for nq in QS:
        scores[(s, nq)] = run(f, nq)
    print(f"{s:30s}" + "".join(f"{scores[(s, nq)]['recall@5'].mean():22.3f}" for nq in QS))
print("query rewriting helps when the query and the documents use different words; it hurts when the first results")
print("were wrong (it drifts toward them). Measure it on your queries; don't assume it.")

# %% [markdown]
# ## 3. Structure: the cross-reference graph
#
# Lessons point to each other ("(29.1)"). Treat that as a graph and let a lesson's score borrow from its neighbors':
# a result that many strong results point to is probably relevant. The simplest relative of GraphRAG.

# %%
ids = [les.id for les in L]
idx = {l: i for i, l in enumerate(ids)}
A = np.zeros((len(ids), len(ids)))
for les in L:
    for ref in set(re.findall(r"\b(\d{2}\.\d)\b", les.body)):
        if ref in idx and ref != les.id:
            A[idx[les.id], idx[ref]] = A[idx[ref], idx[les.id]] = 1
print(f"\ncross-reference graph: {len(ids)} lessons, {int(A.sum() / 2)} links, "
      f"median degree {np.median(A.sum(1)):.0f}")


def lesson_scores(chunk_scores):
    s = np.full(len(ids), -np.inf)
    for c, v in zip(C, chunk_scores):
        s[idx[c.lesson_id]] = max(s[idx[c.lesson_id]], v)
    return s


def graph_ranking(chunk_scores, alpha, exclude=None):
    s = lesson_scores(chunk_scores)
    z = (s - s.mean()) / (s.std() + 1e-9)
    nb = (A @ np.maximum(z, 0)) / np.maximum(A.sum(1), 1)
    final = z + alpha * nb
    order = [ids[i] for i in np.argsort(-final) if ids[i] != exclude]
    return order[:10]


print(f"{'':30s}" + "".join(f"{nq + ' recall@5':>22s}" for nq in QS))
for alpha in (0.0, 0.3, 1.0):
    row = []
    for nq, Q in QS.items():
        rk = [graph_ranking(rrf(bm.scores(q), E @ QV[nq][i]), alpha, exclude=src) for i, (q, _, src) in enumerate(Q)]
        scores[(f"hybrid + graph a={alpha}", nq)] = per_query(rk, Q)
        row.append(scores[(f"hybrid + graph a={alpha}", nq)]["recall@5"].mean())
    print(f"{'hybrid + graph, alpha=' + str(alpha):30s}" + "".join(f"{v:22.3f}" for v in row))
print("the graph encodes knowledge the text doesn't make explicit (which lessons belong together). Whether it helps")
print("depends on whether your queries need that: specific questions rarely do; 'what do I need to know for X' does.")

# %% [markdown]
# ## 4. Is the difference real?

# %%
def paired_bootstrap(a, b, n=5000, seed=0):
    r = np.random.default_rng(seed)
    idx_ = r.integers(0, len(a), (n, len(a)))
    d = (a - b)[idx_].mean(1)
    return (a - b).mean(), np.percentile(d, [2.5, 97.5])


print()
for nq in QS:
    for s1, s2 in (("hybrid (RRF)", "BM25"), ("BM25 + PRF", "BM25")):
        m, ci = paired_bootstrap(scores[(s1, nq)]["recall@5"], scores[(s2, nq)]["recall@5"])
        verdict = "real" if ci[0] > 0 or ci[1] < 0 else "could be noise"
        print(f"{nq:10s} {s1:14s} - {s2:5s}: recall@5 {m:+.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  -> {verdict}")
print("with a few hundred queries, differences of a point or two are inside the noise. Build the query set big enough")
print("to detect the improvements you care about, and compare systems on the same queries (paired).")

# %% [markdown]
# ## 5. Evaluating generation: faithfulness
#
# Is each sentence of an answer supported by the retrieved sources? A cheap automatic check: the share of the
# sentence's content words found in the source. Test it on sentences whose status we know: copied from the source,
# with a few words dropped, taken from another lesson, and copied with one fact changed (a number, or a negation).

# %%
r = random.Random(0)
sent_pool = [(c_i, s) for c_i, c in enumerate(C) for s in _sentences(c.text) if 10 <= len(s.split()) <= 40]
sample = r.sample(sent_pool, 300)


def overlap(sentence, source):
    w = set(tokenize(sentence))
    return len(w & set(tokenize(source))) / max(1, len(w))


def reword(s):
    w = s.split()
    for _ in range(max(1, len(w) // 6)):
        w.pop(r.randrange(len(w)))
    return " ".join(w)


def change_fact(s):
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", s)
    if nums:
        n = r.choice(nums)
        return s.replace(n, str(int(float(n)) * 3 + 7), 1)
    for a, b in ((" is ", " is not "), (" are ", " are not "), (" can ", " cannot "), (" does ", " does not ")):
        if a in s:
            return s.replace(a, b, 1)
    return s.replace(" ", " never ", 1)


cases = {"copied from the source": [], "with words dropped": [], "from another lesson": [], "one fact changed": []}
for c_i, s in sample:
    src = C[c_i].text
    cases["copied from the source"].append(overlap(s, src))
    cases["with words dropped"].append(overlap(reword(s), src))
    other = r.choice([x for x in sent_pool if C[x[0]].lesson_id != C[c_i].lesson_id])[1]
    cases["from another lesson"].append(overlap(other, src))
    cases["one fact changed"].append(overlap(change_fact(s), src))
print("\nshare of content words found in the source (a sentence 'passes' at >= 0.6):")
for k, v in cases.items():
    print(f"  {k:26s} mean {np.mean(v):.2f}   passes {np.mean(np.array(v) >= 0.6):.0%}")
print("the lexical check separates on-topic from off-topic, and passes a sentence that says the opposite of its source,")
print("or the wrong number, almost every time. That's the error that matters most in RAG, and catching it takes a model")
print("that reads for meaning: an NLI model, or an LLM judge, validated against human labels (43.1). RAGAS-style")
print("'faithfulness' scores are LLM judges; trust them after you've measured their agreement with people, not before.")
assert np.mean(np.array(cases["one fact changed"]) >= 0.6) > 0.8 and np.mean(np.array(cases["from another lesson"]) >= 0.6) < 0.2

print("\nAll checks passed.")
