# %% [markdown]
# # Lab 41.3: RAG done properly
#
# Retrieval-augmented generation over this course, with the exercise questions as test queries.
# 1. Chunking: fixed windows vs structure-aware chunks, sizes, overlap, and a header on each chunk.
# 2. Two stages: hybrid candidate retrieval (BM25 + dense, 41.1), then a reranker trained on half the lessons and
#    tested on the other half.
# 3. Knowing when not to answer: questions whose lesson isn't in the index, and a threshold to abstain.
# 4. Assembling the prompt: a token budget, delimited sources, citations the code can check.

# %%
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from fakellm import count_tokens  # noqa: E402
from retrieval import (BM25, chunk, evaluate, exercise_queries, lesson_ranking, lessons, load_or_train_encoder,  # noqa: E402
                       rrf, tokenize)

torch.set_num_threads(4)
L = lessons()
titles = {les.id: les.title for les in L}
EX = exercise_queries(L)
print(f"{len(L)} lessons, {len(EX)} exercise questions")

# %% [markdown]
# ## 1. Chunking

# %%
def with_header(c):
    return f"{titles[c.lesson_id]}. {c.heading}. {c.text}" if c.heading else f"{titles[c.lesson_id]}. {c.text}"


def within_budget(scores, C, texts, budget=600):
    """Lesson IDs whose chunks make it into a prompt of `budget` words, filling it in rank order."""
    used, out = 0, []
    for i in np.argsort(-scores)[:200]:
        n = len(texts[i].split())
        if used + n > budget:
            break
        used += n
        out.append(C[i].lesson_id)
    return out


print("\nBM25. Lesson-level recall with the top 5 chunks (and how many words they are), and with a fixed prompt budget")
print("of 600 words filled in rank order:")
print(f"  {'chunking':44s} {'chunks':>7s} {'recall@5':>9s} {'words in top 5':>15s} {'recall, 600 words':>18s}")
chunk_res = {}
for name, strategy, size, overlap, header in (
        ("fixed 60 words", "fixed", 60, 0, False),
        ("fixed 120 words", "fixed", 120, 0, False),
        ("fixed 120 words, 30 overlap", "fixed", 120, 30, False),
        ("fixed 300 words", "fixed", 300, 0, False),
        ("sections, paragraphs packed to 120 words", "sections", 120, 0, False),
        ("  same + title and heading on each chunk", "sections", 120, 0, True),
        ("whole sections + header", "whole", 0, 0, True)):
    C = chunk(L, strategy, size, overlap)
    texts = [with_header(c) if header else c.text for c in C]
    bm = BM25(texts)
    ranks, words, budget_hits = [], [], []
    for q, r in EX:
        s = bm.scores(q)
        ranks.append(lesson_ranking(s, C))
        words.append(sum(len(texts[i].split()) for i in np.argsort(-s)[:5]))
        budget_hits.append(r in within_budget(s, C, texts))
    rec, mrr = evaluate(ranks, [r for _, r in EX])
    chunk_res[name] = (rec, np.mean(budget_hits), np.mean(words))
    print(f"  {name:44s} {len(C):7,d} {rec:9.3f} {np.mean(words):15,.0f} {np.mean(budget_hits):18.3f}")
print("with a fixed number of chunks, bigger chunks look better: they carry more of each lesson, and more words into")
print("the prompt. At a fixed prompt budget, which is what you pay for, it reverses: small chunks spend the budget on")
print("the parts that match. Structure-aware chunks don't beat fixed windows on this metric; their case is that a chunk")
print("starting mid-sentence or cutting a table in half is harder for the model to *use*, which a retrieval metric")
print("can't see. Putting the title and heading on each chunk helps recall a little, even after paying for the words")
print("it adds.")
h, nh = chunk_res["  same + title and heading on each chunk"], chunk_res["sections, paragraphs packed to 120 words"]
small, big = chunk_res["fixed 60 words"], chunk_res["fixed 300 words"]
assert h[0] > nh[0] and big[0] > small[0] and small[1] > big[1] + 0.05

# %% [markdown]
# ## 2. Hybrid retrieval and a reranker
#
# First stage: the top 30 chunks by each of BM25, the neural encoder and LSA, merged. Second stage: score each
# candidate with a model that sees all three scores and a few more features. A cross-encoder (a transformer reading query
# and chunk together) is the usual reranker; here gradient-boosted trees on features play the same role, trained on
# the questions of even-numbered lessons and evaluated on the odd ones.

# %%
C = chunk(L, "sections", 120)
texts = [with_header(c) for c in C]
bm = BM25(texts)
enc, _ = load_or_train_encoder(verbose=True)
E = enc.embed(texts)
Qv = enc.embed([q for q, _ in EX])
tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
svd = TruncatedSVD(256, random_state=0)
Z = svd.fit_transform(tfidf.fit_transform(texts)); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
Zq = svd.transform(tfidf.transform([q for q, _ in EX])); Zq /= np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-9
head_tokens = [set(tokenize(f"{titles[c.lesson_id]} {c.heading}")) for c in C]
lengths = np.array([len(t.split()) for t in texts])


def candidates(qi, n=30):
    q = EX[qi][0]
    sb, sd, sl = bm.scores(q), E @ Qv[qi], Z @ Zq[qi]
    cand = np.union1d(np.union1d(np.argsort(-sb)[:n], np.argsort(-sd)[:n]), np.argsort(-sl)[:n])
    rb = np.empty(len(sb)); rb[np.argsort(-sb)] = np.arange(len(sb))
    rd = np.empty(len(sd)); rd[np.argsort(-sd)] = np.arange(len(sd))
    qt = set(tokenize(q))
    feats = np.stack([sb[cand] / (sb.max() + 1e-9), sd[cand], sl[cand], np.log1p(rb[cand]), np.log1p(rd[cand]),
                      [len(qt & head_tokens[i]) / (len(qt) + 1) for i in cand], np.log(lengths[cand])], 1)
    return cand, feats, sb, sd, sl


pools = [candidates(i) for i in range(len(EX))]
lesson_num = lambda lid: int(lid.replace(".", ""))
train_q = [i for i, (_, r) in enumerate(EX) if lesson_num(r) % 2 == 0]
test_q = [i for i, (_, r) in enumerate(EX) if lesson_num(r) % 2 == 1]
Xtr = np.concatenate([pools[i][1] for i in train_q])
ytr = np.concatenate([[C[c].lesson_id == EX[i][1] for c in pools[i][0]] for i in train_q])
reranker = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(Xtr, ytr)


def ranked(i, system):
    cand, feats, sb, sd, sl = pools[i]
    if system == "BM25":
        return lesson_ranking(sb, C)
    if system == "dense":
        return lesson_ranking(sd, C)
    if system == "LSA":
        return lesson_ranking(sl, C)
    if system == "hybrid (RRF of all three)":
        return lesson_ranking(rrf(sb, sd, sl), C)
    full = np.full(len(C), -np.inf)
    full[cand] = reranker.decision_function(feats)
    return lesson_ranking(full, C)


print(f"\nreranker trained on {len(train_q)} questions; tested on {len(test_q)} questions from other lessons")
print(f"  {'system':26s} {'recall@1':>9s} {'recall@3':>9s} {'MRR@10':>8s}")
rr = {}
for system in ("BM25", "dense", "LSA", "hybrid (RRF of all three)", "hybrid + reranker"):
    ranks = [ranked(i, system) for i in test_q]
    rel = [EX[i][1] for i in test_q]
    rr[system] = (evaluate(ranks, rel, 1)[0], evaluate(ranks, rel, 3)[0], evaluate(ranks, rel, 3)[1])
    print(f"  {system:26s} {rr[system][0]:9.3f} {rr[system][1]:9.3f} {rr[system][2]:8.3f}")
print("the first stage is for recall (don't miss it), the second for precision (put it first). Naive fusion with weak")
print("retrievers drags BM25 down; a learned reranker recovers most of it and draws level at recall@3, but a reranker built on")
print("the same weak signals can't beat a strong lexical match at the top position. That takes a cross-encoder that reads")
print("the query and the passage together, affordable because it only sees a few dozen candidates.")
assert rr["hybrid + reranker"][0] > rr["hybrid (RRF of all three)"][0] and rr["hybrid + reranker"][1] >= rr["BM25"][1] - 0.02

# %% [markdown]
# ## 3. Knowing when not to answer
#
# Remove a fifth of the lessons from the index. Their exercise questions have no answer in it; a RAG system that
# answers them anyway is making things up from whatever it retrieved. Abstain when the best reranker score is low.

# %%
held = {les.id for k, les in enumerate(L) if k % 5 == 2}
keep = np.array([c.lesson_id not in held for c in C])
prob_best, answerable, correct_top = [], [], []
for i in test_q:
    cand, feats = pools[i][:2]
    m = keep[cand]
    if not m.any():
        prob_best.append(0.0); answerable.append(EX[i][1] not in held); correct_top.append(False)
        continue
    p = reranker.predict_proba(feats[m])[:, 1]
    j = np.argmax(p)
    prob_best.append(p[j])
    answerable.append(EX[i][1] not in held)
    correct_top.append(C[cand[m][j]].lesson_id == EX[i][1])
prob_best, answerable, correct_top = map(np.array, (prob_best, answerable, correct_top))
print(f"\n{answerable.sum()} answerable and {(~answerable).sum()} unanswerable test questions")
print(f"  {'abstain below':>13s} {'answered, right source':>23s} {'answered, wrong source':>23s} {'abstained, rightly':>19s}")
for thr in (0.0, 0.2, 0.4, 0.6):
    ans = prob_best >= thr
    print(f"  {thr:13.1f} {np.mean(ans & correct_top):23.3f} {np.mean(ans & ~correct_top):23.3f} "
          f"{np.mean(~ans & ~answerable) / max(1e-9, np.mean(~answerable)):19.3f}")
print(f"mean best score: answerable {prob_best[answerable].mean():.3f}, unanswerable {prob_best[~answerable].mean():.3f}")
print("retrieval always returns *something*. A threshold on a calibrated relevance score turns 'nothing relevant' into")
print("'I don't know' at the cost of some good answers; where to set it is a product decision about which error is worse.")
assert prob_best[answerable].mean() > prob_best[~answerable].mean()

# %% [markdown]
# ## 4. Assembling the prompt

# %%
def build_prompt(question, chunk_ids, budget_tokens=1500):
    """Sources in rank order, each delimited and numbered, until the token budget is spent. Returns (prompt, sources)."""
    parts, used, sources = [], 0, []
    for i in chunk_ids:
        block = f'<source id="{len(sources) + 1}" lesson="{C[i].lesson_id}">\n{C[i].text}\n</source>'
        n = count_tokens(block)
        if used + n > budget_tokens:
            continue
        parts.append(block); sources.append(i); used += n
    prompt = ("Answer the question using only the sources below. Cite the sources you use as [n]. If the sources don't "
              "contain the answer, say so.\n\n" + "\n\n".join(parts) + f"\n\n<question>{question}</question>")
    return prompt, sources


def check_citations(answer, sources):
    """Each cited number must exist; each sentence with a citation should share content words with its source."""
    problems = []
    answer = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", answer)
    for sent in re.split(r"(?<=[.!?])\s+", answer.strip()):
        for n in map(int, re.findall(r"\[(\d+)\]", sent)):
            if not 1 <= n <= len(sources):
                problems.append(f"[{n}] does not exist")
                continue
            words = set(tokenize(re.sub(r"\[\d+\]", "", sent)))
            overlap = len(words & set(tokenize(C[sources[n - 1]].text))) / max(1, len(words))
            if overlap < 0.5:
                problems.append(f"[{n}] shares only {overlap:.0%} of the sentence's words")
    return problems


qi = test_q[3]
cand, feats = pools[qi][:2]
order = cand[np.argsort(-reranker.decision_function(feats))]
prompt, sources = build_prompt(EX[qi][0], order)
print(f"\nquestion: {EX[qi][0][:100]}")
print(f"prompt: {count_tokens(prompt):,} tokens, {len(sources)} sources, lessons {[C[i].lesson_id for i in sources]} "
      f"(the answer is in {EX[qi][1]})")
first = re.split(r"(?<=[.!?])\s+", C[sources[0]].text)[0]
other = next(c.text for c in C if c.lesson_id != C[sources[0]].lesson_id and len(c.text.split()) > 30)
foreign = re.split(r"(?<=[.!?])\s+", other)[0]
good = f"{first} [1]"
bad = f"{foreign} [1] This is also covered in [{len(sources) + 3}]."
print(f"citation check, faithful answer: {check_citations(good, sources) or 'no problems'}")
print(f"citation check, a sentence from another lesson and an invented citation: {check_citations(bad, sources)}")
print("citations are only useful if something checks them: numbered, delimited sources make that possible in code.")
print("A lexical check like this catches wrong numbers and unsupported sentences, not subtle misreadings: 41.4 and 43.1")
print("cover faithfulness evaluation properly.")
assert check_citations(good, sources) == [] and len(check_citations(bad, sources)) >= 2

print("\nAll checks passed.")
