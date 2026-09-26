# %% [markdown]
# # Lab 43.1: Building an eval suite
#
# The system under test: a small question-answering system over this course (BM25 retrieval, 41.1; the answer is the
# best-matching passage's first sentences, with a citation). Two versions, as if someone changed the chunking.
# The test set: the course's exercise questions, whose correct source lesson is known.
# 1. Assertions in code, per case, and a report that says which cases broke.
# 2. Comparing versions: paired, with intervals, and how many cases you need.
# 3. An LLM judge: measure it against human labels before trusting it; correct its pass rate; detect position bias.

# %%
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from retrieval import BM25, _sentences, chunk, exercise_queries, lessons  # noqa: E402

L = lessons()
BY_ID = {les.id: les for les in L}
CASES = [{"id": f"ex{i}", "question": q, "gold_lesson": lid} for i, (q, lid) in enumerate(exercise_queries(L))]
print(f"{len(CASES)} test cases")


class QA:
    def __init__(self, headers):
        self.C = chunk(L, "sections", 120)
        texts = [(f"{BY_ID[c.lesson_id].title}. {c.heading}. " if headers else "") + c.text for c in self.C]
        self.bm = BM25(texts)

    def answer(self, question):
        c = self.C[int(np.argmax(self.bm.scores(question)))]
        sents = _sentences(c.text)[:2] or [c.text[:200]]
        return {"text": " ".join(sents) + f" [{c.lesson_id}]", "citations": [c.lesson_id]}


systems = {"v1 (plain chunks)": QA(headers=False), "v2 (chunks with title and heading)": QA(headers=True)}
outputs = {name: [s.answer(c["question"]) for c in CASES] for name, s in systems.items()}

# %% [markdown]
# ## 1. Assertions

# %%
def check(case, out):
    """Every check is a named, deterministic assertion. Returns the names of the failed ones."""
    failed = []
    if not out["text"].strip():
        failed.append("empty")
    if not out["citations"]:
        failed.append("no citation")
    if any(c not in BY_ID for c in out["citations"]):
        failed.append("cites a lesson that doesn't exist")
    if len(out["text"].split()) > 90:
        failed.append("too long (> 90 words)")
    if case["gold_lesson"] not in out["citations"]:
        failed.append("wrong source")
    if re.search(r"\[\d{2}\.\d\]", out["text"]) is None:
        failed.append("citation missing from the text")
    return failed


results = {name: [check(c, o) for c, o in zip(CASES, outs)] for name, outs in outputs.items()}
for name, res in results.items():
    counts = {}
    for f in res:
        for k in f:
            counts[k] = counts.get(k, 0) + 1
    print(f"\n{name}: pass rate {np.mean([not f for f in res]):.3f}; failures by check: {counts}")
v1, v2 = results.values()
fixed = [CASES[i]["id"] for i in range(len(CASES)) if v1[i] and not v2[i]]
broke = [CASES[i]["id"] for i in range(len(CASES)) if not v1[i] and v2[i]]
print(f"v1 -> v2: {len(fixed)} cases fixed, {len(broke)} cases broken, e.g. broken: {broke[:5]}")
print("a single pass rate hides that a change fixes some cases and breaks others. Report both lists: the broken cases")
print("are where to look before shipping, and some of them may matter more than all the fixed ones.")
assert broke and fixed

# %% [markdown]
# ## 2. Comparing versions

# %%
a = np.array([not f for f in v1], float)
b = np.array([not f for f in v2], float)
rs = np.random.default_rng(0)
idx = rs.integers(0, len(a), (5000, len(a)))
diff = (b - a)[idx].mean(1)
lo, hi = np.percentile(diff, [2.5, 97.5])
print(f"\nv2 - v1: {b.mean() - a.mean():+.3f}, paired 95% CI [{lo:+.3f}, {hi:+.3f}]")
for n in (50, 200):
    sub = rs.choice(len(a), n, replace=False)
    d = (b[sub] - a[sub])[rs.integers(0, n, (5000, n))].mean(1)
    print(f"  with only {n:3d} cases: {b[sub].mean() - a[sub].mean():+.3f}, CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
disagree = np.mean(a != b)
print(f"the versions disagree on {disagree:.1%} of cases. For a paired comparison, the cases that matter are the")
print("disagreements: to detect a 2-point improvement you need enough of them, which means hundreds to thousands of")
print("cases, not the 30 someone wrote by hand on a Friday.")

# %% [markdown]
# ## 3. An LLM judge, measured
#
# Most qualities can't be checked in code ("is this answer helpful and correct?"), so teams use a model as a judge. A
# judge is a classifier, with errors and biases. Here: 400 answers labeled by people (the labels are the known truth:
# did the answer come from the right lesson?), and a judge that agrees with people most of the time, prefers longer
# answers, and in pairwise comparisons prefers whichever answer it reads first. We only get to see its outputs.

# %%
r = random.Random(1)
sample = r.sample(range(len(CASES)), 400)
human = np.array([not v2[i] for i in sample], dtype=bool)
lengths = np.array([len(outputs["v2 (chunks with title and heading)"][i]["text"].split()) for i in sample])


def judge(is_good, length, rng):
    p = (0.85 if is_good else 0.25) + 0.004 * (length - 40)                  # verbosity bias
    return rng.random() < min(max(p, 0), 1)


jr = random.Random(2)
judged = np.array([judge(h, n, jr) for h, n in zip(human, lengths)])
tpr, fpr = judged[human].mean(), judged[~human].mean()
po = np.mean(judged == human)
pe = judged.mean() * human.mean() + (1 - judged.mean()) * (1 - human.mean())
kappa = (po - pe) / (1 - pe)
print(f"\njudge vs people on 400 answers: agreement {po:.3f}, Cohen's kappa {kappa:.3f}")
print(f"judge passes {tpr:.2f} of good answers and {fpr:.2f} of bad ones")
print(f"pass rate: people {human.mean():.3f}; judge {judged.mean():.3f}")

# the judge will score thousands of new answers with no human labels. Correct its raw rate with the measured error
# rates (Rogan and Gladen, 1978): true = (observed - fpr) / (tpr - fpr)
new = np.array([not f for f in v1], dtype=bool)                               # a system the judge hasn't been measured on
new_len = np.array([len(o["text"].split()) for o in outputs["v1 (plain chunks)"]])
raw = np.mean([judge(h, n, jr) for h, n in zip(new, new_len)])
corrected = (raw - fpr) / (tpr - fpr)
br = np.random.default_rng(4)
boot = []
for _ in range(2000):                                                         # the error rates are estimates too
    k = br.integers(0, len(human), len(human))
    t_, f_ = judged[k][human[k]].mean(), judged[k][~human[k]].mean()
    boot.append((raw - f_) / (t_ - f_))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"on v1's {len(new)} answers: judge's raw pass rate {raw:.3f}, corrected {corrected:.3f} (95% CI [{lo:.3f}, {hi:.3f}]), "
      f"truth {new.mean():.3f}")
print("the correction removes the judge's systematic bias and pays for it in variance: dividing by tpr - fpr amplifies")
print("the uncertainty of rates measured on 400 labels. And the rates were measured on v2's answers; this judge likes long")
print("answers, v1's are shorter, so its error rates on v1 differ. Correct, report the interval, and re-measure the judge")
print("on the system you're scoring when you can.")

# pairwise: which of two answers is better? A position-biased judge picks the first one shown more often
def pairwise(first_good, second_good, rng):
    if first_good != second_good:
        p_first = 0.8 if first_good else 0.2
    else:
        p_first = 0.5
    return rng.random() < min(1, p_first + 0.15)                              # position bias toward the first


pairs = [(bool(a[i]), bool(b[i])) for i in range(len(a)) if a[i] != b[i]]
pr = random.Random(3)
v1_first = np.mean([pairwise(x, y, pr) for x, y in pairs])                     # "v1 wins" when v1 is shown first
v2_first = np.mean([not pairwise(y, x, pr) for x, y in pairs])                 # "v1 wins" when v1 is shown second
truth_v1 = np.mean([x for x, y in pairs])
print(f"\npairwise judge on the {len(pairs)} cases where v1 and v2 differ: v1 wins {v1_first:.2f} when shown first, "
      f"{v2_first:.2f} when shown second (truth: {truth_v1:.2f})")
print(f"averaged over both orders: {(v1_first + v2_first) / 2:.2f}")
print("never trust a judge you haven't measured: label a few hundred cases yourself, compute agreement and kappa, check")
print("its bias toward length and position (swap the order and compare), and correct aggregate rates with its measured")
print("error rates. Re-measure when you change the judge's model or prompt: it's a model in production like any other.")
assert lo <= new.mean() <= hi and hi - lo > 0.05
assert v1_first - v2_first > 0.15

print("\nAll checks passed.")
