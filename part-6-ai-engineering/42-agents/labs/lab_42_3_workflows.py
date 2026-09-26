# %% [markdown]
# # Lab 42.3: Workflow patterns and multi-agent systems
#
# 1. Routing: send each request to a specialist. A real router (TF-IDF + logistic regression) trained to send course
#    questions to the right Part, and what to do when it's unsure.
# 2. Chains, parallel sections and orchestrator-workers: latency and cost, on the simulated API's clock.
# 3. Handoffs lose information: agents passing summaries to agents, measured on this course's text.
# 4. Voting between agents: when more agents help, and when they're the same agent several times.

# %%
import random
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from fakellm import FakeLLM, VirtualClock, count_tokens  # noqa: E402
from retrieval import _sentences, exercise_queries, lessons, tokenize  # noqa: E402

L = lessons()
part_of = {les.id: les.part for les in L}
PART_NAMES = {0: "orientation", 1: "foundations", 2: "data engineering", 3: "classical ML", 4: "deep learning",
              5: "language models", 6: "AI engineering"}

# %% [markdown]
# ## 1. Routing

# %%
EX = exercise_queries(L)
r = random.Random(0)
by_lesson = sorted({lid for _, lid in EX})
test_lessons = set(r.sample(by_lesson, len(by_lesson) // 4))                     # route questions from unseen lessons
train = [(q, part_of[l]) for q, l in EX if l not in test_lessons]
test = [(q, part_of[l]) for q, l in EX if l in test_lessons]
vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
router = LogisticRegression(C=5, max_iter=2000).fit(vec.fit_transform([q for q, _ in train]), [p for _, p in train])
proba = router.predict_proba(vec.transform([q for q, _ in test]))
pred = router.classes_[proba.argmax(1)]
conf = proba.max(1)
truth = np.array([p for _, p in test])
print(f"router trained on {len(train)} questions, tested on {len(test)} from {len(test_lessons)} unseen lessons")
print(f"accuracy {np.mean(pred == truth):.3f} over {len(router.classes_)} specialists (Parts); majority class "
      f"{np.mean(truth == np.bincount(truth).argmax()):.3f}")
print("  confidence >=   share routed   accuracy of routed   the rest go to a generalist")
for thr in (0.0, 0.4, 0.6, 0.8):
    m = conf >= thr
    print(f"  {thr:13.1f}   {m.mean():12.2f}   {np.mean(pred[m] == truth[m]) if m.any() else float('nan'):18.3f}")
print("a misrouted request goes to a specialist that can't help, with no error to show for it. Route only when")
print("confident, send the rest to a generalist (or ask), and log routing decisions so misroutes can be found.")
assert np.mean(pred[conf >= 0.6] == truth[conf >= 0.6]) > np.mean(pred == truth)

# %% [markdown]
# ## 2. Chains, parallel sections, orchestrator-workers
#
# Writing a four-section report on a topic. Each call has a 1,500-token prompt; each section is 400 tokens of output.
# The simulated API's clock gives each call's latency; calls made in parallel take as long as the slowest one.

# %%
def sized_brain(messages, tools, rng):
    return {"text": " word" * sized_brain.n}                                        # output of the requested length


def call(llm, prompt_tokens, out_tokens):
    sized_brain.n = out_tokens
    before = llm.clock.now()
    llm.create([{"role": "user", "content": " x" * prompt_tokens}], max_tokens=out_tokens + 10)
    return llm.clock.now() - before


def pattern(name, llm):
    llm.clock.t = 0.0
    cost0 = sum(e.get("cost", 0) for e in llm.log)
    if name == "one call (everything at once)":
        t = call(llm, 1500, 1600)
    elif name == "chain (4 sections, one after another)":
        t = sum(call(llm, 1500 + 400 * i, 400) for i in range(4))                   # each call sees what came before
    elif name == "parallel sections":
        t = max(call(llm, 1500, 400) for _ in range(4))
    else:                                                                             # orchestrator-workers
        t = call(llm, 1500, 200)                                                      # plan
        t += max(call(llm, 1700, 400) for _ in range(4))                              # workers, in parallel
        t += call(llm, 1500 + 1600, 300)                                              # synthesize
    return t, sum(e.get("cost", 0) for e in llm.log) - cost0


llm = FakeLLM(brain=sized_brain, clock=VirtualClock(), model="medium", cache_min_tokens=10 ** 9)
print(f"\n{'pattern':52s} {'latency':>8s} {'cost':>9s}")
pat = {}
for name in ("one call (everything at once)", "chain (4 sections, one after another)", "parallel sections",
             "orchestrator-workers (plan, 4 workers, synthesize)"):
    pat[name] = pattern(name, llm)
    print(f"{name:52s} {pat[name][0]:7.1f}s ${pat[name][1]:8.4f}")
print("output tokens dominate latency (37.2), so splitting the output across parallel calls is the big latency lever.")
print("Chains are slowest and let each step see the previous ones; orchestrators pay for planning and synthesis in")
print("exchange for adaptivity. Multi-agent designs typically use several times the tokens of a single call.")
assert pat["parallel sections"][0] < pat["chain (4 sections, one after another)"][0] / 2

# %% [markdown]
# ## 3. Handoffs lose information
#
# In multi-agent systems agents pass work to each other as text, often summarized to fit the next agent's context.
# Here an extractive summarizer (keep the 40% most central sentences, by TF-IDF similarity to the rest) plays each
# agent's handoff. What survives of the specific facts (numbers and code identifiers) after each handoff?

# %%
def summarize(sentences, keep=0.4):
    if len(sentences) <= 2:
        return sentences
    v = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None).fit_transform(sentences)
    centrality = np.asarray((v @ v.T).sum(1)).ravel()
    k = max(1, int(round(keep * len(sentences))))
    top = sorted(np.argsort(-centrality)[:k])
    return [sentences[i] for i in top]


def facts(sentences):
    text = " ".join(sentences)
    return set(re.findall(r"\b\d+(?:\.\d+)?%?|`[^`]+`", text))


docs = []
for les in L:
    for head, body in les.sections:
        s = _sentences(body)
        if len(s) >= 12 and len(facts(s)) >= 5:
            docs.append(s)
print(f"\n{len(docs)} lesson sections with at least 12 sentences and 5 specific facts")
retained = {0: 1.0}
current = docs
for hop in range(1, 4):
    current = [summarize(s) for s in current]
    retained[hop] = np.mean([len(facts(c) & facts(d)) / len(facts(d)) for c, d in zip(current, docs)])
print("  handoffs   share of specific facts (numbers, identifiers) still present")
for hop, v in retained.items():
    print(f"  {hop:8d}   {v:.2f}")
print("each summary keeps the gist and drops specifics, and nobody downstream knows what was dropped. Pass the")
print("artifacts themselves (the data, the file, a reference to it) between agents, not prose about them; give each agent")
print("the original task, not a paraphrase of it.")
assert retained[3] < 0.5 * retained[0]

# %% [markdown]
# ## 4. Voting
#
# Five agents answer the same 2,000 questions; each is right 70% of the time. Majority vote. In one world the agents'
# errors are independent (different models, different information); in the other, they're the same model with the
# same prompt, so a question hard for one is hard for all (difficulty drawn per question, same 70% average).

# %%
rs = np.random.default_rng(0)
n_q, n_agents = 2000, 5
indep = rs.random((n_q, n_agents)) < 0.7
difficulty = rs.beta(0.7, 0.3, n_q)                                              # mean 0.7, spread across questions
corr = rs.random((n_q, n_agents)) < difficulty[:, None]
for name, ok in (("independent errors", indep), ("same model, same prompt", corr)):
    print(f"\n{name:26s} single agent {ok[:, 0].mean():.3f}   majority of 5 {np.mean(ok.sum(1) >= 3):.3f}")
print("voting helps only as much as the voters' errors are independent (Condorcet's jury theorem assumes they are).")
print("Five copies of one model mostly agree with each other, including when they're wrong.")
assert np.mean(indep.sum(1) >= 3) > np.mean(corr.sum(1) >= 3) + 0.05

print("\nAll checks passed.")
