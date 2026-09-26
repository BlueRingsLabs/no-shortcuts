# %% [markdown]
# # Lab 45.1: An LLM gateway: caching, routing, fallbacks
#
# 1. Exact caching on realistic traffic: how much repeats?
# 2. Semantic caching: paraphrases of the same question should hit; near-identical questions with different answers
#    must not. Measured on this course's question templates, where "lesson 29.1" and "lesson 29.2" differ by one
#    character and have different answers.
# 3. Routing and cascades: a cheap model first, the expensive one when needed.
# 4. Fallbacks and circuit breakers through a provider outage.

# %%
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "part-5-language-models" / "_shared"))
from fakellm import PRICES  # noqa: E402
from instructions import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, facts  # noqa: E402
from retrieval import load_or_train_encoder  # noqa: E402

torch.set_num_threads(4)

# %% [markdown]
# ## 1. Exact caching

# %%
rs = np.random.default_rng(0)
n_distinct = 20_000
popularity = 1 / np.arange(1, n_distinct + 1) ** 1.0                               # Zipf: a few questions are very common
popularity /= popularity.sum()
traffic = rs.choice(n_distinct, 100_000, p=popularity)
seen, hits = set(), 0
for q in traffic:
    hits += q in seen
    seen.add(q)
print(f"100,000 requests over 20,000 distinct questions (Zipf): exact-match hit rate {hits / len(traffic):.1%}")


def normalize(text):
    return " ".join(text.lower().split()).rstrip("?!. ")


variants = ["What's our refund policy?", "what's our refund policy", "What's our refund  policy ?"]
print("normalizing before hashing merges trivial variants:", len({hashlib.sha256(normalize(v).encode()).hexdigest() for v in variants}),
      "key for", len(variants), "strings")
print("the key must include everything that changes the answer: model version, system prompt, parameters, and the")
print("user's permissions if answers depend on data they can see. A shared cache across users is a data leak waiting.")

# %% [markdown]
# ## 2. Semantic caching

# %%
F = [(kind, slots, ans) for kind, slots, ans in facts() if kind == "lesson_title"]
random.Random(0).shuffle(F)
F = F[:150]
cached_q, cached_a = [], []
for kind, slots, ans in F:                                                        # the cache: one phrasing per fact
    cached_q.append(TRAIN_TEMPLATES[kind][0].format(**slots)); cached_a.append(ans)
queries = []
for kind, slots, ans in F:
    queries.append((HELDOUT_TEMPLATES[kind][0].format(**slots), ans))             # a paraphrase: should hit
    lid = slots["lid"]
    other = next(((s, a) for k, s, a in facts() if k == "lesson_title" and s["lid"] != lid
                  and s["lid"][:-1] == lid[:-1]), None)                           # a sibling lesson: 29.1 -> 29.2
    if other:
        queries.append((TRAIN_TEMPLATES[kind][0].format(**other[0]), other[1]))   # near-identical, different answer
enc, _ = load_or_train_encoder(verbose=False)                                     # 41.1's neural encoder
Cq = enc.embed(cached_q)
Qv = enc.embed([q for q, _ in queries])
print(f"\ncache: {len(cached_q)} answered questions; incoming: {len(queries)} (paraphrases and near-identical siblings)")
print("  threshold   hit rate   hits with the WRONG answer   share of hits that are wrong")
sem = {}
for thr in (0.5, 0.7, 0.8, 0.9, 0.95):
    hit, wrong = 0, 0
    for (q, a), qv in zip(queries, Qv):
        sims = Cq @ qv
        j = int(sims.argmax())
        if sims[j] >= thr:
            hit += 1
            wrong += cached_a[j] != a
    sem[thr] = (hit / len(queries), wrong / len(queries), wrong / max(1, hit))
    print(f"  {thr:9.2f}   {sem[thr][0]:8.3f}   {sem[thr][1]:26.3f}   {sem[thr][2]:28.3f}")
ex_q = queries[1][0] if len(queries) > 1 else queries[0][0]
print(f"e.g. '{cached_q[0]}' vs '{ex_q}'")
print("an embedding barely sees the difference between '18.1' and '18.2': even at a threshold of 0.95, about one hit")
print("in six returns another lesson's answer, and no threshold separates the paraphrases from the wrong siblings.")
print("Semantic caching is only safe for FAQ-like traffic without near-duplicate questions, with a high threshold,")
print("exact-match checks on entities (IDs, numbers, names) and a measured wrong-hit rate.")
assert sem[0.95][2] > 0.1 and sem[0.5][2] > sem[0.95][2]

# %% [markdown]
# ## 3. Routing and cascades
#
# Assumptions, stated: a small model answers 70% of requests correctly, a large one 90%; the small one's mistakes
# are concentrated on hard requests. A cascade tries the small model and escalates when a check fails; the check
# (a validator, or the small model's own confidence) catches a given share of its mistakes and wrongly escalates some
# correct answers.

# %%
n = 20_000
difficulty = rs.random(n)
small_ok = rs.random(n) > np.clip(difficulty * 0.6, 0, 1)                        # ~70%, worse on hard requests
large_ok = rs.random(n) > np.clip(difficulty * 0.2, 0, 1)                        # ~90%
tokens_in, tokens_out = 2000, 300
cost_call = {m: (tokens_in * PRICES[m]["input"] + tokens_out * PRICES[m]["output"]) / 1e6 for m in ("small", "large")}
print(f"\nsmall model correct {small_ok.mean():.3f}, large {large_ok.mean():.3f}; cost per call ${cost_call['small']:.5f} vs ${cost_call['large']:.4f}")
print(f"  {'strategy':46s} {'correct':>8s} {'cost per 1k requests':>21s}")
casc = {}
for name, catch, false_alarm in (("always small", None, None), ("always large", None, None),
                                 ("cascade, check catches 60%, 10% false alarms", 0.6, 0.1),
                                 ("cascade, check catches 90%, 10% false alarms", 0.9, 0.1)):
    if name == "always small":
        ok, cost = small_ok, np.full(n, cost_call["small"])
    elif name == "always large":
        ok, cost = large_ok, np.full(n, cost_call["large"])
    else:
        escalate = np.where(small_ok, rs.random(n) < false_alarm, rs.random(n) < catch)
        ok = np.where(escalate, large_ok, small_ok)
        cost = cost_call["small"] + escalate * cost_call["large"]
    casc[name] = (ok.mean(), cost.mean() * 1000)
    print(f"  {name:46s} {casc[name][0]:8.3f} ${casc[name][1]:20.2f}")
print("a cascade is only as good as its check. With a good one it matches the large model at about a third of the cost,")
print("and here even beats it: requests the small model gets right stay right, including some the large one would miss")
print("(FrugalGPT, Chen et al., 2023, reported the same). With a weak check it keeps many of the small model's errors.")
assert casc["cascade, check catches 90%, 10% false alarms"][1] < 0.5 * casc["always large"][1]

# %% [markdown]
# ## 4. Fallbacks and circuit breakers
#
# A day of traffic, one request per second. The primary provider fails 1% of requests normally, and 100% during a
# 40-minute outage. Each failed attempt costs 10 s (a timeout).

# %%
T = 24 * 3600
outage = (10 * 3600, 10 * 3600 + 2400)


def primary_fails(t, r):
    return (outage[0] <= t < outage[1]) or r.random() < 0.01


def simulate(strategy):
    r = random.Random(1)
    failed, slow, breaker_open_until, recent = 0, 0, -1, []
    for t in range(0, T):
        if strategy == "breaker" and t < breaker_open_until:
            use_primary = False
        else:
            use_primary = True
        latency = 0
        if use_primary:
            f = primary_fails(t, r)
            recent = (recent + [f])[-20:]
            if strategy == "breaker" and sum(recent) >= 10:                     # half the recent calls failed: open
                breaker_open_until, recent = t + 300, []                          # retry the primary in 5 minutes
            if not f:
                continue
            latency += 10
            if strategy == "none":
                failed += 1; slow += 1
                continue
        ok_fallback = r.random() > 0.01                                          # secondary provider, independent
        failed += not ok_fallback
        slow += latency >= 10
    return failed, slow


for strategy, label in (("none", "no fallback"), ("fallback", "fallback on every failure"),
                        ("breaker", "fallback + circuit breaker")):
    f, s = simulate(strategy)
    print(f"  {label:28s} failed requests {f:6,}   requests that waited for a timeout first {s:6,}")
print("a fallback turns an outage into slow requests; a circuit breaker stops sending traffic to a provider that's down,")
print("so most requests during the outage go straight to the fallback. Fallback models behave differently: evaluate them")
print("on your suite (43.1) and keep their prompts tested, or the fallback is a second outage with better uptime.")

print("\nAll checks passed.")
