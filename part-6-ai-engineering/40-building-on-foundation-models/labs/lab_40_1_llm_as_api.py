# %% [markdown]
# # Lab 40.1: The LLM as an API
#
# Against a simulated API (part-6-ai-engineering/_shared/fakellm.py: offline, deterministic, virtual time):
# 1. Tokens and money: the cost of a real workload, by model size, with prompt caching and a batch discount.
# 2. Latency: time to first token and total time vs prompt and output length; what streaming changes.
# 3. A robust client: timeouts, retries with exponential backoff and jitter, Retry-After, and detecting truncation.
# 4. The thundering herd: many clients retrying at once, with and without jitter.

# %%
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from fakellm import PRICES, FakeLLM, RateLimitError, ServerError, Timeout, VirtualClock, count_tokens  # noqa: E402


def summarizer(messages, tools, rng):
    """A stand-in model: 'summarizes' by returning the first sentence and a few words of commentary."""
    text = messages[-1]["content"]
    first = text.split(". ")[0]
    return {"text": f"Summary: {first}. The customer wants this resolved; priority {rng.choice(['low', 'medium', 'high'])}."}


# %% [markdown]
# ## 1. Tokens and money

# %%
ticket = ("I was charged twice for my subscription this month. The second charge appeared on the 14th and I never "
          "authorized it. I already contacted support last week and got no answer. Please refund the duplicate charge "
          "and confirm that my plan is still active, because I use it for work every day. ") * 3
system = ("You are a support triage assistant for a software company. Summarize the ticket in two sentences, classify "
          "its priority (low, medium, high) and list any account actions requested. Follow the company policy: never "
          "promise refunds, never ask for passwords, escalate legal threats. ") * 4
n_sys, n_ticket = count_tokens(system), count_tokens(ticket)
n_out = 60
print(f"system prompt {n_sys} tokens, ticket {n_ticket} tokens, output ~{n_out} tokens")
print("(English prose here is ~4 characters per token; our course tokenizer has a small vocabulary, so a bit more tokens)")

N = 1_000_000
print(f"\ncost of triaging {N:,} tickets:")
print("  model     plain        with prompt caching   + batch API (50% off, async)")
costs = {}
for model, p in PRICES.items():
    plain = N * ((n_sys + n_ticket) * p["input"] + n_out * p["output"]) / 1e6
    cached = N * (n_sys * p["cached_input"] + n_ticket * p["input"] + n_out * p["output"]) / 1e6
    costs[model] = (plain, cached, cached / 2)
    print(f"  {model:7s} ${plain:11,.0f}   ${cached:17,.0f}   ${cached / 2:13,.0f}")
print("the model choice moves the bill by 60x; caching the stable system prompt and batching non-urgent work each cut it")
print("again. Which model is needed is an eval question (43.1), not a vibes question.")
assert costs["large"][0] > 50 * costs["small"][0] and costs["medium"][1] < costs["medium"][0]

# %% [markdown]
# ## 2. Latency

# %%
clock = VirtualClock()
llm = FakeLLM(brain=lambda m, t, r: {"text": " word" * m[-1]["n_out"]}, clock=clock, model="medium", cache_min_tokens=10**9)
print("\nmedium model, time to first token / total:")
print("  prompt tokens   output tokens   TTFT     total")
for n_in, n_o in ((200, 50), (2000, 50), (20000, 50), (2000, 500)):
    r = llm.create(messages=[{"role": "user", "content": " x" * n_in, "n_out": n_o}], max_tokens=1000)
    print(f"  {r['usage']['input_tokens']:13,}   {r['usage']['output_tokens']:13,}   {r['ttft']:5.2f}s   {r['latency']:5.2f}s")
print("prompt length drives time to first token (prefill); output length drives everything after it (decode, 37.2).")
print("Streaming doesn't make the answer faster; it shows the first words after TTFT instead of after the total.")

# %% [markdown]
# ## 3. A robust client

# %%
def call_naive(llm, messages, **kw):
    return llm.create(messages, **kw)


def call_robust(llm, messages, max_retries=6, base=1.0, cap=30.0, timeout=30.0, rng=None, **kw):
    """Retry transient errors (429, 5xx, timeouts) with exponential backoff and full jitter, honoring Retry-After.
    Never retry client errors (a bad request stays bad). Caller must make the operation safe to repeat."""
    rng = rng or random.Random(0)
    for attempt in range(max_retries + 1):
        try:
            resp = llm.create(messages, timeout=timeout, **kw)
            if resp["stop_reason"] == "max_tokens":
                resp["warning"] = "truncated: raise max_tokens or ask for less"      # don't silently use half an answer
            return resp
        except RateLimitError as e:
            wait = max(e.retry_after, rng.uniform(0, min(cap, base * 2 ** attempt)))
        except (ServerError, Timeout):
            wait = rng.uniform(0, min(cap, base * 2 ** attempt))
        if attempt == max_retries:
            raise
        llm.clock.sleep(wait)


results = {}
for name, fn in (("naive", call_naive), ("retries + backoff + jitter", call_robust)):
    clock = VirtualClock()
    llm = FakeLLM(brain=summarizer, clock=clock, model="small", rpm=30, failure_rate=0.08, timeout_rate=0.03, seed=1)
    ok, failed = 0, 0
    for i in range(200):
        try:
            fn(llm, [{"role": "user", "content": ticket}], max_tokens=100, system=system)
            ok += 1
        except Exception:
            failed += 1
    results[name] = (ok, failed, clock.now())
    print(f"\n{name:28s}: {ok} succeeded, {failed} failed, {clock.now() / 60:.1f} virtual minutes for 200 tickets")
print("the naive client fires as fast as it can, hits the 30 requests/minute limit after 30 calls and loses most of the")
print("work; the few server errors and timeouts (8% and 3%) would have cost it more on top. The robust client loses none:")
print("it waits when told to (Retry-After) and retries what is safe to retry, running at exactly the rate limit.")
assert results["retries + backoff + jitter"][0] == 200 and results["naive"][1] > 20

short = FakeLLM(brain=summarizer, clock=VirtualClock(), model="small")
r = call_robust(short, [{"role": "user", "content": ticket}], max_tokens=10)
print(f"\nmax_tokens=10: stop_reason {r['stop_reason']!r}, content {r['content']!r}, warning: {r.get('warning')}")

# %% [markdown]
# ## 4. The thundering herd
#
# 50 clients hit a server at the same moment. The server can take one request per 0.1 s slot (10 per second); requests
# arriving in a slot that's already taken are rejected. Clients retry with a fixed delay, exponential backoff, or
# exponential backoff with full jitter (AWS Architecture Blog, 2015). Discrete-event simulation.

# %%
def herd(policy, clients=50, capacity=1, seed=0):
    r = random.Random(seed)
    pending = {c: 0.0 for c in range(clients)}                                         # next attempt time
    attempts = {c: 0 for c in range(clients)}
    done_at, rejected = {}, 0
    t = 0.0
    while pending:
        t = min(pending.values())
        batch = sorted(c for c, at in pending.items() if abs(at - t) < 1e-9)
        slot = int(t * 10)                                                              # 0.1-second slots
        served_in_slot = sum(1 for v in done_at.values() if int(v * 10) == slot)
        for c in batch:
            if served_in_slot < capacity:
                done_at[c] = t; served_in_slot += 1; del pending[c]
            else:
                rejected += 1
                attempts[c] += 1
                k = attempts[c]
                if policy == "fixed 1s":
                    pending[c] = t + 1.0
                elif policy == "exponential":
                    pending[c] = t + min(30, 0.5 * 2 ** k)
                else:
                    pending[c] = t + r.uniform(0, min(30, 0.5 * 2 ** k))
    return max(done_at.values()), rejected


print("\n                                 all 50 served after   rejected requests")
herd_res = {}
for policy in ("fixed 1s", "exponential", "exponential + full jitter"):
    herd_res[policy] = herd(policy)
    print(f"  {policy:28s} {herd_res[policy][0]:14.1f}s   {herd_res[policy][1]:10d}")
print("synchronized retries collide again and again; jitter spreads them so the server can absorb them. Use a client")
print("library's retry logic (most SDKs retry with jittered backoff by default) and a shared rate limiter across workers.")
assert herd_res["exponential + full jitter"][1] < herd_res["exponential"][1] / 3
assert herd_res["exponential + full jitter"][0] < herd_res["fixed 1s"][0]

print("\nAll checks passed.")
