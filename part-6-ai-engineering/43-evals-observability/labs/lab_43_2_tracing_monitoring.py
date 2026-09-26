# %% [markdown]
# # Lab 43.2: Tracing, monitoring and feedback
#
# A week of traffic to a course-QA assistant (retrieve, then generate), on the simulated API. On day 4 the provider
# silently updates the model behind the alias we call: answers get longer, and a bit worse.
# 1. Tracing: spans for every step, with the attributes you'll need later.
# 2. Monitoring from traces: latency percentiles, cost, errors, per day. Would you have noticed day 4?
# 3. User feedback: what thumbs-up rates measure, and what they don't.
# 4. Reviewing traces: random samples vs targeted ones.

# %%
import random
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from fakellm import FakeLLM, VirtualClock  # noqa: E402

# %% [markdown]
# ## 1. Tracing

# %%
SPANS = []


class Tracer:
    """A minimal version of what OpenTelemetry gives you: nested spans with timing and attributes."""

    def __init__(self, clock):
        self.clock, self.stack = clock, []

    @contextmanager
    def span(self, name, **attrs):
        s = {"trace_id": self.stack[0]["trace_id"] if self.stack else uuid.uuid4().hex[:12], "span_id": uuid.uuid4().hex[:8],
             "parent": self.stack[-1]["span_id"] if self.stack else None, "name": name, "start": self.clock.now(), **attrs}
        self.stack.append(s)
        try:
            yield s
            s["status"] = "ok"
        except Exception as e:
            s["status"] = f"error: {type(e).__name__}"
            raise
        finally:
            s["end"] = self.clock.now()
            self.stack.pop()
            SPANS.append(s)


def make_brain(state):
    def brain(messages, tools, rng):
        long_answers = state["day"] >= 4                                       # the silent model update
        n_words = int(rng.gauss(90 if long_answers else 55, 15))
        return {"text": " ".join(["word"] * max(10, n_words))}
    return brain


state = {"day": 0}
clock = VirtualClock()
llm = FakeLLM(brain=make_brain(state), clock=clock, model="medium", failure_rate=0.01, seed=0)
tracer = Tracer(clock)
rng = random.Random(0)
PROMPT_VERSION = "qa-v7"


def handle(request_id, user_is_power_user):
    with tracer.span("request", request_id=request_id, prompt_version=PROMPT_VERSION, day=state["day"]) as root:
        with tracer.span("retrieve", k=5):
            clock.sleep(rng.lognormvariate(-3.0, 0.4))                            # ~50 ms vector search
        with tracer.span("generate", model="medium@alias") as g:
            r = llm.create([{"role": "user", "content": "context " * 900 + "question"}], max_tokens=400)
            g.update(input_tokens=r["usage"]["input_tokens"], cached_input_tokens=r["usage"]["cached_input_tokens"],
                     output_tokens=r["usage"]["output_tokens"], cost=r["cost"])
        # hidden truth, for the lab only: was the answer good? Worse after the update.
        root["good"] = rng.random() < (0.80 if state["day"] < 4 else 0.72)
        root["power_user"] = user_is_power_user
    return root


t0 = time.time()
for day in range(7):
    state["day"] = day
    for i in range(400):
        try:
            handle(f"d{day}r{i}", rng.random() < 0.2)
        except Exception:
            pass
        clock.sleep(rng.expovariate(1 / 30))                                      # next request
print(f"simulated 7 days x 400 requests in {time.time() - t0:.1f}s: {len(SPANS):,} spans")
one = [s for s in SPANS if s["trace_id"] == SPANS[5]["trace_id"]]
print("one trace:")
for s in sorted(one, key=lambda s: (s["start"], s["parent"] is not None)):
    extra = {k: v for k, v in s.items() if k in ("model", "input_tokens", "cached_input_tokens", "output_tokens", "prompt_version", "k")}
    print(f"  {'  ' if s['parent'] else ''}{s['name']:10s} {1000 * (s['end'] - s['start']):7.0f} ms  {s['status']}  {extra}")

# %% [markdown]
# ## 2. Monitoring, from the traces

# %%
reqs = [s for s in SPANS if s["name"] == "request"]
gens = {s["trace_id"]: s for s in SPANS if s["name"] == "generate"}
print("\nday   requests   errors   p50 latency   p95 latency   output tokens   cost/1k requests   (hidden) quality")
daily = {}
for d in range(7):
    rq = [s for s in reqs if s["day"] == d]
    ok = [s for s in rq if s["status"] == "ok"]
    lat = np.array([s["end"] - s["start"] for s in ok])
    out = np.array([gens[s["trace_id"]]["output_tokens"] for s in ok])
    cost = sum(gens[s["trace_id"]]["cost"] for s in ok) / len(rq) * 1000
    daily[d] = (np.percentile(lat, 50), np.percentile(lat, 95), out.mean(), cost, np.mean([s["good"] for s in ok]))
    print(f"{d:3d}   {len(rq):8d}   {1 - len(ok) / len(rq):6.1%}   {daily[d][0]:10.2f}s   {daily[d][1]:10.2f}s   "
          f"{daily[d][2]:13.0f}   ${daily[d][3]:15.2f}   {daily[d][4]:.2f}")
base = np.array([daily[d][1] for d in range(4)])
alert = [d for d in range(4, 7) if daily[d][1] > base.mean() + 3 * base.std()]
print(f"alert rule 'p95 latency above the baseline mean + 3 sd': fires on days {alert}")
print("latency, output length and cost moved the day the model changed, with no deploy on our side: that's what")
print("pinning model versions prevents (40.1). Quality dropped too, and nothing in these metrics shows it directly.")
assert alert and alert[0] == 4

# %% [markdown]
# ## 3. User feedback

# %%
fb_rng = random.Random(5)
feedback = []
for s in reqs:
    if s["status"] != "ok":
        continue
    p_give = 0.02 if s["good"] else 0.10                                         # unhappy users speak up more
    p_give *= 2 if s["power_user"] else 1
    if fb_rng.random() < p_give:
        feedback.append((s["day"], s["good"]))
print(f"\n{len(feedback)} feedback events from {len(reqs)} requests ({len(feedback) / len(reqs):.1%})")
for label, days in (("days 0-3", range(4)), ("days 4-6", range(4, 7))):
    f = [g for d, g in feedback if d in days]
    truth = np.mean([s["good"] for s in reqs if s["day"] in days and s["status"] == "ok"])
    print(f"  {label}: thumbs-up share {np.mean(f):.2f} (n={len(f)}); true share of good answers {truth:.2f}")
f03 = np.mean([g for d, g in feedback if d < 4])
print(f"feedback comes mostly from unhappy users, so its level says little about quality: a {f03:.0%} thumbs-up rate")
print("coexists with 79% good answers. Its *changes* are informative, but noisy at these counts. Use feedback to find")
print("failures to look at, not as your quality metric; measure quality with evals on sampled traffic (43.1).")

# %% [markdown]
# ## 4. Which traces to review

# %%
days46 = [s for s in reqs if s["day"] >= 4 and s["status"] == "ok"]
rv = random.Random(6)
random_pick = rv.sample(days46, 50)
negative = [s for s in days46 if not s["good"] and rv.random() < 0.10]            # traces with a thumbs-down
long_ones = sorted(days46, key=lambda s: -gens[s["trace_id"]]["output_tokens"])[:50]
for name, pick in (("random sample", random_pick), ("thumbs-down traces", negative[:50]), ("longest answers", long_ones)):
    print(f"  review {len(pick):2d} traces, {name:20s}: {np.mean([not s['good'] for s in pick]):.0%} are failures")
print("targeted samples find failures faster; random samples estimate rates without bias. Do both: random for")
print("measurement, targeted for discovery, and turn every confirmed failure into a test case.")

print("\nAll checks passed.")
