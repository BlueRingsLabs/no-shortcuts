# %% [markdown]
# # Lab 50.2: Deployment strategies: shadow, canary, A/B, and rolling back
#
# A new model version replaces the current one. Three simulated releases:
#   fine:        the new model is slightly better
#   crashing:    it errors on 3% of requests (a feature it expects is sometimes missing in production)
#   segment:     it's better on average and much worse for one segment (10% of traffic)
# 1. Shadow mode: the new model scores real traffic, its answers are logged and compared, never served.
# 2. A canary with automatic rollback: 1% -> 5% -> 25% -> 100%, gated on error rate and a quality metric.
# 3. An A/B test: how much traffic and time it takes to detect a real difference, and the segment problem.

# %%
import numpy as np

rs = np.random.default_rng(502)
N_DAY = 200_000                                                                     # requests a day


def traffic(n, r):
    segment = r.random(n) < 0.10                                                     # e.g. a new market, a new device type
    return segment


def outcome(version, segment, r):
    """Returns (error, success) per request. Success: the business outcome the model drives (e.g. a conversion)."""
    n = len(segment)
    base = np.where(segment, 0.050, 0.050)
    lift = {"current": 0.0, "fine": 0.002, "crashing": 0.002, "segment": 0.004}[version]
    p = base + lift
    if version == "segment":
        p = np.where(segment, 0.030, p)                                              # much worse for the segment
    err = r.random(n) < (0.03 if version == "crashing" else 0.0005)
    success = (~err) & (r.random(n) < p)
    return err, success


# %% [markdown]
# ## 1. Shadow mode

# %%
r = np.random.default_rng(1)
seg = traffic(50_000, r)
print("shadow mode: the new model runs on a copy of 50,000 real requests; users only ever see the current model")
for version in ("fine", "crashing", "segment"):
    err, _ = outcome(version, seg, r)
    agree = r.random(len(seg)) < (0.97 if version != "segment" else np.where(seg, 0.6, 0.97))
    print(f"  {version:9s}: error rate {err.mean():.2%}; agrees with the current model on {agree.mean():.1%} of requests, "
          f"{agree[seg].mean():.1%} within the segment")
print("shadowing catches crashes, latency and disagreement with zero user risk. It can't measure outcomes, since users")
print("never see the new answers; disagreement concentrated in one segment is the hint to look there.")

# %% [markdown]
# ## 2. A canary with automatic rollback

# %%
STAGES = [(0.01, 1), (0.05, 1), (0.25, 1), (1.00, None)]                            # (traffic share, days at that share)


def canary(version, seed=0):
    r = np.random.default_rng(seed)
    exposed_errors = 0
    for share, days in STAGES:
        if days is None:
            return "promoted", exposed_errors
        n = int(N_DAY * share * days)
        seg = traffic(n, r)
        err, succ = outcome(version, seg, r)
        err_c, succ_c = outcome("current", traffic(n, r), r)
        exposed_errors += err.sum()
        # gate 1: error rate, with a margin (a real system uses an SLO and a burn rate)
        if err.mean() > err_c.mean() + 0.005:
            return f"rolled back at {share:.0%} (error rate {err.mean():.2%})", exposed_errors
        # gate 2: success rate not significantly worse (one-sided z-test on the difference)
        d = succ.mean() - succ_c.mean()
        se = np.sqrt(succ.mean() * (1 - succ.mean()) / n + succ_c.mean() * (1 - succ_c.mean()) / n)
        if d / se < -2.33:
            return f"rolled back at {share:.0%} (success {succ.mean():.4f} vs {succ_c.mean():.4f})", exposed_errors
    return "promoted", exposed_errors


print()
results = {}
for version in ("fine", "crashing", "segment"):
    status, errs = canary(version)
    results[version] = status
    print(f"canary for {version:9s}: {status}; users who hit an error on the way: {errs:,}")
print("a canary limits the blast radius: the crashing version reaches 1% of traffic, fails its error gate within a day,")
print("and rolls back automatically, with a few dozen users affected instead of thousands. Rollback must be one command")
print("(or none): the registry alias moves back (48.1). The segment regression sails through: the aggregate metric")
print("improves, so every gate passes.")
assert results["crashing"].startswith("rolled back at 1%") and results["fine"] == "promoted" and results["segment"] == "promoted"

# %% [markdown]
# ## 3. The A/B test, and the segment

# %%
def sample_size(p, delta, alpha=0.05, power=0.8):
    """Per arm, two-sided test for a difference in proportions (05.3)."""
    z_a, z_b = 1.96, 0.84
    return int(np.ceil((z_a + z_b) ** 2 * 2 * p * (1 - p) / delta ** 2))


n_arm = sample_size(0.05, 0.002)
print(f"\nto detect a lift from 5.0% to 5.2% with 80% power: {n_arm:,} requests per arm, {2 * n_arm / N_DAY:.1f} days at "
      f"full traffic split 50/50")
r = np.random.default_rng(3)
seg_a, seg_b = traffic(n_arm, r), traffic(n_arm, r)
_, s_a = outcome("current", seg_a, r)
_, s_b = outcome("segment", seg_b, r)
print(f"A/B test of the 'segment' model: overall {s_a.mean():.4f} -> {s_b.mean():.4f} ({(s_b.mean() - s_a.mean()) / s_a.mean():+.1%})")
print(f"  in the segment:     {s_a[seg_a].mean():.4f} -> {s_b[seg_b].mean():.4f} "
      f"({(s_b[seg_b].mean() - s_a[seg_a].mean()) / s_a[seg_a].mean():+.0%})")
print(f"  everywhere else:    {s_a[~seg_a].mean():.4f} -> {s_b[~seg_b].mean():.4f}")
print("the average hides a 40% drop for one group of users. Pre-register the segments that matter (new markets, devices,")
print("protected groups, 54.1) and look at them in every test, with the multiple-comparison caution of 05.2. Guardrail")
print("metrics (errors, latency, complaints) are part of the test, not an afterthought.")
assert s_b[seg_b].mean() < 0.8 * s_a[seg_a].mean() and s_b.mean() > s_a.mean()

print("\nAll checks passed.")
