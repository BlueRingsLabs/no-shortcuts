# %% [markdown]
# # Lab 46.1: Bandits and exploration
#
# 10 Bernoulli arms (say, 10 versions of a page, each with an unknown conversion rate). 400 independent problems,
# 10,000 pulls each, all algorithms vectorized across problems.
# 1. Greedy, epsilon-greedy, optimistic initial values, UCB1, KL-UCB and Thompson sampling: regret and how often each
#    finds the best arm.
# 2. When the world changes: sample averages vs a constant step size.
# 3. A/B test or bandit? Regret during the test, and a bias in what the bandit tells you afterwards.

# %%
import numpy as np

R, K, T = 400, 10, 10_000
rs = np.random.default_rng(461)
P = rs.uniform(0.02, 0.12, (R, K))                                          # conversion rates: small and close together
best = P.max(1)


def kl_ucb(Q, N, t, iters=20):
    """Largest q with N * KL(Q, q) <= log t, for Bernoulli KL, by bisection (Garivier and Cappe, 2011)."""
    kl = lambda p, q: p * np.log(np.maximum(p, 1e-12) / q) + (1 - p) * np.log(np.maximum(1 - p, 1e-12) / (1 - q))
    lo, hi = Q.copy(), np.ones_like(Q)
    budget = np.log(t + 1) / np.maximum(N, 1e-9)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = kl(Q, np.clip(mid, 1e-9, 1 - 1e-9)) <= budget
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    return lo


def run(policy, T=T, seed=0, P=P, drift=0.0, step_size=None, record_estimates=False):
    r = np.random.default_rng(seed)
    P = P.copy()
    Q = np.full((R, K), 1.0 if policy == "optimistic" else 0.0)            # value estimates
    N = np.zeros((R, K))
    N0 = 20.0 if policy == "optimistic" else 0.0                            # the optimistic prior counts as 20 pulls
    S, F = np.ones((R, K)), np.ones((R, K))                                 # Beta(1, 1) posteriors for Thompson
    regret = np.zeros(T)
    optimal = np.zeros(T)
    rows = np.arange(R)
    for t in range(T):
        if policy == "greedy" or policy == "optimistic":
            a = Q.argmax(1)
        elif policy.startswith("eps"):
            eps = float(policy.split("=")[1])
            a = np.where(r.random(R) < eps, r.integers(0, K, R), Q.argmax(1))
        elif policy == "UCB1":
            ucb = Q + np.sqrt(2 * np.log(t + 1) / np.maximum(N, 1e-9))
            a = np.where(N.min(1) == 0, N.argmin(1), ucb.argmax(1))           # try every arm once first
        elif policy == "KL-UCB":
            a = np.where(N.min(1) == 0, N.argmin(1), kl_ucb(Q, N, t).argmax(1))
        elif policy == "Thompson":
            a = r.beta(S, F).argmax(1)
        reward = (r.random(R) < P[rows, a]).astype(float)
        N[rows, a] += 1
        alpha = step_size if step_size else 1 / (N[rows, a] + N0)
        Q[rows, a] += alpha * (reward - Q[rows, a])
        S[rows, a] += reward; F[rows, a] += 1 - reward
        regret[t] = np.mean(P.max(1) - P[rows, a])
        optimal[t] = np.mean(a == P.argmax(1))
        if drift:
            P = np.clip(P + r.normal(0, drift, P.shape), 0.0, 0.3)
    out = {"regret": np.cumsum(regret), "optimal": optimal}
    if record_estimates:
        out["Q"], out["N"] = Q, N
    return out


# %% [markdown]
# ## 1. Exploration strategies

# %%
policies = ["greedy", "eps=0.01", "eps=0.1", "optimistic", "UCB1", "KL-UCB", "Thompson"]
res = {p: run(p) for p in policies}
print(f"{R} problems x {T} pulls; arms' rates between 2% and 12%")
print(f"  {'policy':12s} {'regret @1,000':>14s} {'regret @10,000':>15s} {'best arm chosen, last 1,000 pulls':>35s}")
for p in policies:
    print(f"  {p:12s} {res[p]['regret'][999]:14.1f} {res[p]['regret'][-1]:15.1f} {res[p]['optimal'][-1000:].mean():35.2f}")
print("regret: conversions lost compared with always showing the best version. Greedy locks onto whichever arm paid")
print("first. With rates this small and close, 10,000 pulls is not much: a well-tuned epsilon-greedy is competitive with")
print("Thompson here, and UCB1 is poor, because its confidence bound assumes the worst-case variance of a [0, 1] reward")
print("and rates near 5% have far less. KL-UCB uses the right (Bernoulli) confidence bound and fixes most of that.")
assert res["Thompson"]["regret"][-1] < res["eps=0.1"]["regret"][-1] < res["greedy"]["regret"][-1]
assert res["KL-UCB"]["regret"][-1] < res["UCB1"]["regret"][-1]

growth = {p: (res[p]["regret"][-1] - res[p]["regret"][4999]) / (res[p]["regret"][4999] - res[p]["regret"][999])
          for p in ("eps=0.1", "Thompson")}
print(f"regret added in pulls 5,000-10,000 relative to pulls 1,000-5,000: eps=0.1 {growth['eps=0.1']:.2f}, Thompson "
      f"{growth['Thompson']:.2f} (a constant rate gives 1.25, pure log T growth 0.43). Epsilon-greedy")
print("keeps paying for exploration forever; Thompson and UCB-style methods stop when they're sure. Over a long enough")
print("horizon that decides it. Thompson is also the easiest to extend (priors, batches, delayed feedback).")
assert growth["Thompson"] < growth["eps=0.1"]

# %% [markdown]
# ## 2. A changing world

# %%
print("\nrates drift a little every pull (a random walk), 10,000 pulls:")
for name, kw in (("eps=0.1, sample averages", dict(step_size=None)), ("eps=0.1, constant step 0.05", dict(step_size=0.05))):
    out = run("eps=0.1", drift=0.002, **kw)
    print(f"  {name:30s} regret {out['regret'][-1]:6.1f}   best arm chosen, last 1,000 pulls {out['optimal'][-1000:].mean():.2f}")
print("sample averages weight a reward from pull 1 as much as one from pull 10,000; a constant step size forgets")
print("old rewards exponentially and tracks the change. Real conversion rates move (seasons, marketing, novelty).")

# %% [markdown]
# ## 3. A/B test or bandit?
#
# A/B test: split traffic evenly for the first 3,000 pulls, then show the winner. Bandit: Thompson throughout.

# %%
def ab_test(explore=3000, seed=0):
    r = np.random.default_rng(seed)
    rows = np.arange(R)
    N, Wn = np.zeros((R, K)), np.zeros((R, K))
    regret = np.zeros(T)
    for t in range(T):
        a = np.full(R, t % K) if t < explore else (Wn / np.maximum(N, 1)).argmax(1)
        reward = r.random(R) < P[rows, a]
        if t < explore:
            N[rows, a] += 1; Wn[rows, a] += reward
        regret[t] = np.mean(best - P[rows, a])
    return np.cumsum(regret), (Wn / np.maximum(N, 1)), N


ab_regret, ab_est, _ = ab_test()
th = run("Thompson", record_estimates=True)
print(f"\nregret after {T} pulls: A/B test then winner {ab_regret[-1]:.1f}; Thompson {th['regret'][-1]:.1f}")
err_ab = (ab_est - P).mean()
visited = th["N"] > 0
err_th = ((th["Q"] - P)[visited]).mean()
worst = P.argmin(1)
err_th_worst = (th["Q"][np.arange(R), worst] - P[np.arange(R), worst]).mean()
print(f"mean error of the estimated rates (estimate - truth): A/B test {err_ab:+.4f}; Thompson {err_th:+.4f}, "
      f"and {err_th_worst:+.4f} on each problem's worst arm")
print("the bandit loses fewer conversions, and its estimates are biased low: an arm that had a bad start gets pulled")
print("less, so its unlucky early average is never corrected (Nie et al., 2018). If you need unbiased effect sizes")
print("for a decision or a report, run the A/B test (05.3); if you need conversions, run the bandit, and don't")
print("present its averages as measurements. Sometimes the answer is both: test first, then let a bandit run.")
assert th["regret"][-1] < ab_regret[-1] and err_th_worst < 0 and abs(err_ab) < abs(err_th_worst)

print("\nAll checks passed.")
