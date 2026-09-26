# %% [markdown]
# # Lab 05.3: An experimentation toolkit, verified by simulation
#
# 1. Sample size formula vs simulated power.
# 2. SRM detection with a chi-square test.
# 3. CUPED: same answer, half the variance.
# 4. Ratio metrics: naive SE vs delta method vs user-level bootstrap.
# 5. Group sequential testing: O'Brien-Fleming-style boundaries keep alpha at 5% with 5 looks.

# %%
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
z = stats.norm.ppf

# %% [markdown]
# ## 1. Power: formula vs simulation

# %%
def n_per_group(p, delta, alpha=0.05, power=0.8):
    return int(np.ceil(2 * (z(1 - alpha / 2) + z(power)) ** 2 * p * (1 - p) / delta**2))


def simulated_power(p, delta, n, reps=4000):
    a = rng.binomial(n, p, reps) / n
    b = rng.binomial(n, p + delta, reps) / n
    pool = (a + b) / 2
    zstat = (b - a) / np.sqrt(2 * pool * (1 - pool) / n)
    return np.mean(2 * stats.norm.sf(np.abs(zstat)) < 0.05)


n = n_per_group(0.12, 0.01)
print(f"12% baseline, +1pt: n = {n:,} per group, simulated power {simulated_power(0.12, 0.01, n):.3f}")
assert 16_000 < n < 17_500 and abs(simulated_power(0.12, 0.01, n) - 0.8) < 0.03
n_small = n_per_group(0.05, 0.0025)
print(f"5% baseline, +5% relative: n = {n_small:,} per group")
print(f"power of a 2-week test with 20k users/group on that effect: {simulated_power(0.05, 0.0025, 20_000):.2f}")
assert simulated_power(0.05, 0.0025, 20_000) < 0.25

# %% [markdown]
# ## 2. Sample ratio mismatch

# %%
def srm_pvalue(n_a, n_b, expected_ratio=0.5):
    total = n_a + n_b
    return stats.chisquare([n_a, n_b], [total * expected_ratio, total * (1 - expected_ratio)]).pvalue


for na, nb in [(10_120, 9_880), (101_200, 98_800), (50_812, 49_188)]:
    print(f"{na:>7,} vs {nb:>7,}: SRM p-value {srm_pvalue(na, nb):.2e}")
assert srm_pvalue(10_120, 9_880) > 0.05 and srm_pvalue(101_200, 98_800) < 1e-6

# A buggy pipeline: the new variant crashes before logging for 10% of users on one browser (20% of traffic).
# At a million users that small leak (2% of B) is unmistakable. At 20k users it would hide in the noise.
users = 1_000_000
arm = rng.integers(0, 2, users)
browser_x = rng.uniform(size=users) < 0.2
lost = (arm == 1) & browser_x & (rng.uniform(size=users) < 0.10)
logged_arm = arm[~lost]
p_srm = srm_pvalue(np.sum(logged_arm == 0), np.sum(logged_arm == 1))
print(f"2% of B users silently lost -> SRM p-value {p_srm:.1e}")
assert p_srm < 1e-3

# %% [markdown]
# ## 3. CUPED

# %%
n = 20_000
pre = rng.gamma(2.0, 25.0, 2 * n)                        # revenue in the 30 days before the test
treat = np.repeat([0, 1], n)
rev = 0.7 * pre + rng.gamma(2.0, 10.0, 2 * n) + 1.5 * treat    # true effect: +1.5 per user
theta = np.cov(pre, rev)[0, 1] / pre.var(ddof=1)
rev_adj = rev - theta * (pre - pre.mean())


def diff_and_se(y):
    a, b = y[treat == 0], y[treat == 1]
    return b.mean() - a.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


d_raw, se_raw = diff_and_se(rev)
d_cup, se_cup = diff_and_se(rev_adj)
rho = np.corrcoef(pre, rev)[0, 1]
print(f"raw: {d_raw:.2f} +- {1.96 * se_raw:.2f}   CUPED: {d_cup:.2f} +- {1.96 * se_cup:.2f}   (rho={rho:.2f})")
assert np.isclose((se_cup / se_raw) ** 2, 1 - rho**2, rtol=0.05)
assert abs(d_cup - 1.5) < 3 * se_cup

# Unbiasedness across many replications
est_raw, est_cup = [], []
for _ in range(300):
    pre_r = rng.gamma(2.0, 25.0, 2 * 2000)
    t_r = np.repeat([0, 1], 2000)
    y_r = 0.7 * pre_r + rng.gamma(2.0, 10.0, 2 * 2000) + 1.5 * t_r
    th = np.cov(pre_r, y_r)[0, 1] / pre_r.var(ddof=1)
    ya = y_r - th * (pre_r - pre_r.mean())
    est_raw.append(y_r[t_r == 1].mean() - y_r[t_r == 0].mean())
    est_cup.append(ya[t_r == 1].mean() - ya[t_r == 0].mean())
print(f"mean estimate raw {np.mean(est_raw):.2f} (sd {np.std(est_raw):.2f})   CUPED {np.mean(est_cup):.2f} (sd {np.std(est_cup):.2f})")
assert abs(np.mean(est_cup) - 1.5) < 0.15 and np.std(est_cup) < 0.75 * np.std(est_raw)

# %% [markdown]
# ## 4. Ratio metric: revenue per session, randomized by user

# %%
n_users = 5000
sessions = rng.poisson(rng.gamma(2, 3, n_users)) + 1       # heavy users have many sessions
user_value = rng.gamma(2, 2, n_users)                      # per-session spend depends on the user
revenue = np.array([rng.gamma(2, v / 2, s).sum() for v, s in zip(user_value, sessions)])
R = revenue.sum() / sessions.sum()

# naive: treat every session as independent (wrong)
per_session = np.concatenate([rng.gamma(2, v / 2, s) for v, s in zip(user_value, sessions)])
se_naive = per_session.std(ddof=1) / np.sqrt(len(per_session))

# delta method on per-user totals
Xbar, Ybar = sessions.mean(), revenue.mean()
cov = np.cov(sessions, revenue)
var_R = (cov[1, 1] - 2 * R * cov[0, 1] + R**2 * cov[0, 0]) / (n_users * Xbar**2)
se_delta = np.sqrt(var_R)

# user-level bootstrap
idx = rng.integers(0, n_users, (2000, n_users))
boots = revenue[idx].sum(axis=1) / sessions[idx].sum(axis=1)
se_boot = boots.std()
print(f"SE of revenue/session: naive {se_naive:.4f}   delta {se_delta:.4f}   user bootstrap {se_boot:.4f}")
assert np.isclose(se_delta, se_boot, rtol=0.1) and se_naive < 0.8 * se_delta

# %% [markdown]
# ## 5. Group sequential design
#
# 5 equally spaced looks. With a naive 1.96 threshold at every look, alpha inflates. O'Brien-Fleming-style boundaries
# c_k = C * sqrt(K / k) with C calibrated by simulation keep it at 5%.

# %%
K, reps = 5, 200_000
increments = rng.normal(size=(reps, K))
zpath = np.cumsum(increments, axis=1) / np.sqrt(np.arange(1, K + 1))   # z-statistic at each look under H0

naive_fpr = np.mean(np.any(np.abs(zpath) > 1.96, axis=1))
shape = np.sqrt(K / np.arange(1, K + 1))
C = np.quantile(np.max(np.abs(zpath) / shape, axis=1), 0.95)            # calibrate the constant
obf_fpr = np.mean(np.any(np.abs(zpath) > C * shape, axis=1))
print(f"5 looks at 1.96: FPR {naive_fpr:.3f}.  O'Brien-Fleming boundaries {np.round(C * shape, 2)}: FPR {obf_fpr:.3f}")
assert naive_fpr > 0.12 and abs(obf_fpr - 0.05) < 0.003
assert 2.0 < C < 2.1, "the final boundary is only slightly above 1.96: little power lost"

# %%
print("\nAll checks passed.")
