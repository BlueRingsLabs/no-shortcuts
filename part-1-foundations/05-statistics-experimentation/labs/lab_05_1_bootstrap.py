# %% [markdown]
# # Lab 05.1: Error bars on everything
#
# 1. Wald vs Wilson intervals for proportions, and their actual coverage for rare events.
# 2. The bootstrap from scratch vs scipy.stats.bootstrap, for a median.
# 3. Bootstrap CIs for ML metrics, and the paired bootstrap for comparing two models.
# 4. Clustered data: row bootstrap vs cluster bootstrap coverage.
# 5. Bias vs variance of estimators: a shrunk estimator can beat the unbiased one on MSE.

# %%
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Proportions: Wald vs Wilson

# %%
lo, hi = proportion_confint(45, 50, method="normal")
wlo, whi = proportion_confint(45, 50, method="wilson")
print(f"45/50 -> Wald [{lo:.3f}, {hi:.3f}]   Wilson [{wlo:.3f}, {whi:.3f}]")


def coverage(p, n, method, reps=20000):
    k = rng.binomial(n, p, reps)
    lo, hi = proportion_confint(k, n, method=method)
    return np.mean((lo <= p) & (p <= hi))


for p, n in [(0.5, 100), (0.02, 100), (0.005, 400)]:
    cw, cwi = coverage(p, n, "normal"), coverage(p, n, "wilson")
    print(f"p={p:<6} n={n:<4}: coverage Wald {cw:.3f}   Wilson {cwi:.3f}   (target 0.95)")
assert coverage(0.02, 100, "normal") < 0.9, "Wald breaks for rare events"
assert coverage(0.02, 100, "wilson") > 0.9

# %% [markdown]
# ## 2. Bootstrap from scratch

# %%
def bootstrap(data, stat, B=5000, rng=rng):
    idx = rng.integers(0, len(data), size=(B, len(data)))
    return np.array([stat(data[i]) for i in idx])


latency_ms = rng.lognormal(mean=3.5, sigma=0.8, size=300)       # skewed, like real latencies
boots = bootstrap(latency_ms, np.median)
ci_mine = np.quantile(boots, [0.025, 0.975])
ci_scipy = stats.bootstrap((latency_ms,), np.median, n_resamples=5000, method="percentile",
                           random_state=1).confidence_interval
print(f"median latency {np.median(latency_ms):.1f} ms, 95% CI mine [{ci_mine[0]:.1f}, {ci_mine[1]:.1f}]  "
      f"scipy [{ci_scipy.low:.1f}, {ci_scipy.high:.1f}]")
assert abs(ci_mine[0] - ci_scipy.low) < 2.5 and abs(ci_mine[1] - ci_scipy.high) < 2.5

# Fraction of distinct points in a bootstrap sample -> 1 - 1/e
frac = np.mean([len(np.unique(rng.integers(0, 10_000, 10_000))) / 10_000 for _ in range(50)])
assert abs(frac - (1 - 1 / np.e)) < 0.005

# Coverage check: does the percentile bootstrap CI for the median cover the true median ~95% of the time?
true_median = np.exp(3.5)
hits = 0
for _ in range(300):
    sample = rng.lognormal(3.5, 0.8, 300)
    b = bootstrap(sample, np.median, B=1000)
    lo, hi = np.quantile(b, [0.025, 0.975])
    hits += lo <= true_median <= hi
print(f"bootstrap CI coverage for the median: {hits / 300:.3f}")
assert 0.9 < hits / 300 < 0.99

# %% [markdown]
# ## 3. Metrics: individual CIs vs the paired difference

# %%
n = 1500
y = rng.binomial(1, 0.3, n)
difficulty = rng.normal(0, 1.5, n)                       # shared: hard examples are hard for both models
score_a = y * 1.6 + difficulty + rng.normal(0, 1.0, n)
score_b = y * 1.75 + difficulty + rng.normal(0, 1.0, n)  # slightly better model

auc_a, auc_b = roc_auc_score(y, score_a), roc_auc_score(y, score_b)
B = 2000
ia, ib, diff = [], [], []
for _ in range(B):
    i = rng.integers(0, n, n)
    a, b = roc_auc_score(y[i], score_a[i]), roc_auc_score(y[i], score_b[i])
    ia.append(a); ib.append(b); diff.append(b - a)
ci_a, ci_b, ci_d = (np.quantile(v, [0.025, 0.975]) for v in (ia, ib, diff))
print(f"AUC A {auc_a:.3f} {np.round(ci_a, 3)}   AUC B {auc_b:.3f} {np.round(ci_b, 3)}")
print(f"paired difference B - A: {auc_b - auc_a:.4f}  95% CI {np.round(ci_d, 4)}")
assert ci_a[1] > ci_b[0], "individual intervals overlap..."
assert ci_d[0] > 0, "...but the paired difference is clearly positive"

# %% [markdown]
# ## 4. Clustered data: resample the independent unit
#
# 100 users x 40 predictions each. Each user has their own accuracy (some users are just harder).

# %%
def make_clustered(rng, users=100, per_user=40):
    user_acc = np.clip(rng.normal(0.8, 0.12, users), 0.01, 0.99)
    correct = rng.binomial(1, np.repeat(user_acc, per_user))
    return correct, np.repeat(np.arange(users), per_user)


def row_ci(correct, B=400):
    b = bootstrap(correct.astype(float), np.mean, B=B)
    return np.quantile(b, [0.025, 0.975])


def cluster_ci(correct, user, B=400):
    users = np.unique(user)
    per_user = np.array([correct[user == u] for u in users])    # (users, per_user)
    vals = [per_user[rng.integers(0, len(users), len(users))].mean() for _ in range(B)]
    return np.quantile(vals, [0.025, 0.975])


true_acc = 0.8
row_hits = cl_hits = 0
reps = 150
for _ in range(reps):
    correct, user = make_clustered(rng)
    lo, hi = row_ci(correct); row_hits += lo <= true_acc <= hi
    lo, hi = cluster_ci(correct, user); cl_hits += lo <= true_acc <= hi
print(f"coverage with clustered data: row bootstrap {row_hits / reps:.2f}   cluster bootstrap {cl_hits / reps:.2f}")
assert row_hits / reps < 0.8 and cl_hits / reps > 0.88

# %% [markdown]
# ## 5. Bias-variance for estimators
#
# Estimate a mean from n=5 noisy samples. The shrunk estimator c * x_bar is biased, yet has lower MSE when the true
# mean is small relative to the noise.

# %%
theta, sigma, n = 0.5, 2.0, 5
xbar = rng.normal(theta, sigma, size=(200_000, n)).mean(axis=1)
c = theta**2 / (theta**2 + sigma**2 / n)                # the MSE-optimal shrinkage (oracle, for illustration)
mse_unbiased = np.mean((xbar - theta) ** 2)
mse_shrunk = np.mean((c * xbar - theta) ** 2)
print(f"MSE unbiased {mse_unbiased:.4f}   MSE shrunk (c={c:.2f}) {mse_shrunk:.4f}")
assert mse_shrunk < 0.5 * mse_unbiased
bias, var = np.mean(c * xbar) - theta, np.var(c * xbar)
assert np.isclose(mse_shrunk, bias**2 + var, rtol=1e-3)

# %%
print("\nAll checks passed.")
