# %% [markdown]
# # Lab 04.3: When averaging works, and when it doesn't
#
# 1. Linearity of expectation on a dependent problem (fixed points of random permutations).
# 2. Zero correlation, full dependence.
# 3. LLN: standard error ~ sigma / sqrt(n), measured.
# 4. CLT for a skewed distribution: how many samples until the mean looks Gaussian?
# 5. Cauchy: the average never settles.
# 6. Correlated samples: the effective sample size.
# 7. Hoeffding vs CLT sample sizes for estimating an accuracy.

# %%
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Linearity of expectation doesn't care about dependence

# %%
for n in (3, 10, 100):
    perms = np.array([rng.permutation(n) for _ in range(20000)])
    fixed = (perms == np.arange(n)).sum(axis=1)
    print(f"n={n:3d}: average number of fixed points {fixed.mean():.3f} (theory: 1)")
    assert abs(fixed.mean() - 1) < 0.03

# %% [markdown]
# ## 2. Uncorrelated but dependent

# %%
x = rng.uniform(-1, 1, 1_000_000)
y = x**2
print(f"corr(X, X^2) = {np.corrcoef(x, y)[0, 1]:.4f}  (but Y is a function of X)")
assert abs(np.corrcoef(x, y)[0, 1]) < 0.005
# dependence shows up as soon as you condition
assert y[np.abs(x) > 0.9].mean() > 0.8 > 0.01 > y[np.abs(x) < 0.1].mean()

# %% [markdown]
# ## 3. Standard error ~ 1/sqrt(n)

# %%
sigma = 2.0
ses = {}
for n in (100, 400, 1600, 6400):
    means = rng.normal(5, sigma, size=(5000, n)).mean(axis=1)
    ses[n] = means.std()
    print(f"n={n:5d}: sd of the sample mean {ses[n]:.4f}   theory {sigma / np.sqrt(n):.4f}")
    assert np.isclose(ses[n], sigma / np.sqrt(n), rtol=0.05)
assert np.isclose(ses[100] / ses[400], 2, rtol=0.1), "4x the data, half the error"

# Monte Carlo estimate of pi, with its error shrinking like 1/sqrt(n)
for n in (10**3, 10**5, 10**7):
    pts = rng.uniform(size=(n, 2))
    est = 4 * np.mean(np.sum(pts**2, axis=1) < 1)
    print(f"pi with n={n:>9,d}: {est:.5f}  (error {abs(est - np.pi):.1e}, expected ~{4 * np.sqrt(np.pi / 4 * (1 - np.pi / 4) / n):.1e})")

# %% [markdown]
# ## 4. CLT on a skewed distribution (log-normal "revenue per user")

# %%
def skew_of_means(n, reps=20000):
    samples = rng.lognormal(0, 1.2, size=(reps, n))
    return stats.skew(samples.mean(axis=1))


skews = {n: skew_of_means(n) for n in (1, 10, 100, 1000)}
for n, sk in skews.items():
    print(f"n={n:5d}: skewness of the sample mean {sk:6.2f}   (0 for a Gaussian)")
assert skews[1000] < 0.6 and skews[10] > 1.5, "skewness decays like 1/sqrt(n): slowly, for skewed data"

# %% [markdown]
# ## 5. Cauchy: no mean, no LLN

# %%
spread = {}
for n in (1, 100, 10_000):
    means = rng.standard_cauchy(size=(2000, n)).mean(axis=1)
    iqr = np.subtract(*np.percentile(means, [75, 25]))
    spread[n] = iqr
    print(f"Cauchy, n={n:6d}: interquartile range of the sample mean {iqr:.3f}")
assert 0.7 < spread[10_000] / spread[1] < 1.4, "averaging 10,000 Cauchy samples is no better than one"

# %% [markdown]
# ## 6. Correlated samples: you have less data than you think
#
# An AR(1) series x_t = phi * x_{t-1} + noise. For phi = 0.9 the effective sample size is n (1-phi)/(1+phi) ~ n/19.

# %%
def ar1(n, phi, rng):
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


phi, n = 0.9, 2000
means = np.array([ar1(n, phi, rng).mean() for _ in range(1000)])
marginal_sd = 1 / np.sqrt(1 - phi**2)
naive_se = marginal_sd / np.sqrt(n)
n_eff = n * (1 - phi) / (1 + phi)
print(f"actual sd of the mean {means.std():.4f}   naive iid SE {naive_se:.4f}   SE with n_eff={n_eff:.0f}: {marginal_sd / np.sqrt(n_eff):.4f}")
assert means.std() > 3.5 * naive_se
assert np.isclose(means.std(), marginal_sd / np.sqrt(n_eff), rtol=0.15)

# %% [markdown]
# ## 7. How many test examples to trust an accuracy?

# %%
n_clt = int(np.ceil(stats.norm.ppf(0.995) ** 2 * 0.9 * 0.1 / 0.01**2))
n_hoeffding = int(np.ceil(np.log(2 / 0.01) / (2 * 0.01**2)))
print(f"+-1 point with 99% confidence at 90% accuracy: CLT n~{n_clt:,}, Hoeffding n~{n_hoeffding:,}")
acc = rng.binomial(n_clt, 0.9, size=100_000) / n_clt
coverage = np.mean(np.abs(acc - 0.9) <= 0.01)
print(f"simulated coverage with the CLT sample size: {coverage:.3f}")
assert coverage > 0.985 and n_hoeffding > n_clt

# Jensen: average of sigmoids vs sigmoid of the average for positive logits
sig = lambda z: 1 / (1 + np.exp(-z))
logits = rng.uniform(0.5, 4, size=(10000, 5))
assert np.all(sig(logits.mean(1)) >= sig(logits).mean(1) - 1e-12)

# %%
print("\nAll checks passed.")
