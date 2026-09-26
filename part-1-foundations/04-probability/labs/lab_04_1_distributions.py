# %% [markdown]
# # Lab 04.1: Distributions, sampled and checked
#
# 1. Inverse-CDF sampling for the exponential and for a categorical (how an LLM picks a token).
# 2. Moments of the table's distributions, checked against large samples.
# 3. The multivariate Gaussian: Cholesky sampling, Mahalanobis distance, linear maps.
# 4. Heavy tails: how often do "6-sigma" events happen in log-normal data?
# 5. Change of variables: the log-normal density, checked with a histogram.

# %%
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
N = 400_000

# %% [markdown]
# ## 1. Inverse-CDF sampling

# %%
lam = 2.0
u = rng.uniform(size=N)
x_exp = -np.log(1 - u) / lam
assert abs(x_exp.mean() - 1 / lam) < 0.01 and abs(x_exp.var() - 1 / lam**2) < 0.01
ks = stats.kstest(x_exp, stats.expon(scale=1 / lam).cdf)
print(f"exponential via inverse CDF: KS p-value {ks.pvalue:.2f}")
assert ks.pvalue > 0.001


def sample_categorical(probs, size, rng):
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0                               # guard against rounding: last bucket must catch everything
    return np.searchsorted(cdf, rng.uniform(size=size), side="right")


probs = np.array([0.5, 0.2, 0.2, 0.1])          # think: softmax over four tokens
draws = sample_categorical(probs, N, rng)
freq = np.bincount(draws, minlength=4) / N
print("categorical frequencies:", np.round(freq, 4))
assert np.allclose(freq, probs, atol=0.005)

# %% [markdown]
# ## 2. Moments

# %%
checks = {
    "Bernoulli(0.3)": (rng.binomial(1, 0.3, N), 0.3, 0.21),
    "Binomial(20,0.3)": (rng.binomial(20, 0.3, N), 6.0, 4.2),
    "Poisson(3)": (rng.poisson(3, N), 3.0, 3.0),
    "Geometric(0.25)": (rng.geometric(0.25, N), 4.0, 12.0),
    "Uniform(2,5)": (rng.uniform(2, 5, N), 3.5, 0.75),
    "Beta(2,5)": (rng.beta(2, 5, N), 2 / 7, 2 * 5 / (49 * 8)),
    "Laplace(0,1)": (rng.laplace(0, 1, N), 0.0, 2.0),
}
for name, (s, m, v) in checks.items():
    print(f"{name:18s} mean {s.mean():7.4f} (theory {m:7.4f})   var {s.var():7.4f} (theory {v:7.4f})")
    assert abs(s.mean() - m) < 0.02 * max(1, abs(m)) + 0.01 and abs(s.var() - v) < 0.03 * max(1, v)

# Poisson's probabilities from the exercises
assert np.isclose(stats.poisson(3).pmf(0), np.exp(-3))
assert np.isclose(stats.poisson(3).sf(4), 0.1847, atol=1e-4)

# Memorylessness of the exponential
s0, t0 = 0.5, 0.3
survive = x_exp[x_exp > s0]
assert abs(np.mean(survive > s0 + t0) - np.mean(x_exp > t0)) < 0.01

# %% [markdown]
# ## 3. Multivariate Gaussian

# %%
mu = np.array([1.0, -2.0, 0.5])
A = rng.normal(size=(3, 3))
Sigma = A @ A.T + 0.5 * np.eye(3)
L = np.linalg.cholesky(Sigma)
Z = rng.standard_normal((N, 3))
X = mu + Z @ L.T                                  # rows are samples: x = mu + L z
assert np.allclose(X.mean(0), mu, atol=0.02)
assert np.allclose(np.cov(X, rowvar=False), Sigma, rtol=0.03, atol=0.02)

# Density matches scipy, and it is NOT bounded by 1
mvn = stats.multivariate_normal(mu, Sigma)
x0 = X[0]
d2 = (x0 - mu) @ np.linalg.solve(Sigma, x0 - mu)          # squared Mahalanobis distance
dens = np.exp(-0.5 * d2) / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(Sigma))
assert np.isclose(dens, mvn.pdf(x0))
narrow = stats.multivariate_normal(np.zeros(2), 0.001 * np.eye(2))
print(f"density at the mode of a narrow 2-D Gaussian: {narrow.pdf([0, 0]):.1f}  (not a probability!)")
assert narrow.pdf([0, 0]) > 100

# Squared Mahalanobis distances of Gaussian samples follow a chi-square with d degrees of freedom
d2_all = np.einsum("ij,ij->i", X - mu, np.linalg.solve(Sigma, (X - mu).T).T)
assert abs(d2_all.mean() - 3) < 0.03

# Linear map: A x + b is N(A mu + b, A Sigma A^T)
M, b = rng.normal(size=(2, 3)), np.array([3.0, -1.0])
Y = X @ M.T + b
assert np.allclose(Y.mean(0), M @ mu + b, atol=0.05)
assert np.allclose(np.cov(Y, rowvar=False), M @ Sigma @ M.T, rtol=0.03, atol=0.05)
print("multivariate Gaussian: sampling, density, Mahalanobis, linear maps ok")

# %% [markdown]
# ## 4. Tails: Gaussian vs log-normal
#
# Fraction of samples more than 6 standard deviations above the mean.

# %%
gauss = rng.normal(size=2_000_000)
logn = rng.lognormal(mean=0.0, sigma=1.5, size=2_000_000)
frac = lambda s: np.mean(s > s.mean() + 6 * s.std())
print(f"P(> mean + 6 sd): Gaussian {frac(gauss):.2e}   log-normal {frac(logn):.2e}")
assert frac(gauss) < 1e-6 and frac(logn) > 1e-3
print(f"log-normal: mean {logn.mean():.2f} vs median {np.median(logn):.2f}  <- report medians for skewed data")

# %% [markdown]
# ## 5. Change of variables: Y = exp(X)

# %%
m, s = 0.3, 0.6
y = np.exp(rng.normal(m, s, N))
hist, edges = np.histogram(y, bins=200, range=(0.05, 6), density=False)
centers = (edges[:-1] + edges[1:]) / 2
width = edges[1] - edges[0]
empirical = hist / (N * width)
formula = np.exp(-((np.log(centers) - m) ** 2) / (2 * s**2)) / (centers * np.sqrt(2 * np.pi * s**2))
assert np.max(np.abs(empirical - formula)) < 0.03
assert np.isclose(np.median(y), np.exp(m), rtol=0.01) and np.isclose(y.mean(), np.exp(m + s**2 / 2), rtol=0.01)
print("log-normal density from the change-of-variables formula matches the histogram")

# %%
print("\nAll checks passed.")
