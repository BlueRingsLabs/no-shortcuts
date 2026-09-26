# %% [markdown]
# # Lab 06.2: Mutual information, estimated honestly
#
# 1. Discrete MI from counts, and the identities between entropies.
# 2. MI sees what correlation misses (X vs X^2, sin); the Gaussian closed form.
# 3. The upward bias of binned estimates, and the shuffled-target baseline.
# 4. XOR: univariate MI says "useless", jointly the features determine the label.
# 5. Information gain of a decision-tree split = MI between the split and the label.
# 6. MDL / BIC chooses the right polynomial degree where training error doesn't.

# %%
import numpy as np
from sklearn.feature_selection import mutual_info_regression

rng = np.random.default_rng(0)


def entropy_from_counts(c):
    p = c[c > 0] / c.sum()
    return -np.sum(p * np.log2(p))


def mi_discrete(x, y):
    xs, xi = np.unique(x, return_inverse=True)
    ys, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(xs), len(ys)))
    np.add.at(joint, (xi, yi), 1)
    return entropy_from_counts(joint.sum(1)) + entropy_from_counts(joint.sum(0)) - entropy_from_counts(joint.ravel())

# %% [markdown]
# ## 1. X uniform on {0,1,2,3}, Y = X mod 2

# %%
x = rng.integers(0, 4, 200_000)
y = x % 2
mi = mi_discrete(x, y)
print(f"I(X; X mod 2) = {mi:.4f} bits (theory 1)")
assert abs(mi - 1) < 1e-3
assert abs(mi_discrete(x, x) - 2) < 1e-3, "I(X;X) = H(X)"

# %% [markdown]
# ## 2. Nonlinear dependence

# %%
n = 5000
xc = rng.uniform(-2, 2, n)
cases = {
    "y = x + noise": xc + rng.normal(0, 0.5, n),
    "y = x^2 + noise": xc**2 + rng.normal(0, 0.5, n),
    "y = sin(3x) + noise": np.sin(3 * xc) + rng.normal(0, 0.3, n),
    "independent": rng.uniform(-2, 2, n),
}
mis = {}
for name, yc in cases.items():
    mis[name] = mutual_info_regression(xc[:, None], yc, random_state=0)[0]
    print(f"{name:22s} corr {np.corrcoef(xc, yc)[0, 1]:+.3f}   MI {mis[name]:.3f} nats")
assert abs(np.corrcoef(xc, cases["y = x^2 + noise"])[0, 1]) < 0.05 and mis["y = x^2 + noise"] > 0.5
assert mis["independent"] < 0.02

rho = 0.9
g = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 20_000)
est = mutual_info_regression(g[:, :1], g[:, 1], random_state=0)[0]
print(f"Gaussian rho=0.9: kNN estimate {est:.3f}, closed form {-0.5 * np.log(1 - rho**2):.3f} nats")
assert abs(est + 0.5 * np.log(1 - rho**2)) < 0.05

# %% [markdown]
# ## 3. Binned estimates are biased upward

# %%
def mi_binned(a, b, bins):
    qa = np.digitize(a, np.quantile(a, np.linspace(0, 1, bins + 1)[1:-1]))
    qb = np.digitize(b, np.quantile(b, np.linspace(0, 1, bins + 1)[1:-1]))
    return mi_discrete(qa, qb) * np.log(2)          # bits -> nats


a, b = rng.normal(size=300), rng.normal(size=300)   # independent: true MI = 0
for bins in (3, 10, 30):
    print(f"independent data, n=300, {bins:2d} bins: estimated MI {mi_binned(a, b, bins):.3f} nats")
assert mi_binned(a, b, 30) > 0.5 > 0.05 > mi_binned(a, b, 3) * 0.5

shuffled = [mi_binned(a, rng.permutation(b), 10) for _ in range(200)]
print(f"shuffled baseline at 10 bins: {np.mean(shuffled):.3f} +- {np.std(shuffled):.3f}  <- this is what 'zero' looks like")

# %% [markdown]
# ## 4. XOR

# %%
x1, x2 = rng.integers(0, 2, 100_000), rng.integers(0, 2, 100_000)
z = x1 ^ x2
print(f"I(X1;Z)={mi_discrete(x1, z):.4f}  I(X2;Z)={mi_discrete(x2, z):.4f}  I(X1,X2;Z)={mi_discrete(2 * x1 + x2, z):.4f} bits")
assert mi_discrete(x1, z) < 1e-3 and mi_discrete(2 * x1 + x2, z) > 0.999

# %% [markdown]
# ## 5. Information gain of a split

# %%
income = rng.lognormal(10, 0.6, 20_000)
default = rng.uniform(size=20_000) < np.where(income < 20_000, 0.25, 0.05)
split = income < 20_000
h_before = entropy_from_counts(np.bincount(default))
h_after = sum(np.mean(split == s) * entropy_from_counts(np.bincount(default[split == s], minlength=2)) for s in (0, 1))
gain = h_before - h_after
assert np.isclose(gain, mi_discrete(split, default))
print(f"information gain of 'income < 20k': {gain:.4f} bits = I(split; default)")

# %% [markdown]
# ## 6. MDL / BIC for polynomial degree
#
# Noisy points from a line. Training error always prefers the highest degree. BIC = n log(RSS/n) + k log n usually
# picks the truth, but "usually" is the honest word: on any single small dataset it can be fooled, so we look at how
# often it picks each degree over 300 datasets.

# %%
def bic_pick(n, rng, max_deg=10):
    xs = np.linspace(-1, 1, n)
    ys = 1.0 + 2.0 * xs + rng.normal(0, 0.3, n)
    bics, rsss = [], []
    for deg in range(max_deg + 1):
        V = np.vander(xs, deg + 1, increasing=True)
        w, *_ = np.linalg.lstsq(V, ys, rcond=None)
        rss = np.sum((V @ w - ys) ** 2)
        rsss.append(rss)
        bics.append(n * np.log(rss / n) + (deg + 1) * np.log(n))
    return int(np.argmin(bics)), int(np.argmin(rsss))


for n in (20, 60):
    picks = np.array([bic_pick(n, rng) for _ in range(300)])
    print(f"n={n}: BIC picks per degree {np.bincount(picks[:, 0], minlength=11)}   "
          f"min training RSS always picks degree {np.unique(picks[:, 1])}")
    if n == 60:
        assert np.mean(picks[:, 0] == 1) > 0.9 and np.all(picks[:, 1] == 10)

# %%
print("\nAll checks passed.")
