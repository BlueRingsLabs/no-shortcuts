# %% [markdown]
# # Lab 16.2: Bias and variance, measured
#
# We know the true function, so we can draw hundreds of independent training sets, fit a model on each, and compute
# the three terms of the decomposition directly.
#
# 1. Polynomial regression: bias^2, variance and noise vs degree; the U-shaped total.
# 2. k-NN regression: the same trade-off, controlled by k.
# 3. The decomposition adds up (checked against the directly measured test error).
# 4. Learning curves for a high-bias and a high-variance model, saved as a figure.

# %%
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402

rng = np.random.default_rng(16)
f = lambda x: np.sin(2 * x) + 0.3 * x          # the truth
SIGMA = 0.4                                     # noise sd
x_test = np.linspace(-3, 3, 200)
N_TRAIN, N_SETS = 40, 400
OUT = Path(__file__).with_name("outputs")
OUT.mkdir(exist_ok=True)


def draw(n):
    x = rng.uniform(-3, 3, n)
    return x, f(x) + rng.normal(0, SIGMA, n)


def decompose(fit_predict):
    preds = np.empty((N_SETS, len(x_test)))
    for s in range(N_SETS):
        x, y = draw(N_TRAIN)
        preds[s] = fit_predict(x, y, x_test)
    mean_pred = preds.mean(axis=0)
    bias2 = np.mean((f(x_test) - mean_pred) ** 2)
    var = np.mean(preds.var(axis=0))
    # directly measured expected test error: fresh noisy y at the test points, averaged over training sets
    y_test = f(x_test) + rng.normal(0, SIGMA, (N_SETS, len(x_test)))
    direct = np.mean((y_test - preds) ** 2)
    return bias2, var, direct

# %% [markdown]
# ## 1. Polynomials

# %%
poly = lambda deg: (lambda x, y, xt: np.polyval(np.polyfit(x, y, deg), xt))
rows = []
for deg in (1, 2, 3, 5, 7, 9, 12):
    b2, v, direct = decompose(poly(deg))
    rows.append((deg, b2, v, SIGMA**2 + b2 + v, direct))
    print(f"degree {deg:2d}: bias^2 {b2:.3f}  variance {v:.3f}  noise {SIGMA**2:.3f}  -> total {SIGMA**2 + b2 + v:.3f} (measured {direct:.3f})")
rows = np.array(rows)
assert rows[0, 1] > rows[3, 1], "bias falls with capacity"
assert rows[-1, 2] > rows[2, 2], "variance rises with capacity"
best = int(rows[np.argmin(rows[:, 3]), 0])
print(f"sweet spot: degree {best}")
assert 3 <= best <= 9

# %% [markdown]
# ## 2. k-NN

# %%
def knn(k):
    return lambda x, y, xt: KNeighborsRegressor(n_neighbors=k).fit(x[:, None], y).predict(xt[:, None])


krows = []
for k in (1, 3, 7, 15, 30, 40):
    b2, v, direct = decompose(knn(k))
    krows.append((k, b2, v, direct))
    print(f"k = {k:2d}: bias^2 {b2:.3f}  variance {v:.3f}  measured error {direct:.3f}")
krows = np.array(krows)
assert krows[0, 2] > krows[-1, 2] and krows[-1, 1] > krows[0, 1], "small k: variance; large k: bias"

# %% [markdown]
# ## 3. The decomposition adds up

# %%
assert np.allclose(rows[:, 3], rows[:, 4], rtol=0.08), "noise + bias^2 + variance = measured expected test error"
print("decomposition matches the directly measured error (within Monte Carlo noise)")

# %% [markdown]
# ## 4. Learning curves

# %%
def learning_curve(deg, sizes, reps=150):
    tr, va = [], []
    for n in sizes:
        e_tr, e_va = [], []
        for _ in range(reps):
            x, y = draw(n)
            c = np.polyfit(x, y, deg)
            e_tr.append(np.mean((np.polyval(c, x) - y) ** 2))
            xv, yv = draw(500)
            e_va.append(np.mean((np.polyval(c, xv) - yv) ** 2))
        tr.append(np.mean(e_tr)); va.append(np.median(e_va))
    return np.array(tr), np.array(va)


sizes = np.array([15, 25, 40, 70, 120, 250, 500])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, constrained_layout=True)
curves = {}
for ax, (deg, label) in zip(axes, [(1, "degree 1: high bias"), (10, "degree 10: high variance")]):
    tr, va = learning_curve(deg, sizes)
    curves[deg] = (tr, va)
    ax.plot(sizes, tr, "o-", label="training error")
    ax.plot(sizes, va, "o-", label="validation error")
    ax.axhline(SIGMA**2, color="0.5", ls="--", label="noise floor")
    ax.set_xscale("log"); ax.set_xlabel("training set size"); ax.set_title(label)
axes[0].set_ylabel("MSE"); axes[0].set_ylim(0, 1.2); axes[1].legend()
fig.savefig(OUT / "learning_curves.png", dpi=120); plt.close(fig)

tr1, va1 = curves[1]
tr10, va10 = curves[10]
print(f"degree 1 at n=500: train {tr1[-1]:.3f}, val {va1[-1]:.3f} (both far above the noise floor {SIGMA**2:.2f}: bias)")
print(f"degree 10: gap at n=25 {va10[1] - tr10[1]:.2f}, at n=500 {va10[-1] - tr10[-1]:.3f} (more data fixed the variance)")
assert va1[-1] > SIGMA**2 + 0.2 and abs(va1[-1] - tr1[-1]) < 0.05
assert (va10[1] - tr10[1]) > 10 * (va10[-1] - tr10[-1])
print("figure saved to", OUT / "learning_curves.png")
print("\nAll checks passed.")
