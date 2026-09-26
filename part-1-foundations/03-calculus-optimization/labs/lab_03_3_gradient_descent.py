# %% [markdown]
# # Lab 03.3: Gradient descent, the good, the slow and the divergent
#
# 1. The stability threshold eta < 2/L, measured.
# 2. Steps to converge grow linearly with the condition number.
# 3. Feature standardization fixes the conditioning of least squares.
# 4. Backtracking line search (Armijo).
# 5. SGD vs full-batch GD on a large least squares problem, with and without step size decay.

# %%
import numpy as np

rng = np.random.default_rng(0)


def gd(grad, w0, eta, steps):
    w = w0.copy()
    for _ in range(steps):
        w = w - eta * grad(w)
    return w

# %% [markdown]
# ## 1. The 2/L threshold

# %%
H = np.diag([10.0, 1.0])
f = lambda w: 0.5 * w @ H @ w
grad = lambda w: H @ w
L = 10.0
for eta in (0.05, 0.1, 0.19, 0.21):
    w = gd(grad, np.array([1.0, 1.0]), eta, 200)
    print(f"eta = {eta:.2f} ({eta * L:.1f}/L): f after 200 steps = {f(w):.3e}")
assert f(gd(grad, np.array([1.0, 1.0]), 0.19, 200)) < 1e-8
assert f(gd(grad, np.array([1.0, 1.0]), 0.21, 200)) > 1e6, "above 2/L it must diverge"

# %% [markdown]
# ## 2. Condition number vs steps

# %%
def steps_to_tol(kappa, tol=1e-6):
    H = np.diag([kappa, 1.0])
    w = np.array([1.0, 1.0])
    f0 = 0.5 * w @ H @ w
    for t in range(1, 10_000_000):
        w = w - (1.0 / kappa) * (H @ w)          # eta = 1/L
        if 0.5 * w @ H @ w < tol * f0:
            return t


results = {k: steps_to_tol(k) for k in (1, 10, 100, 1000)}
for k, t in results.items():
    print(f"kappa = {k:5d}: {t:6d} steps   (theory ~ kappa * ln(1/tol) / 2 = {k * np.log(1e6) / 2:8.0f})")
assert results[1000] > 50 * results[10], "steps grow roughly linearly with kappa"

# %% [markdown]
# ## 3. Standardizing features fixes conditioning
#
# Least squares on features with wildly different scales (like income in dollars next to age in years).

# %%
n = 2000
age = rng.uniform(18, 80, n)
income = rng.lognormal(10.5, 0.5, n)
tenure = rng.uniform(0, 30, n)
X = np.column_stack([np.ones(n), age, income, tenure])
y = 3 + 0.2 * age + 0.0001 * income - 0.5 * tenure + rng.normal(0, 1, n)

Xs = X.copy()
Xs[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
for name, M in (("raw", X), ("standardized", Xs)):
    print(f"{name:13s} cond(X^T X) = {np.linalg.cond(M.T @ M):.1e}")


def run_ls_gd(M, steps):
    Hm = M.T @ M / len(M)
    eta = 1.0 / np.linalg.eigvalsh(Hm)[-1]
    w = np.zeros(M.shape[1])
    for _ in range(steps):
        w -= eta * (M.T @ (M @ w - y) / len(M))
    return np.mean((M @ w - y) ** 2)


w_star = np.linalg.lstsq(Xs, y, rcond=None)[0]
best = np.mean((Xs @ w_star - y) ** 2)
mse_raw, mse_std = run_ls_gd(X, 500), run_ls_gd(Xs, 500)
print(f"MSE after 500 GD steps: raw {mse_raw:.3f}   standardized {mse_std:.4f}   optimum {best:.4f}")
assert np.isclose(mse_std, best, rtol=1e-6) and mse_raw > 1.5 * best

# %% [markdown]
# ## 4. Backtracking line search

# %%
def gd_backtracking(f, grad, w0, steps=100, eta0=1.0, beta=0.5, c=1e-4):
    w, evals = w0.copy(), 0
    for _ in range(steps):
        g = grad(w)
        eta = eta0
        while f(w - eta * g) > f(w) - c * eta * (g @ g):   # Armijo sufficient decrease
            eta *= beta
            evals += 1
        w = w - eta * g
    return w, evals


H3 = np.diag([50.0, 5.0, 0.5])
w_bt, evals = gd_backtracking(lambda w: 0.5 * w @ H3 @ w, lambda w: H3 @ w, np.ones(3), steps=300)
f0 = 0.5 * np.ones(3) @ H3 @ np.ones(3)
print(f"backtracking: f went from {f0:.1f} to {0.5 * w_bt @ H3 @ w_bt:.2e} in 300 steps "
      f"({evals} extra function evals), no eta tuned by hand")
# It found a stable step size on its own, but it can't beat the condition number (100 here): still linear, still slow.
assert 0.5 * w_bt @ H3 @ w_bt < 1e-6 * f0

# %% [markdown]
# ## 5. SGD vs full-batch GD
#
# 100k samples. We count "gradient evaluations per example" (epochs) as the cost unit.

# %%
n, d = 100_000, 20
rng = np.random.default_rng(0)                   # fresh generator: this section doesn't depend on the ones above
Xb = rng.normal(size=(n, d)) * np.linspace(0.3, 3.0, d)     # condition number ~100, like real features
w_true = rng.normal(size=d)
yb = Xb @ w_true + rng.normal(0, 0.5, n)
w_opt = np.linalg.lstsq(Xb, yb, rcond=None)[0]
excess = lambda w: np.mean((Xb @ w - yb) ** 2) - np.mean((Xb @ w_opt - yb) ** 2)

# Full batch: 3 epochs = 3 steps
w = np.zeros(d)
eta_full = 1.0 / np.linalg.eigvalsh(Xb.T @ Xb / n)[-1]
for _ in range(3):
    w -= eta_full * Xb.T @ (Xb @ w - yb) / n
full_excess = excess(w)


def sgd(epochs, batch, eta0, decay, seed=0):
    r = np.random.default_rng(seed)                  # same shuffles for both runs: a fair comparison
    w = np.zeros(d)
    t = 0
    for _ in range(epochs):
        perm = r.permutation(n)                      # shuffle every epoch
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            g = Xb[idx].T @ (Xb[idx] @ w - yb[idx]) / len(idx)
            eta = eta0 / (1 + t / 200) if decay else eta0
            w -= eta * g
            t += 1
    return w


sgd_const = excess(sgd(3, 32, 0.02, decay=False))
sgd_decay = excess(sgd(3, 32, 0.02, decay=True))
print(f"excess MSE after 3 epochs: full-batch GD {full_excess:.2e}   SGD const {sgd_const:.2e}   SGD decay {sgd_decay:.2e}")
assert sgd_const < full_excess / 10, "for the same data passes, SGD wins by a mile"
assert sgd_decay < sgd_const / 3, "decaying the step size removes most of the noise floor"

# %%
print("\nAll checks passed.")
