# %% [markdown]
# # Lab 02.3: Least squares, four ways, on real data
#
# Dataset: `data/fuel_consumption_co2.csv` (2014 vehicles, Natural Resources Canada). We predict CO2 emissions
# (g/km) from engine size, cylinders and combined fuel consumption.
#
# 1. Solve with the normal equations, QR, lstsq and scikit-learn; check they agree.
# 2. Verify the geometry: residuals orthogonal to every column, P^2 = P, trace(P) = d, Pythagoras.
# 3. Wreck the normal equations with badly scaled polynomial features; watch QR survive.
# 4. Rank deficiency: duplicated features and the minimum-norm solution.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[3]
df = pd.read_csv(ROOT / "data" / "fuel_consumption_co2.csv")
print(df.shape)
print(df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]].describe().round(2))

# %% [markdown]
# ## 1. Four solvers, one answer

# %%
features = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]
X_raw = df[features].to_numpy(dtype=float)
y = df["CO2EMISSIONS"].to_numpy(dtype=float)
X = np.column_stack([np.ones(len(X_raw)), X_raw])      # intercept column

w_normal = np.linalg.solve(X.T @ X, X.T @ y)            # at least use solve, not inv
Q, R = np.linalg.qr(X)
w_qr = scipy.linalg.solve_triangular(R, Q.T @ y)
w_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
sk = LinearRegression().fit(X_raw, y)
w_sklearn = np.concatenate([[sk.intercept_], sk.coef_])

for name, w in [("normal", w_normal), ("qr", w_qr), ("lstsq", w_lstsq), ("sklearn", w_sklearn)]:
    print(f"{name:8s}", np.round(w, 4))
for w in (w_normal, w_qr, w_sklearn):
    np.testing.assert_allclose(w, w_lstsq, rtol=1e-8)

y_hat = X @ w_lstsq
rmse = np.sqrt(np.mean((y - y_hat) ** 2))
print(f"RMSE {rmse:.2f} g/km on the training data (training error, not a real evaluation: see 18.1)")

# %% [markdown]
# ## 2. The geometry, verified

# %%
r = y - y_hat
assert np.allclose(X.T @ r, 0, atol=1e-6), "residual is orthogonal to every column"
assert abs(r.sum()) < 1e-6, "with an intercept, residuals sum to zero"

P = Q @ Q.T                                   # projection matrix, computed stably via Q
assert np.allclose(P @ P, P) and np.allclose(P, P.T)
assert np.isclose(np.trace(P), X.shape[1])    # trace = number of columns
assert np.allclose(P @ y, y_hat)
assert np.isclose(y @ y, y_hat @ y_hat + r @ r)               # Pythagoras
x_bar = X_raw.mean(axis=0)
assert np.isclose(w_lstsq[0] + x_bar @ w_lstsq[1:], y.mean())  # fit passes through the means

# The same decomposition after centering gives R^2
yc, yhc = y - y.mean(), y_hat - y.mean()
r2 = (yhc @ yhc) / (yc @ yc)
assert np.isclose(r2, sk.score(X_raw, y))
print(f"R^2 = {r2:.4f} (explained / total variance, from Pythagoras)")

# %% [markdown]
# ## 3. Polynomial features: normal equations vs QR
#
# Fit a degree-8 polynomial of fuel consumption (values around 5-25) with RAW powers, the classic mistake.
# We know the true coefficients because we generate y from them, so we can measure the error directly.

# %%
x = df["FUELCONSUMPTION_COMB"].to_numpy(dtype=float)
deg = 8
V = np.vander(x, deg + 1, increasing=True)                 # columns 1, x, x^2, ..., x^8
rng = np.random.default_rng(0)
w_true = rng.normal(size=deg + 1) / (10.0 ** np.arange(deg + 1))   # keeps terms of comparable size
y_poly = V @ w_true

print(f"cond(V) = {np.linalg.cond(V):.1e}   cond(V^T V) = {np.linalg.cond(V.T @ V):.1e}")
w_ne = np.linalg.solve(V.T @ V, V.T @ y_poly)
Qv, Rv = np.linalg.qr(V)
w_q = scipy.linalg.solve_triangular(Rv, Qv.T @ y_poly)
err_ne = np.linalg.norm(V @ w_ne - y_poly) / np.linalg.norm(y_poly)
err_q = np.linalg.norm(V @ w_q - y_poly) / np.linalg.norm(y_poly)
print(f"relative fit error: normal equations {err_ne:.1e}   QR {err_q:.1e}")
assert err_q < 1e-6 and err_ne > 100 * err_q

# The fix that matters most in practice: scale the input before building powers.
xs = (x - x.mean()) / x.std()
Vs = np.vander(xs, deg + 1, increasing=True)
print(f"after standardizing x: cond(V) = {np.linalg.cond(Vs):.1e}")
assert np.linalg.cond(Vs) < np.linalg.cond(V) / 1e4

# %% [markdown]
# ## 4. Rank deficiency and the minimum-norm solution

# %%
X_dup = np.column_stack([X, X[:, 1]])        # duplicate ENGINESIZE
assert np.linalg.matrix_rank(X_dup) == X.shape[1]
w_min, _, rank, _ = np.linalg.lstsq(X_dup, y, rcond=None)
print("rank:", rank, " weights on the duplicated pair:", np.round(w_min[[1, -1]], 4))
assert np.isclose(w_min[1], w_min[-1]), "minimum norm splits the weight evenly between identical columns"
assert np.isclose(w_min[1] + w_min[-1], w_lstsq[1])
assert np.allclose(X_dup @ w_min, y_hat)     # same predictions as before
w_other = w_min.copy(); w_other[1] += 100; w_other[-1] -= 100
assert np.allclose(X_dup @ w_other, y_hat) and np.linalg.norm(w_other) > np.linalg.norm(w_min)
np.testing.assert_allclose(np.linalg.pinv(X_dup) @ y, w_min, rtol=1e-6)

# %%
print("\nAll checks passed.")
