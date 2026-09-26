# %% [markdown]
# # Lab 17.1: Linear regression for inference
#
# 1. OLS on the fuel data with statsmodels; standard errors recomputed by hand.
# 2. Do 95% confidence intervals cover the truth 95% of the time? (Yes, when the assumptions hold.)
# 3. Heteroskedasticity: classical intervals under-cover, HC3 intervals fix it.
# 4. Multicollinearity: VIF for engine size and cylinders, and what it does to the standard errors.
# 5. Leverage and Cook's distance: one bad point, and Huber regression shrugging it off.
# 6. Log targets: e^beta - 1, not beta.
# 7. Confidence interval vs prediction interval.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import HuberRegressor, LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

rng = np.random.default_rng(171)
DATA = Path(__file__).resolve().parents[3] / "data" / "fuel_consumption_co2.csv"
fuel = pd.read_csv(DATA)
print(fuel.shape)

# %% [markdown]
# ## 1. OLS and its standard errors, by hand

# %%
X = sm.add_constant(fuel[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]].astype(float))
y = fuel["CO2EMISSIONS"].astype(float)
res = sm.OLS(y, X).fit()
print(res.summary().tables[1])

Xm, ym = X.to_numpy(), y.to_numpy()
n, p = Xm.shape
beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
resid = ym - Xm @ beta
sigma2 = resid @ resid / (n - p)                      # unbiased: divide by n - p
XtX_inv = np.linalg.inv(Xm.T @ Xm)                   # fine for a 4x4 matrix used for its diagonal; never for solving
se = np.sqrt(sigma2 * np.diag(XtX_inv))
t = beta / se
t_crit = stats.t.ppf(0.975, n - p)
ci = np.column_stack([beta - t_crit * se, beta + t_crit * se])
assert np.allclose(beta, res.params) and np.allclose(se, res.bse) and np.allclose(t, res.tvalues)
assert np.allclose(ci, res.conf_int().to_numpy())
print("hand-computed coefficients, SEs, t-stats and CIs match statsmodels")

# %% [markdown]
# ## 2. Coverage when the assumptions hold
#
# Fixed design, known true betas, Gaussian homoskedastic noise. Refit 4,000 times and count how often the
# 95% interval for the slope contains the true slope.

# %%
def coverage(noise_fn, reps=4000, n=60, robust=None):
    x = rng.uniform(0, 10, n)
    Xd = sm.add_constant(x)
    beta_true = np.array([2.0, 0.5])
    hits = 0
    for _ in range(reps):
        yy = Xd @ beta_true + noise_fn(x)
        fit = sm.OLS(yy, Xd).fit(cov_type=robust) if robust else sm.OLS(yy, Xd).fit()
        lo, hi = fit.conf_int()[1]
        hits += lo <= beta_true[1] <= hi
    return hits / reps


cov_ok = coverage(lambda x: rng.normal(0, 1.0, len(x)))
print(f"homoskedastic Gaussian noise: slope CI coverage {cov_ok:.3f}")
assert 0.935 < cov_ok < 0.965

# %% [markdown]
# ## 3. Heteroskedasticity breaks the classical intervals
#
# Noise standard deviation grows with x (think revenue, prices). Same estimator, same formula, wrong answer.

# %%
hetero = lambda x: rng.normal(0, 0.05 + 0.02 * x**2, len(x))
cov_classic = coverage(hetero)
cov_hc3 = coverage(hetero, robust="HC3")
print(f"heteroskedastic noise: classical coverage {cov_classic:.3f}, HC3 coverage {cov_hc3:.3f}")
assert cov_classic < 0.92, "classical SEs are too small here"
assert cov_hc3 > cov_classic + 0.02 and cov_hc3 > 0.93, "sandwich SEs get close to nominal"

# the diagnostic you'd look at: residual spread vs fitted values
x = rng.uniform(0, 10, 500)
yy = 2 + 0.5 * x + hetero(x)
fit = sm.OLS(yy, sm.add_constant(x)).fit()
lo_half = np.abs(fit.resid[fit.fittedvalues < np.median(fit.fittedvalues)]).mean()
hi_half = np.abs(fit.resid[fit.fittedvalues >= np.median(fit.fittedvalues)]).mean()
bp_pvalue = sm.stats.het_breuschpagan(fit.resid, fit.model.exog)[1]
print(f"mean |residual|: low fitted {lo_half:.2f}, high fitted {hi_half:.2f}; Breusch-Pagan p = {bp_pvalue:.1e}")
assert hi_half > 2 * lo_half and bp_pvalue < 1e-3

# %% [markdown]
# ## 4. Multicollinearity
#
# Engine size and cylinder count move together. Each coefficient becomes hard to pin down, the predictions don't care.

# %%
Xf = fuel[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]].astype(float)
Xc = sm.add_constant(Xf).to_numpy()
vifs = {c: variance_inflation_factor(Xc, i + 1) for i, c in enumerate(Xf.columns)}
print("correlation engine/cylinders:", round(Xf["ENGINESIZE"].corr(Xf["CYLINDERS"]), 3))
print("VIF:", {k: round(float(v), 1) for k, v in vifs.items()})
# VIF by its definition: 1 / (1 - R^2 of x_j on the others)
others = sm.add_constant(Xf[["CYLINDERS", "FUELCONSUMPTION_COMB"]])
r2 = sm.OLS(Xf["ENGINESIZE"], others).fit().rsquared
assert np.isclose(vifs["ENGINESIZE"], 1 / (1 - r2))
assert vifs["ENGINESIZE"] > 5

alone = sm.OLS(y, sm.add_constant(Xf[["ENGINESIZE"]])).fit()
print(f"SE of ENGINESIZE: alone {alone.bse['ENGINESIZE']:.2f}, with correlated partners {res.bse['ENGINESIZE']:.2f}")
print(f"ENGINESIZE coefficient: alone {alone.params['ENGINESIZE']:.1f}, with partners {res.params['ENGINESIZE']:.1f}"
      " (the meaning changed: 'holding fuel consumption constant')")
assert res.bse["ENGINESIZE"] > alone.bse["ENGINESIZE"]

# bootstrap: coefficients wobble, predictions don't
coefs, preds = [], []
probe = Xc[:50]
for _ in range(300):
    idx = rng.integers(0, len(Xc), len(Xc))
    b = np.linalg.lstsq(Xc[idx], ym[idx], rcond=None)[0]
    coefs.append(b[1:3]); preds.append(probe @ b)
coefs, preds = np.array(coefs), np.array(preds)
rel_coef = coefs.std(0) / np.abs(coefs.mean(0))
rel_pred = (preds.std(0) / preds.mean(0)).mean()
print(f"bootstrap relative sd: engine coef {rel_coef[0]:.2f}, cylinder coef {rel_coef[1]:.2f}, predictions {rel_pred:.3f}")
assert rel_pred < 0.1 * rel_coef.max()

# %% [markdown]
# ## 5. Leverage, influence and robust regression

# %%
x = rng.uniform(0, 10, 40)
yy = 1 + 2 * x + rng.normal(0, 1, 40)
x_bad = np.append(x, 30.0)          # far from the other x values: high leverage
y_bad = np.append(yy, 5.0)          # and wildly off the line: large residual
fit_bad = sm.OLS(y_bad, sm.add_constant(x_bad)).fit()
infl = fit_bad.get_influence()
lev, cooks = infl.hat_matrix_diag, infl.cooks_distance[0]
print(f"leverage of the bad point {lev[-1]:.2f} (average p/n = {2 / len(x_bad):.3f}); Cook's distance {cooks[-1]:.1f} "
      f"(max of the rest {cooks[:-1].max():.2f})")
assert np.isclose(lev.sum(), 2), "leverages sum to p (trace of the hat matrix)"
assert np.argmax(cooks) == len(x_bad) - 1 and cooks[-1] > 1

ols_clean = LinearRegression().fit(x[:, None], yy).coef_[0]
ols_dirty = LinearRegression().fit(x_bad[:, None], y_bad).coef_[0]
huber = HuberRegressor(epsilon=1.35).fit(x_bad[:, None], y_bad).coef_[0]
print(f"slope: OLS clean {ols_clean:.2f}, OLS with bad point {ols_dirty:.2f}, Huber with bad point {huber:.2f} (truth 2)")
assert abs(ols_dirty - 2) > 1.0 and abs(huber - 2) < 0.2

# %% [markdown]
# ## 6. Log targets: the coefficient is not the percentage

# %%
n6 = 20_000
garden = rng.integers(0, 2, n6)
area = rng.uniform(50, 200, n6)
log_price = 11 + 0.5 * garden + 0.004 * area + rng.normal(0, 0.1, n6)
fit6 = sm.OLS(log_price, sm.add_constant(np.column_stack([garden, area]))).fit()
b_garden = fit6.params[1]
price = np.exp(log_price)
# the effect you'd tell someone: median price with a garden vs without, at similar areas
band = (area > 100) & (area < 110)
ratio = np.median(price[band & (garden == 1)]) / np.median(price[band & (garden == 0)]) - 1
print(f"coefficient {b_garden:.3f}; naive reading +{b_garden:.0%}; exp(b) - 1 = +{np.exp(b_garden) - 1:.1%}; "
      f"observed median uplift +{ratio:.1%}")
assert abs(ratio - (np.exp(b_garden) - 1)) < abs(ratio - b_garden), "exp(b) - 1 is the right reading"

# retransformation: exp(prediction) is the median; the mean needs exp(sigma^2 / 2)
pred_log = fit6.fittedvalues
s2 = fit6.mse_resid
naive, smeared = np.exp(pred_log).mean(), np.exp(pred_log + s2 / 2).mean()
print(f"mean price {price.mean():,.0f}; exp(prediction) {naive:,.0f}; with exp(sigma^2/2) correction {smeared:,.0f}")
assert abs(smeared - price.mean()) < abs(naive - price.mean())

# %% [markdown]
# ## 7. Confidence interval vs prediction interval

# %%
x0 = pd.DataFrame({"const": [1.0], "ENGINESIZE": [3.0], "CYLINDERS": [6.0], "FUELCONSUMPTION_COMB": [11.0]})
frame = res.get_prediction(x0).summary_frame(alpha=0.05)
ci_w = float(frame["mean_ci_upper"].iloc[0] - frame["mean_ci_lower"].iloc[0])
pi_w = float(frame["obs_ci_upper"].iloc[0] - frame["obs_ci_lower"].iloc[0])
print(f"at x0: prediction {frame['mean'].iloc[0]:.0f} g/km; 95% CI width {ci_w:.1f}, 95% PI width {pi_w:.1f}")
assert pi_w > 5 * ci_w
assert abs(pi_w / 2 - 1.96 * np.sqrt(res.mse_resid)) / (pi_w / 2) < 0.05, "PI half-width ~ 1.96 sigma for large n"

# and the PI actually covers ~95% of held-out observations
idx = rng.permutation(len(fuel))
tr, te = idx[:800], idx[800:]
fit_tr = sm.OLS(y.iloc[tr], X.iloc[tr]).fit()
pi = fit_tr.get_prediction(X.iloc[te]).summary_frame(alpha=0.05)
inside = ((y.iloc[te].to_numpy() >= pi["obs_ci_lower"].to_numpy()) & (y.iloc[te].to_numpy() <= pi["obs_ci_upper"].to_numpy())).mean()
print(f"held-out coverage of the 95% prediction interval: {inside:.1%}")
print("(real data: residuals aren't Gaussian here, fuel type matters and isn't in the model; see 16.1)")

print("\nAll checks passed.")
