# %% [markdown]
# # Lab 17.3: Logistic regression and GLMs
#
# 1. Logistic regression three ways (gradient descent, IRLS, statsmodels) on the breast cancer data.
# 2. Odds ratios, marginal effects, and the calibration-in-the-large property.
# 3. Separation: the MLE runs away; L2 brings it back.
# 4. Undersampling negatives and correcting the intercept.
# 5. Poisson regression with exposure: with and without the offset.
# 6. Overdispersion: Pearson statistic and negative binomial.
# 7. Gamma regression for positive skewed amounts.

# %%
import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, PoissonRegressor

rng = np.random.default_rng(173)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# %% [markdown]
# ## 1. Three ways to fit the same model

# %%
data = load_breast_cancer()
cols = ["mean radius", "mean texture", "mean smoothness", "mean concave points"]
Xraw = data.data[:, [list(data.feature_names).index(c) for c in cols]]
Xs = (Xraw - Xraw.mean(0)) / Xraw.std(0)
X = np.column_stack([np.ones(len(Xs)), Xs])
y = (data.target == 0).astype(float)                      # 1 = malignant
n, p = X.shape


def bce(beta):
    pr = np.clip(sigmoid(X @ beta), 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr))


# gradient descent
b_gd = np.zeros(p)
for _ in range(20_000):
    b_gd -= 1.0 * X.T @ (sigmoid(X @ b_gd) - y) / n

# IRLS = Newton
b_irls, history = np.zeros(p), []
for it in range(25):
    pr = sigmoid(X @ b_irls)
    w = pr * (1 - pr)
    z = X @ b_irls + (y - pr) / w                          # working response
    b_new = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))
    history.append(np.abs(b_new - b_irls).max())
    b_irls = b_new
    if history[-1] < 1e-12:
        break

sm_fit = sm.Logit(y, X).fit(disp=0)
print("IRLS       :", np.round(b_irls, 4), f"({len(history)} iterations)")
print("statsmodels:", np.round(sm_fit.params, 4))
print("grad desc  :", np.round(b_gd, 4), f"(20,000 iterations; loss gap {bce(b_gd) - bce(b_irls):.1e})")
assert np.allclose(b_irls, sm_fit.params, atol=1e-6)
assert len(history) < 15 and bce(b_gd) - bce(b_irls) < 1e-6
assert np.abs(b_gd - b_irls).max() < 0.05

sk_default = LogisticRegression().fit(Xs, y)
sk_unreg = LogisticRegression(C=1e12, max_iter=10_000, tol=1e-10).fit(Xs, y)
print("sklearn C=1   :", np.round(np.r_[sk_default.intercept_, sk_default.coef_[0]], 3), "(shrunk: this is not the MLE)")
assert np.allclose(np.r_[sk_unreg.intercept_, sk_unreg.coef_[0]], b_irls, atol=1e-3)
assert np.abs(sk_default.coef_[0]).sum() < np.abs(b_irls[1:]).sum()

# Wald standard errors from the inverse Hessian
pr = sigmoid(X @ b_irls)
cov = np.linalg.inv(X.T @ ((pr * (1 - pr))[:, None] * X))
assert np.allclose(np.sqrt(np.diag(cov)), sm_fit.bse, rtol=1e-5)
print("Wald SEs match statsmodels")

# %% [markdown]
# ## 2. Interpretation

# %%
ors = np.exp(sm_fit.params[1:])
ci = np.exp(sm_fit.conf_int()[1:])
for c, o, (lo, hi) in zip(cols, ors, ci):
    print(f"{c:22s} odds ratio per SD {o:6.2f}  (95% CI {lo:.2f} to {hi:.2f})")

j = cols.index("mean texture") + 1
ame = np.mean(b_irls[j] * pr * (1 - pr))
print(f"average marginal effect of +1 SD texture: {ame:+.3f} probability; at p=0.5 it would be {b_irls[j] * 0.25:+.3f}")
eff = b_irls[j] * pr * (1 - pr)
assert eff.max() / max(eff.min(), 1e-12) > 100, "the same coefficient moves some probabilities 100x more than others"

print(f"mean predicted probability {pr.mean():.6f} vs observed rate {y.mean():.6f}")
assert np.isclose(pr.mean(), y.mean(), atol=1e-8), "with an intercept, sum(p) = sum(y) at the MLE"

# %% [markdown]
# ## 3. Separation

# %%
xs = np.r_[rng.normal(-2, 1, 30), rng.normal(2, 1, 30)]
xs[:30] = np.minimum(xs[:30], -0.1); xs[30:] = np.maximum(xs[30:], 0.1)   # perfectly separable at 0
ys = np.r_[np.zeros(30), np.ones(30)]
Xsep = np.column_stack([np.ones(60), xs])
b, norms = np.zeros(2), []
for step in range(1, 200_001):
    b -= 0.5 * Xsep.T @ (sigmoid(Xsep @ b) - ys) / 60
    if step in (1_000, 10_000, 100_000, 200_000):
        norms.append(abs(b[1]))
print("slope after 1e3, 1e4, 1e5, 2e5 GD steps:", np.round(norms, 1), "(still growing: no MLE exists)")
assert norms[0] < norms[1] < norms[2] < norms[3]

# statsmodels used to raise here; current versions warn and return whatever the optimizer reached when it gave up
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sep_fit = sm.Logit(ys, Xsep).fit(disp=0)
kinds = sorted({w.category.__name__ for w in caught})
print(f"statsmodels: slope {sep_fit.params[1]:.1f}, converged={sep_fit.mle_retvals['converged']}, warnings: {kinds}")
assert "PerfectSeparationWarning" in kinds, "read your warnings: this one means the coefficients are meaningless"
ridge_slope = LogisticRegression(C=1.0).fit(xs[:, None], ys).coef_[0, 0]
print(f"L2-regularized slope: {ridge_slope:.2f} (finite, unique)")
assert ridge_slope < norms[0]

# %% [markdown]
# ## 4. Undersampled negatives

# %%
N = 200_000
Xb = rng.normal(size=(N, 3))
true_b = np.array([-4.5, 1.0, -0.5, 0.8])                   # ~2% positives
yb = rng.random(N) < sigmoid(true_b[0] + Xb @ true_b[1:])
keep_rate = 0.1
keep = yb | (rng.random(N) < keep_rate)
m = LogisticRegression(C=1e12, max_iter=1000).fit(Xb[keep], yb[keep])
print(f"base rate {yb.mean():.3%}; fitted intercept {m.intercept_[0]:.2f} (truth {true_b[0]}); slopes {np.round(m.coef_[0], 2)}")
corrected = m.intercept_[0] + np.log(keep_rate)
print(f"corrected intercept {corrected:.2f}")
assert np.allclose(m.coef_[0], true_b[1:], atol=0.08), "slopes survive undersampling"
assert abs(corrected - true_b[0]) < 0.1 and abs(m.intercept_[0] - true_b[0]) > 2
p_raw = m.predict_proba(Xb)[:, 1]
p_fix = sigmoid(np.log(p_raw / (1 - p_raw)) + np.log(keep_rate))
print(f"mean predicted probability: uncorrected {p_raw.mean():.3%}, corrected {p_fix.mean():.3%}, actual {yb.mean():.3%}")
assert abs(p_fix.mean() - yb.mean()) < 0.002 < abs(p_raw.mean() - yb.mean())

# %% [markdown]
# ## 5. Poisson with exposure
#
# Insurance-style data: young drivers have a higher claim *rate*, but were mostly observed for short periods.

# %%
n5 = 20_000
young = rng.random(n5) < 0.3
exposure = np.where(young & (rng.random(n5) < 0.8), rng.uniform(0.05, 0.25, n5), rng.uniform(0.5, 1.0, n5))
urban = rng.random(n5) < 0.5
rate = np.exp(-2.0 + 0.6 * young + 0.3 * urban)            # claims per policy-year
claims = rng.poisson(rate * exposure)
X5 = sm.add_constant(np.column_stack([young, urban]).astype(float))

with_off = sm.GLM(claims, X5, family=sm.families.Poisson(), offset=np.log(exposure)).fit()
without = sm.GLM(claims, X5, family=sm.families.Poisson()).fit()
print(f"rate ratio for young drivers: truth {np.exp(0.6):.2f}; with offset {np.exp(with_off.params[1]):.2f}; "
      f"without offset {np.exp(without.params[1]):.2f}")
assert abs(with_off.params[1] - 0.6) < 0.1
assert without.params[1] < 0, "without exposure, young drivers look *safer*"

# scikit-learn: model the rate y/exposure with exposure as the sample weight
skp = PoissonRegressor(alpha=0, max_iter=10_000, tol=1e-10).fit(X5[:, 1:], claims / exposure, sample_weight=exposure)
assert np.allclose(np.r_[skp.intercept_, skp.coef_], with_off.params, atol=1e-3)
print("scikit-learn (rate + sample_weight) matches statsmodels (offset)")

# the linear-regression alternative: additive effects, so nothing stops it going below zero
ols = sm.OLS(claims, sm.add_constant(np.column_stack([young, urban, exposure]).astype(float))).fit()
x_short = np.array([[1.0, 0.0, 0.0, 0.02]])                # an older rural driver observed for one week
ols_short = ols.predict(x_short)[0]
glm_short = with_off.predict(x_short[:, :3], offset=np.log(x_short[:, 3]))[0]
print(f"one-week policy: OLS predicts {ols_short:+.4f} claims, Poisson {glm_short:.4f}, truth {np.exp(-2.0) * 0.02:.4f}")
assert ols_short < 0 < glm_short, "OLS on counts predicts a negative number of claims"

# %% [markdown]
# ## 6. Overdispersion

# %%
frailty = rng.gamma(shape=1.0, scale=1.0, size=n5)          # unobserved heterogeneity, mean 1
claims_od = rng.poisson(rate * exposure * frailty * 3)
pois = sm.GLM(claims_od, X5, family=sm.families.Poisson(), offset=np.log(exposure)).fit()
pearson = pois.pearson_chi2 / pois.df_resid
print(f"Pearson chi2/df: well-specified data {with_off.pearson_chi2 / with_off.df_resid:.2f}, overdispersed data {pearson:.2f}")
assert 0.9 < with_off.pearson_chi2 / with_off.df_resid < 1.1 and pearson > 1.3

nb = sm.NegativeBinomial(claims_od, X5, offset=np.log(exposure)).fit(disp=0)
print(f"young coefficient SE: Poisson {pois.bse[1]:.4f}, negative binomial {nb.bse[1]:.4f}")
print(f"estimates: Poisson {pois.params[1]:.3f}, negative binomial {nb.params[1]:.3f} (both near 0.6; only the uncertainty differs)")
assert nb.bse[1] > 1.05 * pois.bse[1]
assert abs(pois.params[1] - 0.6) < 0.1 and abs(nb.params[1] - 0.6) < 0.1

# %% [markdown]
# ## 7. Gamma regression for amounts

# %%
n7 = 5000
severity_x = rng.normal(size=n7)
mu = np.exp(7 + 0.5 * severity_x)                          # mean claim size
shape = 2.0
amount = rng.gamma(shape, mu / shape)                      # sd proportional to mean
gam = sm.GLM(amount, sm.add_constant(severity_x), family=sm.families.Gamma(sm.families.links.Log())).fit()
logols = sm.OLS(np.log(amount), sm.add_constant(severity_x)).fit()
naive_mean = np.exp(logols.fittedvalues).mean()
print(f"Gamma GLM slope {gam.params[1]:.3f} (truth 0.5); mean prediction {gam.fittedvalues.mean():,.0f} vs actual {amount.mean():,.0f}; "
      f"log-OLS without correction {naive_mean:,.0f}")
assert abs(gam.params[1] - 0.5) < 0.05
assert abs(gam.fittedvalues.mean() / amount.mean() - 1) < 0.01 < abs(naive_mean / amount.mean() - 1)

print("\nAll checks passed.")
