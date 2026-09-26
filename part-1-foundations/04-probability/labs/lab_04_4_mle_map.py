# %% [markdown]
# # Lab 04.4: Losses are likelihoods, regularizers are priors
#
# 1. Closed-form MLEs vs numerical optimization of the NLL (Bernoulli, Gaussian, Poisson).
# 2. The biased MLE variance, measured.
# 3. MSE = Gaussian NLL, MAE = Laplace NLL: same minimizers, different robustness.
# 4. Asymptotic normality: Fisher-information standard errors and their coverage.
# 5. MAP with a Gaussian prior = ridge regression.
# 6. Poisson regression vs least squares on count data.

# %%
import numpy as np
from scipy import optimize, stats
from sklearn.linear_model import PoissonRegressor, Ridge

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Closed forms vs numerical NLL minimization

# %%
x_b = rng.binomial(1, 0.3, 500)
nll_b = lambda z: -np.mean(x_b * -np.logaddexp(0, -z) + (1 - x_b) * -np.logaddexp(0, z))  # z = logit(p), stable
z_hat = optimize.minimize_scalar(nll_b).x
assert np.isclose(1 / (1 + np.exp(-z_hat)), x_b.mean(), atol=1e-6)

x_g = rng.normal(3.0, 2.0, 1000)
def nll_g(params):
    mu, log_sigma = params                    # parameterize sigma > 0 away
    return np.mean(0.5 * np.log(2 * np.pi) + log_sigma + (x_g - mu) ** 2 / (2 * np.exp(2 * log_sigma)))
mu_hat, log_sigma_hat = optimize.minimize(nll_g, x0=[0.0, 0.0]).x
assert np.isclose(mu_hat, x_g.mean(), atol=1e-4)
assert np.isclose(np.exp(2 * log_sigma_hat), x_g.var(ddof=0), rtol=1e-3), "MLE variance uses 1/n"

x_p = rng.poisson(4.2, 800)
nll_p = lambda log_lam: np.mean(np.exp(log_lam) - x_p * log_lam)
assert np.isclose(np.exp(optimize.minimize_scalar(nll_p).x), x_p.mean(), rtol=1e-5)
print("numerical NLL minimization reproduces the closed-form MLEs")

# %% [markdown]
# ## 2. The MLE variance is biased (by a factor (n-1)/n)

# %%
n = 5
samples = rng.normal(0, 1, size=(200_000, n))
mle_var = samples.var(axis=1, ddof=0)
print(f"E[MLE variance] with n={n}: {mle_var.mean():.4f}   theory (n-1)/n = {(n - 1) / n:.4f}")
assert np.isclose(mle_var.mean(), (n - 1) / n, atol=0.01)
assert np.isclose(samples.var(axis=1, ddof=1).mean(), 1.0, atol=0.01)

# %% [markdown]
# ## 3. MSE vs MAE: Gaussian vs Laplace noise models
#
# The best constant under MSE is the mean, under MAE the median. Add one outlier and watch.

# %%
y = rng.normal(10, 1, 99)
y_out = np.append(y, 1000.0)                  # one corrupted row
best_mse = optimize.minimize_scalar(lambda c: np.mean((y_out - c) ** 2)).x
best_mae = optimize.minimize_scalar(lambda c: np.mean(np.abs(y_out - c)), bounds=(0, 2000), method="bounded").x
print(f"with one outlier: MSE-optimal constant {best_mse:.2f} (mean {y_out.mean():.2f}), "
      f"MAE-optimal {best_mae:.2f} (median {np.median(y_out):.2f})")
assert np.isclose(best_mse, y_out.mean(), atol=1e-3) and abs(best_mae - np.median(y_out)) < 0.05
assert best_mse > 19 and best_mae < 11

# %% [markdown]
# ## 4. Fisher information and standard errors
#
# For a Bernoulli, I(p) = 1 / (p(1-p)), so SE(p_hat) ~ sqrt(p(1-p)/n). Check the coverage of p_hat +- 1.96 SE.

# %%
p_true, n = 0.3, 400
p_hats = rng.binomial(n, p_true, 100_000) / n
se = np.sqrt(p_hats * (1 - p_hats) / n)
coverage = np.mean(np.abs(p_hats - p_true) <= 1.96 * se)
print(f"coverage of the Wald 95% interval at n={n}: {coverage:.3f}")
assert 0.93 < coverage < 0.96

# The Hessian of the NLL at the optimum gives the same SE (here for the Gaussian mean: sigma/sqrt(n))
h = 1e-4
f = lambda m: len(x_g) * nll_g([m, log_sigma_hat])
hess = (f(mu_hat + h) - 2 * f(mu_hat) + f(mu_hat - h)) / h**2
se_hess = 1 / np.sqrt(hess)
assert np.isclose(se_hess, np.exp(log_sigma_hat) / np.sqrt(len(x_g)), rtol=1e-3)
print(f"SE of the mean from the NLL curvature: {se_hess:.4f}")

# %% [markdown]
# ## 5. MAP with a Gaussian prior is ridge

# %%
n, d = 60, 8
X = rng.normal(size=(n, d))
w_true = rng.normal(size=d)
sigma, tau = 1.5, 0.5
y = X @ w_true + rng.normal(0, sigma, n)

neg_log_post = lambda w: np.sum((X @ w - y) ** 2) / (2 * sigma**2) + np.sum(w**2) / (2 * tau**2)
w_map = optimize.minimize(neg_log_post, np.zeros(d), method="L-BFGS-B").x
lam = sigma**2 / tau**2
w_ridge = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
w_closed = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
assert np.allclose(w_map, w_closed, atol=1e-4) and np.allclose(w_ridge, w_closed, atol=1e-8)
print(f"MAP (prior sd {tau}) == ridge with lambda = sigma^2/tau^2 = {lam:.1f}")

# Laplace smoothing = MAP with a Beta(2, 2) prior
k, n_flips = 3, 3
assert (k + 1) / (n_flips + 2) == 0.8
rule_of_three = 1 - 0.05 ** (1 / 50)
print(f"0 failures in 50 runs: MLE 0, Beta(1,1) posterior mean {1 / 52:.3f}, 95% upper bound {rule_of_three:.3f}")
assert abs(rule_of_three - 3 / 50) < 0.002

# %% [markdown]
# ## 6. Counts: Poisson likelihood vs squared error
#
# Orders per store per day, with a multiplicative effect of store size. The Poisson GLM (log link) matches the data
# generating process; a straight least squares fit on the counts doesn't.

# %%
n = 3000
size = rng.uniform(0, 3, n)
promo = rng.binomial(1, 0.3, n)
lam_true = np.exp(0.5 + 0.8 * size + 0.4 * promo)
orders = rng.poisson(lam_true)
Xc = np.column_stack([size, promo])
pois = PoissonRegressor(alpha=0.0, max_iter=1000).fit(Xc, orders)
print(f"Poisson GLM coefficients {np.round(pois.coef_, 3)} intercept {pois.intercept_:.3f} (truth 0.8, 0.4, 0.5)")
assert np.allclose(pois.coef_, [0.8, 0.4], atol=0.05)

ols = np.linalg.lstsq(np.column_stack([np.ones(n), Xc]), orders, rcond=None)[0]
pred_ols = np.column_stack([np.ones(n), Xc]) @ ols
print(f"least squares predicts negative orders for {np.mean(pred_ols < 0):.1%} of stores")
def dev(mu):
    # Poisson deviance; the y*log(y/mu) term is defined as 0 when y = 0 (scipy's xlogy handles that)
    from scipy.special import xlogy
    return 2 * np.mean(xlogy(orders, orders) - xlogy(orders, mu) - (orders - mu))

assert dev(pois.predict(Xc)) < dev(np.clip(pred_ols, 0.1, None))

# %%
print("\nAll checks passed.")
