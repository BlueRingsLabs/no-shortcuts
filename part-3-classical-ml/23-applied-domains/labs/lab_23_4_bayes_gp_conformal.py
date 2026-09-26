# %% [markdown]
# # Lab 23.4: Bayesian inference, Gaussian processes and conformal intervals
#
# 1. A Bayesian A/B test with Beta-Binomial posteriors.
# 2. Partial pooling: 50 stores, empirical Bayes shrinkage vs no pooling vs complete pooling, against the true rates.
# 3. Bayesian linear regression: predictive uncertainty that grows away from the data.
# 4. Metropolis-Hastings for logistic regression: posterior vs MLE and Wald intervals; R-hat.
# 5. A Gaussian process from scratch vs scikit-learn; interval coverage; marginal likelihood.
# 6. Conformal prediction: exact coverage, adaptive widths (CQR), and failure under shift.

# %%
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.special import betaln, expit
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import BayesianRidge

rng = np.random.default_rng(234)

# %% [markdown]
# ## 1. Bayesian A/B test

# %%
nA, sA, nB, sB = 4000, 200, 4000, 236                           # 5.0% vs 5.9%
postA, postB = stats.beta(1 + sA, 1 + nA - sA), stats.beta(1 + sB, 1 + nB - sB)
draws_A, draws_B = postA.rvs(200_000, random_state=1), postB.rvs(200_000, random_state=2)
p_b_better = np.mean(draws_B > draws_A)
loss_A = np.mean(np.maximum(draws_B - draws_A, 0))              # expected conversion lost if we ship A and B was better
loss_B = np.mean(np.maximum(draws_A - draws_B, 0))
lift = (draws_B - draws_A) / draws_A
print(f"P(B > A) = {p_b_better:.3f}; expected loss if we ship A {loss_A:.5f}, if we ship B {loss_B:.5f}")
print(f"relative lift: posterior median {np.median(lift):+.1%}, 95% credible interval [{np.percentile(lift, 2.5):+.1%}, {np.percentile(lift, 97.5):+.1%}]")
pval = sm.stats.proportions_ztest([sB, sA], [nB, nA], alternative="larger")[1]
print(f"(frequentist one-sided p-value {pval:.3f}: close to 1 - P(B > A) with a flat prior, as it often is)")
assert 0.9 < p_b_better < 0.99 and abs(pval - (1 - p_b_better)) < 0.02

# %% [markdown]
# ## 2. Partial pooling

# %%
n_stores = 50
a_true, b_true = 8.0, 92.0                                      # true rates ~ Beta(8, 92): mean 8%
rates = rng.beta(a_true, b_true, n_stores)
visits = rng.choice([12, 30, 80, 300, 2000], n_stores)
conv = rng.binomial(visits, rates)
no_pool = conv / visits
full_pool = np.full(n_stores, conv.sum() / visits.sum())


def neg_marginal(params):                                       # beta-binomial marginal likelihood of all stores
    a, b = np.exp(params)
    return -np.sum(betaln(a + conv, b + visits - conv) - betaln(a, b))


a_hat, b_hat = np.exp(minimize(neg_marginal, [0.0, 2.0], method="Nelder-Mead").x)
partial = (a_hat + conv) / (a_hat + b_hat + visits)             # posterior means under the fitted prior
for name, est in [("no pooling", no_pool), ("complete pooling", full_pool), ("partial pooling", partial)]:
    small = visits <= 30
    print(f"{name:17s} RMSE vs true rates: all stores {np.sqrt(np.mean((est - rates) ** 2)):.4f}, "
          f"stores with <= 30 visits {np.sqrt(np.mean((est[small] - rates[small]) ** 2)):.4f}")
print(f"fitted prior Beta({a_hat:.1f}, {b_hat:.1f}) (truth Beta({a_true:.0f}, {b_true:.0f})); "
      f"a store with 3/12 conversions: raw {3 / 12:.0%}, shrunk {(a_hat + 3) / (a_hat + b_hat + 12):.1%}")
rmse = lambda e: np.sqrt(np.mean((e - rates) ** 2))
print("for the tiniest stores complete pooling is about as good (they have almost no information of their own);")
print("partial pooling is the only estimator that's good for small and large stores at once")
assert rmse(partial) < min(rmse(no_pool), rmse(full_pool))

# %% [markdown]
# ## 3. Bayesian linear regression

# %%
x = rng.uniform(0, 5, 40)
y = 1.0 + 0.8 * x + rng.normal(0, 0.5, 40)
Xd = np.column_stack([np.ones_like(x), x])
sigma, tau = 0.5, 10.0
S_post = np.linalg.inv(Xd.T @ Xd / sigma**2 + np.eye(2) / tau**2)
m_post = S_post @ Xd.T @ y / sigma**2
x_new = np.array([2.5, 5.0, 10.0, 20.0])
Xn = np.column_stack([np.ones_like(x_new), x_new])
pred_sd = np.sqrt(sigma**2 + np.einsum("ij,jk,ik->i", Xn, S_post, Xn))
param_sd = np.sqrt(np.einsum("ij,jk,ik->i", Xn, S_post, Xn))
for xv, ps, tot in zip(x_new, param_sd, pred_sd):
    print(f"x = {xv:4.1f} (training range 0-5): parameter uncertainty sd {ps:.3f}, total predictive sd {tot:.3f}")
br = BayesianRidge().fit(x[:, None], y)
_, br_sd = br.predict(x_new[:, None], return_std=True)
print(f"BayesianRidge (estimates sigma and tau itself) predictive sd: {np.round(br_sd, 3)}")
assert param_sd[-1] > 5 * param_sd[0]

# %% [markdown]
# ## 4. Metropolis-Hastings for a logistic regression

# %%
n4 = 300
X4 = np.column_stack([np.ones(n4), rng.normal(size=(n4, 2))])
y4 = rng.random(n4) < expit(X4 @ np.array([-0.5, 1.0, -0.7]))


def log_post(b):                                                # flat-ish prior N(0, 10^2)
    z = X4 @ b
    return np.sum(y4 * z - np.logaddexp(0, z)) - np.sum(b**2) / 200


def metropolis(start, n_steps=20_000, step=0.12, seed=0):
    r = np.random.default_rng(seed)
    cur = np.array(start, float)
    lp = log_post(cur)
    out, acc = np.empty((n_steps, 3)), 0
    for t in range(n_steps):
        prop = cur + step * r.normal(size=3)
        lp_prop = log_post(prop)
        if np.log(r.random()) < lp_prop - lp:                   # only the ratio: p(D) cancels
            cur, lp = prop, lp_prop; acc += 1
        out[t] = cur
    return out[n_steps // 4:], acc / n_steps                    # drop burn-in


chains = [metropolis(s, seed=i) for i, s in enumerate([[0, 0, 0], [3, -3, 3], [-3, 3, -3], [2, 2, 2]])]
samples = np.vstack([c[0] for c in chains])


def rhat(chs):
    m = np.array([c.mean(0) for c in chs]); v = np.array([c.var(0, ddof=1) for c in chs]); n = len(chs[0])
    W, B = v.mean(0), n * m.var(0, ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W)


mle = sm.Logit(y4.astype(float), X4).fit(disp=0)
print(f"acceptance rates {[round(c[1], 2) for c in chains]}; R-hat {np.round(rhat([c[0] for c in chains]), 3)}")
print(f"posterior mean {np.round(samples.mean(0), 3)} vs MLE {np.round(mle.params, 3)}")
print(f"posterior sd   {np.round(samples.std(0), 3)} vs Wald SE {np.round(mle.bse, 3)} (the Laplace approximation)")
assert np.all(rhat([c[0] for c in chains]) < 1.02)
assert np.allclose(samples.mean(0), mle.params, atol=0.05) and np.allclose(samples.std(0), mle.bse, rtol=0.15)

# %% [markdown]
# ## 5. A Gaussian process from scratch

# %%
f_true = lambda x_: np.sin(x_) + 0.3 * np.cos(3 * x_)
x_tr = np.sort(rng.uniform(0, 6, 40)); y_tr = f_true(x_tr) + rng.normal(0, 0.15, 40)
x_te = np.linspace(-1, 9, 300)


def rbf(a, b, ell, s2):
    return s2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)


def gp_fit_predict(ell, s2, noise, xt=x_te):
    K = rbf(x_tr, x_tr, ell, s2) + noise * np.eye(len(x_tr))
    L = np.linalg.cholesky(K)                                    # never an explicit inverse
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
    Ks = rbf(x_tr, xt, ell, s2)
    mu = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = s2 - (v**2).sum(0)
    log_ml = -0.5 * y_tr @ alpha - np.log(np.diag(L)).sum() - 0.5 * len(x_tr) * np.log(2 * np.pi)
    return mu, var, log_ml


res = minimize(lambda p: -gp_fit_predict(*np.exp(p))[2], np.log([1.0, 1.0, 0.1]), method="Nelder-Mead", options={"maxiter": 2000})
ell, s2, noise = np.exp(res.x)
mu, var, lml = gp_fit_predict(ell, s2, noise)
sk = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1), n_restarts_optimizer=3, random_state=0).fit(x_tr[:, None], y_tr)
mu_sk, sd_sk = sk.predict(x_te[:, None], return_std=True)
print(f"my GP: length scale {ell:.2f}, signal var {s2:.2f}, noise var {noise:.4f}, log marginal likelihood {lml:.2f}")
print(f"scikit-learn: {sk.kernel_}, log marginal likelihood {sk.log_marginal_likelihood_value_:.2f}")
inside = (x_te >= 0) & (x_te <= 6)
print(f"max |mean difference| vs scikit-learn inside the data {np.abs(mu - mu_sk)[inside].max():.4f}")
sd = np.sqrt(var + noise)
y_fresh = f_true(x_te) + rng.normal(0, 0.15, len(x_te))
cov = np.mean(np.abs(y_fresh[inside] - mu[inside]) <= 1.96 * sd[inside])
print(f"95% predictive interval coverage on fresh data inside [0, 6]: {cov:.3f}")
print(f"predictive sd: inside the data ~{sd[inside].mean():.2f}; at x = 9 {sd[-1]:.2f} (back to the prior); mean at x = 9 {mu[-1]:.2f} (back to 0)")
assert abs(lml - sk.log_marginal_likelihood_value_) < 0.5 and np.abs(mu - mu_sk)[inside].max() < 0.05
assert cov > 0.88 and sd[-1] > 3 * sd[inside].mean()

# %% [markdown]
# ## 6. Conformal prediction

# %%
def make(n, shift=0.0):
    x_ = rng.uniform(0, 10, n) + shift
    return x_, np.sin(x_) * 2 + rng.normal(0, 0.2 + 0.15 * x_)           # noise grows with x


x_a, y_a = make(2000); x_c, y_c = make(1000); x_t, y_t = make(20_000)
model = GradientBoostingRegressor(max_depth=3, n_estimators=200, random_state=0).fit(x_a[:, None], y_a)
alpha = 0.1
scores = np.abs(y_c - model.predict(x_c[:, None]))
qhat = np.quantile(scores, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
pred_t = model.predict(x_t[:, None])
cov_conf = np.mean(np.abs(y_t - pred_t) <= qhat)
naive = 1.645 * np.std(y_a - model.predict(x_a[:, None]))       # "use the training residuals": too optimistic
cov_naive = np.mean(np.abs(y_t - pred_t) <= naive)
print(f"90% intervals: split conformal coverage {cov_conf:.3f} (constant width {2 * qhat:.2f}); training-residual interval coverage {cov_naive:.3f}")

lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, max_depth=3, n_estimators=200, random_state=0).fit(x_a[:, None], y_a)
hi = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, max_depth=3, n_estimators=200, random_state=0).fit(x_a[:, None], y_a)
s_cqr = np.maximum(lo.predict(x_c[:, None]) - y_c, y_c - hi.predict(x_c[:, None]))
q_cqr = np.quantile(s_cqr, np.ceil((len(s_cqr) + 1) * (1 - alpha)) / len(s_cqr))
L_t, U_t = lo.predict(x_t[:, None]) - q_cqr, hi.predict(x_t[:, None]) + q_cqr
cov_cqr = np.mean((y_t >= L_t) & (y_t <= U_t))
w_small, w_large = np.mean((U_t - L_t)[x_t < 2]), np.mean((U_t - L_t)[x_t > 8])
print(f"CQR coverage {cov_cqr:.3f}; width for x < 2: {w_small:.2f}, for x > 8: {w_large:.2f} (adapts to the noise)")
cond_small = np.mean(np.abs(y_t - pred_t)[x_t < 2] <= qhat); cond_large = np.mean(np.abs(y_t - pred_t)[x_t > 8] <= qhat)
print(f"plain split conformal is only right on average: coverage for x < 2 {cond_small:.3f}, for x > 8 {cond_large:.3f}")
assert abs(cov_conf - 0.9) < 0.02 and abs(cov_cqr - 0.9) < 0.02 and cov_naive < cov_conf and w_large > 2 * w_small

x_s, y_s = make(20_000, shift=3.0)                              # the future drifts to larger x: exchangeability is gone
cov_shift = np.mean(np.abs(y_s - model.predict(x_s[:, None])) <= qhat)
print(f"under distribution shift (x moved up by 3): split conformal coverage {cov_shift:.3f}. The guarantee needed exchangeability.")
assert cov_shift < 0.85

print("\nAll checks passed.")
