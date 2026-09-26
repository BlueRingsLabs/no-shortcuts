# %% [markdown]
# # Lab 21.2: Gaussian mixtures and EM
#
# 1. EM from scratch: the log-likelihood never decreases; the ELBO sits below it and touches it after each E-step.
# 2. Same fit as scikit-learn.
# 3. Elongated clusters: k-means vs GMM.
# 4. k-means is the hard, spherical limit of EM.
# 5. A collapsing component, and reg_covar.
# 6. BIC picks the number of components and the covariance type.
# 7. Density for anomaly detection, and sampling new data.

# %%
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(212)

# three elongated, rotated clusters with different weights
def rotated(n, mean, angle, sds):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rng.normal(size=(n, 2)) * sds @ R.T + mean


X = np.vstack([rotated(600, [0, 0], 0.6, [2.5, 0.35]), rotated(300, [1.5, 3.0], -0.4, [1.8, 0.3]), rotated(150, [5, -1], 1.2, [0.6, 0.6])])
y = np.r_[np.zeros(600), np.ones(300), 2 * np.ones(150)].astype(int)

# %% [markdown]
# ## 1. EM from scratch, with the ELBO checked every iteration

# %%
def log_joint(X, pi, mu, Sigma):                                # log pi_k + log N(x | mu_k, Sigma_k), shape (n, K)
    return np.column_stack([np.log(pi[k]) + multivariate_normal(mu[k], Sigma[k]).logpdf(X) for k in range(len(pi))])


def em(X, K, iters=200, reg=1e-6, seed=0):
    n, d = X.shape
    km = KMeans(K, n_init=1, random_state=seed).fit(X)          # k-means init, as scikit-learn does
    mu = km.cluster_centers_.copy()
    Sigma = np.array([np.cov(X[km.labels_ == k].T) + reg * np.eye(d) for k in range(K)])
    pi = np.bincount(km.labels_, minlength=K) / n
    trace = []
    for _ in range(iters):
        LJ = log_joint(X, pi, mu, Sigma)
        loglik = logsumexp(LJ, axis=1).sum()
        gamma = np.exp(LJ - logsumexp(LJ, axis=1, keepdims=True))            # E-step
        elbo_after_E = np.sum(gamma * (LJ - np.log(np.clip(gamma, 1e-300, None))))
        Nk = gamma.sum(0)                                                    # M-step
        pi = Nk / n
        mu = (gamma.T @ X) / Nk[:, None]
        Sigma = np.array([((gamma[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / Nk[k] + reg * np.eye(d) for k in range(K)])
        LJ_new = log_joint(X, pi, mu, Sigma)
        elbo_after_M = np.sum(gamma * (LJ_new - np.log(np.clip(gamma, 1e-300, None))))   # old q, new theta
        trace.append((loglik, elbo_after_E, elbo_after_M))
        if len(trace) > 1 and abs(trace[-1][0] - trace[-2][0]) < 1e-8:
            break
    return pi, mu, Sigma, gamma, np.array(trace)


pi, mu, Sigma, gamma, trace = em(X, 3)
ll = trace[:, 0]
print(f"EM converged in {len(trace)} iterations; log-likelihood {ll[0]:.1f} -> {ll[-1]:.1f}")
assert np.all(np.diff(ll) >= -1e-6), "the log-likelihood never decreases"
assert np.allclose(trace[:, 1], trace[:, 0], atol=1e-6), "after the E-step the ELBO equals the log-likelihood"
assert np.all(trace[:, 2] >= trace[:, 1] - 1e-6), "the M-step raises the ELBO"
assert np.all(trace[1:, 0] >= trace[:-1, 2] - 1e-6), "and the new log-likelihood is at least the raised ELBO"
print("ELBO checks: equal after E, raised by M, below the next log-likelihood. All iterations.")

# %% [markdown]
# ## 2. Against scikit-learn

# %%
sk = GaussianMixture(3, covariance_type="full", reg_covar=1e-6, n_init=1, init_params="kmeans", random_state=0, tol=1e-10, max_iter=500).fit(X)
print(f"average log-likelihood per point: mine {ll[-1] / len(X):.6f}, scikit-learn {sk.score(X):.6f}")
assert abs(ll[-1] / len(X) - sk.score(X)) < 1e-4
print(f"ARI vs the true components: {adjusted_rand_score(y, gamma.argmax(1)):.3f}")

# %% [markdown]
# ## 3. Elongated clusters: k-means vs GMM

# %%
ari_km = adjusted_rand_score(y, KMeans(3, n_init=10, random_state=0).fit_predict(X))
ari_gmm = adjusted_rand_score(y, GaussianMixture(3, n_init=5, random_state=0).fit_predict(X))
print(f"ARI: k-means {ari_km:.3f}, GMM {ari_gmm:.3f}")
assert ari_gmm > ari_km + 0.15
uncertain = np.mean(gamma.max(1) < 0.8)
print(f"{uncertain:.1%} of points have no component with responsibility >= 0.8 (the honest 'in between' points k-means hides)")

# %% [markdown]
# ## 4. k-means as the hard limit of EM

# %%
Xb = np.vstack([rng.normal(c, 0.5, size=(200, 2)) for c in ([0, 0], [4, 0], [2, 3])])
km = KMeans(3, n_init=10, random_state=0).fit(Xb)
for sigma in (2.0, 0.5, 0.05):
    d2 = ((Xb[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    g = np.exp(-d2 / (2 * sigma**2) - logsumexp(-d2 / (2 * sigma**2), axis=1, keepdims=True))
    print(f"shared spherical sigma={sigma:4}: mean max responsibility {g.max(1).mean():.3f}, agreement with k-means {np.mean(g.argmax(1) == km.labels_):.3f}")
assert g.max(1).mean() > 0.999

# %% [markdown]
# ## 5. Collapse
#
# Six identical rows (a data glitch) and a component that starts on them. First the math: with that component's
# covariance eps * I, the likelihood grows without bound as eps -> 0. Then EM, with and without reg_covar.

# %%
Xc = np.vstack([rng.normal(0, 1, size=(200, 2)), np.repeat([[3.0, 3.0]], 6, axis=0)])
base_mu, base_S = Xc[:200].mean(0), np.cov(Xc[:200].T)
for eps in (1e-1, 1e-3, 1e-6, 1e-9):
    LJ = log_joint(Xc, np.array([0.97, 0.03]), [base_mu, np.array([3.0, 3.0])], [base_S, eps * np.eye(2)])
    print(f"component on the duplicates with variance {eps:.0e}: log-likelihood {logsumexp(LJ, axis=1).sum():10.1f}")


def em_from(X, pi, mu, Sigma, reg, iters=60):
    for _ in range(iters):
        try:
            LJ = log_joint(X, pi, mu, Sigma)
        except np.linalg.LinAlgError:                           # the covariance became exactly singular: fully collapsed
            return Sigma, np.inf, True
        gamma = np.exp(LJ - logsumexp(LJ, axis=1, keepdims=True))
        Nk = gamma.sum(0); pi = Nk / len(X); mu = (gamma.T @ X) / Nk[:, None]
        Sigma = np.array([((gamma[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / Nk[k] + reg * np.eye(2) for k in range(len(pi))])
        if np.linalg.det(Sigma[1]) < 1e-20:
            return Sigma, logsumexp(log_joint(X, pi, mu, Sigma), axis=1).sum(), True
    return Sigma, logsumexp(log_joint(X, pi, mu, Sigma), axis=1).sum(), False


start = (np.array([0.5, 0.5]), [base_mu, np.array([3.0, 3.0])], [base_S, 0.5 * np.eye(2)])
S0, ll0, collapsed = em_from(Xc, *start, reg=0.0)
S1, ll1, collapsed_reg = em_from(Xc, *start, reg=1e-3)
print(f"EM, reg_covar=0:    component 2 determinant {np.linalg.det(S0[1]):.1e}, log-likelihood {ll0:.1f}  (collapsed: {collapsed})")
print(f"EM, reg_covar=1e-3: component 2 determinant {np.linalg.det(S1[1]):.1e}, log-likelihood {ll1:.1f}  (collapsed: {collapsed_reg})")
print("the 'best' likelihood is a spike on six duplicate rows: a bug report, not a model")
assert collapsed and not collapsed_reg and ll0 > ll1

# %% [markdown]
# ## 6. BIC

# %%
table = {}
for cov in ("spherical", "diag", "full"):
    for K in range(1, 7):
        table[(cov, K)] = GaussianMixture(K, covariance_type=cov, n_init=3, random_state=0).fit(X).bic(X)
best = min(table, key=table.get)
for cov in ("spherical", "diag", "full"):
    print(f"{cov:9s} BIC: " + " ".join(f"K={K}:{table[(cov, K)]:.0f}" for K in range(1, 7)))
print(f"lowest BIC: covariance_type={best[0]}, K={best[1]}")
assert best == ("full", 3)
print(f"spherical's BIC is still falling at K={min(range(1, 7), key=lambda K: table[('spherical', K)])}, the largest tried: "
      "it keeps adding round blobs to tile 3 elongated clusters")

# %% [markdown]
# ## 7. Density: anomalies and sampling

# %%
gm = GaussianMixture(3, random_state=0).fit(X)
outliers = rng.uniform(X.min(0) - 1, X.max(0) + 1, size=(60, 2))
scores = -gm.score_samples(np.vstack([X, outliers]))            # negative log-density = anomaly score
labels = np.r_[np.zeros(len(X)), np.ones(60)]
print(f"anomaly detection by low density: ROC-AUC {roc_auc_score(labels, scores):.3f}")
assert roc_auc_score(labels, scores) > 0.9

samples, comp = gm.sample(3000)
print(f"sampled 3000 new points; component shares {np.round(np.bincount(comp) / 3000, 3)} vs fitted weights {np.round(gm.weights_, 3)}")
print(f"mean log-density of samples under the model {gm.score(samples):.3f} vs real data {gm.score(X):.3f}")
assert abs(gm.score(samples) - gm.score(X)) < 0.1

print("\nAll checks passed.")
