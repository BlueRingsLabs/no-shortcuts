# %% [markdown]
# # Lab 19.2: Naive Bayes and generative classifiers
#
# 1. Multinomial naive Bayes from scratch on a synthetic text corpus, matched to scikit-learn.
# 2. Smoothing: one unseen word vetoes everything without it.
# 3. Gaussian NB and double-counted evidence: duplicate the features, watch the probabilities.
# 4. LDA from scratch: pooled covariance, a linear (logistic) posterior, matched to scikit-learn.
# 5. LDA vs QDA when the covariances differ.
# 6. Ng and Jordan: naive Bayes vs logistic regression as the training set grows.
# 7. Generative perks: missing features and a new class prior, with no retraining.

# %%
import numpy as np
from scipy.special import logsumexp
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

rng = np.random.default_rng(192)

# %% [markdown]
# ## 1. Multinomial naive Bayes on a synthetic corpus
#
# Vocabulary of 500 words. "Spam" and "ham" draw words from different (but overlapping) distributions; documents are word counts.

# %%
V = 500
base = rng.dirichlet(np.full(V, 0.3))
spam_words = base.copy(); spam_words[:25] *= 3; spam_words /= spam_words.sum()   # 25 words much more common in spam
ham_words = base.copy(); ham_words[25:50] *= 3; ham_words /= ham_words.sum()


def corpus(n, p_spam=0.4):
    y = (rng.random(n) < p_spam).astype(int)
    lengths = rng.poisson(15, n) + 3
    X = np.vstack([rng.multinomial(L, spam_words if c else ham_words) for L, c in zip(lengths, y)])
    return X, y


X_tr, y_tr = corpus(1500)
X_te, y_te = corpus(3000)


class MyMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        counts = np.vstack([X[y == c].sum(0) for c in self.classes_]) + self.alpha
        self.log_theta = np.log(counts / counts.sum(1, keepdims=True))
        self.log_prior = np.log(np.array([np.mean(y == c) for c in self.classes_]))
        return self

    def joint_log(self, X):
        finite = np.isfinite(self.log_theta)                    # only matters with alpha = 0 (section 2)
        J = self.log_prior + X @ np.where(finite, self.log_theta, 0.0).T   # log P(c) + sum_j x_j log theta_cj
        J[((X > 0).astype(int) @ (~finite).astype(int).T) > 0] = -np.inf   # a word the class never produced: log 0
        return J

    def predict_proba(self, X):
        J = self.joint_log(X)
        return np.exp(J - logsumexp(J, axis=1, keepdims=True))


mine = MyMultinomialNB(alpha=1.0).fit(X_tr, y_tr)
sk = MultinomialNB(alpha=1.0).fit(X_tr, y_tr)
assert np.allclose(mine.predict_proba(X_te), sk.predict_proba(X_te))
print(f"multinomial NB accuracy {accuracy_score(y_te, mine.predict_proba(X_te).argmax(1)):.3f} (matches scikit-learn)")
ratio = mine.log_theta[1] - mine.log_theta[0]
top_raw = np.argsort(ratio)[-10:]
frequent = X_tr.sum(0) >= 30
top_freq = np.flatnonzero(frequent)[np.argsort(ratio[frequent])[-10:]]
print("most spam-indicative word ids, all words:        ", sorted(top_raw.tolist()), "(the planted ones are 0-24)")
print("most spam-indicative word ids, seen >= 30 times: ", sorted(top_freq.tolist()))
print("rare words have wildly noisy ratios: a word seen twice, both times in spam, looks like the best spam signal there is")
assert np.mean(top_freq < 25) >= 0.8 and np.mean(top_raw < 25) < np.mean(top_freq < 25)

# %% [markdown]
# ## 2. Smoothing

# %%
with np.errstate(divide="ignore"):
    nb0 = MyMultinomialNB(alpha=0.0).fit(X_tr, y_tr)            # no smoothing: unseen words get log(0) = -inf
in_spam = X_tr[y_tr == 1].sum(0) > 0
in_ham = X_tr[y_tr == 0].sum(0) > 0
spam_only = np.flatnonzero(in_spam & ~in_ham)
print(f"{len(spam_only)} words appeared in training spam but never in training ham")
ham_docs = X_te[y_te == 0]
clean = ham_docs[((ham_docs > 0) <= (in_spam & in_ham)).all(1)]  # legitimate emails using only words seen in both classes
doc = clean[np.argmax(mine.predict_proba(clean)[:, 0])].copy()   # the most confidently legitimate of them
p_before = mine.predict_proba(doc[None])[0, 1]
doc[spam_only[0]] += 1                                          # add one word ham never used in training
with np.errstate(invalid="ignore"):
    p_unsmoothed = nb0.predict_proba(doc[None])[0, 1]
p_smoothed = mine.predict_proba(doc[None])[0, 1]
print(f"a clearly legitimate email (P(spam) = {p_before:.4f}) plus one unseen-in-ham word: "
      f"unsmoothed P(spam) = {p_unsmoothed:.4f}, alpha=1 P(spam) = {p_smoothed:.4f}")
assert p_unsmoothed == 1.0 and p_smoothed < 0.5, "without smoothing, one word vetoes all the other evidence"

for a in (0.01, 0.1, 1.0, 10.0):
    print(f"alpha={a:5}: accuracy {MultinomialNB(alpha=a).fit(X_tr, y_tr).score(X_te, y_te):.3f}")

# %% [markdown]
# ## 3. Gaussian NB double-counts correlated evidence

# %%
bc = load_breast_cancer()
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(bc.data, bc.target, test_size=0.4, random_state=0, stratify=bc.target)
for copies in (1, 3, 10):
    g = GaussianNB().fit(np.tile(Xb_tr, copies), yb_tr)
    p = g.predict_proba(np.tile(Xb_te, copies))[:, 1]
    print(f"features x{copies:2d}: accuracy {accuracy_score(yb_te, p > 0.5):.3f}, "
          f"share of predictions beyond 1e-6/1-1e-6 {np.mean((p < 1e-6) | (p > 1 - 1e-6)):.0%}, log loss {log_loss(yb_te, np.clip(p, 1e-15, 1 - 1e-15)):.3f}")
g1 = GaussianNB().fit(Xb_tr, yb_tr); g10 = GaussianNB().fit(np.tile(Xb_tr, 10), yb_tr)
jl1 = g1.predict_joint_log_proba(Xb_te); jl10 = g10.predict_joint_log_proba(np.tile(Xb_te, 10))
lo1 = (jl1[:, 1] - jl1[:, 0]) - np.log(g1.class_prior_[1] / g1.class_prior_[0])
lo10 = (jl10[:, 1] - jl10[:, 0]) - np.log(g10.class_prior_[1] / g10.class_prior_[0])
assert np.allclose(lo10, 10 * lo1, rtol=1e-3, atol=1e-3), "the evidence part of the log-odds scales exactly with the number of copies"
print("log-odds evidence x10 with 10 copies: confirmed")

# %% [markdown]
# ## 4. LDA from scratch

# %%
def lda_fit(X, y):
    classes = np.unique(y)
    mus = np.array([X[y == c].mean(0) for c in classes])
    resid = X - mus[np.searchsorted(classes, y)]
    Sigma = resid.T @ resid / len(X)                            # pooled within-class covariance (MLE; n - K would be unbiased)
    priors = np.array([np.mean(y == c) for c in classes])
    return mus, Sigma, priors


cov_true = np.array([[1.0, 0.6], [0.6, 1.5]])
Xg = np.vstack([rng.multivariate_normal([0, 0], cov_true, 3000), rng.multivariate_normal([1.5, 0.5], cov_true, 3000)])
yg = np.r_[np.zeros(3000, int), np.ones(3000, int)]
mus, Sigma, priors = lda_fit(Xg, yg)
w = np.linalg.solve(Sigma, mus[1] - mus[0])
b = -0.5 * (mus[1] @ np.linalg.solve(Sigma, mus[1]) - mus[0] @ np.linalg.solve(Sigma, mus[0])) + np.log(priors[1] / priors[0])
p_mine = 1 / (1 + np.exp(-(Xg @ w + b)))                        # the LDA posterior IS a logistic function
p_sk = LinearDiscriminantAnalysis(solver="lsqr").fit(Xg, yg).predict_proba(Xg)[:, 1]
assert np.allclose(p_mine, p_sk, atol=1e-6)
lr = LogisticRegression(C=1e6).fit(Xg, yg)
print(f"LDA weights {np.round(w, 3)}, intercept {b:.3f}; logistic regression {np.round(lr.coef_[0], 3)}, {lr.intercept_[0]:.3f}")
print("(same functional form, different fitting criteria; on Gaussian data they agree)")
assert np.allclose(w, lr.coef_[0], atol=0.15)

# %% [markdown]
# ## 5. When covariances differ: QDA

# %%
Xq = np.vstack([rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], 2000), rng.multivariate_normal([0, 0], [[4, 0], [0, 0.25]], 2000)])
yq = np.r_[np.zeros(2000, int), np.ones(2000, int)]
Xq_tr, Xq_te, yq_tr, yq_te = train_test_split(Xq, yq, test_size=0.5, random_state=0)
acc_lda = LinearDiscriminantAnalysis().fit(Xq_tr, yq_tr).score(Xq_te, yq_te)
acc_qda = QuadraticDiscriminantAnalysis().fit(Xq_tr, yq_tr).score(Xq_te, yq_te)
print(f"same means, different covariances: LDA {acc_lda:.3f} (no linear boundary exists), QDA {acc_qda:.3f}")
assert acc_lda < 0.6 and acc_qda > acc_lda + 0.15

# %% [markdown]
# ## 6. Ng and Jordan: who wins depends on n (and on regularization)
#
# 50 features, mildly correlated within each class (so naive Bayes' assumption is wrong, but not wildly).
# Ng and Jordan compared naive Bayes with *unregularized* logistic regression; we add a regularized one too.

# %%
d = 50
A = rng.normal(size=(d, d)) / np.sqrt(d)
cov_c = np.eye(d) + 0.3 * A @ A.T
shift = rng.normal(0, 0.3, d)


def draw(n):
    y = rng.integers(0, 2, n)
    X = rng.multivariate_normal(np.zeros(d), cov_c, n) + y[:, None] * shift
    return X, y


X_big, y_big = draw(20_000)
rows = []
for n in (100, 300, 1000, 5000):
    e = {"nb": [], "lr_unreg": [], "lr_reg": []}
    for rep in range(10):
        Xn, yn = draw(n)
        e["nb"].append(1 - GaussianNB().fit(Xn, yn).score(X_big, y_big))
        e["lr_unreg"].append(1 - LogisticRegression(C=1e4, max_iter=5000).fit(Xn, yn).score(X_big, y_big))
        e["lr_reg"].append(1 - LogisticRegression(C=0.1, max_iter=5000).fit(Xn, yn).score(X_big, y_big))
    m = {k: np.mean(v) for k, v in e.items()}
    rows.append((n, m["nb"], m["lr_unreg"], m["lr_reg"]))
    print(f"n={n:5d}: error naive Bayes {m['nb']:.3f}, unregularized logistic {m['lr_unreg']:.3f}, regularized logistic {m['lr_reg']:.3f}")
rows = np.array(rows)
assert rows[0, 1] < rows[0, 2], "n ~ 2d: the generative model beats the unregularized discriminative one"
assert rows[-1, 2] < rows[-1, 1], "large n: the discriminative model wins"
assert (rows[:, 3] <= rows[:, 1] + 0.01).all(), "a regularized discriminative model erases most of the small-sample gap"

# %% [markdown]
# ## 7. Missing features and a new prior, without retraining

# %%
g = GaussianNB().fit(Xb_tr, yb_tr)
missing = np.array([0, 2, 3, 20, 22, 23])                       # the radius/perimeter/area features are unavailable


def nb_marginal_proba(model, X, observed, prior=None):
    prior = model.class_prior_ if prior is None else prior
    ll = np.stack([-0.5 * (np.log(2 * np.pi * model.var_[c, observed]) + (X[:, observed] - model.theta_[c, observed]) ** 2
                           / model.var_[c, observed]).sum(1) for c in range(2)], axis=1)
    J = np.log(prior) + ll
    return np.exp(J - logsumexp(J, axis=1, keepdims=True))


observed = np.setdiff1d(np.arange(Xb_te.shape[1]), missing)
acc_all = accuracy_score(yb_te, nb_marginal_proba(g, Xb_te, np.arange(30)).argmax(1))
acc_marg = accuracy_score(yb_te, nb_marginal_proba(g, Xb_te, observed).argmax(1))
assert np.isclose(acc_all, g.score(Xb_te, yb_te))
print(f"NB with all features {acc_all:.3f}; with 6 features missing, marginalized out: {acc_marg:.3f} (same model, no imputation)")
assert acc_marg > 0.85

new_prior = np.array([0.9, 0.1])                                # deployed where malignant is 90% (class 0) ... just change P(y)
p_new = nb_marginal_proba(g, Xb_te, np.arange(30), prior=new_prior)
print(f"mean P(class 0): with the training prior {nb_marginal_proba(g, Xb_te, np.arange(30))[:, 0].mean():.3f}, "
      f"with the new prior {p_new[:, 0].mean():.3f}")
assert p_new[:, 0].mean() > nb_marginal_proba(g, Xb_te, np.arange(30))[:, 0].mean()
# it barely moves: section 3 showed these probabilities are already jammed against 0 and 1, and a prior shift of a few nats
# can't move a log-odds of hundreds. Changing the prior is exact for the model; it's only as useful as the model's calibration.

print("\nAll checks passed.")
