# %% [markdown]
# # Lab 18.3: Calibration, thresholds and costs
#
# 1. Reliability of four model families on the same data; ECE; a saved reliability diagram.
# 2. The Brier decomposition, verified.
# 3. Recalibration (sigmoid, isotonic) on held-out data: log loss improves, ranking doesn't move much.
# 4. Class weights break calibration; recalibration repairs it.
# 5. Cost-optimal thresholds: the formula vs a validation sweep vs 0.5.
# 6. Prior shift: train at 30% positives, deploy at 5%, correct with Bayes.
# 7. The reject option: coverage vs accuracy.
# 8. Temperature scaling for an overconfident multiclass network.

# %%
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

rng = np.random.default_rng(183)
OUT = Path(__file__).with_name("outputs")
OUT.mkdir(exist_ok=True)

X, y = make_classification(n_samples=30_000, n_features=20, n_informative=6, n_redundant=6, class_sep=0.8,
                           flip_y=0.03, random_state=0)
X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=10_000, random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_rest, y_rest, train_size=10_000, random_state=0)

# %% [markdown]
# ## 1. Reliability by model family

# %%
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum(np.sum(idx == b) / len(p) * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if np.any(idx == b))


models = {
    "logistic": LogisticRegression(max_iter=1000),
    "naive Bayes": GaussianNB(),
    "random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1),
    "SVM + Platt (CV)": CalibratedClassifierCV(SVC(), method="sigmoid", cv=5, ensemble=False),
}
probs = {}
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
ax.plot([0, 1], [0, 1], "k--", lw=1)
for name, m in models.items():
    p = m.fit(X_tr[:5000], y_tr[:5000]).predict_proba(X_te)[:, 1]
    probs[name] = (m, p)
    frac, mean_pred = calibration_curve(y_te, p, n_bins=10)
    ax.plot(mean_pred, frac, "o-", label=name)
    print(f"{name:25s} AUC {roc_auc_score(y_te, p):.3f}  log loss {log_loss(y_te, p):.3f}  ECE {ece(y_te, p):.3f}")
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed positive rate"); ax.legend(fontsize=8)
fig.savefig(OUT / "reliability.png", dpi=120); plt.close(fig)

p_nb = probs["naive Bayes"][1]
print(f"naive Bayes puts {np.mean((p_nb < 0.02) | (p_nb > 0.98)):.0%} of predictions beyond 0.02/0.98; "
      f"the forest puts {np.mean((probs['random forest'][1] < 0.02) | (probs['random forest'][1] > 0.98)):.0%}")
assert ece(y_te, p_nb) > 2 * ece(y_te, probs["logistic"][1])

# %% [markdown]
# ## 2. Brier = reliability - resolution + uncertainty
#
# Exact when predictions are grouped by their distinct values; we bin into 20 groups by rounding, and compute the decomposition on the binned forecasts.

# %%
p = np.round(probs["random forest"][1] * 20) / 20
base = y_te.mean()
rel = res = 0.0
for v in np.unique(p):
    m = p == v
    w = m.mean()
    rel += w * (v - y_te[m].mean()) ** 2
    res += w * (y_te[m].mean() - base) ** 2
unc = base * (1 - base)
print(f"Brier {brier_score_loss(y_te, p):.5f} = reliability {rel:.5f} - resolution {res:.5f} + uncertainty {unc:.5f} = {rel - res + unc:.5f}")
assert np.isclose(brier_score_loss(y_te, p), rel - res + unc)

# %% [markdown]
# ## 3. Recalibration on held-out data

# %%
nb = GaussianNB().fit(X_tr, y_tr)
p_raw = nb.predict_proba(X_te)[:, 1]
eces = {}
for method in ("sigmoid", "isotonic"):
    cal = CalibratedClassifierCV(GaussianNB(), method=method, cv=5).fit(np.vstack([X_tr, X_cal]), np.r_[y_tr, y_cal])
    p_c = cal.predict_proba(X_te)[:, 1]
    eces[method] = ece(y_te, p_c)
    print(f"naive Bayes + {method:8s}: log loss {log_loss(y_te, p_raw):.3f} -> {log_loss(y_te, p_c):.3f}; "
          f"ECE {ece(y_te, p_raw):.3f} -> {eces[method]:.3f}; AUC {roc_auc_score(y_te, p_raw):.3f} -> {roc_auc_score(y_te, p_c):.3f}")
    assert log_loss(y_te, p_c) < log_loss(y_te, p_raw) - 0.05
# scikit-learn fits the sigmoid to NB's *probabilities*, which are piled up at 0 and 1; a sigmoid of a saturated score
# can't undo that. Isotonic has no shape assumption. (Platt on the log-odds does better; next cell uses it.)
print("sigmoid helps, isotonic helps more: the distortion here isn't sigmoid-shaped")
assert eces["isotonic"] < 0.03 and eces["isotonic"] < eces["sigmoid"]

# calibrating on the training data does nothing useful: the model's training scores share its overconfidence
platt_train = LogisticRegression().fit(nb.predict_log_proba(X_tr)[:, [1]] - nb.predict_log_proba(X_tr)[:, [0]], y_tr)
platt_cal = LogisticRegression().fit(nb.predict_log_proba(X_cal)[:, [1]] - nb.predict_log_proba(X_cal)[:, [0]], y_cal)
logit_te = nb.predict_log_proba(X_te)[:, [1]] - nb.predict_log_proba(X_te)[:, [0]]
print(f"Platt fitted on training data: log loss {log_loss(y_te, platt_train.predict_proba(logit_te)[:, 1]):.4f}; "
      f"on a separate calibration set: {log_loss(y_te, platt_cal.predict_proba(logit_te)[:, 1]):.4f}")
# (for a simple model like NB the difference is small; for a model that overfits, like a deep tree, it's large)
tree_like = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
s_tr, s_cal, s_te = (tree_like.predict_proba(A)[:, [1]] for A in (X_tr, X_cal, X_te))
ll_bad = log_loss(y_te, LogisticRegression().fit(s_tr, y_tr).predict_proba(s_te)[:, 1])
ll_good = log_loss(y_te, LogisticRegression().fit(s_cal, y_cal).predict_proba(s_te)[:, 1])
print(f"fully grown forest, Platt on training scores: log loss {ll_bad:.3f}; on held-out scores: {ll_good:.3f}")
assert ll_good < ll_bad

# %% [markdown]
# ## 4. Class weights and calibration

# %%
Xi, yi = make_classification(n_samples=40_000, n_features=10, n_informative=5, weights=[0.95], flip_y=0.01, random_state=1)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.5, random_state=1)
plain = LogisticRegression(max_iter=1000).fit(Xi_tr, yi_tr).predict_proba(Xi_te)[:, 1]
weighted = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xi_tr, yi_tr).predict_proba(Xi_te)[:, 1]
print(f"prevalence {yi_te.mean():.3f}; mean prediction plain {plain.mean():.3f}, class_weight='balanced' {weighted.mean():.3f}")
print(f"log loss plain {log_loss(yi_te, plain):.3f}, balanced {log_loss(yi_te, weighted):.3f}; AUC {roc_auc_score(yi_te, plain):.3f} vs {roc_auc_score(yi_te, weighted):.3f}")
assert weighted.mean() > 3 * yi_te.mean() and abs(roc_auc_score(yi_te, plain) - roc_auc_score(yi_te, weighted)) < 0.01

# %% [markdown]
# ## 5. Thresholds from costs

# %%
C_FP, C_FN = 1.0, 20.0
t_star = C_FP / (C_FP + C_FN)
cal_lr = LogisticRegression(max_iter=1000).fit(Xi_tr[:10_000], yi_tr[:10_000])
p_val, y_val = cal_lr.predict_proba(Xi_tr[10_000:])[:, 1], yi_tr[10_000:]
p_test = cal_lr.predict_proba(Xi_te)[:, 1]


def cost(y, p, t):
    act = p > t
    return (C_FP * np.sum(act & (y == 0)) + C_FN * np.sum(~act & (y == 1))) / len(y)


grid = np.linspace(0.005, 0.9, 400)
t_val = grid[np.argmin([cost(y_val, p_val, t) for t in grid])]
print(f"analytic threshold {t_star:.3f}; best on validation {t_val:.3f}")
print(f"test cost per case: t=0.5 {cost(yi_te, p_test, 0.5):.4f}, t*={t_star:.3f} {cost(yi_te, p_test, t_star):.4f}, "
      f"validation-chosen {cost(yi_te, p_test, t_val):.4f}, act on nobody {C_FN * yi_te.mean():.4f}")
assert cost(yi_te, p_test, t_star) < 0.7 * cost(yi_te, p_test, 0.5)
assert abs(np.log(t_val / t_star)) < np.log(3), "with calibrated probabilities the theory and the sweep agree roughly"

# %% [markdown]
# ## 6. Prior shift

# %%
def sample(n, prev):
    npos = rng.binomial(n, prev)
    Xp = rng.normal(1.0, 1, size=(npos, 3)); Xn = rng.normal(0, 1, size=(n - npos, 3))
    return np.vstack([Xp, Xn]), np.r_[np.ones(npos), np.zeros(n - npos)]


Xs_tr, ys_tr = sample(20_000, 0.30)
Xs_dep, ys_dep = sample(50_000, 0.05)
m6 = LogisticRegression().fit(Xs_tr, ys_tr)
p_old = m6.predict_proba(Xs_dep)[:, 1]
pi_old, pi_new = 0.30, 0.05
odds = p_old / (1 - p_old) * (pi_new / (1 - pi_new)) / (pi_old / (1 - pi_old))
p_new = odds / (1 + odds)
print(f"deployment prevalence {ys_dep.mean():.3f}; mean prediction raw {p_old.mean():.3f}, prior-corrected {p_new.mean():.3f}")
print(f"log loss raw {log_loss(ys_dep, p_old):.3f}, corrected {log_loss(ys_dep, p_new):.3f}")
assert abs(p_new.mean() - ys_dep.mean()) < 0.01 and log_loss(ys_dep, p_new) < log_loss(ys_dep, p_old)

# %% [markdown]
# ## 7. The reject option

# %%
p7 = probs["logistic"][1]
conf = np.abs(p7 - 0.5)
for coverage in (1.0, 0.9, 0.7, 0.5):
    keep = conf >= np.quantile(conf, 1 - coverage)
    print(f"automate {coverage:.0%} of cases: accuracy on those {accuracy_score(y_te[keep], p7[keep] > 0.5):.3f}")
keep50 = conf >= np.quantile(conf, 0.5)
assert accuracy_score(y_te[keep50], p7[keep50] > 0.5) > accuracy_score(y_te, p7 > 0.5) + 0.05

# %% [markdown]
# ## 8. Temperature scaling
#
# A two-hidden-layer network with no regularization, trained to 100% training accuracy on noisy 5-class data: overconfident.
# (On clean, easy data like the digits, the same network is barely miscalibrated; overconfidence shows up where it makes mistakes.)

# %%
Xm, ym = make_classification(n_samples=6000, n_features=20, n_informative=10, n_classes=5, n_clusters_per_class=1,
                             flip_y=0.05, class_sep=1.0, random_state=3)
Xm_tr, Xm_tmp, ym_tr, ym_tmp = train_test_split(Xm, ym, train_size=2000, random_state=0)
Xm_val, Xm_te, ym_val, ym_te = train_test_split(Xm_tmp, ym_tmp, test_size=0.5, random_state=0)
mlp = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=0.0, max_iter=500, random_state=0).fit(Xm_tr, ym_tr)


def logits(model, X):
    h = X
    for W, b in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        h = np.maximum(h @ W + b, 0)
    return h @ model.coefs_[-1] + model.intercepts_[-1]


def nll(Z, y, T):
    Z = Z / T
    Z = Z - Z.max(1, keepdims=True)
    return -(Z[np.arange(len(y)), y] - np.log(np.exp(Z).sum(1))).mean()


Z_val, Z_te = logits(mlp, Xm_val), logits(mlp, Xm_te)
P_te = np.exp(Z_te - Z_te.max(1, keepdims=True)); P_te /= P_te.sum(1, keepdims=True)
assert np.allclose(P_te, mlp.predict_proba(Xm_te), atol=1e-6), "our logits reproduce the network's probabilities"
T = minimize_scalar(lambda T: nll(Z_val, ym_val, T), bounds=(0.05, 50), method="bounded").x
acc_before, acc_after = np.mean(Z_te.argmax(1) == ym_te), np.mean((Z_te / T).argmax(1) == ym_te)
print(f"training accuracy {mlp.score(Xm_tr, ym_tr):.3f}, test accuracy {acc_before:.3f}, mean confidence {P_te.max(1).mean():.3f}")
print(f"fitted temperature {T:.2f}; test NLL {nll(Z_te, ym_te, 1.0):.3f} -> {nll(Z_te, ym_te, T):.3f}; accuracy {acc_before:.3f} -> {acc_after:.3f}")
assert P_te.max(1).mean() > acc_before + 0.03, "confidence exceeds accuracy: overconfident"
assert T > 1.5 and nll(Z_te, ym_te, T) < 0.8 * nll(Z_te, ym_te, 1.0) and acc_before == acc_after

print("\nfigure saved to", OUT / "reliability.png")
print("All checks passed.")
