# %% [markdown]
# # Lab 18.2: Metrics, and how each one lies
#
# 1. Confusion-matrix metrics from scratch, checked against scikit-learn; precision moves with the base rate.
# 2. ROC-AUC is the Mann-Whitney probability; computed three ways.
# 3. The self-check: great ROC-AUC, useless alerts, at 0.1% prevalence. PR curves tell the truth.
# 4. Proper scoring rules: a reckless model wins on accuracy and loses on log loss and Brier.
# 5. Macro vs micro vs weighted.
# 6. Regression: which constant each loss prefers; MAPE under-forecasts; negative R^2.
# 7. Bootstrap confidence intervals for AUC and for a difference in AUC.

# %%
import numpy as np
from scipy import stats
from scipy.integrate import trapezoid
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
                             mean_pinball_loss, precision_recall_curve, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from scipy.optimize import minimize_scalar

rng = np.random.default_rng(182)

# %% [markdown]
# ## 1. The confusion matrix, by hand

# %%
def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1))); fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0))); fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


y = rng.random(5000) < 0.1
score = np.where(y, rng.normal(1.5, 1, 5000), rng.normal(0, 1, 5000))
pred = (score > 1.0).astype(int)
tp, fp, tn, fn = confusion(y, pred)
prec, rec, spec = tp / (tp + fp), tp / (tp + fn), tn / (tn + fp)
f1 = 2 * prec * rec / (prec + rec)
assert np.isclose(prec, precision_score(y, pred)) and np.isclose(rec, recall_score(y, pred)) and np.isclose(f1, f1_score(y, pred))
print(f"precision {prec:.3f}, recall {rec:.3f}, specificity {spec:.3f}, F1 {f1:.3f}, accuracy {(tp + tn) / len(y):.3f}"
      f" (always-negative baseline {1 - y.mean():.3f})")

# the same classifier at different prevalences: recall/specificity fixed, precision not
tpr, fpr = rec, 1 - spec
for pi in (0.5, 0.1, 0.01, 0.001):
    print(f"prevalence {pi:>6}: precision {tpr * pi / (tpr * pi + fpr * (1 - pi)):.3f}")
print("99%/99% test at 0.1% prevalence:", round(0.99 * 0.001 / (0.99 * 0.001 + 0.01 * 0.999), 3))
assert abs(0.99 * 0.001 / (0.99 * 0.001 + 0.01 * 0.999) - 0.09) < 0.001

# %% [markdown]
# ## 2. ROC-AUC three ways

# %%
pos, neg = score[y], score[~y]
auc_pairs = np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])
U = stats.mannwhitneyu(pos, neg).statistic
fpr_c, tpr_c, _ = roc_curve(y, score)
auc_trap = trapezoid(tpr_c, fpr_c)
print(f"AUC: P(pos > neg) {auc_pairs:.4f}; Mann-Whitney U/(n+ n-) {U / (len(pos) * len(neg)):.4f}; "
      f"area under the curve {auc_trap:.4f}; sklearn {roc_auc_score(y, score):.4f}")
assert np.allclose([auc_pairs, U / (len(pos) * len(neg)), auc_trap], roc_auc_score(y, score))

# AUC ignores calibration: any monotone transform leaves it unchanged
assert np.isclose(roc_auc_score(y, np.exp(3 * score)), roc_auc_score(y, score))

# %% [markdown]
# ## 3. 0.1% positives: ROC looks great, the alerts don't

# %%
n_neg, n_pos = 1_000_000, 1_000
yr = np.r_[np.zeros(n_neg, bool), np.ones(n_pos, bool)]
sr = np.r_[rng.normal(0, 1, n_neg), rng.normal(3.0, 1, n_pos)]
auc = roc_auc_score(yr, sr)
ap = average_precision_score(yr, sr)
order = np.argsort(-sr)
top500_precision = yr[order[:500]].mean()
p_c, r_c, _ = precision_recall_curve(yr, sr)
prec_at_80 = p_c[r_c >= 0.8].max()
fpr_c, tpr_c, thr = roc_curve(yr, sr)
i = np.searchsorted(tpr_c, 0.8)
print(f"ROC-AUC {auc:.3f} (looks superb); average precision {ap:.3f} (random baseline = prevalence {yr.mean():.4f})")
print(f"to catch 80% of positives: FPR {fpr_c[i]:.3%} -> {int(fpr_c[i] * n_neg):,} false alarms for {int(0.8 * n_pos)} catches; "
      f"best precision at recall >= 0.8: {prec_at_80:.1%}")
print(f"precision among the top 500 alerts: {top500_precision:.1%}")
assert auc > 0.97 and prec_at_80 < 0.2 and ap < 0.5

# %% [markdown]
# ## 4. Proper scoring rules
#
# Two models with the same ranking. One reports honest probabilities; the other pushes them toward 0 and 1.

# %%
n4 = 20_000
x4 = rng.normal(size=n4)
p_true = 1 / (1 + np.exp(-(1.5 * x4 - 0.3)))
y4 = rng.random(n4) < p_true
honest = p_true
reckless = 1 / (1 + np.exp(-4 * (1.5 * x4 - 0.3)))            # same ranking, overconfident
for name, p in [("honest", honest), ("reckless", reckless)]:
    print(f"{name:9s}: accuracy {accuracy_score(y4, p > 0.5):.3f}  AUC {roc_auc_score(y4, p):.3f}  "
          f"log loss {log_loss(y4, p):.3f}  Brier {brier_score_loss(y4, p):.3f}")
assert np.isclose(accuracy_score(y4, honest > 0.5), accuracy_score(y4, reckless > 0.5))
assert np.isclose(roc_auc_score(y4, honest), roc_auc_score(y4, reckless))
assert log_loss(y4, reckless) > log_loss(y4, honest) + 0.1 and brier_score_loss(y4, reckless) > brier_score_loss(y4, honest)

# properness, directly: for a true probability of 0.3, the expected score is optimized by reporting 0.3
qs = np.linspace(0.01, 0.99, 99)
exp_log = -(0.3 * np.log(qs) + 0.7 * np.log(1 - qs))
exp_brier = 0.3 * (1 - qs) ** 2 + 0.7 * qs**2
exp_acc = 0.3 * (qs > 0.5) + 0.7 * (qs <= 0.5)
print(f"best report for p=0.3: log loss {qs[exp_log.argmin()]:.2f}, Brier {qs[exp_brier.argmin()]:.2f}, "
      f"accuracy: anything <= 0.5 ties ({np.sum(exp_acc == exp_acc.max())} values)")
assert np.isclose(qs[exp_log.argmin()], 0.3) and np.isclose(qs[exp_brier.argmin()], 0.3)

# %% [markdown]
# ## 5. Averaging over classes

# %%
y5 = rng.choice(3, 3000, p=[0.85, 0.12, 0.03])
p5 = y5.copy()
p5[(y5 == 2) & (rng.random(3000) < 0.8)] = 0                   # the rare class is mostly missed
p5[(y5 == 0) & (rng.random(3000) < 0.02)] = 1
for avg in ("micro", "macro", "weighted"):
    print(f"{avg:8s} F1 {f1_score(y5, p5, average=avg):.3f}")
print("per class:", np.round(f1_score(y5, p5, average=None), 3), " accuracy", round(accuracy_score(y5, p5), 3))
assert np.isclose(f1_score(y5, p5, average="micro"), accuracy_score(y5, p5))
assert f1_score(y5, p5, average="macro") < f1_score(y5, p5, average="weighted") - 0.15

# %% [markdown]
# ## 6. Regression metrics prefer different answers

# %%
y6 = rng.lognormal(3, 0.8, 5000)                             # skewed, like sales or prices
best = lambda loss: minimize_scalar(lambda c: loss(np.full_like(y6, c)), bounds=(0, 200), method="bounded").x
c_mse = best(lambda f: np.mean((y6 - f) ** 2))
c_mae = best(lambda f: np.mean(np.abs(y6 - f)))
c_mape = best(lambda f: np.mean(np.abs((y6 - f) / y6)))
c_q90 = best(lambda f: mean_pinball_loss(y6, f, alpha=0.9))
print(f"best constant: MSE {c_mse:.1f} (mean {y6.mean():.1f}); MAE {c_mae:.1f} (median {np.median(y6):.1f}); "
      f"MAPE {c_mape:.1f}; pinball 0.9 {c_q90:.1f} (90th pct {np.quantile(y6, 0.9):.1f})")
assert abs(c_mse - y6.mean()) < 0.1 and abs(c_mae - np.median(y6)) < 0.3 and abs(c_q90 - np.quantile(y6, 0.9)) < 0.5
assert c_mape < c_mae < c_mse, "MAPE's favorite is the lowest: it pays to under-forecast"

# R^2 below zero, and R^2 that depends on the test set's spread
y_te = rng.normal(10, 1, 1000)
print(f"R^2 of predicting a constant 11 when the mean is 10: {r2_score(y_te, np.full(1000, 11.0)):.2f}")
assert r2_score(y_te, np.full(1000, 11.0)) < 0
x_all = rng.uniform(0, 10, 4000); y_all = x_all + rng.normal(0, 1, 4000)
narrow = (x_all > 4) & (x_all < 6)
print(f"same model, same noise: R^2 on the full range {r2_score(y_all, x_all):.2f}, on x in (4, 6) {r2_score(y_all[narrow], x_all[narrow]):.2f}; "
      f"RMSE {np.sqrt(np.mean((y_all - x_all) ** 2)):.2f} vs {np.sqrt(np.mean((y_all[narrow] - x_all[narrow]) ** 2)):.2f}")
assert r2_score(y_all, x_all) - r2_score(y_all[narrow], x_all[narrow]) > 0.4

# %% [markdown]
# ## 7. Bootstrap intervals

# %%
yb = rng.random(2000) < 0.2
sa = np.where(yb, rng.normal(1.2, 1, 2000), rng.normal(0, 1, 2000))
sb = sa + rng.normal(0, 0.6, 2000)                           # model B: a noisier version of A
B = 1000
aucs_a, diffs = np.empty(B), np.empty(B)
for i in range(B):
    idx = rng.integers(0, 2000, 2000)
    a_i = roc_auc_score(yb[idx], sa[idx])
    aucs_a[i], diffs[i] = a_i, a_i - roc_auc_score(yb[idx], sb[idx])
lo, hi = np.percentile(aucs_a, [2.5, 97.5])
dlo, dhi = np.percentile(diffs, [2.5, 97.5])
print(f"AUC of A {roc_auc_score(yb, sa):.3f}, 95% CI [{lo:.3f}, {hi:.3f}]; A - B {roc_auc_score(yb, sa) - roc_auc_score(yb, sb):+.3f}, "
      f"paired 95% CI [{dlo:+.3f}, {dhi:+.3f}]")
print(f"note: the paired CI width {dhi - dlo:.3f} is much narrower than A's own CI width {hi - lo:.3f}")
assert dlo > 0 and (dhi - dlo) < (hi - lo)

print("\nAll checks passed.")
