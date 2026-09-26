# %% [markdown]
# # Lab 51.1: Drift, data quality and model monitoring
#
# A credit model in production for a year; labels (did the loan default?) arrive 90 days after each prediction.
# 1. Detecting input drift: PSI and the Kolmogorov-Smirnov test, and why a p-value is the wrong alarm at production
#    sample sizes.
# 2. Many features, every day: how many false alarms does a naive monitor raise?
# 3. Does drift matter? Covariate shift that doesn't hurt, and concept drift that input monitors can't see.
# 4. Estimating performance before the labels arrive: confidence-based estimation, and where it fails.

# %%
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rs = np.random.default_rng(511)


def psi(ref, cur, bins=10):
    """Population stability index over the reference distribution's deciles."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


# %% [markdown]
# ## 1. PSI and KS

# %%
ref = rs.normal(0, 1, 50_000)
print(f"{'current window':36s} {'n':>7s} {'PSI':>7s} {'KS stat':>8s} {'KS p-value':>11s}")
for name, shift, n in (("same distribution", 0.0, 50_000), ("mean shifted by 0.02 sd (trivial)", 0.02, 50_000),
                       ("mean shifted by 0.02 sd, n = 500", 0.02, 500), ("mean shifted by 0.5 sd (real)", 0.5, 50_000),
                       ("mean shifted by 0.5 sd, n = 200", 0.5, 200)):
    cur = rs.normal(shift, 1, n)
    ks = stats.ks_2samp(ref, cur)
    print(f"{name:36s} {n:7,d} {psi(ref, cur):7.3f} {ks.statistic:8.3f} {ks.pvalue:11.2e}")
print("a p-value mixes the size of a change with the amount of data: with 50,000 rows a trivial 0.02 sd shift is")
print("'significant', and any window can cross p < 0.05 by chance. PSI measures how much the distribution moved, and")
print("stays near zero for trivial shifts at any sample size. Alarm on an effect size (PSI > 0.1 'look', > 0.25 'act'")
print("are the usual rules of thumb, or the KS statistic itself); use tests to check that the window is big enough.")

# %% [markdown]
# ## 2. Many features, every day

# %%
n_feat, days, n_day = 50, 90, 5_000
ref_X = rs.normal(size=(20_000, n_feat))
alarms_p, alarms_psi = 0, 0
for d in range(days):
    cur_X = rs.normal(size=(n_day, n_feat))                                         # no drift at all
    p = np.array([stats.ks_2samp(ref_X[:, j], cur_X[:, j]).pvalue for j in range(n_feat)])
    alarms_p += (p < 0.05).sum()
    alarms_psi += sum(psi(ref_X[:, j], cur_X[:, j]) > 0.1 for j in range(n_feat))
print(f"\n{n_feat} features, {days} days, no drift: KS p < 0.05 raises {alarms_p} alarms ({alarms_p / days:.1f} a day); "
      f"PSI > 0.1 raises {alarms_psi}")
print("5% of 50 tests fire every day by chance. A monitor that cries wolf twice a day gets muted within a week, and")
print("then misses the real event. Use effect sizes, correct for multiple testing (or monitor fewer, important")
print("features plus the model's output), and require persistence (two days in a row) before paging anyone.")
assert alarms_p > days and alarms_psi == 0

# %% [markdown]
# ## 3. Does the drift matter?

# %%
def world(n, income_shift=0.0, concept=False, seed=0):
    r = np.random.default_rng(seed)
    income = r.normal(income_shift, 1, n)
    debt = r.normal(0, 1, n)
    history = r.normal(0, 1, n)
    X = np.column_stack([income, debt, history])
    w = np.array([-1.0, 1.2, -0.8]) if not concept else np.array([-1.0, 0.2, -0.8])    # concept drift: debt stops mattering
    logit = -1.5 + X @ w
    y = r.random(n) < 1 / (1 + np.exp(-logit))
    return X, y


X_tr, y_tr = world(30_000, seed=1)
model = LogisticRegression().fit(X_tr, y_tr)
X_ref = X_tr
print(f"\n{'production scenario':44s} {'PSI income':>10s} {'PSI score':>10s} {'true AUC':>9s} {'true default rate':>18s}")
scen = {}
for name, kw in (("like training", {}), ("covariate shift: incomes fall 0.7 sd", {"income_shift": -0.7}),
                 ("concept drift: debt stops predicting default", {"concept": True})):
    X, y = world(20_000, seed=2, **kw)
    s_ref, s = model.predict_proba(X_ref)[:, 1], model.predict_proba(X)[:, 1]
    scen[name] = (X, y, s)
    print(f"{name:44s} {psi(X_ref[:, 0], X[:, 0]):10.3f} {psi(s_ref, s):10.3f} {roc_auc_score(y, s):9.3f} {y.mean():18.3f}")
print("the covariate shift is loud in the inputs and in the score distribution, and the model is fine: it's the same")
print("relationship on a different population. The concept drift is silent in every input and score monitor, and the")
print("model is much worse. Input drift is a reason to look, not a verdict; only outcomes measure performance.")

# %% [markdown]
# ## 4. Performance before the labels arrive
#
# Labels take 90 days. If the model is calibrated, its own probabilities predict its error: expected accuracy and
# expected defaults can be computed from scores alone (confidence-based performance estimation).

# %%
print(f"\n{'scenario':44s} {'estimated default rate':>23s} {'actual':>7s}   estimated vs actual accuracy at 0.5")
for name, (X, y, s) in scen.items():
    est_rate = s.mean()
    pred = s > 0.5
    est_acc = np.mean(np.where(pred, s, 1 - s))
    print(f"{name:44s} {est_rate:23.3f} {y.mean():7.3f}   {est_acc:.3f} vs {np.mean(pred == y):.3f}")
print("under covariate shift the estimates track the truth, because P(y | x) didn't change and the model is calibrated")
print("for it. Under concept drift they're confidently wrong: the model can't know its probabilities are stale. So:")
print("estimate early, confirm with real labels as they arrive (and with faster proxies: first missed payment at 30")
print("days), and treat a gap between estimated and realized performance as the concept-drift alarm.")
X_c, y_c, s_c = scen["concept drift: debt stops predicting default"]
X_s, y_s, s_s = scen["covariate shift: incomes fall 0.7 sd"]
assert abs(s_s.mean() - y_s.mean()) < 0.02 and abs(s_c.mean() - y_c.mean()) > 0.03

print("\nAll checks passed.")
