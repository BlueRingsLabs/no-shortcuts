# %% [markdown]
# # Lab 18.4: Leakage, imbalance and other ways to fool yourself
#
# 1. Target leakage: a feature computed after the outcome, and how permutation importance exposes it.
# 2. Point-in-time joins: today's customer attributes vs as-of attributes.
# 3. Target encoding of a random high-cardinality ID: naive vs out-of-fold.
# 4. A row index that knows the label.
# 5. Oversampling before the split vs inside the fold.
# 6. Class weights vs threshold moving: same ranking, same cost, worse probabilities.
# 7. Seed variance: is the "improvement" bigger than the noise?
# 8. Selective labels: a model trained on approved loans only.

# %%
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

rng = np.random.default_rng(184)

# %% [markdown]
# ## 1. Target leakage
#
# Churn data. `days_inactive` was computed at extraction time, weeks after the prediction date, so churners have piled up
# inactivity that didn't exist when the prediction would have been made.

# %%
n = 8000
tenure = rng.exponential(24, n)
support_calls = rng.poisson(1.5, n)
monthly_fee = rng.normal(50, 15, n)
logit = -1.5 - 0.03 * tenure + 0.35 * support_calls + 0.02 * (monthly_fee - 50)
churn = rng.random(n) < 1 / (1 + np.exp(-logit))
days_inactive_at_extraction = np.where(churn, rng.uniform(20, 60, n), rng.exponential(3, n))   # after the fact
days_inactive_at_prediction = rng.exponential(3, n) + 2 * churn                                  # the honest version: weakly informative
honest = np.column_stack([tenure, support_calls, monthly_fee, days_inactive_at_prediction])
leaky = np.column_stack([tenure, support_calls, monthly_fee, days_inactive_at_extraction])
names = ["tenure", "support_calls", "monthly_fee", "days_inactive"]

Xl_tr, Xl_te, Xh_tr, Xh_te, y_tr, y_te = train_test_split(leaky, honest, churn, test_size=0.3, random_state=0)
gb_leaky = HistGradientBoostingClassifier(random_state=0).fit(Xl_tr, y_tr)
gb_honest = HistGradientBoostingClassifier(random_state=0).fit(Xh_tr, y_tr)
print(f"AUC with the leaky feature {roc_auc_score(y_te, gb_leaky.predict_proba(Xl_te)[:, 1]):.3f}; "
      f"with the as-of-prediction feature {roc_auc_score(y_te, gb_honest.predict_proba(Xh_te)[:, 1]):.3f}")
imp = permutation_importance(gb_leaky, Xl_te, y_te, scoring="roc_auc", n_repeats=5, random_state=0).importances_mean
print("permutation importance (AUC drop):", {k: round(float(v), 3) for k, v in zip(names, imp)})
assert roc_auc_score(y_te, gb_leaky.predict_proba(Xl_te)[:, 1]) > 0.98
assert imp.argmax() == 3 and imp[3] > 5 * np.sort(imp)[-2], "one feature carries everything: investigate before celebrating"

# %% [markdown]
# ## 2. Point-in-time joins
#
# Customers change segment over time (a type 2 history). Premium customers default less. Some customers were *moved*
# to a "watchlist" segment *because* they defaulted. Joining today's segment leaks that.

# %%
n_c = 4000
cust = pd.DataFrame({"customer_id": np.arange(n_c), "segment": rng.choice(["basic", "premium"], n_c, p=[0.7, 0.3])})
events = pd.DataFrame({"customer_id": rng.integers(0, n_c, 20_000),
                       "ts": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, 20_000), unit="D")}).sort_values("ts")
seg_at = events.merge(cust, on="customer_id")["segment"].to_numpy()
events["default"] = rng.random(len(events)) < np.where(seg_at == "premium", 0.03, 0.08)
events = events.reset_index(drop=True)

# history: initial segment from 2024, and a later move to "watchlist" after a default
history = cust.assign(valid_from=pd.Timestamp("2024-01-01"))
first_default = events[events["default"]].groupby("customer_id")["ts"].min()
moved = first_default.sample(frac=0.9, random_state=0)
history = pd.concat([history, pd.DataFrame({"customer_id": moved.index, "segment": "watchlist",
                                            "valid_from": moved.to_numpy() + pd.Timedelta(days=1)})]).sort_values("valid_from")
current = history.sort_values("valid_from").groupby("customer_id").last().reset_index()

naive = events.merge(current[["customer_id", "segment"]], on="customer_id")
asof = pd.merge_asof(events, history[["customer_id", "segment", "valid_from"]], left_on="ts", right_on="valid_from", by="customer_id")


def auc_by_segment(df):
    X = pd.get_dummies(df["segment"]).astype(float)
    cut = df["ts"] < pd.Timestamp("2025-10-01")
    m = LogisticRegression().fit(X[cut], df.loc[cut, "default"])
    return roc_auc_score(df.loc[~cut, "default"], m.predict_proba(X[~cut])[:, 1])


print(f"AUC, segment joined as of today {auc_by_segment(naive):.3f}; joined as of the event {auc_by_segment(asof):.3f}")
print(f"'watchlist' share of training rows: today-join {np.mean(naive['segment'] == 'watchlist'):.1%}, as-of {np.mean(asof['segment'] == 'watchlist'):.1%}")
assert auc_by_segment(naive) > auc_by_segment(asof) + 0.1

# %% [markdown]
# ## 3. Target encoding a random ID

# %%
n3 = 5000
user_id = rng.integers(0, 2500, n3).astype(str)                # ~2 rows per ID, and the ID means nothing
x_real = rng.normal(size=n3)
y3 = rng.random(n3) < 1 / (1 + np.exp(-0.8 * x_real))
df3 = pd.DataFrame({"user_id": user_id, "x": x_real})
cv = StratifiedKFold(5, shuffle=True, random_state=0)

means = pd.Series(y3).groupby(df3["user_id"]).mean()           # computed on ALL rows, labels included
df_naive = pd.DataFrame({"user_te": df3["user_id"].map(means).to_numpy(), "x": x_real})
auc_naive = cross_val_score(HistGradientBoostingClassifier(random_state=0), df_naive, y3, cv=cv, scoring="roc_auc").mean()
pipe = make_pipeline(make_column_transformer((TargetEncoder(cv=KFold(5, shuffle=True, random_state=0)), ["user_id"]), remainder="passthrough"),
                     HistGradientBoostingClassifier(random_state=0))
auc_oof = cross_val_score(pipe, df3, y3, cv=cv, scoring="roc_auc").mean()
auc_x = cross_val_score(HistGradientBoostingClassifier(random_state=0), x_real[:, None], y3, cv=cv, scoring="roc_auc").mean()
print(f"CV AUC: naive target encoding {auc_naive:.3f}; out-of-fold TargetEncoder {auc_oof:.3f}; same model on x alone {auc_x:.3f}")
print("the honest encoding adds nothing (the ID is noise); the naive one invents a large gain")
assert auc_naive > auc_x + 0.1 and auc_oof < auc_x + 0.02

# %% [markdown]
# ## 4. The row index knows the label

# %%
Xr = rng.normal(size=(3000, 5))
yr = rng.random(3000) < 0.3
order = np.argsort(yr, kind="stable")                           # the export was sorted by label
Xr, yr = Xr[order], yr[order]
with_index = np.column_stack([np.arange(3000), Xr])
auc_idx = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), with_index, yr, cv=cv, scoring="roc_auc").mean()
auc_noidx = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), Xr, yr, cv=cv, scoring="roc_auc").mean()
print(f"pure-noise features: CV AUC with the row index {auc_idx:.3f}, without {auc_noidx:.3f}")
assert auc_idx > 0.95 and abs(auc_noidx - 0.5) < 0.05

# %% [markdown]
# ## 5. Oversampling before the split

# %%
Xo = rng.normal(size=(2000, 20))
yo = rng.random(2000) < 1 / (1 + np.exp(-(Xo[:, 0] - 3.5)))      # ~5% positives, weak signal
pos = np.flatnonzero(yo)
dup = rng.choice(pos, 15 * len(pos))                            # oversample positives 16x ... before splitting
X_bad, y_bad = np.vstack([Xo, Xo[dup]]), np.r_[yo, yo[dup]]
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
ap_bad = cross_val_score(rf, X_bad, y_bad, cv=cv, scoring="average_precision").mean()
ap_bad_base = y_bad.mean()


def oversampled_fit_score(tr, te):
    p_tr = tr[yo[tr]]
    tr_os = np.r_[tr, rng.choice(p_tr, 15 * len(p_tr))]
    m = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(Xo[tr_os], yo[tr_os])
    return average_precision_score(yo[te], m.predict_proba(Xo[te])[:, 1])


ap_good = np.mean([oversampled_fit_score(tr, te) for tr, te in cv.split(Xo, yo)])
print(f"average precision: oversample-then-split {ap_bad:.3f} (base rate {ap_bad_base:.2f}); "
      f"split-then-oversample {ap_good:.3f} (base rate {yo.mean():.2f})")
assert ap_bad > 0.9 and ap_good < 0.5

# %% [markdown]
# ## 6. Class weights vs moving the threshold

# %%
Xi = rng.normal(size=(60_000, 8))
yi = rng.random(60_000) < 1 / (1 + np.exp(-(Xi[:, :4] @ np.array([1.0, -0.8, 0.6, 0.5]) - 4.0)))
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.5, random_state=0, stratify=yi)
plain = LogisticRegression().fit(Xi_tr, yi_tr).predict_proba(Xi_te)[:, 1]
weighted = LogisticRegression(class_weight="balanced").fit(Xi_tr, yi_tr).predict_proba(Xi_te)[:, 1]
C_FP, C_FN = 1, 25


def min_cost(p):
    ts = np.quantile(p, np.linspace(0.5, 0.999, 300))
    return min((C_FP * np.sum((p > t) & ~yi_te) + C_FN * np.sum((p <= t) & yi_te)) / len(p) for t in ts)


print(f"prevalence {yi_te.mean():.3f}")
for name, p in [("unweighted", plain), ("balanced weights", weighted)]:
    print(f"{name:17s} AP {average_precision_score(yi_te, p):.3f}  min cost {min_cost(p):.4f}  "
          f"log loss {log_loss(yi_te, p):.3f}  mean p {p.mean():.3f}")
assert abs(average_precision_score(yi_te, plain) - average_precision_score(yi_te, weighted)) < 0.01
assert abs(min_cost(plain) - min_cost(weighted)) < 0.01 * max(min_cost(plain), 1e-9) + 0.002
assert log_loss(yi_te, weighted) > 3 * log_loss(yi_te, plain)

# %% [markdown]
# ## 7. Seed variance
#
# Model B is model A with a different random seed. Would you have shipped the "improvement"?

# %%
Xs = rng.normal(size=(3000, 20))
ys = rng.random(3000) < 1 / (1 + np.exp(-(Xs[:, :5].sum(1) * 0.6)))
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.3, random_state=0)
scores = []
for seed in range(12):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = GradientBoostingClassifier(subsample=0.7, max_features=0.5, random_state=seed).fit(Xs_tr, ys_tr)
    scores.append(roc_auc_score(ys_te, m.predict_proba(Xs_te)[:, 1]))
scores = np.array(scores)
print(f"same model, 12 seeds: AUC {scores.mean():.4f} ± {scores.std():.4f} (range {scores.min():.4f} to {scores.max():.4f})")
print(f"best seed vs worst seed: +{scores.max() - scores.min():.4f}, the size of many 'improvements' in a results table")
assert scores.max() - scores.min() > 0.003

# %% [markdown]
# ## 8. Selective labels
#
# The old policy approved applicants with feature 0 above 0.3. We only see repayment for the approved. Risk rises gently
# over the approved range and steeply below it, a change the approved-only data cannot reveal.

# %%
n8 = 40_000
X8 = rng.normal(size=(n8, 3))
logit8 = -1.8 - 0.4 * X8[:, 0] - 1.6 * np.minimum(X8[:, 0], 0) + 0.5 * X8[:, 1]
default = rng.random(n8) < 1 / (1 + np.exp(-logit8))
approved = X8[:, 0] > 0.3
m_sel = LogisticRegression().fit(X8[approved], default[approved])
m_all = LogisticRegression().fit(X8, default)                   # what you'd get with labels for everyone (you don't have them)
rej = ~approved
pred_sel, pred_all = m_sel.predict_proba(X8[rej])[:, 1].mean(), m_all.predict_proba(X8[rej])[:, 1].mean()
print(f"approved {approved.mean():.0%} of applicants; default rate approved {default[approved].mean():.3f}, rejected {default[rej].mean():.3f}")
print(f"predicted default on the rejected: approved-only model {pred_sel:.3f}, full-label model {pred_all:.3f}, truth {default[rej].mean():.3f}")
print("the approved-only model would happily approve the rejected population; its own validation set could never tell you")
assert default[rej].mean() - pred_sel > 0.1 and abs(pred_all - default[rej].mean()) < default[rej].mean() - pred_sel

print("\nAll checks passed.")
