# %% [markdown]
# # Lab 47.1: The numbers in a design doc
#
# A card-fraud detection system, designed from the business problem backwards. Synthetic transactions: 0.3% fraud,
# amounts log-normal, fraud more likely on new devices, at night and for unusual amounts.
# 1. The baseline you have to beat: the rules the fraud team already uses.
# 2. From a score to a decision: the review team can check 150 alerts a day. Metrics at that capacity, and money.
# 3. Two models with the same AUC and different value.
# 4. A feature that isn't available when the decision is made.

# %%
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

rs = np.random.default_rng(471)
DAYS, PER_DAY = 60, 20_000
n = DAYS * PER_DAY
day = np.repeat(np.arange(DAYS), PER_DAY)
amount = np.exp(rs.normal(3.5, 1.1, n))
new_device = rs.random(n) < 0.08
night = rs.random(n) < 0.15
merchant_risk = rs.beta(1, 8, n)
velocity = rs.poisson(1.5, n)                                                     # transactions in the last hour
dist_home = rs.exponential(20, n)
logit = (-8.6 + 1.8 * new_device + 0.9 * night + 3.0 * merchant_risk + 0.35 * velocity
         + 0.25 * np.log1p(dist_home) + 0.35 * (np.log(amount) - 3.5) + rs.normal(0, 1.0, n))
fraud = rs.random(n) < 1 / (1 + np.exp(-logit))
X = np.column_stack([np.log(amount), new_device, night, merchant_risk, velocity, np.log1p(dist_home)])
train, test = day < 45, day >= 45                                                  # split by time, not at random
print(f"{n:,} transactions over {DAYS} days, fraud rate {fraud.mean():.2%}, {fraud[test].sum():,} frauds in the "
      f"15 test days (${amount[test & fraud].sum():,.0f})")

# %% [markdown]
# ## 1. The baseline

# %%
rules = (new_device & (amount > 150)) | (velocity >= 5)
model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[train], fraud[train])
score = model.predict_proba(X[test])[:, 1]
print(f"\nrules: flag {rules[test].mean():.2%} of test transactions, precision {fraud[test][rules[test]].mean():.3f}, "
      f"recall {fraud[test][rules[test]].sum() / fraud[test].sum():.3f}")
print(f"model: AUC {roc_auc_score(fraud[test], score):.3f}, average precision {average_precision_score(fraud[test], score):.3f} "
      f"(a random score would have {fraud[test].mean():.3f})")
print("the rules are the baseline. A design doc that can't say how much better than the rules the model must be, and")
print("at what, hasn't defined success.")

# %% [markdown]
# ## 2. From a score to a decision: capacity and money

# %%
CAPACITY, REVIEW_COST = 150, 4.0                                                    # alerts a day, dollars per review
test_days = np.unique(day[test])


def daily_top(scores, k=CAPACITY):
    """The alerts the team actually reviews: the k highest scores each day."""
    flagged = np.zeros(test.sum(), bool)
    d = day[test]
    for dd in test_days:
        idx = np.flatnonzero(d == dd)
        flagged[idx[np.argsort(-scores[idx])[:k]]] = True
    return flagged


amt, fr = amount[test], fraud[test]
print(f"\nthe review team checks {CAPACITY} alerts a day; each review costs ${REVIEW_COST:.0f}; a caught fraud saves its amount")
print(f"  {'policy':38s} {'precision':>9s} {'recall':>7s} {'fraud $ caught':>15s} {'net value / day':>16s}")
value = {}
tie = np.random.default_rng(2).random(test.sum())
for name, flagged in (("rules (150 of the flagged, at random)", daily_top(rules[test].astype(float) + 1e-6 * tie)),
                      ("model, top 150 by probability", daily_top(score)),
                      ("model, top 150 by expected loss", daily_top(score * amt))):
    caught = amt[flagged & fr].sum()
    net = (caught - REVIEW_COST * flagged.sum()) / len(test_days)
    value[name] = net
    print(f"  {name:38s} {fr[flagged].mean():9.3f} {fr[flagged].sum() / fr.sum():7.3f} ${caught:14,.0f} ${net:15,.0f}")
print("the operating point is set by the team's capacity, not by a 0.5 threshold, and the objective is money, so")
print("ranking by expected loss (probability x amount) beats ranking by probability at the same capacity. Precision")
print("at capacity and dollars caught are the metrics this system should report; AUC is a diagnostic.")
assert value["model, top 150 by expected loss"] > max(value["model, top 150 by probability"], value["rules (150 of the flagged, at random)"])

# %% [markdown]
# ## 3. Same AUC, different value

# %%
rng = np.random.default_rng(1)
big = amt > np.quantile(amt, 0.8)
rand20 = rng.random(test.sum()) < 0.2
vals = {}
for name, mask in (("model A (scrambled on the largest 20% of amounts)", big), ("model B (scrambled on a random 20%)", rand20)):
    s_ = score.copy()
    idx = np.flatnonzero(mask)
    s_[idx] = rng.permutation(s_[idx])                                              # no information left inside the subset
    f = daily_top(s_ * amt)
    vals[name] = (roc_auc_score(fr, s_), (amt[f & fr].sum() - REVIEW_COST * f.sum()) / len(test_days))
    print(f"  {name:50s} AUC {vals[name][0]:.3f}   net value/day ${vals[name][1]:8,.0f}")
print("a metric that treats every fraud alike can't see where the money is. Weight the offline metric the way the")
print("business weighs errors, or report the money directly.")
(auc_a, v_a), (auc_b, v_b) = vals.values()
assert abs(auc_a - auc_b) < 0.03 and v_b > 1.5 * v_a

# %% [markdown]
# ## 4. Available at decision time?
#
# The warehouse has a column `chargeback_filed`: whether the cardholder later disputed the transaction. It's in the
# same table as the features. It's also filed days after the transaction, when the decision was long made.

# %%
chargeback = fraud & (rs.random(n) < 0.7) | (~fraud & (rs.random(n) < 0.0002))
X_leak = np.column_stack([X, chargeback])
m_leak = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X_leak[train], fraud[train])
auc_offline = roc_auc_score(fr, m_leak.predict_proba(X_leak[test])[:, 1])
X_online = X_leak[test].copy(); X_online[:, -1] = 0                                  # at decision time: never filed yet
auc_online = roc_auc_score(fr, m_leak.predict_proba(X_online)[:, 1])
print(f"\nwith the chargeback column: offline AUC {auc_offline:.3f}; in production, where it's always 0 at decision time, "
      f"{auc_online:.3f} (the honest model: {roc_auc_score(fr, score):.3f})")
print("the design doc has to list, for every feature, when it becomes available relative to the decision, and the")
print("training data has to be built 'as of' that moment (point-in-time joins, 09.2 and 14.2). Leakage like")
print("this is the most common reason an offline result doesn't survive contact with production.")
assert auc_offline > roc_auc_score(fr, score) + 0.05 and auc_online < roc_auc_score(fr, score)

print("\nAll checks passed.")
