# %% [markdown]
# # Lab 54.1: Fairness, measured
#
# A synthetic lending decision with two groups, A and B, whose repayment rates differ (as they do in real data,
# for reasons that include past discrimination). The model decides who gets a loan.
# 1. Group metrics: selection rates, error rates, calibration. They disagree about whether the model is fair.
# 2. "Fairness through unawareness": remove the group column. A proxy (the neighborhood) brings it back.
# 3. The impossibility result, numerically: a calibrated score can't have equal error rates when base rates differ.
# 4. A mitigation (group-specific thresholds for equal opportunity), and what it costs.

# %%
import numpy as np
from sklearn.linear_model import LogisticRegression

rs = np.random.default_rng(541)
n = 60_000
group = rs.random(n) < 0.3                                                          # True = group B (30% of applicants)
neighborhood = np.where(group, rs.normal(1.0, 0.6, n), rs.normal(-0.4, 0.6, n))     # correlated with group: a proxy
income = rs.normal(0, 1, n) - 0.4 * group
history = rs.normal(0, 1, n) - 0.3 * group
logit = 1.0 + 1.2 * income + 1.0 * history - 0.6 * group + 0.4 * rs.normal(0, 1, n)   # group: factors the data lacks
repaid = rs.random(n) < 1 / (1 + np.exp(-logit))
train, test = np.arange(n) < 40_000, np.arange(n) >= 40_000
print(f"repayment rate: group A {repaid[~group].mean():.3f}, group B {repaid[group].mean():.3f}")


def report(approve, score=None, label=""):
    out = {}
    for name, g in (("A", ~group[test]), ("B", group[test])):
        y, a = repaid[test][g], approve[g]
        out[name] = {"approval rate": a.mean(),
                     "TPR (repayers approved)": a[y].mean(),
                     "FPR (defaulters approved)": a[~y].mean(),
                     "PPV (approved who repay)": y[a].mean()}
        if score is not None:
            bins = np.digitize(score[g], [0.6, 0.7, 0.8, 0.9])
            out[name]["calibration gap"] = np.mean([abs(score[g][bins == b].mean() - y[bins == b].mean()) for b in range(5) if (bins == b).sum() > 50])
    print(f"\n{label}")
    for k in out["A"]:
        print(f"  {k:28s} A {out['A'][k]:.3f}   B {out['B'][k]:.3f}   gap {out['A'][k] - out['B'][k]:+.3f}")
    return out


# %% [markdown]
# ## 1 and 2. Group metrics, with and without the group column

# %%
X_full = np.column_stack([income, history, neighborhood, group])
X_blind = np.column_stack([income, history, neighborhood])
m_full = LogisticRegression().fit(X_full[train], repaid[train])
m_blind = LogisticRegression().fit(X_blind[train], repaid[train])
s_full, s_blind = m_full.predict_proba(X_full[test])[:, 1], m_blind.predict_proba(X_blind[test])[:, 1]
r_full = report(s_full > 0.7, s_full, "model with the group column, approve if P(repay) > 0.7:")
r_blind = report(s_blind > 0.7, s_blind, "model WITHOUT the group column (it still sees the neighborhood):")
m_proxy = LogisticRegression().fit(neighborhood[train, None], group[train])
print(f"\nthe neighborhood alone predicts group membership with accuracy {m_proxy.score(neighborhood[test, None], group[test]):.3f}")
print("dropping the protected attribute narrows the approval gap only partly: the model leans on the neighborhood, which")
print("encodes most of the same information. And B's scores get less calibrated, so the same score means less for them.")
print("Unawareness is not fairness, and it removes your ability to measure the disparity. Keep the attribute for")
print("evaluation (where lawful), even when the model must not use it.")
assert r_blind["A"]["approval rate"] - r_blind["B"]["approval rate"] > 0.5 * (r_full["A"]["approval rate"] - r_full["B"]["approval rate"])

# %% [markdown]
# ## 3. The impossibility result
#
# Kleinberg, Mullainathan and Raghavan (2016) and Chouldechova (2017): when base rates differ, a score that is
# calibrated within each group cannot also give both groups equal false-positive and false-negative rates (except in
# degenerate cases). Here: the TRUE probability of repaying, the best calibrated score there is.

# %%
p_true = 1 / (1 + np.exp(-(1.0 + 1.2 * income + 1.0 * history - 0.6 * group)))
oracle = report(p_true[test] > 0.7, p_true[test], "the true repayment probability (perfectly calibrated), threshold 0.7:")
print("\neven the true probabilities, calibrated for everyone, approve the groups at different rates and with different")
print("error rates, because the groups' score distributions differ. No threshold rule on a calibrated score equalizes")
print("calibration, false positive rates and false negative rates at once. You have to choose which to equalize, and")
print("that's a decision about values and law, not statistics.")
assert abs(oracle["A"]["TPR (repayers approved)"] - oracle["B"]["TPR (repayers approved)"]) > 0.05

# %% [markdown]
# ## 4. Equal opportunity by group thresholds

# %%
def threshold_for_tpr(score, y, target):
    for t in np.linspace(0.99, 0.01, 197):
        if (score[y] > t).mean() >= target:
            return t
    return 0.01


gA, gB = ~group[test], group[test]
yA, yB = repaid[test][gA], repaid[test][gB]
target = (s_full[gA][yA] > 0.7).mean()                                              # group A's TPR at 0.7
tB = threshold_for_tpr(s_full[gB], yB, target)
approve_eo = np.where(gB, s_full > tB, s_full > 0.7)
r_eo = report(approve_eo, s_full, f"group-specific thresholds (A 0.70, B {tB:.2f}) to equalize the TPR:")
acc = lambda a: np.mean(a == repaid[test])
print(f"\naccuracy: single threshold {acc(s_full > 0.7):.3f}, equal-opportunity thresholds {acc(approve_eo):.3f}")
print("equalizing the true positive rate (repayers get loans at the same rate in both groups) closes that gap, and here")
print("the false positive rates happen to line up too. Accuracy even rises, because 0.7 was never the accuracy-optimal")
print("threshold: the lender chose it for its own costs. That cost shows up in B's PPV: more of the approved B applicants")
print("default. Whether group-specific thresholds are allowed depends on the jurisdiction and the domain; some require")
print("them as remediation, others prohibit them as disparate treatment. Ask counsel, decide explicitly, and document.")
assert abs(r_eo["A"]["TPR (repayers approved)"] - r_eo["B"]["TPR (repayers approved)"]) < 0.02
assert r_eo["B"]["PPV (approved who repay)"] < r_full["B"]["PPV (approved who repay)"]

print("\nAll checks passed.")
