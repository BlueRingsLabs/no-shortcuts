# %% [markdown]
# # Lab 50.1: Testing ML systems
#
# A loan-default model and the code around it, and the test suite that gates it in CI. Two candidate releases:
#   good:   retrained on fresh data
#   broken: retrained after someone "cleaned up" the feature code, which now computes income in thousands during
#           training while the serving path still sends dollars (training/serving skew). Its offline metrics look fine.
# 1. Tests of code: unit and property-based tests of the feature functions.
# 2. Tests of data: schema, ranges, missingness, and distribution against the last training set.
# 3. Tests of the model: a quality gate against the champion, slices, invariance and directional expectations.
# 4. A train/serve consistency test: the same raw record through both paths.

# %%
import numpy as np
from hypothesis import given, settings, strategies as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

rs = np.random.default_rng(501)


def make_data(n, seed):
    r = np.random.default_rng(seed)
    income = r.lognormal(10.8, 0.5, n)                                              # dollars a year
    debt = income * r.beta(2, 5, n)
    age = r.integers(18, 80, n)
    late = r.poisson(0.3, n)
    region = r.integers(0, 4, n)
    logit = -2.0 + 3.0 * debt / income + 0.6 * late - 0.4 * np.log(income / 50_000) - 0.01 * (age - 40)
    y = r.random(n) < 1 / (1 + np.exp(-logit))
    raw = [{"income": float(i), "debt": float(d), "age": int(a), "late_payments": int(l), "region": int(g)}
           for i, d, a, l, g in zip(income, debt, age, late, region)]
    return raw, y


# the feature code: training and serving must call the same function
def features(rec, income_unit=1.0):
    income = rec["income"] / income_unit
    return [np.log1p(max(income, 0.0)), rec["debt"] / max(rec["income"], 1.0), rec["age"], rec["late_payments"], rec["region"]]


def featurize(records, income_unit=1.0):
    return np.array([features(r, income_unit) for r in records])


train_raw, y_train = make_data(20_000, 1)
test_raw, y_test = make_data(8_000, 2)
champion = HistGradientBoostingClassifier(random_state=0).fit(featurize(train_raw), y_train)
good = HistGradientBoostingClassifier(random_state=1).fit(featurize(make_data(20_000, 3)[0]), make_data(20_000, 3)[1])
broken = HistGradientBoostingClassifier(random_state=1).fit(featurize(train_raw, income_unit=1000.0), y_train)
serving_path = lambda model, records: model.predict_proba(featurize(records))[:, 1]          # production: dollars
training_path = {"good": lambda recs: good.predict_proba(featurize(recs))[:, 1],
                 "broken": lambda recs: broken.predict_proba(featurize(recs, 1000.0))[:, 1]}   # how it was evaluated
models = {"good": good, "broken": broken}
report = {name: {} for name in models}

# %% [markdown]
# ## 1. Code: unit and property tests

# %%
def test_features_unit():
    f = features({"income": 60_000, "debt": 15_000, "age": 30, "late_payments": 0, "region": 2})
    assert abs(f[1] - 0.25) < 1e-12 and f[2] == 30


@settings(max_examples=300, deadline=None)
@given(income=st.floats(0, 1e7), debt=st.floats(0, 1e7), age=st.integers(18, 100))
def test_features_properties(income, debt, age):
    """For any valid input: finite outputs, and a debt ratio that grows with debt."""
    rec = {"income": income, "debt": debt, "age": age, "late_payments": 0, "region": 0}
    f = features(rec)
    assert all(np.isfinite(f))
    assert features({**rec, "debt": debt + 1000})[1] >= f[1]


test_features_unit()
test_features_properties()
print("code tests: unit test and 300 property-based cases (hypothesis) pass for the shared feature function")
print("property tests find the inputs you didn't think of: zero income, huge debts, NaN-producing corners.")

# %% [markdown]
# ## 2. Data tests

# %%
def data_tests(records, reference):
    X, R = featurize(records), featurize(reference)
    out = {}
    out["schema: required fields present"] = all({"income", "debt", "age", "late_payments", "region"} <= set(r) for r in records)
    out["ranges: 18 <= age <= 100, income >= 0"] = all(18 <= r["age"] <= 100 and r["income"] >= 0 for r in records)
    out["no missing values"] = bool(np.isfinite(X).all())
    q = np.quantile(R, [0.01, 0.99], axis=0)
    share_outside = ((X < q[0]) | (X > q[1])).mean(0)
    out["distribution: each feature within its 1-99% range for >= 95% of rows"] = bool((share_outside < 0.05).all())
    return out


for name, recs in (("new training batch", make_data(5_000, 3)[0]),
                   ("a batch with ages in months", [{**r, "age": r["age"] * 12} for r in make_data(2_000, 4)[0]])):
    res = data_tests(recs, train_raw)
    print(f"\ndata tests on {name}: " + ", ".join(f"{k.split(':')[0]} {'ok' if v else 'FAIL'}" for k, v in res.items()))

# %% [markdown]
# ## 3. Model tests

# %%
def paired_gate(p_new, p_old, y, n_boot=2000, seed=0):
    """Is the candidate no worse than the champion? Paired bootstrap on the same test rows (05.1, 43.1)."""
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(y), (n_boot, len(y)))
    diffs = [roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]) for i in idx[:300]]
    return np.percentile(diffs, 2.5)


p_champ = champion.predict_proba(featurize(test_raw))[:, 1]
X_test = featurize(test_raw)
region = X_test[:, 4]
for name in models:
    p_offline = training_path[name](test_raw)                                        # the numbers in the training report
    auc = roc_auc_score(y_test, p_offline)
    lower = paired_gate(p_offline, p_champ, y_test)
    report[name]["quality gate (offline, vs champion)"] = lower > -0.01
    slice_aucs = [roc_auc_score(y_test[region == g], p_offline[region == g]) for g in range(4)]
    report[name]["no slice below 0.60 AUC"] = min(slice_aucs) > 0.60
    # invariance: region should not change the score much (a policy requirement here)
    recs2 = [{**r, "region": (r["region"] + 1) % 4} for r in test_raw[:2000]]
    report[name]["invariance to region (mean |change| < 0.02)"] = float(np.abs(training_path[name](recs2) - p_offline[:2000]).mean()) < 0.02
    # directional expectation: more late payments should never lower the risk on average
    recs3 = [{**r, "late_payments": r["late_payments"] + 2} for r in test_raw[:2000]]
    report[name]["directional: more late payments -> higher risk"] = float((training_path[name](recs3) - p_offline[:2000]).mean()) > 0
    print(f"\n{name}: offline AUC {auc:.3f} (champion {roc_auc_score(y_test, p_champ):.3f}), lower bound of the paired "
          f"difference {lower:+.4f}, slice AUCs {np.round(slice_aucs, 3).tolist()}")

# %% [markdown]
# ## 4. Train/serve consistency

# %%
for name, model in models.items():
    p_train = training_path[name](test_raw[:500])
    p_serve = serving_path(model, test_raw[:500])
    gap = float(np.abs(p_train - p_serve).max())
    report[name]["train/serve consistency (max |diff| < 1e-6)"] = gap < 1e-6
    report[name]["_online_auc"] = roc_auc_score(y_test, serving_path(model, test_raw))

print("\n" + " " * 52 + "   good   broken")
for test in report["good"]:
    if test.startswith("_"):
        continue
    print(f"  {test:50s} " + "   ".join(f"{'pass' if report[n][test] else 'FAIL':>5s}" for n in models))
print(f"\nAUC in production (the serving path): good {report['good']['_online_auc']:.3f}, broken {report['broken']['_online_auc']:.3f}")
print("the broken model passes every offline test: its evaluation used its own (wrong) feature code consistently. Only")
print("the test that runs the same raw records through the training path *and* the serving path catches it. Put one in")
print("CI, and compute features in one shared function or a feature store, never twice.")
assert all(v for k, v in report["good"].items() if not k.startswith("_"))
assert not report["broken"]["train/serve consistency (max |diff| < 1e-6)"] and report["broken"]["quality gate (offline, vs champion)"]

print("\nAll checks passed.")
