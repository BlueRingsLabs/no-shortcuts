# %% [markdown]
# # Lab 18.1: Splits and cross-validation that don't lie
#
# Every section builds a situation where we know the true performance, then compares honest and dishonest estimates of it.
#
# 1. Reusing the test set: 200 looks make a useless tweak look good.
# 2. Grouped data: random k-fold vs GroupKFold vs the truth on new groups.
# 3. Temporal data: random k-fold vs forward chaining vs the future.
# 4. Feature selection outside the fold: 90% accuracy on pure noise.
# 5. Nested CV vs the best score from a grid search.
# 6. Comparing two models with paired, corrected differences.
# 7. Adversarial validation spots a shifted test set.

# %%
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit,
                                     cross_val_predict, cross_val_score)
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

rng = np.random.default_rng(181)

# %% [markdown]
# ## 1. Looking at the test set 200 times
#
# The true signal lives in 5 features. We "tune" by trying random subsets of 30 noise features added to them,
# keeping whichever subset scores best on the test set. None of the noise can help; the best test score still climbs.

# %%
def make_clf_data(n):
    X = rng.normal(size=(n, 35))
    logits = X[:, :5] @ np.array([1.0, -1.0, 0.8, 0.6, -0.5])
    return X, (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)


X_tr, y_tr = make_clf_data(600)
X_te, y_te = make_clf_data(400)
X_new, y_new = make_clf_data(50_000)                         # "production": the truth we never get to see
best_score, best_cols = -1, None
for _ in range(200):
    cols = np.r_[np.arange(5), 5 + rng.choice(30, 10, replace=False)]
    m = LogisticRegression().fit(X_tr[:, cols], y_tr)
    s = m.score(X_te[:, cols], y_te)
    if s > best_score:
        best_score, best_cols, best_model = s, cols, m
true_best = best_model.score(X_new[:, best_cols], y_new)
honest = LogisticRegression().fit(X_tr[:, :5], y_tr)
print(f"'tuned' model: test {best_score:.3f}, production {true_best:.3f}; "
      f"untuned 5-feature model: test {honest.score(X_te[:, :5], y_te):.3f}, production {honest.score(X_new[:, :5], y_new):.3f}")
assert best_score - true_best > 0.015, "repeated looks inflated the test score"

# %% [markdown]
# ## 2. Grouped data
#
# 150 patients, 20 measurements each. Each patient has an idiosyncratic offset the model can memorize.

# %%
n_pat, per = 150, 20
groups = np.repeat(np.arange(n_pat), per)
patient_effect = rng.normal(0, 2.0, n_pat)[groups]
patient_id_signal = rng.normal(size=(n_pat, 4))[groups] + 0.05 * rng.normal(size=(n_pat * per, 4))   # features that fingerprint the patient
x_real = rng.normal(size=(n_pat * per, 2))
Xg = np.column_stack([x_real, patient_id_signal])
yg = x_real @ np.array([1.0, 0.5]) + patient_effect + rng.normal(0, 0.5, n_pat * per)

rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=0, n_jobs=-1)
r2_random = cross_val_score(rf, Xg, yg, cv=KFold(5, shuffle=True, random_state=0)).mean()
r2_group = cross_val_score(rf, Xg, yg, cv=GroupKFold(5), groups=groups).mean()
# truth: new patients
new_groups = np.repeat(np.arange(200), per)
Xg_new = np.column_stack([rng.normal(size=(200 * per, 2)), rng.normal(size=(200, 4))[new_groups]])
yg_new = Xg_new[:, :2] @ np.array([1.0, 0.5]) + rng.normal(0, 2.0, 200)[new_groups] + rng.normal(0, 0.5, 200 * per)
r2_true = rf.fit(Xg, yg).score(Xg_new, yg_new)
print(f"R^2: random KFold {r2_random:.2f}, GroupKFold {r2_group:.2f}, truth on new patients {r2_true:.2f}")
assert abs(r2_group - r2_true) < 0.1 and r2_random - r2_true > 0.4, "random folds let the forest recognize patients"

# %% [markdown]
# ## 3. Temporal data
#
# The relationship drifts slowly over time. Random folds interpolate; production extrapolates.

# %%
T = 3000
t = np.arange(T)
Xt = rng.normal(size=(T, 3))
coef_t = np.column_stack([np.linspace(1, -1, T), np.full(T, 0.5), np.linspace(0, 1, T)])
yt = (Xt * coef_t).sum(1) + 0.02 * (t / 100) ** 1.5 + rng.normal(0, 0.3, T)
feat = np.column_stack([Xt, t])                               # a time index as a feature: memorizes the period
train, future = slice(0, 2500), slice(2500, 3000)
rfr = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
r2_rand = cross_val_score(rfr, feat[train], yt[train], cv=KFold(5, shuffle=True, random_state=0)).mean()
r2_tss = cross_val_score(rfr, feat[train], yt[train], cv=TimeSeriesSplit(5, test_size=250)).mean()
r2_future = rfr.fit(feat[train], yt[train]).score(feat[future], yt[future])
print(f"R^2: random KFold {r2_rand:.2f}, TimeSeriesSplit {r2_tss:.2f}, the actual future {r2_future:.2f}")
assert r2_rand - r2_future > 0.2 and abs(r2_tss - r2_future) < abs(r2_rand - r2_future)

# %% [markdown]
# ## 4. Feature selection outside the fold
#
# 100 samples, 10,000 noise features, random labels. The right answer is 50%.

# %%
Xn = rng.normal(size=(100, 10_000))
yn = rng.integers(0, 2, 100)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
top = SelectKBest(f_classif, k=20).fit(Xn, yn).get_support()  # selected on ALL rows
wrong = cross_val_score(LogisticRegression(), Xn[:, top], yn, cv=cv).mean()
right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression()), Xn, yn, cv=cv).mean()
print(f"CV accuracy on pure noise: selection outside the fold {wrong:.2f}, inside the pipeline {right:.2f}")
assert wrong > 0.75 and abs(right - 0.5) < 0.15

# %% [markdown]
# ## 5. Nested CV
#
# Many hyperparameter settings, a modest dataset: the best inner CV score flatters; the nested estimate doesn't.

# %%
Xh, yh = make_clf_data(300)
Xh_new, yh_new = make_clf_data(30_000)
grid = {"C": np.logspace(-2, 3, 6), "gamma": np.logspace(-4, 1, 6)}
search = GridSearchCV(SVC(), grid, cv=StratifiedKFold(5, shuffle=True, random_state=1)).fit(Xh, yh)
nested = cross_val_score(GridSearchCV(SVC(), grid, cv=StratifiedKFold(5, shuffle=True, random_state=1)),
                         Xh, yh, cv=StratifiedKFold(5, shuffle=True, random_state=2)).mean()
true_acc = search.best_estimator_.score(Xh_new, yh_new)
print(f"best grid-search CV score {search.best_score_:.3f}; nested CV {nested:.3f}; true accuracy of the chosen model {true_acc:.3f}")
# nested CV estimates the tune-then-train *procedure* averaged over datasets like this one, so it won't match this particular
# model exactly; what matters is that it's much less flattering than the best inner score
assert search.best_score_ > true_acc and abs(nested - true_acc) < search.best_score_ - true_acc + 0.01

# %% [markdown]
# ## 6. Comparing two models honestly

# %%
Xc, yc = make_clf_data(1500)
cvr = [StratifiedKFold(10, shuffle=True, random_state=s) for s in range(5)]   # 5 x 10-fold, same folds for both models
a = np.concatenate([cross_val_score(LogisticRegression(), Xc, yc, cv=c) for c in cvr])
b = np.concatenate([cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), Xc, yc, cv=c) for c in cvr])
d = a - b
k = len(d)
n_test, n_train = len(yc) / 10, len(yc) * 9 / 10
t_naive = d.mean() / (d.std(ddof=1) * np.sqrt(1 / k))
t_corr = d.mean() / (d.std(ddof=1) * np.sqrt(1 / k + n_test / n_train))
p_naive = 2 * stats.t.sf(abs(t_naive), k - 1)
p_corr = 2 * stats.t.sf(abs(t_corr), k - 1)
print(f"logistic {a.mean():.3f} vs forest {b.mean():.3f}; mean difference {d.mean():+.3f}, fold sd of each model {a.std():.3f}")
print(f"paired t: naive p = {p_naive:.2g}, Nadeau-Bengio corrected p = {p_corr:.2g}")
assert p_corr > p_naive, "the correction is always more conservative"

# %% [markdown]
# ## 7. Adversarial validation

# %%
Xa_train, _ = make_clf_data(2000)
Xa_same, _ = make_clf_data(2000)
Xa_shift, _ = make_clf_data(2000)
Xa_shift[:, 2] += 0.8                                         # one upstream feature changed (a unit change, a new source...)


def adversarial_auc(A, B):
    Xall = np.vstack([A, B]); lab = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1)
    proba = cross_val_predict(clf, Xall, lab, cv=5, method="predict_proba")[:, 1]
    imp = clf.fit(Xall, lab).feature_importances_
    return roc_auc_score(lab, proba), int(np.argmax(imp))


auc_same, _ = adversarial_auc(Xa_train, Xa_same)
auc_shift, culprit = adversarial_auc(Xa_train, Xa_shift)
print(f"adversarial AUC: same distribution {auc_same:.2f}; shifted {auc_shift:.2f}, most telling feature: {culprit}")
assert abs(auc_same - 0.5) < 0.05 and auc_shift > 0.6 and culprit == 2

print("\nAll checks passed.")
