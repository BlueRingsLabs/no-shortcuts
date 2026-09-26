# %% [markdown]
# # Lab 20.4: Gradient-boosted trees in practice
#
# A synthetic credit dataset with the usual trouble: a 300-level categorical, informative missing values, a feature
# that should act monotonically, and 10% positives.
#
# 1. XGBoost, LightGBM, HistGradientBoosting (and CatBoost if installed) with early stopping, side by side.
# 2. Categorical handling: integer codes vs one-hot vs out-of-fold target encoding vs native.
# 3. Missing values: native vs sentinel vs mean imputation.
# 4. Monotone constraints: verified on a grid, and what they cost.
# 5. A custom asymmetric objective in XGBoost.
# 6. Three built-in importances and permutation importance disagree.
# 7. scale_pos_weight and calibration.
# 8. Save in native formats, reload, identical predictions.

# %%
import tempfile
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

try:
    import catboost
except ImportError:                                             # optional: pip install catboost (requirements-extras.txt)
    catboost = None

warnings.filterwarnings("ignore", category=UserWarning)
rng = np.random.default_rng(204)

# %% [markdown]
# ## The data

# %%
n = 40_000
income = rng.lognormal(10.8, 0.5, n)
age = rng.uniform(21, 75, n)
debt_ratio = rng.beta(2, 5, n)
merchant = rng.integers(0, 300, n)
merchant_effect = rng.normal(0, 0.8, 300)
history = rng.gamma(3, 30, n)
history_missing = rng.random(n) < 0.15
logit = (-2.4 - 1.1 * (np.log(income) - 10.8) + 2.5 * (debt_ratio - 0.28) - 0.01 * (age - 45)
         + merchant_effect[merchant] - 0.006 * (history - 90) + 1.0 * history_missing)   # no history = riskier
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
history_obs = np.where(history_missing, np.nan, history)
df = pd.DataFrame({"income": income, "age": age, "debt_ratio": debt_ratio,
                   "merchant": pd.Categorical(merchant), "history_months": history_obs})
print(f"{n:,} rows, default rate {y.mean():.1%}, missing history {np.isnan(history_obs).mean():.0%}")

X_trv, X_te, y_trv, y_te = train_test_split(df, y, test_size=0.25, random_state=0, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_trv, y_trv, test_size=0.2, random_state=0, stratify=y_trv)   # val = early stopping only


def report(name, p, t, rounds):
    print(f"{name:26s} AUC {roc_auc_score(y_te, p):.4f}  log loss {log_loss(y_te, p):.4f}  fit {t:5.1f}s  rounds {rounds}")
    return roc_auc_score(y_te, p)

# %% [markdown]
# ## 1. Four libraries, early stopping on a validation set

# %%
aucs = {}
t0 = time.perf_counter()
xgb_m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                          enable_categorical=True, tree_method="hist", early_stopping_rounds=100, eval_metric="logloss",
                          random_state=0, n_jobs=4)
xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
aucs["xgboost"] = report("XGBoost", xgb_m.predict_proba(X_te)[:, 1], time.perf_counter() - t0, xgb_m.best_iteration + 1)

t0 = time.perf_counter()
lgb_m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, cat_smooth=20, random_state=0, n_jobs=4, verbose=-1)
lgb_m.fit(X_tr, y_tr, eval_X=(X_val,), eval_y=(y_val,), callbacks=[lgb.early_stopping(100, verbose=False)])
aucs["lightgbm"] = report("LightGBM", lgb_m.predict_proba(X_te)[:, 1], time.perf_counter() - t0, lgb_m.best_iteration_)

# scikit-learn's native categorical support stops at 255 levels (one histogram bin per category): with 300 merchants,
# HistGradientBoosting refuses. A real gotcha. Target-encode the merchant (out-of-fold, inside the pipeline) instead.
try:
    HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=5).fit(X_tr, y_tr)
except ValueError as e:
    print("HistGB native categorical:", str(e).split(" but")[0])
t0 = time.perf_counter()
hgb = make_pipeline(make_column_transformer((TargetEncoder(cv=KFold(5, shuffle=True, random_state=0)), ["merchant"]), remainder="passthrough"),
                    HistGradientBoostingClassifier(max_iter=3000, learning_rate=0.05, early_stopping=True, validation_fraction=0.2,
                                                   n_iter_no_change=100, random_state=0))
hgb.fit(X_trv.assign(merchant=X_trv["merchant"].astype(int)), y_trv)   # uses its own internal validation split
aucs["histgb"] = report("sklearn HistGB + target enc.", hgb.predict_proba(X_te.assign(merchant=X_te["merchant"].astype(int)))[:, 1],
                        time.perf_counter() - t0, hgb[-1].n_iter_)

if catboost is not None:
    t0 = time.perf_counter()
    cb_tr, cb_val, cb_te = (d.assign(merchant=d["merchant"].astype(int)) for d in (X_tr, X_val, X_te))
    cb = catboost.CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=6, cat_features=["merchant"],
                                     early_stopping_rounds=100, random_seed=0, verbose=False, thread_count=4,
                                     allow_writing_files=False)
    cb.fit(cb_tr, y_tr, eval_set=(cb_val, y_val))
    aucs["catboost"] = report("CatBoost", cb.predict_proba(cb_te)[:, 1], time.perf_counter() - t0, cb.get_best_iteration() + 1)
else:
    print("CatBoost not installed; skipping (pip install catboost)")

print(f"spread between libraries: {max(aucs.values()) - min(aucs.values()):.4f} AUC")
assert max(aucs.values()) - min(aucs.values()) < 0.02, "tuned sensibly, the libraries land close together"

# %% [markdown]
# ## 2. Categorical handling (LightGBM, everything else equal)

# %%
def lgb_auc(Xa, Xb, Xc, categorical="auto"):
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=31, min_child_samples=50, cat_smooth=20,
                           random_state=0, n_jobs=4, verbose=-1)
    m.fit(Xa, y_tr, eval_X=(Xb,), eval_y=(y_val,), categorical_feature=categorical, callbacks=[lgb.early_stopping(100, verbose=False)])
    return roc_auc_score(y_te, m.predict_proba(Xc)[:, 1])


codes = [d.assign(merchant=d["merchant"].cat.codes) for d in (X_tr, X_val, X_te)]
onehot = [pd.get_dummies(d, columns=["merchant"], dtype=float) for d in (X_tr, X_val, X_te)]
te = TargetEncoder(cv=KFold(5, shuffle=True, random_state=0), smooth="auto")
te_tr = te.fit_transform(X_tr[["merchant"]].astype(int), y_tr)[:, 0]                  # out-of-fold on training rows
te_val, te_te = (te.transform(d[["merchant"]].astype(int))[:, 0] for d in (X_val, X_te))
tenc = [d.assign(merchant=v) for d, v in zip((X_tr, X_val, X_te), (te_tr, te_val, te_te))]
cat_results = {
    "integer codes as numbers": lgb_auc(*codes, categorical=[]),
    "one-hot (300 columns)": lgb_auc(*onehot),
    "out-of-fold target encoding": lgb_auc(*tenc, categorical=[]),
    "native categorical": lgb_auc(X_tr, X_val, X_te),
}
drop = lgb_auc(*[d.drop(columns="merchant") for d in (X_tr, X_val, X_te)])
for k, v in cat_results.items():
    print(f"{k:30s} AUC {v:.4f}")
print(f"{'(merchant dropped entirely)':30s} AUC {drop:.4f}")
assert cat_results["native categorical"] > cat_results["integer codes as numbers"]
assert max(cat_results.values()) > drop + 0.01

# %% [markdown]
# ## 3. Missing values

# %%
def xgb_auc(A, B, C):
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=5, enable_categorical=True, early_stopping_rounds=100,
                          eval_metric="logloss", random_state=0, n_jobs=4)
    m.fit(A, y_tr, eval_set=[(B, y_val)], verbose=False)
    return roc_auc_score(y_te, m.predict_proba(C)[:, 1])


mean_hist = X_tr["history_months"].mean()
observed_hist = X_tr["history_months"].dropna().to_numpy()


def random_fill(d):                                             # "realistic" imputation: draw from the observed values
    d = d.copy(); m = d["history_months"].isna()
    d.loc[m, "history_months"] = rng.choice(observed_hist, m.sum()); return d


variants = {
    "native NaN handling": (X_tr, X_val, X_te),
    "sentinel -999": [d.fillna({"history_months": -999}) for d in (X_tr, X_val, X_te)],
    "mean imputation": [d.fillna({"history_months": mean_hist}) for d in (X_tr, X_val, X_te)],
    "random-draw imputation": [random_fill(d) for d in (X_tr, X_val, X_te)],
}
miss = {k: xgb_auc(*v) for k, v in variants.items()}
for k, v in miss.items():
    print(f"trees, {k:24s} AUC {v:.4f}")
print("any *constant* fill is a value a tree can split off, so trees keep the signal; filling with varying values destroys it")
assert miss["native NaN handling"] > miss["random-draw imputation"] + 0.005
assert abs(miss["native NaN handling"] - miss["mean imputation"]) < 0.005

# a linear model can't split a constant off: it needs the missingness as an explicit feature
from sklearn.linear_model import LogisticRegression  # noqa: E402
num = ["income", "age", "debt_ratio", "history_months"]


def lin_design(d, indicator):
    Z = d[num].copy(); Z["income"] = np.log(Z["income"])
    ind = Z["history_months"].isna().astype(float)
    Z["history_months"] = Z["history_months"].fillna(mean_hist)
    if indicator:
        Z["history_missing"] = ind
    return (Z - Z.mean()) / Z.std()


for ind in (False, True):
    lr = LogisticRegression(max_iter=1000).fit(lin_design(X_tr, ind), y_tr)
    print(f"logistic regression, mean imputation {'+ missing indicator' if ind else 'alone':22s} AUC "
          f"{roc_auc_score(y_te, lr.predict_proba(lin_design(X_te, ind))[:, 1]):.4f}")

# %% [markdown]
# ## 4. Monotone constraints

# %%
cols = list(X_tr.columns)
mono = tuple(-1 if c == "income" else 0 for c in cols)          # default risk must not increase with income
free = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=5, enable_categorical=True, early_stopping_rounds=100,
                         eval_metric="logloss", random_state=0, n_jobs=4).fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
cons = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=5, enable_categorical=True, early_stopping_rounds=100,
                         eval_metric="logloss", monotone_constraints=mono, random_state=0, n_jobs=4).fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
grid = np.quantile(X_tr["income"], np.linspace(0.01, 0.99, 60))
sample = X_te.sample(200, random_state=0)


def violations(model):
    count = 0
    for _, row in sample.iterrows():
        rows = pd.DataFrame([row] * len(grid)).astype(X_te.dtypes.to_dict()); rows["income"] = grid
        p = model.predict_proba(rows)[:, 1]
        count += np.any(np.diff(p) > 1e-9)
    return count


v_free, v_cons = violations(free), violations(cons)
auc_free, auc_cons = roc_auc_score(y_te, free.predict_proba(X_te)[:, 1]), roc_auc_score(y_te, cons.predict_proba(X_te)[:, 1])
print(f"customers whose risk goes UP somewhere as income rises: unconstrained {v_free}/200, constrained {v_cons}/200")
print(f"AUC unconstrained {auc_free:.4f}, constrained {auc_cons:.4f}")
assert v_cons == 0 and v_free > 0 and auc_cons > auc_free - 0.005

# %% [markdown]
# ## 5. A custom asymmetric objective
#
# Demand forecasting where under-forecasting costs 5x more than over-forecasting.

# %%
Xd = rng.normal(size=(20_000, 4))
demand = np.exp(1.5 + 0.4 * Xd[:, 0] - 0.3 * Xd[:, 1]) * rng.gamma(4, 0.25, 20_000)
Xd_tr, Xd_te, d_tr, d_te = train_test_split(Xd, demand, test_size=0.3, random_state=0)
W_UNDER = 5.0


def asymmetric(preds, dtrain):
    r = dtrain.get_label() - preds
    w = np.where(r > 0, W_UNDER, 1.0)
    return -2 * w * r, 2 * w                                    # gradient and Hessian of w * r^2


params = {"max_depth": 4, "eta": 0.05, "base_score": float(np.mean(d_tr))}
sym = xgb.train({**params, "objective": "reg:squarederror"}, xgb.DMatrix(Xd_tr, d_tr), 400)
asy = xgb.train(params, xgb.DMatrix(Xd_tr, d_tr), 400, obj=asymmetric)
cost = lambda p: np.mean(np.where(d_te > p, W_UNDER, 1.0) * (d_te - p) ** 2)
for name, b in [("squared error", sym), ("asymmetric", asy)]:
    p = b.predict(xgb.DMatrix(Xd_te))
    print(f"{name:14s}: under-forecast on {np.mean(p < d_te):.0%} of days, asymmetric cost {cost(p):.2f}")
p_sym, p_asy = sym.predict(xgb.DMatrix(Xd_te)), asy.predict(xgb.DMatrix(Xd_te))
assert np.mean(p_asy < d_te) < np.mean(p_sym < d_te) - 0.15 and cost(p_asy) < cost(p_sym)

# %% [markdown]
# ## 6. Importance: four answers to one question

# %%
booster = xgb_m.get_booster()
imp = {kind: booster.get_score(importance_type=kind) for kind in ("weight", "gain", "cover")}
perm = permutation_importance(xgb_m, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=0).importances_mean
print(f"{'feature':15s} {'weight':>8s} {'gain':>8s} {'cover':>8s} {'perm AUC drop':>14s}")
for i, c in enumerate(cols):
    row = [imp[k].get(c, 0.0) for k in ("weight", "gain", "cover")]
    tot = [sum(imp[k].values()) for k in ("weight", "gain", "cover")]
    print(f"{c:15s} {row[0] / tot[0]:8.1%} {row[1] / tot[1]:8.1%} {row[2] / tot[2]:8.1%} {perm[i]:14.4f}")
ranks = {k: sorted(cols, key=lambda c: -imp[k].get(c, 0)) for k in imp}
print("top feature by weight:", ranks["weight"][0], "| by gain:", ranks["gain"][0], "| by permutation:", cols[int(np.argmax(perm))])

# %% [markdown]
# ## 7. scale_pos_weight and calibration

# %%
spw = (y_tr == 0).sum() / (y_tr == 1).sum()
weighted = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=5, enable_categorical=True, early_stopping_rounds=100,
                             eval_metric="logloss", scale_pos_weight=spw, random_state=0, n_jobs=4).fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
p_plain, p_w = xgb_m.predict_proba(X_te)[:, 1], weighted.predict_proba(X_te)[:, 1]
print(f"scale_pos_weight={spw:.1f}: AUC {roc_auc_score(y_te, p_plain):.4f} -> {roc_auc_score(y_te, p_w):.4f}; "
      f"mean predicted {p_plain.mean():.3f} -> {p_w.mean():.3f} (actual {y_te.mean():.3f}); log loss {log_loss(y_te, p_plain):.3f} -> {log_loss(y_te, p_w):.3f}")
# Probabilities inflate, as 18.4 predicted. And here the ranking got worse too: weighting positives by 5.8 multiplies their
# Hessian mass, which loosens min_child_weight and the lambda shrinkage of leaf values (both are in Hessian units), so the
# weighted model effectively has less regularization. Weights change more than the prior.
assert p_w.mean() > 1.5 * y_te.mean() and log_loss(y_te, p_w) > log_loss(y_te, p_plain)
assert roc_auc_score(y_te, p_w) <= roc_auc_score(y_te, p_plain) + 0.005, "no ranking gain to pay for the broken probabilities"

# %% [markdown]
# ## 8. Save natively, reload, compare

# %%
with tempfile.TemporaryDirectory() as tmp:
    xgb_m.save_model(Path(tmp) / "model.ubj")
    re_xgb = xgb.XGBClassifier(); re_xgb.load_model(Path(tmp) / "model.ubj")
    lgb_m.booster_.save_model(str(Path(tmp) / "model.txt"))
    re_lgb = lgb.Booster(model_file=str(Path(tmp) / "model.txt"))
    same_xgb = np.array_equal(re_xgb.predict_proba(X_te)[:, 1], xgb_m.predict_proba(X_te)[:, 1])
    same_lgb = np.allclose(re_lgb.predict(X_te), lgb_m.predict_proba(X_te)[:, 1], atol=1e-12)
print(f"reloaded predictions identical: XGBoost {same_xgb}, LightGBM {same_lgb}")
print(f"library versions to pin: xgboost {xgb.__version__}, lightgbm {lgb.__version__}")
assert same_xgb and same_lgb

print("\nAll checks passed.")
