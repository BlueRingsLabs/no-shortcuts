# %% [markdown]
# # Lab 22.1: Features and pipelines, done as of the right time
#
# Task: "will this customer place an order in the next 60 days?", built from the shop's raw tables.
#
# 1. Point-in-time training data: features strictly before each cutoff, labels strictly after. Several cutoffs stacked.
# 2. Recency alone vs engineered features, validated on a later cutoff.
# 3. The same features computed with all the data (a leak): what it does to the offline score.
# 4. Scaling inside vs outside CV: the size of the leak for a scaler.
# 5. Unseen categories: the pipeline copes, pd.get_dummies silently doesn't.
# 6. Cyclical time features for a linear model.
# 7. Training-serving skew: a "reimplemented" feature at serving time.

# %%
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "part-2-data-engineering" / "_shared"))
from shop import make_shop  # noqa: E402

rng = np.random.default_rng(221)
shop = make_shop(n_customers=6000, seed=22)
customers, orders, items, events, products = (shop[k] for k in ("customers", "orders", "order_items", "events", "products"))
revenue = items.assign(rev=items["quantity"] * items["unit_price"]).groupby("order_id")["rev"].sum()
orders = orders.merge(revenue.rename("revenue"), left_on="order_id", right_index=True)
cat_of_order = (items.merge(products[["product_id", "category"]], on="product_id")
                .groupby("order_id")["category"].agg(lambda c: c.mode().iloc[0]))
orders = orders.merge(cat_of_order.rename("main_category"), left_on="order_id", right_index=True, how="left")
HORIZON = pd.Timedelta(days=60)

# %% [markdown]
# ## 1. Point-in-time features

# %%
def build(cutoff, leak=False, orders_lag_days=0):
    cutoff = pd.Timestamp(cutoff)
    active = customers[customers["signup_date"] < cutoff]
    o = orders if leak else orders[orders["order_ts"] < cutoff - pd.Timedelta(days=orders_lag_days)]   # leak=True: the bug
    e = events if leak else events[events["ts"] < cutoff]
    ref = cutoff
    g = o.groupby("customer_id")
    f = pd.DataFrame(index=active["customer_id"])
    f["n_orders"] = g.size()
    f["n_orders_90d"] = o[o["order_ts"] >= ref - pd.Timedelta(days=90)].groupby("customer_id").size()
    f["spend"] = g["revenue"].sum()
    f["avg_basket"] = g["revenue"].mean()
    f["days_since_last_order"] = (ref - g["order_ts"].max()).dt.days
    f["days_since_first_order"] = (ref - g["order_ts"].min()).dt.days
    f["share_app"] = o.assign(app=o["channel"].eq("app")).groupby("customer_id")["app"].mean()
    f["top_category"] = g["main_category"].agg(lambda c: c.mode().iloc[0] if len(c) else None)
    f["events_30d"] = e[e["ts"] >= ref - pd.Timedelta(days=30)].groupby("customer_id").size()
    f["cart_adds_30d"] = e[(e["ts"] >= ref - pd.Timedelta(days=30)) & (e["event_type"] == "add_to_cart")].groupby("customer_id").size()
    f["last_order_hour"] = g["order_ts"].max().dt.hour
    f = f.join(active.set_index("customer_id")[["country", "segment", "signup_date"]])
    f["tenure_days"] = (ref - f.pop("signup_date")).dt.days
    for col in ("n_orders", "n_orders_90d", "spend", "events_30d", "cart_adds_30d"):
        f[col] = f[col].fillna(0)
    future = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] < cutoff + HORIZON)]
    y = f.index.isin(future["customer_id"]).astype(int)
    return f.reset_index(drop=True).assign(top_category=lambda d: d["top_category"].fillna("none")), y


train_cutoffs = ["2024-09-01", "2024-11-01", "2025-01-01", "2025-03-01"]
test_cutoff = "2025-04-15"
parts = [build(c) for c in train_cutoffs]
X_tr = pd.concat([p[0] for p in parts], ignore_index=True); y_tr = np.concatenate([p[1] for p in parts])
X_te, y_te = build(test_cutoff)
print(f"training rows {len(X_tr):,} from {len(train_cutoffs)} cutoffs (positive rate {y_tr.mean():.1%}); "
      f"test rows {len(X_te):,} at cutoff {test_cutoff} (positive rate {y_te.mean():.1%})")
assert events["ts"].max() > pd.Timestamp(test_cutoff) + HORIZON - pd.Timedelta(days=60)

# %% [markdown]
# ## 2. Recency alone vs engineered features (temporal validation)

# %%
num = ["n_orders", "n_orders_90d", "spend", "avg_basket", "days_since_last_order", "days_since_first_order",
       "share_app", "events_30d", "cart_adds_30d", "tenure_days", "last_order_hour"]
cat = ["country", "segment", "top_category"]
gbm = make_pipeline(
    ColumnTransformer([("num", "passthrough", num),
                       ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False), cat)]),
    HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0))
gbm.fit(X_tr, y_tr)
auc_full = roc_auc_score(y_te, gbm.predict_proba(X_te)[:, 1])
auc_recency = roc_auc_score(y_te, -X_te["days_since_last_order"].fillna(10_000))
print(f"test AUC: recency alone {auc_recency:.3f}; engineered features + boosting {auc_full:.3f}")
assert auc_full > auc_recency

# %% [markdown]
# ## 3. The same features, computed with all the data

# %%
parts_leak = [build(c, leak=True) for c in train_cutoffs]
X_leak = pd.concat([p[0] for p in parts_leak], ignore_index=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
cv_leak = cross_val_score(gbm, X_leak, y_tr, cv=cv, scoring="roc_auc").mean()
cv_clean = cross_val_score(gbm, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
print(f"cross-validated AUC: leaky features {cv_leak:.3f}, point-in-time features {cv_clean:.3f}; the real test score is {auc_full:.3f}")
leaky_model = gbm.fit(X_leak, y_tr)
print(f"the leaky model on honest test features: {roc_auc_score(y_te, leaky_model.predict_proba(X_te)[:, 1]):.3f}")
assert cv_leak > cv_clean + 0.1
gbm.fit(X_tr, y_tr)                                              # restore the honest model

# %% [markdown]
# ## 4. Scaling inside vs outside CV

# %%
lin = lambda: make_pipeline(ColumnTransformer([("num", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()),
                                                   ["n_orders", "n_orders_90d", "spend", "events_30d", "cart_adds_30d"])]),
                             LogisticRegression(max_iter=2000))
Xn = X_tr[["n_orders", "n_orders_90d", "spend", "events_30d", "cart_adds_30d"]]
inside = cross_val_score(lin(), Xn, y_tr, cv=cv, scoring="roc_auc").mean()
pre_scaled = pd.DataFrame(StandardScaler().fit_transform(np.log1p(Xn)), columns=Xn.columns)       # fitted on everything
outside = cross_val_score(make_pipeline(LogisticRegression(max_iter=2000)), pre_scaled, y_tr, cv=cv, scoring="roc_auc").mean()
print(f"logistic regression CV AUC: scaler inside the folds {inside:.4f}, fitted on all data first {outside:.4f} "
      f"(difference {outside - inside:+.4f})")
print("tiny for a scaler on 20k rows. The rule is about the steps where it isn't tiny, and about shipping one pipeline.")
assert abs(outside - inside) < 0.01

# %% [markdown]
# ## 5. Unseen categories

# %%
serve = X_te.head(5).copy()
serve.loc[serve.index[0], "country"] = "JP"                     # the business launched in Japan last week
print("pipeline prediction with a new country:", np.round(gbm.predict_proba(serve)[:, 1], 3))
train_dummies = pd.get_dummies(X_tr[cat])
serve_dummies = pd.get_dummies(serve[cat])
missing_cols = set(train_dummies.columns) - set(serve_dummies.columns)
extra_cols = set(serve_dummies.columns) - set(train_dummies.columns)
print(f"pd.get_dummies at serving time: {len(serve_dummies.columns)} columns vs {len(train_dummies.columns)} in training; "
      f"missing {len(missing_cols)}, unexpected {sorted(extra_cols)}")
assert "country_JP" in extra_cols and len(missing_cols) > 0

# %% [markdown]
# ## 6. Cyclical encoding for a linear model

# %%
hours = rng.integers(0, 24, 20_000)
p_buy = 1 / (1 + np.exp(-(-1 + 1.5 * np.cos(2 * np.pi * (hours - 21) / 24))))   # peak around 21:00, low in the morning
yb = rng.random(20_000) < p_buy
raw = cross_val_score(LogisticRegression(), hours[:, None], yb, cv=cv, scoring="roc_auc").mean()
cyc = np.column_stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)])
cyclic = cross_val_score(LogisticRegression(), cyc, yb, cv=cv, scoring="roc_auc").mean()
tree = cross_val_score(HistGradientBoostingClassifier(), hours[:, None], yb, cv=cv, scoring="roc_auc").mean()
print(f"hour of day -> purchase: linear on raw hour {raw:.3f}, linear on sin/cos {cyclic:.3f}, boosting on raw hour {tree:.3f}")
assert cyclic > raw + 0.05 and abs(tree - cyclic) < 0.01

# %% [markdown]
# ## 7. Training-serving skew

# %%
blob = pickle.dumps(gbm)                                        # ship the fitted pipeline itself
served = pickle.loads(blob)
same = np.array_equal(served.predict_proba(X_te)[:, 1], gbm.predict_proba(X_te)[:, 1])

skewed, _ = build(test_cutoff, orders_lag_days=14)             # online, the orders table loads with a 14-day delay:
                                                                # the most recent orders simply aren't there yet
auc_skew = roc_auc_score(y_te, gbm.predict_proba(skewed)[:, 1])
p_ok, p_skew = gbm.predict_proba(X_te)[:, 1], gbm.predict_proba(skewed)[:, 1]
recent = (X_te["days_since_last_order"] < 14).to_numpy()        # the slice the delay hits: customers who just bought
print(f"reloaded pipeline gives identical predictions: {same}")
print(f"serving features built from an orders table 14 days behind: overall AUC {auc_full:.3f} -> {auc_skew:.3f} (looks fine)")
print(f"but for the {recent.sum()} customers who ordered in the last 14 days: mean predicted probability {p_ok[recent].mean():.3f} -> "
      f"{p_skew[recent].mean():.3f}, actual rate {y_te[recent].mean():.3f}")
print("your most engaged customers get scored as lapsed. Nothing errors; the aggregate metric barely moves. Slice, and compare")
print("feature distributions online vs offline (51.1).")
assert same and p_ok[recent].mean() - p_skew[recent].mean() > 0.05 and abs(p_skew[recent].mean() - y_te[recent].mean()) > 2 * abs(p_ok[recent].mean() - y_te[recent].mean())

print("\nAll checks passed.")
