# %% [markdown]
# # Lab 23.1: Forecasting, backtested honestly
#
# Synthetic daily sales for 40 stores over 3 years: store level, growth trend, weekly and yearly seasonality,
# promotions (known in advance) and holidays, multiplicative noise.
#
# 1. Random K-fold vs time-ordered CV for the same model: the self-check, measured.
# 2. One store, rolling-origin backtest: naive, seasonal naive, ETS (damped), SARIMA. MASE by method and horizon.
# 3. A global gradient-boosting model with lag features, direct 7-day-ahead forecasts, across all stores.
# 4. Trees can't extrapolate a trend: level target vs ratio-to-recent-level target.
# 5. Prediction intervals: coverage of quantile boosting and of split conformal.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score  # noqa: E402
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng(231)

# %% [markdown]
# ## The data

# %%
n_stores, n_days = 40, 3 * 365
dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
t = np.arange(n_days)
weekly = np.array([0.8, 0.85, 0.9, 0.95, 1.15, 1.35, 1.0])      # Mon..Sun
holidays = set(pd.to_datetime(["2023-12-25", "2024-12-25", "2025-12-25", "2023-01-01", "2024-01-01", "2025-01-01",
                                "2023-11-24", "2024-11-29", "2025-11-28"]))
rows = []
for s in range(n_stores):
    level = rng.lognormal(4.5, 0.5)
    growth = rng.normal(0.0006, 0.0005)                          # per day
    yearly_amp = rng.uniform(0.05, 0.25)
    promo = rng.random(n_days) < 0.08
    mult = (level * (1 + growth * t) * weekly[dates.dayofweek] * (1 + yearly_amp * np.sin(2 * np.pi * (t - 80) / 365.25))
            * np.where(promo, 1.4, 1.0) * np.where(dates.isin(holidays), 0.3, 1.0))
    sales = rng.poisson(mult * rng.lognormal(0, 0.1, n_days))
    rows.append(pd.DataFrame({"store": s, "date": dates, "sales": sales.astype(float), "promo": promo.astype(int)}))
df = pd.concat(rows, ignore_index=True)
print(f"{n_stores} stores x {n_days} days; mean daily sales {df['sales'].mean():.0f}")

# %% [markdown]
# ## Features for the ML approach (direct model for horizon H = 7: only lags >= 7)

# %%
H = 7


def add_features(d):
    d = d.sort_values(["store", "date"]).copy()
    g = d.groupby("store")["sales"]
    for lag in (7, 14, 21, 28, 364):
        d[f"lag_{lag}"] = g.shift(lag)
    d["roll_mean_28"] = g.transform(lambda x: x.shift(H).rolling(28).mean())
    d["roll_std_28"] = g.transform(lambda x: x.shift(H).rolling(28).std())
    d["dow"] = d["date"].dt.dayofweek
    d["doy"] = d["date"].dt.dayofyear
    d["holiday"] = d["date"].isin(holidays).astype(int)
    return d


feat = add_features(df).dropna().reset_index(drop=True)
FEATS = ["store", "promo", "lag_7", "lag_14", "lag_21", "lag_28", "lag_364", "roll_mean_28", "roll_std_28", "dow", "doy", "holiday"]

# %% [markdown]
# ## 1. Random K-fold vs time-ordered CV
#
# Target: the ratio of sales to the recent level (section 4 explains why). Same model, two ways of splitting.

# %%
feat["ratio"] = feat["sales"] / feat["roll_mean_28"]
by_time = feat.sort_values("date").reset_index(drop=True)
gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features=[0], random_state=0)
r2_random = cross_val_score(gbm, by_time[FEATS], by_time["ratio"], cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
r2_time = cross_val_score(gbm, by_time[FEATS], by_time["ratio"], cv=TimeSeriesSplit(5, gap=H * n_stores), scoring="r2").mean()
print(f"R^2: random K-fold {r2_random:.3f}, time-ordered CV {r2_time:.3f}")
assert r2_random > r2_time

# %% [markdown]
# ## 2. One store: statistical models, rolling-origin backtest

# %%
def mase(y_true, y_pred, y_train, m=7):
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return np.mean(np.abs(y_true - y_pred)) / scale


store0 = df[df["store"] == 3].set_index("date")["sales"]
origins = pd.date_range("2025-03-01", periods=6, freq="28D")
HZ = 28
results = {k: [] for k in ("naive", "seasonal naive", "ETS damped", "SARIMA")}
by_h = {k: np.zeros(HZ) for k in results}
for origin in origins:
    train = store0[store0.index < origin].to_numpy()
    test = store0[(store0.index >= origin) & (store0.index < origin + pd.Timedelta(days=HZ))].to_numpy()
    preds = {
        "naive": np.repeat(train[-1], HZ),
        "seasonal naive": np.tile(train[-7:], HZ // 7),
        "ETS damped": ExponentialSmoothing(train[-730:], trend="add", damped_trend=True, seasonal="mul", seasonal_periods=7).fit().forecast(HZ),
        "SARIMA": SARIMAX(np.log1p(train[-365:]), order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False).forecast(HZ),
    }
    preds["SARIMA"] = np.expm1(preds["SARIMA"])
    for k, p in preds.items():
        results[k].append(mase(test, p, train))
        by_h[k] += np.abs(test - p) / len(origins)
for k in results:
    print(f"{k:15s} MASE {np.mean(results[k]):.3f}   MAE day 1-7 {by_h[k][:7].mean():6.1f}, day 22-28 {by_h[k][21:].mean():6.1f}")
assert np.mean(results["seasonal naive"]) < np.mean(results["naive"])
assert min(np.mean(results["ETS damped"]), np.mean(results["SARIMA"])) < np.mean(results["seasonal naive"])

# %% [markdown]
# ## 3. A global model across all stores, backtested

# %%
cut_dates = pd.date_range("2025-06-01", periods=4, freq="28D")
wape = {"seasonal naive": [], "ETS per store": [], "global GBDT": []}
for cut in cut_dates:
    tr = feat[feat["date"] < cut - pd.Timedelta(days=H - 1)]      # labels must be known by the cut: gap of H-1 days
    te = feat[(feat["date"] >= cut) & (feat["date"] < cut + pd.Timedelta(days=H))]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features=[0], random_state=0)
    model.fit(tr[FEATS], tr["ratio"])
    p_gbm = model.predict(te[FEATS]) * te["roll_mean_28"]
    wape["global GBDT"].append(np.abs(te["sales"] - p_gbm).sum() / te["sales"].sum())
    wape["seasonal naive"].append(np.abs(te["sales"] - te["lag_7"]).sum() / te["sales"].sum())
    err = tot = 0.0
    for s in range(n_stores):
        hist = df[(df["store"] == s) & (df["date"] < cut)]["sales"].to_numpy()[-730:]
        actual = te[te["store"] == s].sort_values("date")["sales"].to_numpy()
        f = ExponentialSmoothing(hist, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=7).fit().forecast(H)
        err += np.abs(actual - f[: len(actual)]).sum(); tot += actual.sum()
    wape["ETS per store"].append(err / tot)
for k, v in wape.items():
    print(f"{k:15s} WAPE over {len(cut_dates)} backtest weeks, 40 stores: {np.mean(v):.3f}")
print("the global model knows about promotions and holidays in advance; ETS doesn't. Covariates are where ML earns its keep.")
assert np.mean(wape["global GBDT"]) < np.mean(wape["seasonal naive"])

# %% [markdown]
# ## 4. Trees and trends

# %%
tt = np.arange(900)
trend_series = 100 + 0.5 * tt + 10 * weekly[tt % 7] + rng.normal(0, 5, 900)
td = pd.DataFrame({"y": trend_series, "dow": tt % 7})
td["lag_7"] = td["y"].shift(7); td["roll"] = td["y"].shift(7).rolling(28).mean()
td = td.dropna()
train_td, test_td = td.iloc[:700], td.iloc[700:]
level = HistGradientBoostingRegressor(random_state=0).fit(train_td[["dow", "lag_7", "roll"]], train_td["y"])
ratio = HistGradientBoostingRegressor(random_state=0).fit(train_td[["dow", "lag_7", "roll"]], train_td["y"] / train_td["roll"])
p_level = level.predict(test_td[["dow", "lag_7", "roll"]])
p_ratio = ratio.predict(test_td[["dow", "lag_7", "roll"]]) * test_td["roll"]
print(f"trending series, test period beyond the training range: max training y {train_td['y'].max():.0f}, max test y {test_td['y'].max():.0f}")
print(f"GBDT on the level: max prediction {p_level.max():.0f}, MAE {np.mean(np.abs(p_level - test_td['y'])):.1f}; "
      f"on the ratio to the recent level: MAE {np.mean(np.abs(p_ratio - test_td['y'])):.1f}")
assert p_level.max() <= train_td["y"].max() + 1 and np.mean(np.abs(p_ratio - test_td["y"])) < 0.3 * np.mean(np.abs(p_level - test_td["y"]))

# %% [markdown]
# ## 5. Prediction intervals: quantile boosting vs split conformal

# %%
cal_cut, test_cut = pd.Timestamp("2025-08-01"), pd.Timestamp("2025-10-01")
tr = feat[feat["date"] < cal_cut - pd.Timedelta(days=H - 1)]
cal = feat[(feat["date"] >= cal_cut) & (feat["date"] < test_cut - pd.Timedelta(days=H - 1))]
te = feat[feat["date"] >= test_cut]
q = {a: HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=300, learning_rate=0.05, categorical_features=[0],
                                      random_state=0).fit(tr[FEATS], tr["ratio"]) for a in (0.05, 0.95)}
lo, hi = (q[a].predict(te[FEATS]) * te["roll_mean_28"] for a in (0.05, 0.95))
cov_q = np.mean((te["sales"] >= lo) & (te["sales"] <= hi))
point = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features=[0], random_state=0).fit(tr[FEATS], tr["ratio"])
resid_cal = np.abs(cal["sales"] - point.predict(cal[FEATS]) * cal["roll_mean_28"]) / cal["roll_mean_28"]   # scaled residuals
qhat = np.quantile(resid_cal, np.ceil((len(resid_cal) + 1) * 0.9) / len(resid_cal))
center = point.predict(te[FEATS]) * te["roll_mean_28"]
cov_c = np.mean(np.abs(te["sales"] - center) <= qhat * te["roll_mean_28"])
print(f"90% intervals on the last months: quantile boosting coverage {cov_q:.3f}, split conformal coverage {cov_c:.3f}")
assert abs(cov_c - 0.9) < 0.04

print("\nAll checks passed.")
