# %% [markdown]
# # Lab 16.1: Baselines first, and targets are decisions
#
# 1. Regression baselines on the fuel consumption data: mean, median, a one-feature rule, then a linear model.
# 2. Classification baselines on the telecom customers: majority class, prior probability (log loss), then k-NN.
# 3. The churn target: how the definition (window length) changes the base rate and who counts as a churner.
# 4. ERM over two hypothesis classes: empirical risk vs held-out risk.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "part-2-data-engineering" / "_shared"))
from shop import make_shop  # noqa: E402

# %% [markdown]
# ## 1. Regression: CO2 emissions

# %%
fuel = pd.read_csv(ROOT / "data" / "fuel_consumption_co2.csv")
X = fuel[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]]
y = fuel["CO2EMISSIONS"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

results = {
    "predict the mean": DummyRegressor(strategy="mean").fit(Xtr, ytr).predict(Xte),
    "predict the median": DummyRegressor(strategy="median").fit(Xtr, ytr).predict(Xte),
}
# A domain rule: CO2 (g/km) is roughly proportional to fuel burned (~23 g per litre per 100 km for gasoline)
results["physics rule: 23 x L/100km"] = 23.0 * Xte["FUELCONSUMPTION_COMB"].to_numpy()
results["linear regression (3 features)"] = LinearRegression().fit(Xtr, ytr).predict(Xte)
maes = {k: mean_absolute_error(yte, v) for k, v in results.items()}
for k, v in maes.items():
    print(f"{k:34s} MAE {v:6.1f} g/km")
assert maes["physics rule: 23 x L/100km"] < maes["predict the median"] / 2, "a one-line domain rule is a strong baseline"
assert maes["physics rule: 23 x L/100km"] < maes["linear regression (3 features)"], \
    "and here it BEATS the generic model: the baseline just told us the model is missing something"

# What is it missing? The rule's residuals are concentrated in diesel (D) and ethanol (E) cars: CO2 per litre depends on the fuel.
test = fuel.loc[Xte.index]
resid = (yte - results["physics rule: 23 x L/100km"]).abs().groupby(test["FUELTYPE"]).mean()
print("physics-rule error by fuel type:", resid.round(1).to_dict())
ratio = (fuel["CO2EMISSIONS"] / fuel["FUELCONSUMPTION_COMB"]).groupby(fuel["FUELTYPE"]).median()
print("g CO2 per (L/100km), by fuel type:", ratio.round(1).to_dict())

# Domain-informed features: one slope per fuel type (fuel consumption x fuel-type indicator)
dummies = pd.get_dummies(fuel["FUELTYPE"], dtype=float)
Xd = pd.concat([dummies.mul(fuel["FUELCONSUMPTION_COMB"], axis=0).add_suffix("_x_consumption"), dummies], axis=1)
Xd_tr, Xd_te = Xd.loc[Xtr.index], Xd.loc[Xte.index]
mae_domain = mean_absolute_error(yte, LinearRegression().fit(Xd_tr, ytr).predict(Xd_te))
print(f"{'linear, fuel-type x consumption':34s} MAE {mae_domain:6.2f} g/km")
assert mae_domain < 1.0, "with the right features the relationship is almost exactly linear"

# %% [markdown]
# ## 2. Classification: customer category (4 classes)

# %%
tele = pd.read_csv(ROOT / "data" / "telecust_1000.csv")
Xc, yc = tele.drop(columns="custcat"), tele["custcat"]
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0, stratify=yc)
maj = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
prior = DummyClassifier(strategy="prior").fit(Xtr, ytr)
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr, ytr)
rows = [
    ("majority class", accuracy_score(yte, maj.predict(Xte)), np.nan),
    ("class prior probabilities", accuracy_score(yte, prior.predict(Xte)), log_loss(yte, prior.predict_proba(Xte))),
    ("k-NN (k=25, scaled)", accuracy_score(yte, knn.predict(Xte)), log_loss(yte, knn.predict_proba(Xte))),
]
print(pd.DataFrame(rows, columns=["model", "accuracy", "log_loss"]).round(3).to_string(index=False))
print(f"class balance: {yc.value_counts(normalize=True).round(3).to_dict()}")
assert rows[2][1] > rows[0][1], "k-NN beats the majority class..."
assert rows[2][1] - rows[0][1] < 0.2, "...but not by much: this is a hard problem, and only the baseline tells you so"
assert rows[2][2] < rows[1][2]

# %% [markdown]
# ## 3. The churn target depends on its definition

# %%
t = make_shop(n_customers=8_000, seed=16)
orders = t["orders"][t["orders"]["status"] != "cancelled"]
PRED_DATE = pd.Timestamp("2025-03-01")
hist = orders[orders["order_ts"] < PRED_DATE]
active = hist.loc[hist["order_ts"] >= PRED_DATE - pd.Timedelta(days=90), "customer_id"].unique()
future = orders[orders["order_ts"] >= PRED_DATE]
churn = {}
for window in (30, 60, 90, 180):
    bought = future.loc[future["order_ts"] < PRED_DATE + pd.Timedelta(days=window), "customer_id"].unique()
    churn[window] = pd.Series(~np.isin(active, bought), index=active)
    print(f"churn = no order in the next {window:3d} days: base rate {churn[window].mean():.1%} of {len(active):,} active customers")
agree = (churn[30] == churn[180]).mean()
print(f"the 30-day and 180-day definitions disagree on {1 - agree:.0%} of customers")
assert churn[30].mean() > churn[180].mean() + 0.2 and agree < 0.8

# %% [markdown]
# ## 4. ERM: empirical risk vs held-out risk for two hypothesis classes

# %%
rng = np.random.default_rng(0)
xs = rng.uniform(-3, 3, 60)
ys = np.sin(xs) + rng.normal(0, 0.3, 60)
x_new = rng.uniform(-3, 3, 5000)
y_new = np.sin(x_new) + rng.normal(0, 0.3, 5000)
for deg in (1, 3, 15):
    coef = np.polyfit(xs, ys, deg)
    emp = np.mean((np.polyval(coef, xs) - ys) ** 2)
    true = np.mean((np.polyval(coef, x_new) - y_new) ** 2)
    print(f"polynomials of degree {deg:2d}: empirical risk {emp:.3f}   risk on new data {true:.3f}")
    if deg == 15:
        assert emp < 0.09 and true > emp, "the richest class wins on the sample and pays on new data"
print("\nAll checks passed.")
