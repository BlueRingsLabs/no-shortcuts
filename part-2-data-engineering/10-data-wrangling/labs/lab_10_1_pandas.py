# %% [markdown]
# # Lab 10.1: pandas 3, properly
#
# 1. Dtypes: what type guessing does to IDs and ZIP codes, and how to prevent it.
# 2. Copy-on-write: chained assignment does nothing; .loc does the job.
# 3. Index alignment surprises.
# 4. merge(validate=...) catching a fan-out, indicator=True finding orphans.
# 5. groupby transform as a window function.
# 6. apply vs vectorized: measured.
# 7. Memory: category and narrower dtypes.
# 8. Time series: resample fills empty days.

# %%
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

t = make_shop(n_customers=5_000, seed=6)
print("pandas", pd.__version__)

# %% [markdown]
# ## 1. Dtypes

# %%
csv = "zip,customer_id,signup\n02134,17,2025-01-03\n10001,,2025-02-11\n00501,23,2025-03-09\n"
guessed = pd.read_csv(io.StringIO(csv))
explicit = pd.read_csv(io.StringIO(csv), dtype={"zip": "str", "customer_id": "Int64"}, parse_dates=["signup"])
print(guessed.dtypes.to_dict(), "\n", explicit.dtypes.to_dict())
assert guessed["zip"].iloc[0] == 2134, "type guessing ate the leading zero"
assert guessed["customer_id"].dtype == "float64", "a missing value turned IDs into floats"
assert explicit["zip"].iloc[0] == "02134" and str(explicit["customer_id"].dtype) == "Int64"
assert pd.api.types.is_datetime64_any_dtype(explicit["signup"])

# %% [markdown]
# ## 2. Copy-on-write

# %%
df = t["orders"].copy()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    df[df["status"] == "returned"]["channel"] = "REFUNDED"          # chained assignment
assert (df["channel"] != "REFUNDED").all(), "chained assignment modifies a temporary copy, never df"
assert any("Chained" in type(w.message).__name__ or "chained" in str(w.message) for w in caught)

df.loc[df["status"] == "returned", "channel"] = "REFUNDED"           # the right way
assert (df.loc[df["status"] == "returned", "channel"] == "REFUNDED").all()

sub = df[df["channel"] == "web"]
sub.loc[:, "channel"] = "changed-in-sub"
assert (df["channel"] != "changed-in-sub").all(), "modifying a subset never touches the parent under CoW"
print("copy-on-write: chained assignment is a no-op, .loc works, subsets are independent")

# %% [markdown]
# ## 3. Index alignment

# %%
a = pd.Series([10, 20, 30], index=[0, 1, 2])
b = pd.Series([1, 2, 3], index=[1, 2, 3])
s = a + b
assert s.isna().sum() == 2 and s.loc[1] == 21, "aligned by label, NaN where labels don't match"
filtered = t["orders"][t["orders"]["channel"] == "app"]                   # index is no longer 0..n-1
scores = np.arange(len(filtered))
bad = filtered["order_id"] + pd.Series(scores)                           # aligned by label: mostly NaN!
good = filtered["order_id"].to_numpy() + scores
print(f"adding a fresh Series to a filtered one: {bad.isna().mean():.0%} NaN")
assert bad.isna().mean() > 0.5 and not np.isnan(good).any()

# %% [markdown]
# ## 4. merge seatbelts

# %%
customers_dup = pd.concat([t["customers"], t["customers"].sample(40, random_state=0)])   # duplicated customer rows
try:
    t["orders"].merge(customers_dup, on="customer_id", validate="many_to_one")
    raise AssertionError("validate should have caught duplicated keys")
except pd.errors.MergeError as e:
    print("validate='many_to_one' caught it:", str(e).splitlines()[0])
silent = t["orders"].merge(customers_dup, on="customer_id")
print(f"without validate: {len(t['orders']):,} orders became {len(silent):,} rows")
assert len(silent) > len(t["orders"])

orders_orphans = t["orders"].copy()
orders_orphans.loc[orders_orphans.sample(25, random_state=1).index, "customer_id"] = 999_999
check = orders_orphans.merge(t["customers"][["customer_id"]], on="customer_id", how="left", indicator=True)
orphans = int((check["_merge"] == "left_only").sum())
assert orphans == 25
print(f"indicator=True found {orphans} orphan orders")

# %% [markdown]
# ## 5. transform = window function

# %%
items = t["order_items"].assign(amount=lambda d: d["quantity"] * d["unit_price"])
items["share"] = items["amount"] / items.groupby("order_id")["amount"].transform("sum")
assert np.allclose(items.groupby("order_id")["share"].sum(), 1.0)
items["line_rank"] = items.groupby("order_id")["amount"].rank(method="first", ascending=False)
assert (items.groupby("order_id")["line_rank"].min() == 1).all()

# %% [markdown]
# ## 6. apply vs vectorized

# %%
big = pd.concat([items] * 4, ignore_index=True)
t0 = time.perf_counter()
b1 = big.apply(lambda r: "big" if r.amount > 1000 else ("mid" if r.amount > 100 else "small"), axis=1)
t_apply = time.perf_counter() - t0
t0 = time.perf_counter()
b2 = np.select([big["amount"] > 1000, big["amount"] > 100], ["big", "mid"], default="small")
t_vec = time.perf_counter() - t0
assert (b1.to_numpy() == b2).all()
print(f"{len(big):,} rows: apply(axis=1) {t_apply:.2f}s   np.select {t_vec * 1000:.1f} ms   ({t_apply / t_vec:.0f}x)")
assert t_apply > 20 * t_vec

# %% [markdown]
# ## 7. Memory

# %%
ev = t["events"].copy()
ev["event_type_obj"] = ev["event_type"].astype(object)
before = ev["event_type_obj"].memory_usage(deep=True)
after = ev["event_type"].astype("category").memory_usage(deep=True)
print(f"event_type: object {before / 1e6:.2f} MB   category {after / 1e6:.2f} MB")
assert after < before / 5

# %% [markdown]
# ## 8. resample fills empty periods

# %%
o = t["orders"]
daily = o.set_index("order_ts").resample("D")["order_id"].count()
assert daily.index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all()
naive = o.groupby(o["order_ts"].dt.date)["order_id"].count()
print(f"groupby by date: {len(naive)} days; resample: {len(daily)} days ({(daily == 0).sum()} with zero orders)")
assert len(daily) >= len(naive)

# %%
print("\nAll checks passed.")
