# %% [markdown]
# # Lab 10.2: One question, three engines
#
# 1. Write the shop to Parquet.
# 2. Revenue by country and month in pandas, Polars (lazy) and DuckDB: identical results, different speed.
# 3. The Polars optimizer at work: explain() shows predicate and projection pushdown.
# 4. Window expressions: Polars .over() vs pandas transform vs SQL.
# 5. Larger-than-memory: DuckDB aggregating a Parquet dataset with a tiny memory limit.
# 6. Arrow interop: moving tables between engines without copying.

# %%
import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

tmp = Path(tempfile.mkdtemp())
t = make_shop(n_customers=40_000, seed=7)
lines = (t["order_items"].merge(t["orders"], on="order_id", validate="many_to_one")
         .assign(amount=lambda d: d["quantity"] * d["unit_price"]))
lines = pd.concat([lines] * 4, ignore_index=True)        # ~1M rows: enough to see differences
lines.to_parquet(tmp / "lines.parquet")
t["customers"].to_parquet(tmp / "customers.parquet")
print(f"lines: {len(lines):,} rows")

# %% [markdown]
# ## 2. Same query, three engines

# %%
def timed(fn):
    t0 = time.perf_counter(); out = fn(); return out, time.perf_counter() - t0


def with_pandas():
    li = pd.read_parquet(tmp / "lines.parquet")
    cu = pd.read_parquet(tmp / "customers.parquet", columns=["customer_id", "country"])
    d = li[li["status"] == "delivered"].merge(cu, on="customer_id", validate="many_to_one")
    d["month"] = d["order_ts"].dt.to_period("M").dt.to_timestamp()
    return d.groupby(["country", "month"], as_index=False)["amount"].sum().sort_values(["country", "month"]).reset_index(drop=True)


def with_polars():
    return (pl.scan_parquet(tmp / "lines.parquet")
            .filter(pl.col("status") == "delivered")
            .join(pl.scan_parquet(tmp / "customers.parquet").select("customer_id", "country"), on="customer_id")
            .group_by("country", pl.col("order_ts").dt.truncate("1mo").alias("month"))
            .agg(pl.col("amount").sum())
            .sort("country", "month")
            .collect())


def with_duckdb():
    return duckdb.sql(f"""
        SELECT c.country, date_trunc('month', l.order_ts) AS month, SUM(l.amount) AS amount
        FROM '{tmp / "lines.parquet"}' l JOIN '{tmp / "customers.parquet"}' c USING (customer_id)
        WHERE l.status = 'delivered'
        GROUP BY ALL ORDER BY country, month
    """).df()


r_pd, t_pd = timed(with_pandas)
r_pl, t_pl = timed(with_polars)
r_dk, t_dk = timed(with_duckdb)
print(f"pandas {t_pd:.2f}s   polars lazy {t_pl:.2f}s   duckdb {t_dk:.2f}s   ({len(r_pd)} result rows)")
r_pl = r_pl.to_pandas()
assert len(r_pd) == len(r_pl) == len(r_dk)
assert (r_pd["country"].to_numpy() == r_pl["country"].to_numpy()).all()
assert np.allclose(r_pd["amount"], r_pl["amount"]) and np.allclose(r_pd["amount"], r_dk["amount"])
assert (pd.to_datetime(r_pd["month"]).to_numpy() == pd.to_datetime(r_dk["month"]).to_numpy()).all()

# %% [markdown]
# ## 3. What the optimizer did

# %%
plan = (pl.scan_parquet(tmp / "lines.parquet")
        .join(pl.scan_parquet(tmp / "customers.parquet"), on="customer_id")
        .filter(pl.col("status") == "delivered")            # written AFTER the join...
        .select("country", "amount")
        .explain())
print(plan)
left_block = plan[plan.index("LEFT PLAN"):plan.index("RIGHT PLAN")]
assert "lines.parquet" in left_block and 'SELECTION: col("status")' in left_block, \
    "the filter written after the join was pushed down into the lines.parquet scan"
assert "PROJECT" in left_block, "and only the needed columns are read"

# %% [markdown]
# ## 4. Window expressions in three dialects

# %%
small = lines.head(50_000).assign(rn=np.arange(50_000))
pd_share = small["amount"] / small.groupby("customer_id")["amount"].transform("sum")
pl_share = pl.from_pandas(small).select(pl.col("amount") / pl.col("amount").sum().over("customer_id")).to_series().to_numpy()
dk_share = duckdb.sql("SELECT amount / SUM(amount) OVER (PARTITION BY customer_id) AS s FROM small ORDER BY rn").df()["s"]
assert np.allclose(pd_share.to_numpy(), pl_share) and np.allclose(dk_share.to_numpy(), pl_share)

# %% [markdown]
# ## 5. Larger than memory
#
# We write ~4M rows to 8 Parquet files and aggregate them with DuckDB capped at 64 MB of RAM and one thread: it
# streams through the files and spills if it must. The raw data is several times the memory limit.

# %%
big_dir = tmp / "events"
big_dir.mkdir()
rng = np.random.default_rng(0)
for i in range(8):
    n = 500_000
    pd.DataFrame({
        "user_id": rng.integers(0, 200_000, n),
        "country": rng.choice(["AR", "ES", "US", "DE", "PL"], n),
        "amount": rng.gamma(2, 20, n),
        "payload": rng.integers(0, 1 << 62, n),
    }).to_parquet(big_dir / f"part-{i}.parquet")
on_disk = sum(f.stat().st_size for f in big_dir.iterdir())
con = duckdb.connect()
con.execute("SET memory_limit = '64MB'; SET threads = 1;")
con.execute(f"SET temp_directory = '{tmp / 'spill'}'")
res = con.execute(f"""
    SELECT country, COUNT(*) AS n, COUNT(DISTINCT user_id) AS users, SUM(amount) AS revenue
    FROM '{big_dir}/*.parquet' GROUP BY country ORDER BY country
""").df()
print(f"{on_disk / 1e6:.0f} MB of Parquet (~{4_000_000 * 32 / 1e6:.0f} MB in memory as raw columns), aggregated under a 64 MB limit:")
print(res)
assert res["n"].sum() == 4_000_000

# %% [markdown]
# ## 6. Arrow: zero-copy hand-offs

# %%
arrow_tbl = duckdb.sql(f"SELECT * FROM '{tmp / 'lines.parquet'}' LIMIT 200000").arrow()
if isinstance(arrow_tbl, pa.RecordBatchReader):
    arrow_tbl = arrow_tbl.read_all()
t0 = time.perf_counter()
pl_df = pl.from_arrow(arrow_tbl)
t_conv = time.perf_counter() - t0
back = duckdb.sql("SELECT SUM(amount) FROM pl_df").fetchone()[0]
assert np.isclose(back, arrow_tbl.column("amount").to_numpy().sum())
print(f"DuckDB -> Arrow -> Polars -> DuckDB: conversion took {t_conv * 1000:.2f} ms for {arrow_tbl.num_rows:,} rows")

shutil.rmtree(tmp)
print("\nAll checks passed.")
