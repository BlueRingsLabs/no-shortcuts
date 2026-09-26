# %% [markdown]
# # Lab 08.3: Reading plans and making queries fast
#
# SQLite plays the operational (row-store, B-tree) database; DuckDB plays the analytical (columnar) one.
#
# 1. A point lookup: full scan vs index, with EXPLAIN QUERY PLAN and timings.
# 2. Composite index column order and the leftmost-prefix rule.
# 3. Non-sargable predicates that ignore the index, and their rewrites.
# 4. "Latest order of a customer" served straight from an index (no sort).
# 5. DuckDB: EXPLAIN ANALYZE of a hash join, and column pruning (bytes read) in a columnar engine.

# %%
import sqlite3
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

t = make_shop(n_customers=40_000, seed=2)
orders = t["orders"].assign(order_ts=t["orders"]["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
print(f"orders: {len(orders):,} rows")

db = sqlite3.connect(":memory:")
orders.to_sql("orders", db, index=False)


def plan(sql, params=()):
    return " | ".join(row[-1] for row in db.execute("EXPLAIN QUERY PLAN " + sql, params))


def timeit_once(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


def timed(sql, params=(), reps=50):
    t0 = time.perf_counter()
    for _ in range(reps):
        db.execute(sql, params).fetchall()
    return (time.perf_counter() - t0) / reps

# %% [markdown]
# ## 1. Point lookup

# %%
lookup = "SELECT * FROM orders WHERE customer_id = ?"
before_plan, before_t = plan(lookup, (123,)), timed(lookup, (123,))
db.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
db.execute("ANALYZE")
after_plan, after_t = plan(lookup, (123,)), timed(lookup, (123,))
print(f"without index: {before_plan:45s} {before_t * 1e6:8.0f} us")
print(f"with index:    {after_plan:45s} {after_t * 1e6:8.0f} us")
assert "SCAN" in before_plan and "USING INDEX" in after_plan
assert before_t > 20 * after_t

# %% [markdown]
# ## 2. Composite index column order

# %%
db.execute("DROP INDEX idx_orders_customer")
db.execute("CREATE INDEX idx_ts_cust ON orders(order_ts, customer_id)")          # range column first: wrong order
q2 = "SELECT order_id FROM orders WHERE customer_id = ? AND order_ts >= ?"
p_wrong, t_wrong = plan(q2, (123, "2024-06-01")), timed(q2, (123, "2024-06-01"))
db.execute("CREATE INDEX idx_cust_ts ON orders(customer_id, order_ts)")          # equality first, then range
db.execute("ANALYZE")
p_right, t_right = plan(q2, (123, "2024-06-01")), timed(q2, (123, "2024-06-01"))
print(f"(order_ts, customer_id) only: {p_wrong}  {t_wrong * 1e6:.0f} us")
print(f"with (customer_id, order_ts): {p_right}  {t_right * 1e6:.0f} us")
assert "idx_cust_ts" in p_right

# Leftmost prefix: the (customer_id, order_ts) index can't serve a filter on order_ts alone
p_ts_only = plan("SELECT COUNT(*) FROM orders WHERE order_ts >= '2025-06-01'")
print("filter on order_ts alone uses:", p_ts_only)
assert "idx_cust_ts" not in p_ts_only

# %% [markdown]
# ## 3. Sargable vs non-sargable

# %%
db.execute("CREATE INDEX idx_ts ON orders(order_ts)")
db.execute("ANALYZE")
non_sarg = "SELECT COUNT(*) FROM orders WHERE date(order_ts) = '2024-09-15'"
sarg = "SELECT COUNT(*) FROM orders WHERE order_ts >= '2024-09-15' AND order_ts < '2024-09-16'"
assert db.execute(non_sarg).fetchone() == db.execute(sarg).fetchone(), "same answer"
p_ns, p_s = plan(non_sarg), plan(sarg)
t_ns, t_s = timed(non_sarg), timed(sarg)
print(f"date(order_ts) = ...     {p_ns:55s} {t_ns * 1e6:7.0f} us")
print(f"range on raw order_ts    {p_s:55s} {t_s * 1e6:7.0f} us")
assert "SCAN" in p_ns and "SEARCH" in p_s and t_ns > 5 * t_s

# %% [markdown]
# ## 4. Latest order per customer, straight from the index

# %%
latest = "SELECT order_id, order_ts FROM orders WHERE customer_id = ? ORDER BY order_ts DESC LIMIT 1"
p_latest = plan(latest, (123,))
print("latest order plan:", p_latest)
assert "USE TEMP B-TREE FOR ORDER BY" not in p_latest, "no sort needed: the index is already ordered"

# %% [markdown]
# ## 5. DuckDB: hash joins and column pruning

# %%
con = duckdb.connect()
for name, df in t.items():
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
prof = con.execute("""
EXPLAIN ANALYZE
SELECT c.country, SUM(oi.quantity * oi.unit_price) AS revenue
FROM orders o JOIN customers c USING (customer_id) JOIN order_items oi USING (order_id)
WHERE o.order_ts >= TIMESTAMP '2025-01-01'
GROUP BY c.country
""").fetchall()[0][1]
print("\n".join(line for line in prof.splitlines() if any(k in line for k in ("HASH_JOIN", "SEQ_SCAN", "HASH_GROUP_BY", "Total Time"))))
assert "HASH_JOIN" in prof

# Columnar storage: reading one column of a wide table is much cheaper than reading all of them.
wide = pd.DataFrame(np.random.default_rng(0).normal(size=(300_000, 40)), columns=[f"c{i}" for i in range(40)])
path = Path(__file__).with_name("_wide.parquet")
wide.to_parquet(path)
all_cols = " + ".join(f"c{i}" for i in range(40))
t_one = min(timeit_once(lambda: con.execute(f"SELECT AVG(c0) FROM '{path}'").fetchall()) for _ in range(3))
t_all = min(timeit_once(lambda: con.execute(f"SELECT AVG({all_cols}) FROM '{path}'").fetchall()) for _ in range(3))
path.unlink()
print(f"Parquet, 40 columns: aggregate reading 1 column {t_one * 1000:.1f} ms   reading all 40 {t_all * 1000:.1f} ms")
assert t_all > 3 * t_one, "a columnar engine only reads the columns the query touches"

# %%
print("\nAll checks passed.")
