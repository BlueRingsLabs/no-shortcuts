# %% [markdown]
# # Lab 13.1: Pipelines you can re-run
#
# Source: an "operational" SQLite database of orders that get inserted, updated (status changes) and occasionally
# deleted. Target: a DuckDB "warehouse".
#
# 1. Naive append: re-running a day duplicates it.
# 2. Idempotent partition overwrite: re-running is harmless.
# 3. Incremental extraction by watermark: a late commit is lost without overlap, recovered with overlap + MERGE.
# 4. Hard deletes are invisible to watermarks; reconciliation finds them.
# 5. A backfill over a date range, parameterized by logical date.

# %%
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

t = make_shop(n_customers=4_000, seed=13)
src = sqlite3.connect(":memory:")
orders = t["orders"].assign(order_ts=t["orders"]["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
orders["updated_at"] = orders["order_ts"]
orders.to_sql("orders", src, index=False)
items = t["order_items"]
items.to_sql("order_items", src, index=False)
wh = duckdb.connect()


def extract_day(ds: str) -> pd.DataFrame:
    """Revenue lines for orders placed on logical date ds (a parameter, never now())."""
    return pd.read_sql("""
        SELECT substr(o.order_ts, 1, 10) AS order_date, o.order_id, SUM(i.quantity * i.unit_price) AS amount
        FROM orders o JOIN order_items i USING (order_id)
        WHERE substr(o.order_ts, 1, 10) = ? AND o.status NOT IN ('cancelled', 'returned')
        GROUP BY 1, 2""", src, params=(ds,))

# %% [markdown]
# ## 1-2. Append vs overwrite

# %%
wh.execute("CREATE TABLE rev_append (order_date VARCHAR, order_id BIGINT, amount DOUBLE)")
wh.execute("CREATE TABLE rev_overwrite (order_date VARCHAR, order_id BIGINT, amount DOUBLE)")


def load_append(ds):
    df = extract_day(ds)
    wh.execute("INSERT INTO rev_append SELECT * FROM df")


def load_overwrite(ds):
    df = extract_day(ds)
    wh.execute("BEGIN")
    wh.execute("DELETE FROM rev_overwrite WHERE order_date = ?", [ds])
    wh.execute("INSERT INTO rev_overwrite SELECT * FROM df")
    wh.execute("COMMIT")


ds = "2025-02-14"
for _ in range(3):                                 # the task "retried" twice
    load_append(ds)
    load_overwrite(ds)
n_true = len(extract_day(ds))
n_app = wh.execute("SELECT COUNT(*) FROM rev_append").fetchone()[0]
n_ovw = wh.execute("SELECT COUNT(*) FROM rev_overwrite").fetchone()[0]
print(f"{ds}: {n_true} lines; after 3 runs -> append table {n_app}, overwrite table {n_ovw}")
assert n_app == 3 * n_true and n_ovw == n_true

# %% [markdown]
# ## 3. Watermarks, late commits, overlap + MERGE
#
# We extract orders changed since the last watermark into a warehouse copy with MERGE (upsert by order_id).
# Then a "late commit" happens: a status update stamped BEFORE the watermark becomes visible only AFTER the run.

# %%
wh.execute("CREATE TABLE dim_orders (order_id BIGINT PRIMARY KEY, status VARCHAR, updated_at VARCHAR)")


def incremental_sync(watermark: str, overlap: timedelta) -> str:
    since = (datetime.fromisoformat(watermark) - overlap).strftime("%Y-%m-%d %H:%M:%S")
    batch = pd.read_sql("SELECT order_id, status, updated_at FROM orders WHERE updated_at > ?", src, params=(since,))
    wh.execute("""
        INSERT INTO dim_orders SELECT * FROM batch
        ON CONFLICT (order_id) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at
    """)                                            # MERGE / upsert: applying the same rows twice changes nothing
    return max(watermark, batch["updated_at"].max() if len(batch) else watermark)


def run_scenario(overlap):
    wh.execute("DELETE FROM dim_orders")
    src.execute("UPDATE orders SET status = 'delivered', updated_at = order_ts WHERE order_id = 1")
    wm = incremental_sync("1970-01-01 00:00:00", overlap)                  # initial load
    late_ts = (datetime.fromisoformat(wm) - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    src.execute("UPDATE orders SET status = 'returned', updated_at = ? WHERE order_id = 1", (late_ts,))
    wm = incremental_sync(wm, overlap)                                       # next run
    return wh.execute("SELECT status FROM dim_orders WHERE order_id = 1").fetchone()[0]


no_overlap = run_scenario(timedelta(0))
with_overlap = run_scenario(timedelta(hours=1))
print(f"late-committed status change: without overlap the warehouse says '{no_overlap}', with overlap '{with_overlap}'")
assert no_overlap == "delivered" and with_overlap == "returned"

before = wh.execute("SELECT COUNT(*), SUM(hash(status)) FROM dim_orders").fetchone()
incremental_sync("2025-01-01 00:00:00", timedelta(days=30))                  # re-run with a huge overlap
assert wh.execute("SELECT COUNT(*), SUM(hash(status)) FROM dim_orders").fetchone() == before, "MERGE is idempotent"

# %% [markdown]
# ## 4. Hard deletes and reconciliation

# %%
deleted = [r[0] for r in src.execute("SELECT order_id FROM orders WHERE status = 'cancelled' LIMIT 30")]
src.executemany("DELETE FROM orders WHERE order_id = ?", [(i,) for i in deleted])
incremental_sync("2026-12-31 00:00:00", timedelta(hours=1))                 # nothing new: watermarks can't see deletes
src_ids = pd.read_sql("SELECT order_id FROM orders", src)
wh_ids = wh.execute("SELECT order_id FROM dim_orders").df()
ghosts = set(wh_ids["order_id"]) - set(src_ids["order_id"])
print(f"reconciliation: {len(ghosts)} orders exist in the warehouse but were deleted at the source")
assert ghosts == set(deleted)
wh.execute(f"DELETE FROM dim_orders WHERE order_id IN ({','.join(map(str, ghosts))})")
assert set(wh.execute("SELECT order_id FROM dim_orders").df()["order_id"]) == set(src_ids["order_id"])

# %% [markdown]
# ## 5. Backfill

# %%
def backfill(start: date, end: date):
    d, runs = start, 0
    while d <= end:
        load_overwrite(d.isoformat())             # same idempotent task, one logical date at a time
        d += timedelta(days=1)
        runs += 1
    return runs


runs = backfill(date(2025, 2, 1), date(2025, 2, 28))
runs += backfill(date(2025, 2, 10), date(2025, 2, 20))   # overlapping re-run: harmless
got = wh.execute("SELECT order_date, COUNT(*) AS n FROM rev_overwrite GROUP BY 1 ORDER BY 1").df()
expected = pd.DataFrame([(d.date().isoformat(), len(extract_day(d.date().isoformat())))
                         for d in pd.date_range("2025-02-01", "2025-02-28")], columns=["order_date", "n"])
expected = expected[expected["n"] > 0].reset_index(drop=True)
print(f"{runs} task runs (with an overlapping re-run); {len(got)} days loaded, all counts correct")
assert (got["n"].to_numpy() == expected["n"].to_numpy()).all()

print("\nAll checks passed.")
