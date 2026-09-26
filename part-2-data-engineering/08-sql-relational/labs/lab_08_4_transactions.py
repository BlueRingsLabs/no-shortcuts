# %% [markdown]
# # Lab 08.4: Atomicity, lost updates, and OLTP vs OLAP
#
# 1. Atomicity: a transfer that fails halfway is rolled back completely.
# 2. The lost update, reproduced with two real connections to the same database file.
# 3. Three fixes: atomic UPDATE, optimistic concurrency with a version column, and BEGIN IMMEDIATE (pessimistic lock).
# 4. Concurrent threads hammering a counter: naive read-modify-write vs atomic update.
# 5. OLTP vs OLAP: point lookups and big aggregations on SQLite (row store) vs DuckDB (column store).

# %%
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

tmp = tempfile.TemporaryDirectory()
DB = str(Path(tmp.name) / "bank.db")


def connect():
    c = sqlite3.connect(DB, timeout=10, isolation_level=None, check_same_thread=False)   # autocommit; we manage BEGIN/COMMIT
    c.execute("PRAGMA journal_mode=WAL")
    return c

# %% [markdown]
# ## 1. Atomicity

# %%
a = connect()
a.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER NOT NULL CHECK (balance >= 0))")
a.execute("INSERT INTO accounts VALUES (1, 100), (2, 50)")


def transfer(conn, src, dst, amount):
    conn.execute("BEGIN")
    try:
        conn.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, dst))   # credit first...
        conn.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, src))   # ...then debit (may violate CHECK)
        conn.execute("COMMIT")
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK")
        raise


transfer(a, 1, 2, 30)
try:
    transfer(a, 1, 2, 500)                      # would make balance negative
except sqlite3.IntegrityError:
    pass
balances = dict(a.execute("SELECT id, balance FROM accounts").fetchall())
print("after one good and one failed transfer:", balances)
assert balances == {1: 70, 2: 80}, "the failed transfer's credit must have been rolled back too"
assert sum(balances.values()) == 150, "money is neither created nor destroyed"

# %% [markdown]
# ## 2. The lost update, step by step
#
# Two clerks sell from the same stock of 10 units. Each reads, computes, writes.

# %%
a.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, stock INTEGER, version INTEGER DEFAULT 0)")
a.execute("INSERT INTO products (id, stock) VALUES (1, 10)")
c1, c2 = connect(), connect()

s1 = c1.execute("SELECT stock FROM products WHERE id = 1").fetchone()[0]      # clerk 1 reads 10
s2 = c2.execute("SELECT stock FROM products WHERE id = 1").fetchone()[0]      # clerk 2 reads 10
c1.execute("UPDATE products SET stock = ? WHERE id = 1", (s1 - 3,))           # writes 7
c2.execute("UPDATE products SET stock = ? WHERE id = 1", (s2 - 4,))           # writes 6: clerk 1's sale is lost
stock = a.execute("SELECT stock FROM products WHERE id = 1").fetchone()[0]
print(f"sold 3 + 4 from 10, stock is {stock} (should be 3)")
assert stock == 6

# %% [markdown]
# ## 3. Fixes

# %%
# (a) Atomic update: the database does the read-modify-write in one statement, with a guard.
a.execute("UPDATE products SET stock = 10 WHERE id = 1")


def sell_atomic(conn, qty):
    cur = conn.execute("UPDATE products SET stock = stock - ? WHERE id = 1 AND stock >= ?", (qty, qty))
    return cur.rowcount == 1                    # False: not enough stock, the sale fails cleanly


assert sell_atomic(c1, 3) and sell_atomic(c2, 4) and not sell_atomic(c1, 5)
assert a.execute("SELECT stock FROM products WHERE id = 1").fetchone()[0] == 3
print("atomic UPDATE: 10 - 3 - 4 = 3, and an oversell is refused")

# (b) Optimistic concurrency with a version column
a.execute("UPDATE products SET stock = 10, version = 0 WHERE id = 1")


def sell_optimistic(conn, qty, before_write=None, max_retries=5):
    for attempt in range(max_retries):
        stock, version = conn.execute("SELECT stock, version FROM products WHERE id = 1").fetchone()
        if stock < qty:
            return False, attempt
        if before_write:
            before_write()                      # lets us inject a competing write at the worst moment
            before_write = None
        cur = conn.execute("UPDATE products SET stock = ?, version = version + 1 WHERE id = 1 AND version = ?",
                           (stock - qty, version))
        if cur.rowcount == 1:
            return True, attempt
    raise RuntimeError("too much contention")


ok, retries = sell_optimistic(c1, 3, before_write=lambda: sell_optimistic(c2, 4))
stock, version = a.execute("SELECT stock, version FROM products WHERE id = 1").fetchone()
print(f"optimistic: clerk 1 retried {retries} time(s); stock {stock}, version {version}")
assert ok and retries == 1 and stock == 3 and version == 2

# (c) Pessimistic: BEGIN IMMEDIATE takes the write lock before reading, so nobody can sneak in between.
a.execute("UPDATE products SET stock = 10 WHERE id = 1")
c1.execute("BEGIN IMMEDIATE")
s = c1.execute("SELECT stock FROM products WHERE id = 1").fetchone()[0]
c3 = sqlite3.connect(DB, timeout=0.2, isolation_level=None)
try:
    c3.execute("BEGIN IMMEDIATE")               # blocked: clerk 1 holds the lock
    blocked = False
except sqlite3.OperationalError as e:
    blocked = "locked" in str(e)
c1.execute("UPDATE products SET stock = ? WHERE id = 1", (s - 3,))
c1.execute("COMMIT")
assert blocked, "the second writer must wait (here: time out) while the first holds the lock"
print("pessimistic lock: a concurrent writer was blocked until commit")

# %% [markdown]
# ## 4. Many threads, one counter

# %%
a.execute("CREATE TABLE counters (id INTEGER PRIMARY KEY, n INTEGER)")


def hammer(increment, threads=8, per_thread=200):
    a.execute("DELETE FROM counters"); a.execute("INSERT INTO counters VALUES (1, 0)")
    conns = [connect() for _ in range(threads)]
    workers = [threading.Thread(target=lambda c=c: [increment(c) for _ in range(per_thread)]) for c in conns]
    for w in workers: w.start()
    for w in workers: w.join()
    return a.execute("SELECT n FROM counters").fetchone()[0], threads * per_thread


def naive_increment(c):
    n = c.execute("SELECT n FROM counters WHERE id = 1").fetchone()[0]
    time.sleep(0)                                # yield: makes the race window real
    c.execute("UPDATE counters SET n = ? WHERE id = 1", (n + 1,))


def atomic_increment(c):
    c.execute("UPDATE counters SET n = n + 1 WHERE id = 1")


got_naive, expected = hammer(naive_increment)
got_atomic, _ = hammer(atomic_increment)
print(f"8 threads x 200 increments: naive {got_naive}/{expected}, atomic {got_atomic}/{expected}")
assert got_atomic == expected and got_naive < expected

# %% [markdown]
# ## 5. OLTP vs OLAP

# %%
t = make_shop(n_customers=60_000, seed=3)
items = t["order_items"]
lite = sqlite3.connect(":memory:")
items.to_sql("order_items", lite, index=False)
lite.execute("CREATE INDEX idx_items ON order_items(order_id, product_id)")
duck = duckdb.connect()
duck.execute("CREATE TABLE order_items AS SELECT * FROM items")
print(f"order_items: {len(items):,} rows")

keys = items.sample(300, random_state=0)[["order_id", "product_id"]].to_numpy().tolist()
point = "SELECT quantity FROM order_items WHERE order_id = ? AND product_id = ?"
agg = "SELECT product_id, SUM(quantity * unit_price) AS rev FROM order_items GROUP BY product_id ORDER BY rev DESC LIMIT 5"


def bench(conn, sql, params_list):
    t0 = time.perf_counter()
    for p in params_list:
        conn.execute(sql, p).fetchall()
    return (time.perf_counter() - t0) / len(params_list)


lookup_lite, lookup_duck = bench(lite, point, keys), bench(duck, point, keys)
agg_lite, agg_duck = bench(lite, agg, [()] * 3), bench(duck, agg, [()] * 3)
print(f"point lookup:  SQLite {lookup_lite * 1e6:7.0f} us   DuckDB {lookup_duck * 1e6:7.0f} us")
print(f"aggregation:   SQLite {agg_lite * 1e3:7.1f} ms   DuckDB {agg_duck * 1e3:7.1f} ms")
assert lookup_lite < lookup_duck, "row store + B-tree wins single-row lookups"
assert agg_duck < agg_lite, "column store + vectorized execution wins big scans"
assert lite.execute(agg).fetchall()[0][0] == duck.execute(agg).fetchall()[0][0]

tmp.cleanup()
print("\nAll checks passed.")
