# %% [markdown]
# # Lab 08.1: SQL on the Rat & Co. shop
#
# 1. Load the shop into DuckDB, with constraints, and watch the constraints reject bad rows.
# 2. Queries whose answers we verify independently with pandas.
# 3. NULL traps: <>, NOT IN, COUNT(col) vs COUNT(*), SUM of all-NULL.
# 4. SQL injection against SQLite, then the parameterized fix.

# %%
import sqlite3
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

t = make_shop(n_customers=5_000, seed=0)
con = duckdb.connect()        # in-memory database

# %% [markdown]
# ## 1. Schema with constraints

# %%
con.execute("""
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        VARCHAR NOT NULL,
    email       VARCHAR UNIQUE,
    country     VARCHAR,
    signup_date DATE NOT NULL,
    segment     VARCHAR CHECK (segment IN ('consumer', 'pro', 'enterprise'))
);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR, category VARCHAR, price DECIMAL(10, 2) CHECK (price >= 0));
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_ts TIMESTAMP NOT NULL, status VARCHAR NOT NULL, channel VARCHAR
);
CREATE TABLE order_items (
    order_id INTEGER REFERENCES orders(order_id), product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER CHECK (quantity > 0), unit_price DECIMAL(10, 2),
    PRIMARY KEY (order_id, product_id)
);
""")
for name in ("customers", "products", "orders", "order_items"):
    df = t[name]
    con.execute(f"INSERT INTO {name} SELECT * FROM df")     # DuckDB can read a pandas DataFrame by variable name
    print(f"{name:12s} {con.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]:>7,} rows")

rejections = []
for bad_sql in [
    "INSERT INTO customers VALUES (1, 'Dup', 'x@y.z', 'AR', '2024-01-01', 'consumer')",          # duplicate PK
    "INSERT INTO customers VALUES (999999, 'Bad', 'b@y.z', 'AR', '2024-01-01', 'vip')",          # CHECK
    "INSERT INTO orders VALUES (999999, 424242, '2024-01-01', 'delivered', 'web')",              # FK: no such customer
    "INSERT INTO order_items VALUES (1, 1, -3, 10.0)",                                           # CHECK quantity > 0
]:
    try:
        con.execute(bad_sql)
    except duckdb.Error as e:
        rejections.append(type(e).__name__)
print("rejected by constraints:", rejections)
assert len(rejections) == 4

# %% [markdown]
# ## 2. Queries, verified against pandas

# %%
q_country = """
SELECT c.country, COUNT(DISTINCT o.order_id) AS n_orders, SUM(oi.quantity * oi.unit_price) AS revenue
FROM orders o
JOIN customers c    ON c.customer_id = o.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.status = 'delivered'
GROUP BY c.country
ORDER BY revenue DESC
"""
res = con.execute(q_country).df()
print(res)

# Independent check in pandas
m = t["orders"].merge(t["customers"], on="customer_id").merge(t["order_items"], on="order_id")
m = m[m["status"] == "delivered"]
m["line"] = m["quantity"] * m["unit_price"]
check = m.groupby("country").agg(n_orders=("order_id", "nunique"), revenue=("line", "sum")).sort_values("revenue", ascending=False)
assert list(res["country"]) == list(check.index)
assert np.allclose(res["revenue"].astype(float), check["revenue"], rtol=1e-9)
assert (res["n_orders"].to_numpy() == check["n_orders"].to_numpy()).all()

# Channel shares with a CTE and a window function
shares = con.execute("""
WITH d AS (SELECT channel, COUNT(*) AS n FROM orders WHERE status = 'delivered' GROUP BY channel)
SELECT channel, n, ROUND(100.0 * n / SUM(n) OVER (), 2) AS pct FROM d ORDER BY n DESC
""").df()
print(shares)
assert abs(shares["pct"].sum() - 100) < 0.05

# Lifetime value by segment (CTEs as named steps)
ltv = con.execute("""
WITH order_totals AS (
    SELECT o.order_id, o.customer_id, SUM(oi.quantity * oi.unit_price) AS total
    FROM orders o JOIN order_items oi USING (order_id)
    WHERE o.status NOT IN ('cancelled', 'returned')
    GROUP BY o.order_id, o.customer_id
), customer_value AS (
    SELECT customer_id, COUNT(*) AS n_orders, SUM(total) AS lifetime_value FROM order_totals GROUP BY customer_id
)
SELECT c.segment, COUNT(*) AS customers, AVG(cv.lifetime_value) AS avg_ltv, MEDIAN(cv.lifetime_value) AS median_ltv
FROM customer_value cv JOIN customers c USING (customer_id)
GROUP BY c.segment ORDER BY avg_ltv DESC
""").df()
print(ltv)
assert (ltv["avg_ltv"] > ltv["median_ltv"]).all(), "LTV is right-skewed: mean above median in every segment"

# %% [markdown]
# ## 3. NULL traps

# %%
con.execute("""
CREATE TABLE people AS SELECT * FROM (VALUES (1, 'AR', 10), (2, 'ES', NULL), (3, NULL, 5), (4, 'AR', NULL)) v(id, country, score);
CREATE TABLE blocked AS SELECT * FROM (VALUES (2), (NULL)) v(id);
""")
q = lambda sql: con.execute(sql).fetchall()

assert q("SELECT COUNT(*) FROM people WHERE country <> 'AR'") == [(1,)], "the NULL-country row silently disappears"
assert q("SELECT COUNT(*) FROM people WHERE country IS DISTINCT FROM 'AR'") == [(2,)]
assert q("SELECT COUNT(*), COUNT(score), AVG(score), AVG(COALESCE(score, 0)) FROM people") == [(4, 2, 7.5, 3.75)]
assert q("SELECT SUM(score) FROM people WHERE score IS NULL") == [(None,)], "SUM of nothing is NULL, not 0"
assert q("SELECT COUNT(*) FROM people WHERE id NOT IN (SELECT id FROM blocked)") == [(0,)], "NOT IN + NULL = nothing"
assert q("SELECT COUNT(*) FROM people p WHERE NOT EXISTS (SELECT 1 FROM blocked b WHERE b.id = p.id)") == [(3,)]
assert q("SELECT NULL = NULL, NULL IS NULL, NULL IS NOT DISTINCT FROM NULL") == [(None, True, True)]
assert q("SELECT 10 / NULLIF(0, 0)") == [(None,)], "NULLIF avoids division by zero"
print("NULL semantics: all traps demonstrated")

# %% [markdown]
# ## 4. SQL injection
#
# A tiny "admin tool" that looks customers up by email, written the wrong way and the right way.

# %%
sq = sqlite3.connect(":memory:")
t["customers"].assign(signup_date=t["customers"]["signup_date"].astype(str)).to_sql("customers", sq, index=False)
sq.execute("CREATE TABLE secrets (api_key TEXT)")
sq.execute("INSERT INTO secrets VALUES ('sk-live-do-not-leak')")


def lookup_unsafe(email):
    return sq.execute(f"SELECT customer_id, name FROM customers WHERE email = '{email}'").fetchall()


def lookup_safe(email):
    return sq.execute("SELECT customer_id, name FROM customers WHERE email = ?", (email,)).fetchall()


legit = t["customers"]["email"].iloc[0]
assert lookup_unsafe(legit) == lookup_safe(legit) and len(lookup_safe(legit)) == 1

payload_all = "' OR '1'='1"
payload_union = "' UNION SELECT 1, api_key FROM secrets --"
leak_all = lookup_unsafe(payload_all)
leak_secret = lookup_unsafe(payload_union)
print(f"unsafe: '{payload_all}' returns {len(leak_all):,} customers; UNION payload returns {leak_secret}")
assert len(leak_all) == len(t["customers"]) and ("sk-live-do-not-leak",) in [(r[1],) for r in leak_secret]
assert lookup_safe(payload_all) == [] and lookup_safe(payload_union) == []
print("safe (parameterized): both payloads return nothing, because they're just strange email addresses")


def products_sorted(sort_col):
    allowed = {"price", "name", "category"}                     # identifiers can't be parameters: allow-list them
    if sort_col not in allowed:
        raise ValueError(f"cannot sort by {sort_col!r}")
    return con.execute(f"SELECT name FROM products ORDER BY {sort_col} LIMIT 3").fetchall()


assert len(products_sorted("price")) == 3
try:
    products_sorted("price; DROP TABLE products")
    raise AssertionError("should have been rejected")
except ValueError:
    pass

# %%
print("\nAll checks passed.")
