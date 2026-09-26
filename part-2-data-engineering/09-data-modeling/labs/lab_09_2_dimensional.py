# %% [markdown]
# # Lab 09.2: A star schema with a Type 2 customer dimension
#
# 1. Build dim_date, dim_product, dim_customer (SCD2) and fact_sales at the order-line grain.
# 2. Simulate customers moving country; apply an SCD2 merge (close old rows, insert new ones).
# 3. As-of join: facts get the dimension row valid at event time. Compare Type 1 vs Type 2 reports.
# 4. The overlapping-interval bug from a double load, and a test that catches it.
# 5. Semi-additive measures: summing balances across time is nonsense.

# %%
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

t = make_shop(n_customers=3_000, seed=5)
con = duckdb.connect()
for name, df in t.items():
    con.register(name, df)
rng = np.random.default_rng(5)

# %% [markdown]
# ## 1. Dimensions and facts

# %%
con.execute("""
CREATE TABLE dim_date AS
SELECT CAST(strftime(d, '%Y%m%d') AS INTEGER) AS date_key, CAST(d AS DATE) AS date,
       year(d) AS year, quarter(d) AS quarter, month(d) AS month, strftime(d, '%Y-%m') AS year_month,
       dayofweek(d) AS dow, dayofweek(d) IN (0, 6) AS is_weekend
FROM range(DATE '2024-01-01', DATE '2026-12-31', INTERVAL 1 DAY) r(d);

CREATE TABLE dim_product AS
SELECT row_number() OVER (ORDER BY product_id) AS product_key, product_id, name, category FROM products;

-- initial customer dimension: one current row per customer, valid since signup
CREATE TABLE dim_customer AS
SELECT row_number() OVER (ORDER BY customer_id) AS customer_key, customer_id, name, country, segment,
       CAST(signup_date AS TIMESTAMP) AS valid_from, TIMESTAMP '9999-12-31' AS valid_to, TRUE AS is_current
FROM customers;
INSERT INTO dim_customer VALUES (-1, -1, 'Unknown', 'N/A', 'N/A', TIMESTAMP '1900-01-01', TIMESTAMP '9999-12-31', TRUE);
""")
print(con.execute("SELECT COUNT(*) FROM dim_date").fetchone()[0], "days in dim_date")

# %% [markdown]
# ## 2. SCD2 merge: 300 customers move country on random dates

# %%
movers = con.execute("SELECT customer_id, country FROM customers USING SAMPLE 300 ROWS (reservoir, 7)").df()
countries = ["AR", "ES", "US", "DE", "PL", "BR", "MX", "FR"]
movers["new_country"] = [rng.choice([c for c in countries if c != old]) for old in movers["country"]]
movers["changed_at"] = pd.Timestamp("2024-06-01") + pd.to_timedelta(rng.integers(0, 365, len(movers)), unit="D")
con.register("changes", movers[["customer_id", "new_country", "changed_at"]])


def scd2_merge(con):
    """Close the current row of every changed customer and insert a new current row. Idempotent: running it
    twice with the same changes must not create extra rows (the WHERE clauses check for real changes)."""
    con.execute("""
    CREATE OR REPLACE TEMP TABLE to_apply AS
    SELECT ch.* FROM changes ch
    JOIN dim_customer d ON d.customer_id = ch.customer_id AND d.is_current
    WHERE d.country <> ch.new_country AND ch.changed_at > d.valid_from;

    INSERT INTO dim_customer
    SELECT (SELECT MAX(customer_key) FROM dim_customer) + row_number() OVER (ORDER BY a.customer_id),
           d.customer_id, d.name, a.new_country, d.segment, a.changed_at, TIMESTAMP '9999-12-31', TRUE
    FROM to_apply a JOIN dim_customer d ON d.customer_id = a.customer_id AND d.is_current;

    UPDATE dim_customer d SET valid_to = a.changed_at, is_current = FALSE
    FROM to_apply a
    WHERE d.customer_id = a.customer_id AND d.is_current AND d.valid_from < a.changed_at;
    """)


scd2_merge(con)
n_rows_1 = con.execute("SELECT COUNT(*) FROM dim_customer").fetchone()[0]
scd2_merge(con)                                  # run again: nothing should change
n_rows_2 = con.execute("SELECT COUNT(*) FROM dim_customer").fetchone()[0]
print(f"dim_customer rows: {n_rows_1:,} after merge, {n_rows_2:,} after re-running it (idempotent)")
signup = t["customers"].set_index("customer_id")["signup_date"]
applicable = int((movers["changed_at"].to_numpy() > signup.loc[movers["customer_id"]].to_numpy()).sum())
print(f"{applicable} of {len(movers)} changes apply (the rest predate the customer's signup and are rejected)")
assert n_rows_1 == len(t["customers"]) + 1 + applicable and n_rows_2 == n_rows_1
assert con.execute("SELECT COUNT(*) FROM dim_customer WHERE is_current GROUP BY customer_id HAVING COUNT(*) > 1").fetchall() == []

# %% [markdown]
# ## 3. Facts with an as-of join, and Type 1 vs Type 2 reports

# %%
con.execute("""
CREATE TABLE fact_sales AS
SELECT CAST(strftime(o.order_ts, '%Y%m%d') AS INTEGER) AS date_key,
       COALESCE(dc.customer_key, -1) AS customer_key,
       dp.product_key, o.order_id, oi.quantity, oi.quantity * oi.unit_price AS amount
FROM orders o
JOIN order_items oi USING (order_id)
JOIN dim_product dp ON dp.product_id = oi.product_id
LEFT JOIN dim_customer dc ON dc.customer_id = o.customer_id
                         AND o.order_ts >= dc.valid_from AND o.order_ts < dc.valid_to
WHERE o.status NOT IN ('cancelled', 'returned');
""")
n_lines = con.execute("""SELECT COUNT(*) FROM orders o JOIN order_items USING (order_id)
                         WHERE o.status NOT IN ('cancelled', 'returned')""").fetchone()[0]
assert con.execute("SELECT COUNT(*) FROM fact_sales").fetchone()[0] == n_lines, "grain preserved: one row per order line"

type2 = con.execute("""
SELECT c.country, ROUND(SUM(f.amount)) AS revenue FROM fact_sales f
JOIN dim_customer c USING (customer_key) JOIN dim_date d USING (date_key)
WHERE d.year = 2024 GROUP BY ALL ORDER BY ALL""").df().set_index("country")
type1 = con.execute("""
SELECT cur.country, ROUND(SUM(f.amount)) AS revenue FROM fact_sales f
JOIN dim_customer c USING (customer_key)
JOIN dim_customer cur ON cur.customer_id = c.customer_id AND cur.is_current
JOIN dim_date d USING (date_key)
WHERE d.year = 2024 GROUP BY ALL ORDER BY ALL""").df().set_index("country")
cmp = type2.join(type1, lsuffix="_as_it_was", rsuffix="_current_country")
print(cmp)
assert np.isclose(cmp.iloc[:, 0].sum(), cmp.iloc[:, 1].sum()), "same total, different attribution"
assert (cmp.iloc[:, 0] != cmp.iloc[:, 1]).any(), "moving customers change the per-country history under Type 1"

# %% [markdown]
# ## 4. The overlapping-interval bug
#
# A buggy re-run inserts a second 'current' row for some customers without closing the previous one.
# Facts in the overlap match two dimension rows and get duplicated. A validity test catches it.

# %%
def overlap_test(con):
    return con.execute("""
    SELECT COUNT(*) FROM dim_customer a JOIN dim_customer b
      ON a.customer_id = b.customer_id AND a.customer_key < b.customer_key
     AND a.valid_from < b.valid_to AND b.valid_from < a.valid_to
    """).fetchone()[0]


assert overlap_test(con) == 0
con.execute("CREATE TABLE dim_customer_bad AS SELECT * FROM dim_customer")
con.execute("""
INSERT INTO dim_customer_bad
SELECT customer_key + 100000, customer_id, name, 'XX', segment, TIMESTAMP '2024-01-01', TIMESTAMP '9999-12-31', TRUE
FROM dim_customer WHERE customer_id BETWEEN 1 AND 50 AND is_current
""")
dup = con.execute("""
SELECT COUNT(*) FROM orders o JOIN order_items USING (order_id)
JOIN dim_customer_bad dc ON dc.customer_id = o.customer_id AND o.order_ts >= dc.valid_from AND o.order_ts < dc.valid_to
WHERE o.status NOT IN ('cancelled', 'returned')""").fetchone()[0]
con.execute("ALTER TABLE dim_customer RENAME TO dim_customer_good")
con.execute("ALTER TABLE dim_customer_bad RENAME TO dim_customer")
n_overlaps = overlap_test(con)
print(f"after the buggy load: {n_overlaps} overlapping intervals, fact rows {dup:,} instead of {n_lines:,}")
assert n_overlaps > 0 and dup > n_lines

# %% [markdown]
# ## 5. Semi-additive measures

# %%
days = pd.date_range("2025-01-01", "2025-01-31", freq="D")
bal = pd.DataFrame([(d, a, 1000 + 10 * i) for i, d in enumerate(days) for a in (1, 2)], columns=["day", "account", "balance"])
con.register("balances", bal)
wrong = con.execute("SELECT SUM(balance) FROM balances").fetchone()[0]
month_end = con.execute("SELECT SUM(balance) FROM balances WHERE day = (SELECT MAX(day) FROM balances)").fetchone()[0]
avg_daily = con.execute("SELECT AVG(total) FROM (SELECT day, SUM(balance) AS total FROM balances GROUP BY day)").fetchone()[0]
print(f"'total balance' summing across days: {wrong:,}  <- meaningless. Month-end total: {month_end:,}. Average daily total: {avg_daily:,.0f}")
assert month_end == 2 * (1000 + 10 * 30) and wrong > 30 * month_end / 2

# %%
print("\nAll checks passed.")
