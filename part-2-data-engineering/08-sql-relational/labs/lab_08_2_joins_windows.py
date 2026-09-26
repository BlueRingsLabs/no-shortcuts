# %% [markdown]
# # Lab 08.2: Fan-outs, windows, sessions and cohorts
#
# 1. The fan-out bug: reproduce it, measure how wrong it is, fix it.
# 2. LEFT JOIN + WHERE on the right table silently becomes an INNER JOIN.
# 3. Latest row per key, top-N per group, deterministic tie-breaking.
# 4. Running totals and a gap-safe 7-day moving average (calendar join).
# 5. Sessionization (30-minute gap), verified against a pandas implementation.
# 6. Monthly cohort retention.

# %%
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

t = make_shop(n_customers=3_000, seed=1)
con = duckdb.connect()
for name, df in t.items():
    con.register(name, df)
q = lambda sql: con.execute(sql).df()

# %% [markdown]
# ## 1. The fan-out

# %%
wrong = q("""
SELECT c.customer_id, SUM(oi.quantity * oi.unit_price) AS revenue, COUNT(e.event_id) AS events
FROM customers c
JOIN orders o       ON o.customer_id = c.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
JOIN events e       ON e.customer_id = c.customer_id
GROUP BY c.customer_id
""")
right = q("""
WITH rev AS (
    SELECT o.customer_id, SUM(oi.quantity * oi.unit_price) AS revenue
    FROM orders o JOIN order_items oi USING (order_id) GROUP BY o.customer_id
), ev AS (SELECT customer_id, COUNT(*) AS events FROM events GROUP BY customer_id)
SELECT c.customer_id, COALESCE(rev.revenue, 0) AS revenue, COALESCE(ev.events, 0) AS events
FROM customers c LEFT JOIN rev USING (customer_id) LEFT JOIN ev USING (customer_id)
""")
true_revenue = float((t["order_items"]["quantity"] * t["order_items"]["unit_price"]).sum())
print(f"total revenue: true {true_revenue:,.0f}   fan-out query {wrong['revenue'].sum():,.0f} "
      f"({wrong['revenue'].sum() / true_revenue:.0f}x)   fixed query {right['revenue'].sum():,.0f}")
assert np.isclose(right["revenue"].sum(), true_revenue)
assert wrong["revenue"].sum() > 5 * true_revenue
assert right["events"].sum() == len(t["events"])

# Row count check: an enrichment join that should be 1:1 must not change the number of rows
n_before = len(t["orders"])
n_after = q("SELECT COUNT(*) AS n FROM orders o JOIN customers c USING (customer_id)")["n"][0]
assert n_after == n_before, "customers.customer_id is unique, so the join keeps the grain"

# %% [markdown]
# ## 2. LEFT JOIN + WHERE

# %%
kept_where = q("""SELECT COUNT(DISTINCT c.customer_id) AS n FROM customers c
                  LEFT JOIN orders o ON o.customer_id = c.customer_id WHERE o.status = 'delivered'""")["n"][0]
kept_on = q("""SELECT COUNT(DISTINCT c.customer_id) AS n FROM customers c
               LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'delivered'""")["n"][0]
print(f"customers kept: condition in WHERE {kept_where:,}   condition in ON {kept_on:,} (all of them)")
assert kept_on == len(t["customers"]) and kept_where < kept_on

# %% [markdown]
# ## 3. Latest row per key and top-N per group

# %%
latest = q("""
SELECT customer_id, order_id, order_ts FROM orders
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_ts DESC, order_id DESC) = 1
""")
pd_latest = t["orders"].sort_values(["order_ts", "order_id"]).groupby("customer_id").tail(1)
assert set(latest["order_id"]) == set(pd_latest["order_id"])
assert latest["customer_id"].is_unique

top3 = q("""
WITH r AS (SELECT p.category, p.product_id, SUM(oi.quantity * oi.unit_price) AS revenue
           FROM order_items oi JOIN products p USING (product_id) GROUP BY ALL)
SELECT * FROM r QUALIFY ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC, product_id) <= 3
ORDER BY category, revenue DESC
""")
assert (top3.groupby("category").size() == 3).all()
print(top3.head(6))

# Ties: RANK vs DENSE_RANK vs ROW_NUMBER
ties = q("""SELECT x, ROW_NUMBER() OVER (ORDER BY x DESC, id) rn, RANK() OVER (ORDER BY x DESC) rk,
                   DENSE_RANK() OVER (ORDER BY x DESC) drk
            FROM (VALUES (1, 10), (2, 10), (3, 7)) v(id, x) ORDER BY rn""")
assert ties["rk"].tolist() == [1, 1, 3] and ties["drk"].tolist() == [1, 1, 2] and ties["rn"].tolist() == [1, 2, 3]

# %% [markdown]
# ## 4. Running totals and a gap-safe moving average

# %%
daily = q("""
WITH rev AS (
    SELECT CAST(o.order_ts AS DATE) AS day, SUM(oi.quantity * oi.unit_price) AS revenue
    FROM orders o JOIN order_items oi USING (order_id)
    WHERE o.status = 'delivered' GROUP BY ALL
), cal AS (
    SELECT CAST(range AS DATE) AS day FROM range((SELECT MIN(day) FROM rev), (SELECT MAX(day) FROM rev) + INTERVAL 1 DAY, INTERVAL 1 DAY)
)
SELECT cal.day, COALESCE(rev.revenue, 0) AS revenue,
       SUM(COALESCE(rev.revenue, 0)) OVER (ORDER BY cal.day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running,
       AVG(COALESCE(rev.revenue, 0)) OVER (ORDER BY cal.day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7
FROM cal LEFT JOIN rev USING (day) ORDER BY cal.day
""")
assert daily["day"].diff().dropna().dt.days.eq(1).all(), "one row per calendar day, no gaps"
pd_ma7 = daily["revenue"].rolling(7, min_periods=1).mean()
assert np.allclose(daily["ma7"], pd_ma7)
assert np.isclose(daily["running"].iloc[-1], daily["revenue"].sum())
print(f"{len(daily)} days, {int((daily['revenue'] == 0).sum())} with zero delivered revenue (they'd be missing without the calendar)")

# %% [markdown]
# ## 5. Sessionization

# %%
sessions = q("""
WITH flagged AS (
    SELECT *, CASE WHEN LAG(ts) OVER w IS NULL OR ts - LAG(ts) OVER w > INTERVAL 30 MINUTES THEN 1 ELSE 0 END AS new_session
    FROM events WINDOW w AS (PARTITION BY customer_id ORDER BY ts, event_id)
)
SELECT event_id, customer_id,
       SUM(new_session) OVER (PARTITION BY customer_id ORDER BY ts, event_id ROWS UNBOUNDED PRECEDING) AS session_number
FROM flagged
""").sort_values("event_id").reset_index(drop=True)

ev = t["events"].sort_values(["customer_id", "ts", "event_id"])
gap = ev.groupby("customer_id")["ts"].diff()
new = gap.isna() | (gap > pd.Timedelta(minutes=30))
ev["session_number"] = new.astype(int).groupby(ev["customer_id"]).cumsum()
ev = ev.sort_values("event_id").reset_index(drop=True)
assert (sessions["session_number"].to_numpy() == ev["session_number"].to_numpy()).all()
n_sessions = int(sessions.groupby("customer_id")["session_number"].max().sum())
print(f"{len(t['events']):,} events -> {n_sessions:,} sessions ({len(t['events']) / n_sessions:.1f} events per session)")

# %% [markdown]
# ## 6. Cohort retention

# %%
cohorts = q("""
WITH o AS (
    SELECT customer_id, DATE_TRUNC('month', order_ts) AS month,
           MIN(DATE_TRUNC('month', order_ts)) OVER (PARTITION BY customer_id) AS cohort
    FROM orders WHERE status <> 'cancelled'
), active AS (
    SELECT DISTINCT customer_id, cohort, DATE_DIFF('month', cohort, month) AS age FROM o
)
SELECT cohort, age, COUNT(*) AS customers,
       COUNT(*) / FIRST_VALUE(COUNT(*)) OVER (PARTITION BY cohort ORDER BY age) AS retention
FROM active GROUP BY cohort, age ORDER BY cohort, age
""")
pivot = cohorts.pivot(index="cohort", columns="age", values="retention")
print(pivot.iloc[:6, :7].round(2))
assert np.allclose(pivot[0].dropna(), 1.0), "month 0 retention is 100% by definition"
assert np.nanmax(pivot.iloc[:, 1:].to_numpy()) <= 1, "nobody can come back more than once per month"

# %%
print("\nAll checks passed.")
