# %% [markdown]
# # Lab 09.1: Normalizing a flat export
#
# 1. Build the flat "export" table (one row per order line, everything repeated).
# 2. Discover its functional dependencies from the data.
# 3. Demonstrate an update anomaly.
# 4. Normalize to 3NF with DuckDB and prove the decomposition is lossless.
# 5. Measure the redundancy removed.

# %%
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

t = make_shop(n_customers=2_000, seed=4)
con = duckdb.connect()
for name, df in t.items():
    con.register(name, df)

flat = con.execute("""
SELECT o.order_id, o.order_ts, o.status, c.customer_id, c.name AS customer_name, c.email AS customer_email,
       c.country, p.product_id, p.name AS product_name, p.category, oi.unit_price, oi.quantity
FROM order_items oi
JOIN orders o USING (order_id) JOIN customers c USING (customer_id) JOIN products p USING (product_id)
ORDER BY order_id, product_id
""").df()
print(f"flat export: {len(flat):,} rows x {flat.shape[1]} columns")

# %% [markdown]
# ## 2. Functional dependencies, found in the data
#
# A -> B holds (in this data) if every value of A maps to exactly one value of B.

# %%
def holds(df, lhs, rhs):
    return bool((df.groupby(lhs)[rhs].nunique(dropna=False) <= 1).all())


fds = {
    ("customer_id", "customer_email"): holds(flat, "customer_id", "customer_email"),
    ("order_id", "customer_id"): holds(flat, "order_id", "customer_id"),
    ("order_id", "order_ts"): holds(flat, "order_id", "order_ts"),
    ("product_id", "category"): holds(flat, "product_id", "category"),
    ("customer_id", "order_id"): holds(flat, "customer_id", "order_id"),        # should NOT hold: a customer has many orders
    ("product_id", "unit_price"): holds(flat, "product_id", "unit_price"),      # should NOT hold: discounts per order
}
for (a, b), ok in fds.items():
    print(f"{a:>12s} -> {b:<15s} {'holds' if ok else 'does not hold'}")
assert fds[("customer_id", "customer_email")] and fds[("order_id", "customer_id")]
assert not fds[("customer_id", "order_id")] and not fds[("product_id", "unit_price")]
assert holds(flat, ["order_id", "product_id"], "quantity"), "(order_id, product_id) is the key of the line"

# %% [markdown]
# ## 3. The update anomaly
#
# Customer changes email; the "application" updates only the rows of their latest order.

# %%
victim = flat["customer_id"].value_counts().index[0]           # the customer with the most lines
rows = flat.index[flat["customer_id"] == victim]
latest_order = flat.loc[rows, "order_id"].max()
flat_bad = flat.copy()
mask = (flat_bad["customer_id"] == victim) & (flat_bad["order_id"] == latest_order)
flat_bad.loc[mask, "customer_email"] = "new.address@example.com"
n_emails = flat_bad.loc[flat_bad["customer_id"] == victim, "customer_email"].nunique()
print(f"customer {victim} now has {n_emails} different emails in the export: which one is true?")
assert n_emails == 2 and not holds(flat_bad, "customer_id", "customer_email")

# %% [markdown]
# ## 4. Normalize and prove it's lossless

# %%
con.register("flat", flat)
con.execute("""
CREATE TABLE n_customers AS SELECT DISTINCT customer_id, customer_name, customer_email, country FROM flat;
CREATE TABLE n_products  AS SELECT DISTINCT product_id, product_name, category FROM flat;
CREATE TABLE n_orders    AS SELECT DISTINCT order_id, order_ts, status, customer_id FROM flat;
CREATE TABLE n_lines     AS SELECT order_id, product_id, unit_price, quantity FROM flat;
""")
for tbl, key in [("n_customers", "customer_id"), ("n_products", "product_id"), ("n_orders", "order_id")]:
    n, n_keys = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT {key}) FROM {tbl}").fetchone()
    assert n == n_keys, f"{tbl}: {key} must be unique after normalization"

rejoined = con.execute("""
SELECT o.order_id, o.order_ts, o.status, c.customer_id, c.customer_name, c.customer_email, c.country,
       p.product_id, p.product_name, p.category, l.unit_price, l.quantity
FROM n_lines l JOIN n_orders o USING (order_id) JOIN n_customers c USING (customer_id) JOIN n_products p USING (product_id)
ORDER BY order_id, product_id
""").df()
pd.testing.assert_frame_equal(rejoined.reset_index(drop=True), flat.reset_index(drop=True), check_dtype=False)
print("lossless: joining the 3NF tables reproduces the export exactly")

# The same decomposition applied to the corrupted export fails the uniqueness check: normalization would have
# made the anomaly impossible in the first place (one email per customer, stored once).
con.register("flat_bad", flat_bad)
n, n_keys = con.execute("SELECT COUNT(*), COUNT(DISTINCT customer_id) FROM (SELECT DISTINCT customer_id, customer_email FROM flat_bad)").fetchone()
assert n == n_keys + 1

# %% [markdown]
# ## 5. Redundancy removed

# %%
cells_flat = flat.size
cells_norm = sum(con.execute(f"SELECT COUNT(*) FROM {x}").fetchone()[0] * len(con.execute(f"SELECT * FROM {x} LIMIT 0").description)
                 for x in ("n_customers", "n_products", "n_orders", "n_lines"))
repeats = len(flat) / flat["customer_id"].nunique()
print(f"cells: flat {cells_flat:,}   normalized {cells_norm:,} ({cells_flat / cells_norm:.1f}x fewer); "
      f"each customer's email was stored {repeats:.1f} times on average")
assert cells_norm < cells_flat

# %%
print("\nAll checks passed.")
