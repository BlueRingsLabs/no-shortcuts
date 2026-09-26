# requires: dbt
# %% [markdown]
# # Lab 13.3: A real dbt project on DuckDB
#
# The lab writes a small dbt project to a temporary directory and runs it with dbt-duckdb:
#
#   sources (raw shop tables) -> staging models (clean, typed) -> marts (fct_orders, dim_customers, customer_revenue)
#
# 1. `dbt build` runs models and tests in dependency order.
# 2. We break the raw data (a duplicated customer); the uniqueness test fails and downstream models are skipped.
# 3. An incremental model processes only new rows on the second run.
# 4. We read the manifest to see the lineage graph dbt built from ref() and source().

# %%
import json
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

from dbt.cli.main import dbtRunner  # noqa: E402

proj = Path(tempfile.mkdtemp()) / "shop_dbt"
DB = proj / "warehouse.duckdb"


def write(rel, text):
    p = proj / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text).lstrip())


write("dbt_project.yml", """
    name: shop
    version: "1.0"
    profile: shop
    model-paths: ["models"]
    models:
      shop:
        staging: {+materialized: view}
        marts: {+materialized: table}
""")
write("profiles.yml", f"""
    shop:
      target: dev
      outputs:
        dev:
          type: duckdb
          path: "{DB}"
          threads: 4
""")
write("models/sources.yml", """
    version: 2
    sources:
      - name: raw
        schema: main
        tables:
          - name: customers
          - name: orders
          - name: order_items
          - name: events
""")
write("models/staging/stg_customers.sql", """
    select customer_id, trim(lower(email)) as email, upper(country) as country, cast(signup_date as date) as signup_date, segment
    from {{ source('raw', 'customers') }}
""")
write("models/staging/stg_orders.sql", """
    select order_id, customer_id, cast(order_ts as timestamp) as order_ts, lower(trim(status)) as status, channel
    from {{ source('raw', 'orders') }}
""")
write("models/staging/stg_order_items.sql", """
    select order_id, product_id, quantity, unit_price, quantity * unit_price as amount
    from {{ source('raw', 'order_items') }}
""")
write("models/staging/schema.yml", """
    version: 2
    models:
      - name: stg_customers
        columns:
          - name: customer_id
            data_tests: [unique, not_null]
          - name: country
            data_tests:
              - accepted_values: {arguments: {values: ['AR', 'ES', 'US', 'DE', 'PL', 'BR', 'MX', 'FR']}}
      - name: stg_orders
        columns:
          - name: order_id
            data_tests: [unique, not_null]
          - name: customer_id
            data_tests:
              - relationships: {arguments: {to: ref('stg_customers'), field: customer_id}}
          - name: status
            data_tests:
              - accepted_values: {arguments: {values: ['delivered', 'shipped', 'cancelled', 'returned']}}
""")
write("models/marts/fct_orders.sql", """
    -- grain: one row per order
    select o.order_id, o.customer_id, o.order_ts, o.status, o.channel,
           sum(i.amount) as order_total, count(*) as n_lines
    from {{ ref('stg_orders') }} o
    join {{ ref('stg_order_items') }} i using (order_id)
    group by all
""")
write("models/marts/dim_customers.sql", """
    select customer_id, email, country, segment, signup_date from {{ ref('stg_customers') }}
""")
write("models/marts/customer_revenue.sql", """
    select c.customer_id, c.country, c.segment,
           count(f.order_id) as orders,
           coalesce(sum(f.order_total) filter (where f.status = 'delivered'), 0) as delivered_revenue
    from {{ ref('dim_customers') }} c
    left join {{ ref('fct_orders') }} f using (customer_id)
    group by all
""")
write("models/marts/schema.yml", """
    version: 2
    models:
      - name: fct_orders
        columns:
          - name: order_id
            data_tests: [unique, not_null]
      - name: customer_revenue
        data_tests:
          - non_negative_revenue
""")
write("tests/generic/non_negative_revenue.sql", """
    {% test non_negative_revenue(model) %}
    -- a custom generic test: revenue can never be negative
    select * from {{ model }} where delivered_revenue < 0
    {% endtest %}
""")
write("models/marts/events_daily.sql", """
    {{ config(materialized='incremental', unique_key='event_date') }}
    select cast(ts as date) as event_date, count(*) as events
    from {{ source('raw', 'events') }}
    {% if is_incremental() %}
    where cast(ts as date) >= (select max(event_date) from {{ this }})
    {% endif %}
    group by 1
""")

def load_raw(tables):
    con = duckdb.connect(str(DB))
    for name in ("customers", "orders", "order_items", "events"):
        df = tables[name]
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")
    con.close()


def dbt(*args):
    res = dbtRunner().invoke([*args, "--project-dir", str(proj), "--profiles-dir", str(proj), "--quiet"])
    statuses = {}
    if res.result is not None and hasattr(res.result, "results"):
        for r in res.result.results:
            statuses[r.node.name] = str(r.status)
    return res.success, statuses

# %% [markdown]
# ## 1. dbt build

# %%
t = make_shop(n_customers=3_000, seed=14)
load_raw(t)
ok, statuses = dbt("build")
print(f"dbt build success={ok}: {sum(s in ('success', 'pass') for s in statuses.values())} nodes ok out of {len(statuses)}")
assert ok, statuses
con = duckdb.connect(str(DB))
rev_dbt = con.execute("select sum(delivered_revenue) from customer_revenue").fetchone()[0]
con.close()
d = t["order_items"].merge(t["orders"], on="order_id")
rev_pd = (d.loc[d["status"] == "delivered", "quantity"] * d.loc[d["status"] == "delivered", "unit_price"]).sum()
assert abs(rev_dbt - rev_pd) < 1e-6 * rev_pd
print(f"delivered revenue from the dbt mart: {rev_dbt:,.2f} (matches pandas)")

# %% [markdown]
# ## 2. Break the data: a duplicated customer

# %%
bad = dict(t)
bad["customers"] = pd.concat([t["customers"], t["customers"].iloc[[0]]], ignore_index=True)   # a double load
load_raw(bad)
ok, statuses = dbt("build")
failed = [n for n, s in statuses.items() if s in ("fail", "error")]
skipped = [n for n, s in statuses.items() if s == "skipped"]
print(f"dbt build success={ok}; failed: {failed}; skipped downstream: {skipped}")
assert not ok and any("unique_stg_customers_customer_id" in n for n in failed)
assert "dim_customers" in skipped and "customer_revenue" in skipped, "dbt build stops the bad data from propagating"
load_raw(t)

# %% [markdown]
# ## 3. Incremental model

# %%
ok, _ = dbt("build", "--select", "events_daily")
con = duckdb.connect(str(DB))
n_days_1 = con.execute("select count(*) from events_daily").fetchone()[0]
last_day = con.execute("select max(event_date) from events_daily").fetchone()[0]
con.execute("insert into events select * from events where cast(ts as date) = (select max(cast(ts as date)) from events)")
con.close()
ok, _ = dbt("build", "--select", "events_daily")
con = duckdb.connect(str(DB))
n_days_2 = con.execute("select count(*) from events_daily").fetchone()[0]
last_count = con.execute("select events from events_daily where event_date = ?", [last_day]).fetchone()[0]
true_last = con.execute("select count(*) from events where cast(ts as date) = ?", [last_day]).fetchone()[0]
con.close()
print(f"incremental: {n_days_1} days after the first run, {n_days_2} after the second; last day recomputed = {last_count} events")
assert ok and n_days_1 == n_days_2 and last_count == true_last

# %% [markdown]
# ## 4. Lineage from the manifest

# %%
manifest = json.loads((proj / "target" / "manifest.json").read_text())
parents = {n["name"]: sorted(p.split(".")[-1] for p in n["depends_on"]["nodes"])
           for n in manifest["nodes"].values() if n["resource_type"] == "model"}
for model in ("fct_orders", "customer_revenue"):
    print(f"{model} <- {parents[model]}")
assert parents["customer_revenue"] == ["dim_customers", "fct_orders"]
assert parents["fct_orders"] == ["stg_order_items", "stg_orders"]

shutil.rmtree(proj.parent)
print("\nAll checks passed.")
