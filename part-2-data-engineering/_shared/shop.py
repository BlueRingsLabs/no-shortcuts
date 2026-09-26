"""The "Rat & Co." online shop: a small, deterministic, synthetic dataset used across Part II.

Why synthetic? Because I know the ground truth. When a lab says "this query should return 1,337 rows", it's
because I generated the data and know it does. Real data is for capstones.

Usage from a lab (labs run with their own folder as the working directory):

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
    from shop import make_shop
    tables = make_shop(n_customers=5_000, seed=0)            # dict of pandas DataFrames
    dirty = make_shop(n_customers=5_000, seed=0, dirty=True)  # same data, with realistic garbage mixed in

Tables:
    customers(customer_id, name, email, country, signup_date, segment)
    products(product_id, name, category, price)
    orders(order_id, customer_id, order_ts, status, channel)
    order_items(order_id, product_id, quantity, unit_price)
    events(event_id, customer_id, ts, event_type, product_id)      # clickstream
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNTRIES = ["AR", "ES", "US", "DE", "PL", "BR", "MX", "FR"]
COUNTRY_P = [0.18, 0.16, 0.2, 0.1, 0.08, 0.12, 0.1, 0.06]
CATEGORIES = {
    "laptops": (600, 2500), "gpus": (250, 2000), "keyboards": (30, 200), "monitors": (120, 900),
    "cables": (5, 40), "books": (15, 80), "coffee": (8, 30), "hoodies": (35, 90),
}
FIRST = ["Ana", "Luis", "Marta", "Jan", "Olga", "Pedro", "Sofia", "Tom", "Ines", "Kai", "Lena", "Hugo"]
LAST = ["Garcia", "Kowalski", "Smith", "Muller", "Rossi", "Silva", "Lopez", "Novak", "Martin", "Costa"]
START = pd.Timestamp("2024-01-01")


def make_shop(n_customers: int = 5_000, n_products: int = 400, seed: int = 0, dirty: bool = False,
              events_per_customer: float = 12.0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # --- customers
    cid = np.arange(1, n_customers + 1)
    first = rng.choice(FIRST, n_customers)
    last = rng.choice(LAST, n_customers)
    customers = pd.DataFrame({
        "customer_id": cid,
        "name": [f"{f} {l}" for f, l in zip(first, last)],
        "email": [f"{f.lower()}.{l.lower()}{i}@example.com" for f, l, i in zip(first, last, cid)],
        "country": rng.choice(COUNTRIES, n_customers, p=COUNTRY_P),
        "signup_date": START + pd.to_timedelta(rng.integers(0, 540, n_customers), unit="D"),
        "segment": rng.choice(["consumer", "pro", "enterprise"], n_customers, p=[0.8, 0.17, 0.03]),
    })

    # --- products
    cats = rng.choice(list(CATEGORIES), n_products)
    lo = np.array([CATEGORIES[c][0] for c in cats])
    hi = np.array([CATEGORIES[c][1] for c in cats])
    products = pd.DataFrame({
        "product_id": np.arange(1, n_products + 1),
        "name": [f"{c[:-1].title()} #{i}" for i, c in enumerate(cats, start=1)],
        "category": cats,
        "price": np.round(lo + (hi - lo) * rng.beta(2, 5, n_products), 2),
    })

    # --- orders: heavy users order a lot (gamma-poisson), never before signup
    activity = rng.gamma(0.8, 2.5, n_customers)
    n_orders = rng.poisson(activity)
    o_cust = np.repeat(cid, n_orders)
    signup = customers.set_index("customer_id").loc[o_cust, "signup_date"].to_numpy()
    days_after = rng.integers(0, 365, len(o_cust))
    order_ts = pd.to_datetime(signup) + pd.to_timedelta(days_after, unit="D") + pd.to_timedelta(
        rng.integers(0, 86_400, len(o_cust)), unit="s")
    orders = pd.DataFrame({
        "order_id": np.arange(1, len(o_cust) + 1),
        "customer_id": o_cust,
        "order_ts": order_ts,
        "status": rng.choice(["delivered", "shipped", "cancelled", "returned"], len(o_cust), p=[0.82, 0.08, 0.06, 0.04]),
        "channel": rng.choice(["web", "app", "marketplace"], len(o_cust), p=[0.55, 0.35, 0.10]),
    }).sort_values("order_ts", kind="stable").reset_index(drop=True)
    orders["order_id"] = np.arange(1, len(orders) + 1)

    # --- order items: 1-5 lines per order, popular products more likely (Zipf-ish)
    lines = rng.integers(1, 6, len(orders))
    it_order = np.repeat(orders["order_id"].to_numpy(), lines)
    popularity = 1.0 / np.arange(1, n_products + 1) ** 0.9
    popularity /= popularity.sum()
    it_prod = rng.choice(products["product_id"].to_numpy(), len(it_order), p=popularity)
    items = pd.DataFrame({
        "order_id": it_order,
        "product_id": it_prod,
        "quantity": rng.choice([1, 1, 1, 1, 2, 2, 3], len(it_order)),
    })
    items = items.drop_duplicates(["order_id", "product_id"]).reset_index(drop=True)
    base_price = products.set_index("product_id").loc[items["product_id"], "price"].to_numpy()
    discount = rng.choice([1.0, 1.0, 1.0, 0.9, 0.8], len(items))
    items["unit_price"] = np.round(base_price * discount, 2)

    # --- clickstream events, clustered in sessions
    n_ev = rng.poisson(events_per_customer * activity / activity.mean())
    ev_cust = np.repeat(cid, n_ev)
    n_total = len(ev_cust)
    session_start = START + pd.to_timedelta(rng.integers(0, 540 * 86_400, n_total), unit="s")
    # events come in bursts: 70% happen within minutes of the previous one for the same customer
    ev = pd.DataFrame({"customer_id": ev_cust, "ts": session_start})
    ev = ev.sort_values(["customer_id", "ts"]).reset_index(drop=True)
    same = ev["customer_id"].eq(ev["customer_id"].shift())
    burst = same & (rng.uniform(size=n_total) < 0.7)
    gaps = pd.to_timedelta(rng.exponential(90, n_total).astype(int) + 1, unit="s")
    ts = ev["ts"].to_numpy().copy()
    for i in np.flatnonzero(burst.to_numpy()):
        ts[i] = ts[i - 1] + gaps[i]
    ev["ts"] = ts
    ev["event_type"] = rng.choice(["page_view", "product_view", "add_to_cart", "checkout"], n_total, p=[0.5, 0.35, 0.1, 0.05])
    ev["product_id"] = np.where(ev["event_type"] == "page_view", pd.NA,
                                rng.choice(products["product_id"].to_numpy(), n_total, p=popularity))
    ev["product_id"] = ev["product_id"].astype("Int64")
    ev = ev.sort_values("ts", kind="stable").reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(1, len(ev) + 1))

    tables = {"customers": customers, "products": products, "orders": orders, "order_items": items, "events": ev}
    if dirty:
        tables = _add_dirt(tables, rng)
    return tables


def _add_dirt(t: dict[str, pd.DataFrame], rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """The things real data does to you. Each one is documented, because lesson 10.3 asks you to find them all."""
    c = t["customers"].copy()
    n = len(c)
    # 1. duplicated customers (same person signed up twice, different id, same email with different case/spaces)
    dup = c.sample(frac=0.02, random_state=1).copy()
    dup["customer_id"] = np.arange(c["customer_id"].max() + 1, c["customer_id"].max() + 1 + len(dup))
    dup["email"] = "  " + dup["email"].str.upper() + " "
    c = pd.concat([c, dup], ignore_index=True)
    # 2. missing and malformed emails
    idx = rng.choice(len(c), int(0.01 * n), replace=False)
    c.loc[idx[: len(idx) // 2], "email"] = None
    c.loc[idx[len(idx) // 2:], "email"] = "not-an-email"
    # 3. inconsistent country codes
    idx = rng.choice(len(c), int(0.02 * n), replace=False)
    c.loc[idx, "country"] = c.loc[idx, "country"].map({"AR": "Argentina", "ES": "es", "US": "USA", "DE": "de"}).fillna(c.loc[idx, "country"])
    # 4. signup dates in the future (a timezone/parsing bug upstream)
    idx = rng.choice(len(c), 15, replace=False)
    c.loc[idx, "signup_date"] = pd.Timestamp("2099-01-01")

    it = t["order_items"].copy()
    # 5. negative and absurd quantities
    idx = rng.choice(len(it), 20, replace=False)
    it.loc[idx[:10], "quantity"] = -1
    it.loc[idx[10:], "quantity"] = 9999
    # 6. exact duplicate rows (a retried load)
    it = pd.concat([it, it.sample(n=50, random_state=2)], ignore_index=True)

    o = t["orders"].copy()
    # 7. orders pointing to customers that don't exist (orphans)
    idx = rng.choice(len(o), 25, replace=False)
    o.loc[idx, "customer_id"] = o["customer_id"].max() + 10_000
    # 8. status with stray whitespace and case
    idx = rng.choice(len(o), 40, replace=False)
    o.loc[idx, "status"] = o.loc[idx, "status"].str.upper() + " "

    p = t["products"].copy()
    # 9. a price stored as a string with a currency symbol, and a zero price
    p["price"] = p["price"].astype(object)
    p.loc[3, "price"] = "EUR 129.99"
    p.loc[7, "price"] = 0.0
    return {**t, "customers": c, "order_items": it, "orders": o, "products": p}


if __name__ == "__main__":
    for name, df in make_shop().items():
        print(f"{name:12s} {len(df):>8,} rows  {list(df.columns)}")
