# %% [markdown]
# # Lab 15.2: Classification, pseudonymization, lineage and erasure
#
# 1. A column scanner that suggests PII classifications from values.
# 2. Keyed pseudonymization: consistent, joinable, not readable.
# 3. k-anonymity: measure it, then generalize quasi-identifiers until it holds.
# 4. Lineage parsed from SQL with sqlglot, and an erasure plan computed from lineage + column tags.
# 5. A data contract check: compatible vs breaking schema changes.

# %%
import hashlib
import hmac
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

t = make_shop(n_customers=5_000, seed=16)
rng = np.random.default_rng(16)

# %% [markdown]
# ## 1. Suggest classifications from the data

# %%
DETECTORS = {
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I),
    "phone": re.compile(r"^\+?\d[\d\s-]{7,}\d$"),
    "card_number": re.compile(r"^\d{13,19}$"),
}


def suggest(df: pd.DataFrame, min_share=0.8) -> dict[str, str]:
    out = {}
    for col in df.columns:
        vals = df[col].dropna().astype(str).head(500)
        for label, pat in DETECTORS.items():
            if len(vals) and vals.str.match(pat).mean() >= min_share:
                out[col] = label
    return out


customers = t["customers"].assign(phone=[f"+34 6{rng.integers(10**7, 10**8)}" for _ in range(len(t["customers"]))])
found = suggest(customers)
print("first attempt:", found)
assert found.get("signup_date") == "phone", "dates like 2024-03-05 are digits and dashes: a false positive"

# Refine: a phone number needs at least 9 digits; an ISO date has 8.
DETECTORS["phone"] = re.compile(r"^(?=(?:\D*\d){9,})\+?\d[\d\s-]{7,}\d$")
found = suggest(customers)
print("refined detector:", found)
assert found == {"email": "email", "phone": "phone"}
print("a human owner still confirms: 'name' is personal data too, and no regex will tell you that reliably")

# %% [markdown]
# ## 2. Keyed pseudonymization (HMAC)
#
# A plain hash of an email can be reversed by hashing a list of candidate emails. A keyed hash (HMAC) can't be
# recomputed without the key, which lives in the secrets manager, not in the warehouse.

# %%
KEY = b"from-the-secrets-manager-not-from-git"


def pseudonymize(value: str, key: bytes = KEY) -> str:
    return hmac.new(key, value.strip().lower().encode(), hashlib.sha256).hexdigest()[:24]


customers["email_key"] = customers["email"].map(pseudonymize)
orders_with_email = t["orders"].merge(t["customers"][["customer_id", "email"]], on="customer_id")
orders_with_email["email_key"] = orders_with_email["email"].map(pseudonymize)
joined = orders_with_email.drop(columns="email").merge(customers[["email_key", "country"]], on="email_key")
assert len(joined) == len(t["orders"]), "consistent tokens: tables still join without exposing emails"
assert pseudonymize(" ANA@X.COM ") == pseudonymize("ana@x.com"), "normalize before tokenizing"
guess = hashlib.sha256(customers["email"].iloc[0].encode()).hexdigest()[:24]
assert guess != customers["email_key"].iloc[0], "an unkeyed dictionary attack doesn't reproduce the token"
print("pseudonymized", customers["email_key"].nunique(), "emails; joins preserved")

# %% [markdown]
# ## 3. k-anonymity

# %%
people = pd.DataFrame({
    "birth_date": pd.to_datetime("1950-01-01") + pd.to_timedelta(rng.integers(0, 365 * 55, 5000), unit="D"),
    "postcode": rng.integers(28001, 28100, 5000).astype(str),
    "gender": rng.choice(["F", "M"], 5000),
    "diagnosis": rng.choice(["A", "B", "C"], 5000),
})


def k_anonymity(df, quasi):
    return int(df.groupby(quasi).size().min())


raw_k = k_anonymity(people, ["birth_date", "postcode", "gender"])
unique_share = (people.groupby(["birth_date", "postcode", "gender"]).size() == 1).mean()
print(f"raw quasi-identifiers: k = {raw_k}; {unique_share:.0%} of combinations are unique people")
assert raw_k == 1

gen = people.assign(birth_decade=(people["birth_date"].dt.year // 10) * 10,
                    postcode_prefix=people["postcode"].str[:3])
k_gen = k_anonymity(gen, ["birth_decade", "postcode_prefix", "gender"])
print(f"after generalizing to birth decade + postcode prefix: k = {k_gen}")
assert k_gen >= 10

# %% [markdown]
# ## 4. Lineage from SQL, and an erasure plan

# %%
MODELS = {
    "stg_customers": "SELECT customer_id, lower(email) AS email, country FROM raw.customers",
    "stg_orders": "SELECT order_id, customer_id, order_ts, status FROM raw.orders",
    "stg_order_items": "SELECT order_id, product_id, quantity * unit_price AS amount FROM raw.order_items",
    "fct_orders": "SELECT o.order_id, o.customer_id, SUM(i.amount) AS total FROM stg_orders o JOIN stg_order_items i ON o.order_id = i.order_id GROUP BY 1, 2",
    "dim_customers": "SELECT customer_id, email, country FROM stg_customers",
    "customer_revenue": "SELECT c.customer_id, c.country, SUM(f.total) AS revenue FROM dim_customers c JOIN fct_orders f ON c.customer_id = f.customer_id GROUP BY 1, 2",
    "revenue_by_country": "SELECT country, SUM(revenue) AS revenue FROM customer_revenue GROUP BY 1",
}

g = nx.DiGraph()
output_cols = {}
for model, sql in MODELS.items():
    tree = sqlglot.parse_one(sql)
    for tbl in tree.find_all(exp.Table):
        src = f"{tbl.db}.{tbl.name}" if tbl.db else tbl.name
        g.add_edge(src, model)
    output_cols[model] = [s.alias_or_name for s in tree.selects]

print("upstream of revenue_by_country:", sorted(nx.ancestors(g, "revenue_by_country")))
assert "raw.customers" in nx.ancestors(g, "revenue_by_country")

# Erasure: tables downstream of personal-data sources that still carry a person identifier in their output
PERSON_KEYS = {"customer_id", "email"}
personal_sources = {"raw.customers", "raw.orders"}
affected = set(personal_sources)
for s in personal_sources:
    affected |= nx.descendants(g, s)
must_delete = sorted(n for n in affected if n in personal_sources or PERSON_KEYS & set(output_cols.get(n, [])))
no_action = sorted(affected - set(must_delete))
print("erasure must process:", must_delete)
print("derived but carries no person identifier:", no_action)
assert "revenue_by_country" in no_action and "customer_revenue" in must_delete and "fct_orders" in must_delete

# %% [markdown]
# ## 5. Contract check

# %%
contract = {"order_id": ("string", False), "customer_id": ("string", False), "placed_at": ("timestamp", False),
            "status": ("string", False), "total_amount_minor": ("int", False)}


def check_change(contract, proposed):
    problems = []
    for col, (typ, nullable) in contract.items():
        if col not in proposed:
            problems.append(f"removed required field {col}")
        elif proposed[col][0] != typ:
            problems.append(f"type of {col} changed {typ} -> {proposed[col][0]}")
        elif proposed[col][1] and not nullable:
            problems.append(f"{col} became nullable")
    for col, (typ, nullable) in proposed.items():
        if col not in contract and not nullable:
            problems.append(f"new field {col} is required (old producers can't provide it)")
    return problems


compatible = {**contract, "coupon_code": ("string", True)}
breaking = {k: v for k, v in contract.items() if k != "total_amount_minor"} | {"total_amount": ("float", False)}
assert check_change(contract, compatible) == []
problems = check_change(contract, breaking)
print("breaking change detected:", problems)
assert any("removed required field total_amount_minor" in p for p in problems)

print("\nAll checks passed.")
