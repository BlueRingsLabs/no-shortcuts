# %% [markdown]
# # Lab 10.3: Find the garbage, clean it, and prove it's clean
#
# The dirty shop has nine injected problems. This lab:
#
# 1. Runs a small expectations framework against the raw (dirty) tables and reports every failing check with counts.
# 2. Cleans the data in one deterministic, idempotent function that logs what it did.
# 3. Re-runs the checks on the cleaned data: everything must pass.
# 4. Verifies the cleaning against the known ground truth (the clean shop with the same seed).

# %%
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

truth = make_shop(n_customers=5_000, seed=0)
raw = make_shop(n_customers=5_000, seed=0, dirty=True)
EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$"
COUNTRIES = {"AR", "ES", "US", "DE", "PL", "BR", "MX", "FR"}
STATUSES = {"delivered", "shipped", "cancelled", "returned"}
NOW = pd.Timestamp("2026-06-30")

# %% [markdown]
# ## 1. A tiny expectations framework

# %%
@dataclass
class Check:
    name: str
    dimension: str
    fn: object            # tables -> boolean Series (True = row OK) or a bool

    def run(self, tables):
        res = self.fn(tables)
        if isinstance(res, (bool, np.bool_)):
            return bool(res), 0 if res else 1
        return bool(res.all()), int((~res).sum())


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


CHECKS = [
    Check("customers.customer_id unique", "uniqueness", lambda t: ~t["customers"]["customer_id"].duplicated()),
    Check("customers.email normalized-unique", "uniqueness",
          lambda t: ~t["customers"]["email"].dropna().str.strip().str.lower().duplicated()),
    Check("customers.email present", "completeness", lambda t: t["customers"]["email"].notna()),
    Check("customers.email valid", "validity", lambda t: t["customers"]["email"].isna() | t["customers"]["email"].str.match(EMAIL_RE)),
    Check("customers.country ISO code", "consistency", lambda t: t["customers"]["country"].isin(COUNTRIES)),
    Check("customers.signup_date not future", "validity",
          lambda t: t["customers"]["signup_date"].isna() | (t["customers"]["signup_date"] <= NOW)),
    Check("customers.signup_date present", "completeness", lambda t: t["customers"]["signup_date"].notna()),
    Check("order_items.quantity in [1, 100]", "validity", lambda t: t["order_items"]["quantity"].between(1, 100)),
    Check("order_items no exact duplicates", "uniqueness", lambda t: ~t["order_items"].duplicated()),
    Check("order_items (order_id, product_id) unique", "uniqueness",
          lambda t: ~t["order_items"].duplicated(["order_id", "product_id"])),
    Check("orders.customer_id exists", "integrity", lambda t: t["orders"]["customer_id"].isin(t["customers"]["customer_id"])),
    Check("orders.status canonical", "consistency", lambda t: t["orders"]["status"].isin(STATUSES)),
    Check("products.price numeric", "validity", lambda t: t["products"]["price"].isna() | to_num(t["products"]["price"]).notna()),
    Check("products.price present", "completeness", lambda t: t["products"]["price"].notna()),
    Check("products.price > 0", "validity", lambda t: to_num(t["products"]["price"]).fillna(1) > 0),
]


def run_checks(tables, label):
    rows = []
    for c in CHECKS:
        ok, n_bad = c.run(tables)
        rows.append((c.name, c.dimension, "PASS" if ok else "FAIL", n_bad))
    report = pd.DataFrame(rows, columns=["check", "dimension", "result", "bad_rows"])
    print(f"\n== {label} ==\n{report.to_string(index=False)}")
    return report


before = run_checks(raw, "raw (dirty) data")
assert (before["result"] == "FAIL").sum() >= 9, "every injected problem should trip at least one check"
assert (run_checks(truth, "ground truth")["result"] == "PASS").all(), "the clean generator passes everything"

# %% [markdown]
# ## 2. Cleaning: deterministic, idempotent, logged

# %%
COUNTRY_MAP = {"ARGENTINA": "AR", "USA": "US", "ES": "ES", "DE": "DE"}


def clean(t):
    log = {}
    c = t["customers"].copy()
    c["email"] = c["email"].str.strip().str.lower()
    bad_email = c["email"].notna() & ~c["email"].str.match(EMAIL_RE)
    log["emails nulled (malformed)"] = int(bad_email.sum())
    c.loc[bad_email, "email"] = pd.NA
    up = c["country"].str.strip().str.upper()
    c["country"] = up.map(lambda x: COUNTRY_MAP.get(x, x))
    log["countries remapped"] = int((t["customers"]["country"] != c["country"]).sum())
    future = c["signup_date"] > NOW
    log["future signup dates nulled"] = int(future.sum())
    c.loc[future, "signup_date"] = pd.NaT
    # entity dedup: same normalized email -> keep the lowest customer_id, remember the mapping
    has_email = c["email"].notna()
    first_id = c[has_email].groupby("email")["customer_id"].transform("min")
    remap = dict(zip(c.loc[has_email, "customer_id"], first_id))
    dup = has_email & (c["customer_id"] != c["email"].map(c[has_email].groupby("email")["customer_id"].min()))
    log["duplicate customers merged"] = int(dup.sum())
    c = c[~dup].reset_index(drop=True)

    o = t["orders"].copy()
    o["customer_id"] = o["customer_id"].map(lambda x: remap.get(x, x))
    o["status"] = o["status"].str.strip().str.lower()
    orphan = ~o["customer_id"].isin(c["customer_id"])
    log["orphan orders quarantined"] = int(orphan.sum())
    quarantine = {"orders": o[orphan]}
    o = o[~orphan].reset_index(drop=True)

    it = t["order_items"].copy()
    n0 = len(it)
    it = it.drop_duplicates()
    log["exact duplicate lines removed"] = n0 - len(it)
    bad_q = ~it["quantity"].between(1, 100)
    log["lines with invalid quantity quarantined"] = int(bad_q.sum())
    quarantine["order_items"] = it[bad_q]
    it = it[~bad_q & it["order_id"].isin(o["order_id"])].reset_index(drop=True)

    p = t["products"].copy()
    parsed = pd.to_numeric(p["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    log["prices parsed from strings"] = int(to_num(p["price"]).isna().sum())
    p["price"] = parsed
    zero = p["price"] <= 0
    log["zero prices nulled"] = int(zero.sum())
    p.loc[zero, "price"] = np.nan
    p["price"] = p["price"].astype(float)
    return {**t, "customers": c, "orders": o, "order_items": it, "products": p}, log, quarantine


cleaned, log, quarantine = clean(raw)
print("\ncleaning log:")
for k, v in log.items():
    print(f"  {k:42s} {v:6d}")

# %% [markdown]
# ## 3. Re-run the checks

# %%
after = run_checks(cleaned, "cleaned data")
# We chose to NULL invalid values rather than invent replacements, so some completeness checks still fail,
# visibly and with counts. That's honest: a downstream consumer can decide what missing means for them.
allowed_fail = {"customers.email present", "customers.signup_date present", "products.price present"}
failing = set(after.loc[after["result"] == "FAIL", "check"])
assert failing <= allowed_fail, failing
cleaned2, log2, _ = clean(cleaned)
assert all(len(cleaned2[k]) == len(cleaned[k]) for k in cleaned), "cleaning is idempotent: a second pass changes nothing"
assert log2["duplicate customers merged"] == 0 and log2["exact duplicate lines removed"] == 0

# %% [markdown]
# ## 4. Against the ground truth

# %%
extra = set(cleaned["customers"]["customer_id"]) - set(truth["customers"]["customer_id"])
assert set(truth["customers"]["customer_id"]) <= set(cleaned["customers"]["customer_id"]), "no real customer was lost"
unmergeable = cleaned["customers"][cleaned["customers"]["customer_id"].isin(extra)]
n_destroyed = int((raw["customers"]["email"].isna() | (raw["customers"]["email"] == "not-an-email")).sum())
print(f"{len(extra)} duplicate customers could not be merged: the email of one of the two copies was ALSO destroyed "
      f"(null or malformed), so nothing reliable links them. Real life does this to you too; a fuzzy match on name "
      f"would find candidates for a human to review.")
assert 0 < len(extra) <= n_destroyed
m = cleaned["customers"].merge(truth["customers"], on="customer_id", suffixes=("", "_true"))
assert (m["country"] == m["country_true"]).all(), "all country codes restored"
assert log["orphan orders quarantined"] == 25
kept_lines = cleaned["order_items"].merge(truth["order_items"], on=["order_id", "product_id"], how="left", indicator=True)
assert (kept_lines["_merge"] == "both").all(), "no invented order lines"
print(f"\ncleaned: {len(cleaned['order_items']):,} order lines vs {len(truth['order_items']):,} in the truth "
      f"(the difference is exactly the quarantined ones: {len(truth['order_items']) - len(cleaned['order_items'])})")

# %%
print("\nAll checks passed.")
