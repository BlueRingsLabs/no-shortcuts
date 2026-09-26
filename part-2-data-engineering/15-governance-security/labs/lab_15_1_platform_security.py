# %% [markdown]
# # Lab 15.1: Guardrails for a data platform
#
# 1. A pre-commit style check that keeps credentials out of a repository.
# 2. Role-based access with masked views and row-level policies in DuckDB.
# 3. Field-level encryption of a restricted column before it lands in the warehouse.
# 4. Reviewing an access log: per-user baselines and unusual export volumes.

# %%
import math
import re
import sys
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

# %% [markdown]
# ## 1. Keep credentials out of the repository
#
# A small version of what tools like gitleaks do in a pre-commit hook: known credential patterns plus a
# high-entropy check for long random-looking strings assigned to suspicious names. The goal is to stop the
# commit before a secret reaches Git history.

# %%
PATTERNS = {
    "cloud access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "connection string with password": re.compile(r"\b\w+://[^\s:/]+:[^\s@/]+@[^\s/]+"),
}
SUSPICIOUS_NAME = re.compile(r"(?i)(secret|token|password|passwd|api_key|apikey)\s*[=:]\s*['\"]?([^\s'\"]{16,})")


def shannon_entropy(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def scan(text: str) -> list[tuple[int, str]]:
    findings = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pat in PATTERNS.items():
            if pat.search(line):
                findings.append((lineno, name))
        m = SUSPICIOUS_NAME.search(line)
        if m and shannon_entropy(m.group(2)) > 3.5:
            findings.append((lineno, "high-entropy value assigned to a secret-like name"))
    return findings


# Test fixtures: obviously fake values, built at runtime so this lab file itself doesn't trip a real scanner.
fake_key_id = "AKIA" + "EXAMPLE0" * 2
fake_token = "".join(np.random.default_rng(0).choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), 32))
bad_config = f"""
db_host = "warehouse.internal"
aws_access_key_id = "{fake_key_id}"
API_KEY = "{fake_token}"
dsn = "postgresql://etl:{'hunter2'}@db.internal:5432/shop"
"""
good_config = """
db_host = "warehouse.internal"
api_key = os.environ["SHOP_API_KEY"]          # injected at runtime from the secrets manager
dsn = os.environ["WAREHOUSE_DSN"]
log_level = "INFO"
"""
bad_findings, good_findings = scan(bad_config), scan(good_config)
print("findings in the bad config:", bad_findings)
print("findings in the good config:", good_findings)
assert {f[1] for f in bad_findings} >= {"cloud access key id", "connection string with password",
                                        "high-entropy value assigned to a secret-like name"}
assert good_findings == [], "reading secrets from the environment is the pattern we want"

# %% [markdown]
# ## 2. Roles, masked views and row-level policies
#
# Analysts never get the raw table. Each role gets a view that applies its masking and row filter, and grants
# are to views only. DuckDB has no users/grants, so we model the grant table explicitly and route every query
# through a function that enforces it: the same shape a warehouse's policy engine has.

# %%
t = make_shop(n_customers=2_000, seed=15)
con = duckdb.connect()
con.register("customers_df", t["customers"])
con.execute("CREATE SCHEMA restricted; CREATE TABLE restricted.customers AS SELECT * FROM customers_df")
con.execute("CREATE TABLE salt AS SELECT 'rotate-me-per-environment' AS s")      # in real life: from the secrets manager

con.execute("""
CREATE VIEW v_customers_support AS
SELECT customer_id, name,
       left(email, 1) || '***@' || split_part(email, '@', 2) AS email_masked,
       country, segment
FROM restricted.customers;

CREATE VIEW v_customers_marketing AS
SELECT sha256(email || (SELECT s FROM salt)) AS email_key,       -- pseudonymous: joinable and countable, not readable
       country, segment, signup_date
FROM restricted.customers;

CREATE TABLE region_policy (role VARCHAR, country VARCHAR);
INSERT INTO region_policy VALUES ('support_es', 'ES'), ('support_ar', 'AR');
""")

GRANTS = {
    "support_es": {"v_customers_support"},
    "support_ar": {"v_customers_support"},
    "marketing": {"v_customers_marketing"},
    "fraud": {"restricted.customers"},
}


def query_as(role: str, relation: str, where: str = "true") -> pd.DataFrame:
    if relation not in GRANTS.get(role, set()):
        raise PermissionError(f"role {role!r} has no access to {relation!r}")
    row_filter = ""
    if relation == "v_customers_support":
        row_filter = f" AND country IN (SELECT country FROM region_policy WHERE role = '{role}')"
    return con.execute(f"SELECT * FROM {relation} WHERE ({where}){row_filter}").df()


es = query_as("support_es", "v_customers_support")
assert set(es["country"]) == {"ES"}, "row-level policy: Spanish support only sees Spanish customers"
assert es["email_masked"].str.contains(r"^\w\*\*\*@", regex=True).all() and "email" not in es.columns
mk = query_as("marketing", "v_customers_marketing")
assert "email" not in mk.columns and mk["email_key"].nunique() == t["customers"]["email"].nunique(), \
    "marketing can count distinct customers without ever seeing an email"
try:
    query_as("marketing", "restricted.customers")
    raise AssertionError("marketing must not read the restricted table")
except PermissionError as e:
    print("denied:", e)
full = query_as("fraud", "restricted.customers")
assert full["email"].str.contains("@").all()
print(f"support_es sees {len(es)} masked rows, marketing sees {len(mk)} pseudonymized rows, fraud sees {len(full)} full rows")

# %% [markdown]
# ## 3. Field-level encryption
#
# The most sensitive column is encrypted in the pipeline, before loading. The warehouse stores ciphertext; only a
# service holding the key (from a key management service, never from the repo) can decrypt.

# %%
key = Fernet.generate_key()                 # stand-in for a KMS-managed data key
f = Fernet(key)
ids = pd.DataFrame({"customer_id": t["customers"]["customer_id"].head(5),
                    "national_id": [f"ID-{n:08d}" for n in np.random.default_rng(1).integers(0, 10**8, 5)]})
ids["national_id_enc"] = [f.encrypt(v.encode()).decode() for v in ids["national_id"]]
to_load = ids.drop(columns=["national_id"])
con.register("to_load", to_load)
con.execute("CREATE TABLE restricted.customer_ids AS SELECT * FROM to_load")
stored = con.execute("SELECT national_id_enc FROM restricted.customer_ids").df()["national_id_enc"]
assert not stored.str.contains("ID-").any(), "the warehouse never sees plaintext"
decrypted = [f.decrypt(v.encode()).decode() for v in stored]
assert decrypted == ids["national_id"].tolist()
try:
    Fernet(Fernet.generate_key()).decrypt(stored.iloc[0].encode())
    raise AssertionError("a different key must not decrypt")
except Exception as e:  # noqa: BLE001
    assert type(e).__name__ == "InvalidToken"
print("field-level encryption: ciphertext at rest, decryptable only with the right key")

# %% [markdown]
# ## 4. Reviewing the access log
#
# 60 days of synthetic warehouse query logs. Each user has a normal daily export volume; we flag days far
# above that user's own baseline (robust z-score with median and MAD, 10.3), plus reads of restricted tables by
# roles that shouldn't have them.

# %%
rng = np.random.default_rng(15)
users = {f"analyst_{i}": rng.lognormal(8, 0.4) for i in range(12)}
rows = []
for day in pd.date_range("2025-05-01", periods=60):
    for u, typical in users.items():
        rows.append((day, u, "analyst", "gold.fct_orders", rng.lognormal(np.log(typical), 0.3)))
rows.append((pd.Timestamp("2025-06-20"), "analyst_3", "analyst", "gold.fct_orders", 40 * users["analyst_3"]))
rows.append((pd.Timestamp("2025-06-21"), "analyst_7", "analyst", "restricted.customers", 900.0))
log = pd.DataFrame(rows, columns=["day", "user", "role", "relation", "rows_returned"])


def flag_unusual_volume(log: pd.DataFrame, threshold: float = 6.0) -> pd.DataFrame:
    g = log.groupby("user")["rows_returned"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    z = (log["rows_returned"] - med) / mad
    return log.assign(robust_z=z)[z > threshold]


ALLOWED = {"analyst": ("gold.",), "fraud": ("gold.", "restricted.")}
unusual = flag_unusual_volume(log)
out_of_policy = log[~log.apply(lambda r: r["relation"].startswith(ALLOWED[r["role"]]), axis=1)]
print("unusual export volumes:\n", unusual[["day", "user", "rows_returned", "robust_z"]].round({"rows_returned": 0, "robust_z": 1}).to_string(index=False))
print("reads outside the role's policy:\n", out_of_policy[["day", "user", "relation"]].to_string(index=False))
assert list(unusual["user"]) == ["analyst_3"]
assert list(out_of_policy["user"]) == ["analyst_7"]
print("both would go to a reviewer; neither is proof of wrongdoing, both deserve a question")

print("\nAll checks passed.")
