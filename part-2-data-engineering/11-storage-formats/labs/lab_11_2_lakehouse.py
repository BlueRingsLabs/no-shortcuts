# %% [markdown]
# # Lab 11.2: Build a (tiny) table format
#
# We implement the core of Delta Lake / Iceberg on the local filesystem, treating it like object storage:
# data files are immutable Parquet files, and a transaction log of numbered JSON commits says which files form the table.
#
# 1. Append, delete and read through the log: readers never see uncommitted files.
# 2. A crashed writer leaves orphan files that nobody reads.
# 3. Optimistic concurrency: two writers race for the same version; one retries.
# 4. Time travel.
# 5. Compaction of small files, and vacuum.
# 6. Hive-style partitioning with DuckDB and partition pruning.

# %%
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shop import make_shop  # noqa: E402


class CommitConflict(Exception):
    pass


class TinyTable:
    """Data files in data/, commits in _log/00000000000000000001.json, ... Each commit lists files added and removed."""

    def __init__(self, root: Path):
        self.root = root
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "_log").mkdir(exist_ok=True)

    # --- log
    def version(self) -> int:
        versions = [int(p.stem) for p in (self.root / "_log").glob("*.json")]
        return max(versions, default=0)

    def files_at(self, version: int | None = None) -> set[str]:
        version = self.version() if version is None else version
        live: set[str] = set()
        for v in range(1, version + 1):
            entry = json.loads((self.root / "_log" / f"{v:020d}.json").read_text())
            live |= set(entry["add"])
            live -= set(entry["remove"])
        return live

    def _commit(self, base_version: int, add: list[str], remove: list[str], op: str):
        target = self.root / "_log" / f"{base_version + 1:020d}.json"
        payload = json.dumps({"op": op, "add": add, "remove": remove, "ts": time.time()})
        try:
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # atomic put-if-absent
        except FileExistsError:
            raise CommitConflict(f"version {base_version + 1} already exists") from None
        with os.fdopen(fd, "w") as f:
            f.write(payload)

    # --- data
    def _write_file(self, df: pd.DataFrame) -> str:
        name = f"part-{uuid.uuid4().hex[:12]}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), self.root / "data" / name)
        return name

    def append(self, df, base_version=None, retries=3):
        name = self._write_file(df)                 # 1. write data (invisible until committed)
        for _ in range(retries):
            base = self.version() if base_version is None else base_version
            try:
                self._commit(base, [name], [], "append")   # 2. publish atomically
                return
            except CommitConflict:
                base_version = None                 # appends never conflict logically: retry on the latest version
        raise CommitConflict("gave up")

    def delete_where(self, predicate):
        base = self.version()
        add, remove = [], []
        for f in self.files_at(base):
            df = pd.read_parquet(self.root / "data" / f)
            keep = df[~predicate(df)]
            if len(keep) < len(df):                 # copy-on-write: rewrite only affected files
                remove.append(f)
                if len(keep):
                    add.append(self._write_file(keep))
        self._commit(base, add, remove, "delete")

    def read(self, version=None) -> pd.DataFrame:
        files = sorted(self.files_at(version))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(self.root / "data" / f) for f in files], ignore_index=True)

    def compact(self):
        base = self.version()
        files = sorted(self.files_at(base))
        merged = pd.concat([pd.read_parquet(self.root / "data" / f) for f in files], ignore_index=True)
        self._commit(base, [self._write_file(merged)], files, "compact")

    def vacuum(self, keep_versions: int):
        """Delete files not referenced by any of the last `keep_versions` versions."""
        current = self.version()
        needed = set().union(*(self.files_at(v) for v in range(max(1, current - keep_versions + 1), current + 1)))
        removed = 0
        for p in (self.root / "data").glob("*.parquet"):
            if p.name not in needed:
                p.unlink()
                removed += 1
        return removed


root = Path(tempfile.mkdtemp())
tbl = TinyTable(root / "orders")
t = make_shop(n_customers=3_000, seed=10)
orders = t["orders"]
batches = [orders.iloc[i:i + 500] for i in range(0, len(orders), 500)]

# %% [markdown]
# ## 1. Appends and deletes through the log

# %%
for b in batches:
    tbl.append(b)
assert len(tbl.read()) == len(orders) and tbl.version() == len(batches)
print(f"{tbl.version()} commits, {len(tbl.files_at())} data files, {len(tbl.read()):,} rows")

before_delete = tbl.version()
VICTIM = int(orders["customer_id"].value_counts().index[0])     # a customer with many orders
tbl.delete_where(lambda d: d["customer_id"] == VICTIM)      # e.g. a right-to-erasure request
assert (tbl.read()["customer_id"] != VICTIM).all()
print(f"deleted customer {VICTIM}: now version {tbl.version()}, {len(tbl.read()):,} rows")

# %% [markdown]
# ## 2. A crashed writer

# %%
orphan = tbl._write_file(orders.head(100))               # data written, but the process "dies" before committing
assert orphan not in tbl.files_at()
assert len(tbl.read()) == len(orders) - (orders["customer_id"] == VICTIM).sum(), "readers never see uncommitted files"
naive = pd.concat([pd.read_parquet(p) for p in (tbl.root / "data").glob("*.parquet")])
print(f"a reader that lists files instead of reading the log sees {len(naive):,} rows (wrong, includes the orphan and deleted files)")
assert len(naive) > len(tbl.read())

# %% [markdown]
# ## 3. Optimistic concurrency

# %%
v = tbl.version()
tbl.append(orders.head(10), base_version=v)            # writer A commits v+1
try:
    tbl._commit(v, ["someone-elses-file.parquet"], [], "append")   # writer B also tries to commit v+1
    raise AssertionError("writer B should have conflicted")
except CommitConflict as e:
    print("writer B:", e, "-> re-read the log and retry on top of the new version")
tbl.append(orders.head(5), base_version=v)             # the retry loop inside append() handles it
assert tbl.version() == v + 2

# %% [markdown]
# ## 4. Time travel

# %%
old = tbl.read(before_delete)
assert (old["customer_id"] == VICTIM).any(), "the version before the delete still has the customer"
print(f"time travel to version {before_delete}: {len(old):,} rows, customer {VICTIM} still present")

# %% [markdown]
# ## 5. Compaction and vacuum

# %%
n_files_before = len(tbl.files_at())
t0 = time.perf_counter(); tbl.read(); t_many = time.perf_counter() - t0
tbl.compact()
t0 = time.perf_counter(); rows_after = tbl.read(); t_one = time.perf_counter() - t0
print(f"compaction: {n_files_before} files -> {len(tbl.files_at())}; read {t_many * 1000:.0f} ms -> {t_one * 1000:.0f} ms")
assert len(tbl.files_at()) == 1

removed = tbl.vacuum(keep_versions=1)
print(f"vacuum removed {removed} unreferenced files (including the orphan and the pre-delete files with the erased customer)")
try:
    tbl.read(before_delete)
    time_travel_ok = True
except FileNotFoundError:
    time_travel_ok = False
assert not time_travel_ok, "after vacuum, old versions can't be read: the erased data is physically gone"

# %% [markdown]
# ## 6. Hive-style partitioning and pruning with DuckDB

# %%
lake = root / "lake"
con = duckdb.connect()
con.register("orders_df", t["orders"].assign(order_date=t["orders"]["order_ts"].dt.date))
con.execute(f"COPY (SELECT * FROM orders_df) TO '{lake}' (FORMAT parquet, PARTITION_BY (order_date))")
parts = sorted(p.name for p in lake.iterdir())
print(f"{len(parts)} partitions, e.g. {parts[:2]}")
assert parts[0].startswith("order_date=")

q = f"SELECT COUNT(*) FROM read_parquet('{lake}/*/*.parquet', hive_partitioning=true) WHERE order_date = '2025-01-15'"
plan = con.execute("EXPLAIN ANALYZE " + q).fetchall()[0][1]
files_line = [ln for ln in plan.splitlines() if "Files" in ln or "File" in ln]
print("scan info:", files_line[:3])
count = con.execute(q).fetchone()[0]
assert count == (t["orders"]["order_ts"].dt.date.astype(str) == "2025-01-15").sum()

shutil.rmtree(root)
print("\nAll checks passed.")
