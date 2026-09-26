# %% [markdown]
# # Lab 13.2: An orchestrator in 150 lines
#
# Not a replacement for Airflow or Dagster: a way to see what they do.
#
# 1. DAG definition, cycle detection, topological levels (what can run in parallel).
# 2. Execution with a thread pool, retries with exponential backoff, and upstream_failed propagation.
# 3. Logical dates and a throttled backfill; idempotent tasks make re-runs harmless.
# 4. Asset-style "materialize only what's stale".

# %%
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta

# %% [markdown]
# ## 1. DAGs

# %%
@dataclass
class Task:
    name: str
    fn: callable
    upstream: list = field(default_factory=list)
    retries: int = 2
    retry_delay: float = 0.01


class CycleError(Exception):
    pass


class DAG:
    def __init__(self, name):
        self.name, self.tasks = name, {}

    def task(self, name, upstream=(), retries=2):
        def deco(fn):
            self.tasks[name] = Task(name, fn, list(upstream), retries)
            return fn
        return deco

    def levels(self):
        """Kahn's algorithm, grouped by depth: tasks in the same level can run in parallel."""
        indeg = {n: len(t.upstream) for n, t in self.tasks.items()}
        children = defaultdict(list)
        for n, t in self.tasks.items():
            for u in t.upstream:
                if u not in self.tasks:
                    raise KeyError(f"{n} depends on unknown task {u}")
                children[u].append(n)
        level = sorted(n for n, d in indeg.items() if d == 0)
        out, seen = [], 0
        while level:
            out.append(level)
            seen += len(level)
            nxt = []
            for n in level:
                for c in children[n]:
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        nxt.append(c)
            level = sorted(nxt)
        if seen != len(self.tasks):
            raise CycleError("the graph has a cycle")
        return out

# %% [markdown]
# ## 2. Running a DAG for one logical date

# %%
def run_dag(dag, ds, max_workers=4, log=None):
    state = {n: "pending" for n in dag.tasks}
    lock = threading.Lock()

    def attempt(task):
        for i in range(task.retries + 1):
            try:
                task.fn(ds)
                return "success", i + 1
            except Exception as e:                            # noqa: BLE001 (we record and retry)
                if i == task.retries:
                    return f"failed: {e}", i + 1
                time.sleep(task.retry_delay * 2**i)          # exponential backoff

    with ThreadPoolExecutor(max_workers) as pool:
        for level in dag.levels():
            futures = {}
            for n in level:
                ups = dag.tasks[n].upstream
                if any(state[u] != "success" for u in ups):
                    state[n] = "upstream_failed"
                    continue
                futures[pool.submit(attempt, dag.tasks[n])] = n
            for f in as_completed(futures):
                n = futures[f]
                result, tries = f.result()
                with lock:
                    state[n] = "success" if result == "success" else "failed"
                if log is not None:
                    log.append((ds, n, result, tries))
    return state


# The shop pipeline from the exercise, with an in-memory "warehouse" keyed by (table, logical date)
warehouse: dict = {}
calls = defaultdict(int)
flaky_failures = {"extract_orders": 1}                      # fails once, then succeeds


def make_task(table, flaky=False):
    def fn(ds):
        calls[table] += 1
        if flaky and flaky_failures.get(table, 0) > 0:
            flaky_failures[table] -= 1
            raise ConnectionError("source timed out")
        warehouse[(table, ds)] = f"{table}@{ds}"              # overwrite the partition: idempotent
    return fn


shop = DAG("shop_daily")
for src_name in ("orders", "customers", "products"):
    shop.task(f"extract_{src_name}")(make_task(f"extract_{src_name}", flaky=True))
    shop.task(f"validate_{src_name}", upstream=[f"extract_{src_name}"])(make_task(f"validate_{src_name}"))
shop.task("fact_sales", upstream=["validate_orders", "validate_products"])(make_task("fact_sales"))
shop.task("dim_customer", upstream=["validate_customers"])(make_task("dim_customer"))
shop.task("revenue_dashboard", upstream=["fact_sales", "dim_customer"])(make_task("revenue_dashboard"))

levels = shop.levels()
for i, lv in enumerate(levels):
    print(f"level {i}: {lv}")
assert levels[0] == ["extract_customers", "extract_orders", "extract_products"]
assert levels[-1] == ["revenue_dashboard"]

log = []
state = run_dag(shop, "2025-03-14", log=log)
assert all(s == "success" for s in state.values())
retried = [(n, tries) for _, n, r, tries in log if tries > 1]
print("retried and recovered:", retried)
assert retried == [("extract_orders", 2)]

# Cycles are rejected
bad = DAG("bad")
bad.task("a", upstream=["b"])(lambda ds: None)
bad.task("b", upstream=["a"])(lambda ds: None)
try:
    bad.levels()
    raise AssertionError("cycle not detected")
except CycleError:
    pass

# A permanent failure propagates downstream, and independent branches still run
def always_fails(ds):
    raise RuntimeError("schema changed upstream")


shop.tasks["extract_products"].fn = always_fails
state = run_dag(shop, "2025-03-15")
print({k: v for k, v in state.items() if v != "success"})
assert state["extract_products"] == "failed" and state["validate_products"] == "upstream_failed"
assert state["fact_sales"] == "upstream_failed" and state["revenue_dashboard"] == "upstream_failed"
assert state["dim_customer"] == "success", "the customer branch doesn't depend on products"
shop.tasks["extract_products"].fn = make_task("extract_products")

# %% [markdown]
# ## 3. Backfill with throttling, and idempotency

# %%
active, peak = 0, 0
gauge = threading.Lock()


def throttled_backfill(dag, start, end, max_parallel_runs=3):
    global active, peak
    dates = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]

    def one(ds):
        global active, peak
        with gauge:
            active += 1
            peak = max(peak, active)
        try:
            return ds, run_dag(dag, ds, max_workers=2)
        finally:
            with gauge:
                active -= 1

    with ThreadPoolExecutor(max_parallel_runs) as pool:
        return dict(pool.map(one, dates))


results = throttled_backfill(shop, date(2025, 2, 1), date(2025, 2, 28))
assert all(all(s == "success" for s in st.values()) for st in results.values())
n_partitions = len([k for k in warehouse if k[0] == "revenue_dashboard"])
throttled_backfill(shop, date(2025, 2, 10), date(2025, 2, 20))          # overlapping re-run
assert len([k for k in warehouse if k[0] == "revenue_dashboard"]) == n_partitions, "re-runs overwrite, never duplicate"
print(f"backfilled 28 days with at most {peak} concurrent DAG runs; re-running 11 of them created no duplicates")
assert peak <= 3

# %% [markdown]
# ## 4. Assets: materialize only what's stale

# %%
versions = {"raw_orders": 1, "raw_products": 1, "fact_sales": 0, "dashboard": 0}
asset_deps = {"fact_sales": ["raw_orders", "raw_products"], "dashboard": ["fact_sales"]}
built_from = {}                                         # asset -> versions of its inputs when last built


def stale(asset):
    return built_from.get(asset) != {d: versions[d] for d in asset_deps[asset]}


def materialize_stale():
    rebuilt = []
    for asset in ("fact_sales", "dashboard"):           # topological order
        if stale(asset):
            versions[asset] += 1
            built_from[asset] = {d: versions[d] for d in asset_deps[asset]}
            rebuilt.append(asset)
    return rebuilt


assert materialize_stale() == ["fact_sales", "dashboard"]
assert materialize_stale() == [], "nothing changed, nothing to do"
versions["raw_products"] += 1                           # new products landed
assert materialize_stale() == ["fact_sales", "dashboard"], "changes propagate to everything downstream"
print("asset view: only stale assets are rebuilt, and staleness propagates")

print("\nAll checks passed.")
