# requires: pyspark
# %% [markdown]
# # Lab 12.1: MapReduce by hand, then Spark
#
# 1. MapReduce word count in plain Python with real processes: map, combine, hash-partition, shuffle, reduce.
# 2. Measuring what the combiner saves.
# 3. PySpark (local mode): laziness, narrow vs wide transformations, stages and Exchange operators in the plan.
# 4. Amdahl's law measured: speedup vs number of cores.

# %%
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("PYSPARK_PYTHON", "python3")

# %% [markdown]
# ## 1-2. MapReduce word count

# %%
rng = np.random.default_rng(0)
vocab = [f"w{i}" for i in range(2000)]
zipf = 1 / np.arange(1, 2001) ** 1.1
zipf /= zipf.sum()
documents = [" ".join(rng.choice(vocab, 200, p=zipf)) for _ in range(400)]
N_MAPPERS, N_REDUCERS = 4, 3


def mapper(docs, use_combiner):
    pairs = [(w, 1) for d in docs for w in re.findall(r"\w+", d)]
    if use_combiner:                                   # pre-aggregate locally before the shuffle
        pairs = list(Counter(w for w, _ in pairs).items())
    parts = defaultdict(list)                          # hash partitioning: which reducer gets each key
    for w, c in pairs:
        parts[hash(w) % N_REDUCERS].append((w, c))
    return dict(parts), len(pairs)


def reducer(pairs):
    out = defaultdict(int)
    for w, c in pairs:
        out[w] += c
    return dict(out)


def mapreduce(use_combiner):
    chunks = [documents[i::N_MAPPERS] for i in range(N_MAPPERS)]
    with ProcessPoolExecutor(N_MAPPERS) as pool:
        mapped = list(pool.map(mapper, chunks, [use_combiner] * N_MAPPERS))
    shuffled = defaultdict(list)                       # the shuffle: gather each reducer's partition from every mapper
    sent = 0
    for parts, n in mapped:
        sent += n
        for r, pairs in parts.items():
            shuffled[r].extend(pairs)
    with ProcessPoolExecutor(N_REDUCERS) as pool:
        reduced = list(pool.map(reducer, [shuffled[r] for r in range(N_REDUCERS)]))
    result = {}
    for part in reduced:
        assert not (set(part) & set(result)), "each key must land on exactly one reducer"
        result.update(part)
    return result, sent


if __name__ == "__main__":
    os.environ["PYTHONHASHSEED"] = "0"
    truth = Counter(w for d in documents for w in d.split())
    res_plain, sent_plain = mapreduce(False)
    res_comb, sent_comb = mapreduce(True)
    assert res_plain == truth and res_comb == truth
    print(f"pairs shuffled: without combiner {sent_plain:,}, with combiner {sent_comb:,} ({sent_plain / sent_comb:.1f}x less traffic)")
    assert sent_comb < sent_plain / 5

# %% [markdown]
# ## 3. Spark: laziness, narrow vs wide, stages

# %%
if __name__ == "__main__":
    import sys
    from pathlib import Path

    from pyspark.sql import functions as F

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
    from shop import make_shop
    from spark_local import local_spark

    spark = local_spark(cores=4, name="lab12_1")

    t = make_shop(n_customers=20_000, seed=11)
    items = spark.createDataFrame(t["order_items"]).repartition(6)
    orders = spark.createDataFrame(t["orders"].assign(order_ts=t["orders"]["order_ts"].astype(str)))

    t0 = time.perf_counter()
    lazy = items.withColumn("amount", F.col("quantity") * F.col("unit_price")).filter(F.col("quantity") > 1)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    n = lazy.count()
    t_action = time.perf_counter() - t0
    print(f"building the plan: {t_build * 1000:.1f} ms (nothing ran); count(): {t_action * 1000:.0f} ms -> {n:,} rows")
    assert t_build < t_action

    narrow = items.withColumn("amount", F.col("quantity") * F.col("unit_price")).filter("quantity > 1")
    wide = narrow.groupBy("order_id").agg(F.sum("amount").alias("total"))
    joined = wide.join(orders.select("order_id", "customer_id"), "order_id")

    def plan_of(df):
        return df._jdf.queryExecution().executedPlan().toString()

    exchanges = {name: plan_of(df).count("Exchange") for name, df in
                 [("narrow chain", narrow), ("groupBy", wide), ("groupBy + join", joined)]}
    print("Exchange (shuffle) operators in the physical plan:", exchanges)
    assert exchanges["narrow chain"] <= 1          # only the explicit repartition(6) we did on purpose
    assert exchanges["groupBy"] > exchanges["narrow chain"]
    assert exchanges["groupBy + join"] >= exchanges["groupBy"]

    pd_check = (t["order_items"].assign(amount=lambda d: d["quantity"] * d["unit_price"]).query("quantity > 1")
                .groupby("order_id")["amount"].sum())
    spark_res = wide.toPandas().set_index("order_id")["total"].sort_index()
    assert np.allclose(spark_res.to_numpy(), pd_check.sort_index().to_numpy())
    print(f"Spark result matches pandas on {len(spark_res):,} orders; result has {wide.rdd.getNumPartitions()} partitions "
          f"(= spark.sql.shuffle.partitions after the shuffle, possibly coalesced by AQE)")

# %% [markdown]
# ## 4. Amdahl's law, measured
#
# A CPU-heavy job over 32 partitions with local[1], local[2], local[4]. The fixed driver/scheduling overhead is the
# serial part; speedup flattens.

# %%
if __name__ == "__main__":
    spark.stop()
    cores = os.cpu_count() or 1
    timings = {}
    for k in sorted({1, 2, min(4, cores)}):
        s = local_spark(cores=k, name=f"amdahl{k}")
        rdd = s.sparkContext.parallelize(range(32), 32)
        rdd.map(lambda i: i).count()                          # warm up the Python workers
        t0 = time.perf_counter()
        rdd.map(lambda i: sum((j * i) % 7 for j in range(300_000))).sum()
        timings[k] = time.perf_counter() - t0
        s.stop()
    base = timings[1]
    for k, v in timings.items():
        print(f"local[{k}]: {v:.2f}s  speedup {base / v:.2f}x")
    if cores >= 2:
        assert base / timings[max(timings)] > 1.3, "more cores should help..."
        assert base / timings[max(timings)] < max(timings), "...but never linearly (Amdahl + overhead)"
    print("\nAll checks passed.")
