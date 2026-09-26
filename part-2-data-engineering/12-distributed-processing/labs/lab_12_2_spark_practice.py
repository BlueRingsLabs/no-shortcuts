# requires: pyspark
# %% [markdown]
# # Lab 12.2: The six classic Spark problems, reproduced and fixed
#
# 1. Broadcast vs sort-merge join: read the plans.
# 2. Skew: one hot key, measured by rows per shuffle partition; fixed by salting.
# 3. Python UDF vs pandas UDF vs built-in functions: timing.
# 4. Caching: counting recomputations.
# 5. Output files from partitionBy, with and without repartition.
#
# AQE is disabled in parts 1-2 so you can see the raw behaviour; in real jobs, leave it on.

# %%
import glob
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from spark_local import local_spark  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.functions import pandas_udf  # noqa: E402
from pyspark.sql.types import StringType  # noqa: E402

spark = local_spark(cores=4, name="lab12_2", shuffle_partitions=8,
                    extra={"spark.sql.adaptive.enabled": "false", "spark.sql.autoBroadcastJoinThreshold": "-1"})
rng = np.random.default_rng(12)


def physical_plan(df):
    return df._jdf.queryExecution().executedPlan().toString()

# %% [markdown]
# ## 1. Broadcast vs sort-merge

# %%
n_facts = 400_000
facts = spark.createDataFrame(pd.DataFrame({
    "product_id": rng.integers(1, 500, n_facts), "amount": rng.gamma(2, 20, n_facts)}))
dim = spark.createDataFrame(pd.DataFrame({"product_id": np.arange(1, 500), "category": rng.choice(list("ABCDEFGH"), 499)}))

smj = facts.join(dim, "product_id")
bhj = facts.join(F.broadcast(dim), "product_id")
p_smj, p_bhj = physical_plan(smj), physical_plan(bhj)
print("default (broadcast disabled):", "SortMergeJoin" in p_smj, "| exchanges:", p_smj.count("Exchange"))
print("with F.broadcast():          ", "BroadcastHashJoin" in p_bhj, "| exchanges:", p_bhj.count("Exchange"))
assert "SortMergeJoin" in p_smj and "BroadcastHashJoin" in p_bhj
assert p_bhj.count("Exchange hashpartitioning") == 0, "the big side is not shuffled at all"
assert smj.count() == bhj.count() == n_facts

# %% [markdown]
# ## 2. Skew and salting
#
# 40% of events belong to user 0 (think: all guest checkouts mapped to a default id).

# %%
n = 600_000
users = np.where(rng.uniform(size=n) < 0.4, 0, rng.integers(1, 50_000, n))
events = spark.createDataFrame(pd.DataFrame({"user_id": users, "value": rng.normal(size=n)}))
profiles = spark.createDataFrame(pd.DataFrame({"user_id": np.arange(0, 50_000), "segment": rng.choice(["a", "b", "c"], 50_000)}))


def rows_per_partition(df):
    return df.rdd.glom().map(len).collect()


skewed = events.repartition(8, "user_id")
sizes = rows_per_partition(skewed)
print(f"rows per shuffle partition (hash by user_id): {sizes}")
assert max(sizes) > 3 * np.median(sizes), "one partition carries the hot key"

K = 8
salted_events = events.withColumn("salt", F.when(F.col("user_id") == 0, (F.rand(seed=1) * K).cast("int")).otherwise(F.lit(0)))
salts = spark.range(K).withColumnRenamed("id", "salt").withColumn("salt", F.col("salt").cast("int"))
salted_profiles = (profiles.filter("user_id = 0").crossJoin(salts)
                   .unionByName(profiles.filter("user_id != 0").withColumn("salt", F.lit(0))))
salted = salted_events.repartition(8, "user_id", "salt")
sizes_salted = rows_per_partition(salted)
print(f"after salting the hot key over {K} buckets:        {sizes_salted}")
assert max(sizes_salted) < max(sizes) / 2

plain_join = events.join(profiles, "user_id").groupBy("segment").agg(F.sum("value").alias("v")).toPandas().set_index("segment")
salted_join = (salted_events.join(salted_profiles, ["user_id", "salt"]).groupBy("segment")
               .agg(F.sum("value").alias("v")).toPandas().set_index("segment"))
assert np.allclose(plain_join.sort_index()["v"], salted_join.sort_index()["v"]), "salting doesn't change the answer"

# two-stage (salted) aggregation of events per user
two_stage = (events.withColumn("salt", (F.rand(seed=2) * K).cast("int"))
             .groupBy("user_id", "salt").agg(F.count("*").alias("c"))
             .groupBy("user_id").agg(F.sum("c").alias("events")))
direct = events.groupBy("user_id").agg(F.count("*").alias("events"))
assert two_stage.orderBy("user_id").toPandas().equals(direct.orderBy("user_id").toPandas())
print("salted join and two-stage aggregation give identical results")

# %% [markdown]
# ## 3. UDFs

# %%
emails = spark.createDataFrame(pd.DataFrame({
    "email": [f"user{i}@{d}" for i, d in enumerate(rng.choice(["x.com", "y.org", "z.net"], 300_000))],
    "amount": rng.gamma(2, 300, 300_000)})).cache()
emails.count()


@F.udf(StringType())
def bucket_py(a):
    return "high" if a > 1000 else ("low" if a < 10 else "mid")


@pandas_udf(StringType())
def bucket_pd(a: pd.Series) -> pd.Series:
    return pd.Series(np.select([a > 1000, a < 10], ["high", "low"], default="mid"))


builtin = F.when(F.col("amount") > 1000, "high").when(F.col("amount") < 10, "low").otherwise("mid")
timings = {}
for name, expr in [("python udf", bucket_py("amount")), ("pandas udf", bucket_pd("amount")), ("built-in", builtin)]:
    t0 = time.perf_counter()
    counts = emails.withColumn("b", expr).groupBy("b").count().orderBy("b").toPandas()
    timings[name] = time.perf_counter() - t0
    print(f"{name:11s} {timings[name]:.2f}s  {dict(zip(counts['b'], counts['count']))}")
assert timings["built-in"] < timings["python udf"]
plan_udf = physical_plan(emails.withColumn("b", bucket_py("amount")))
# Spark 4 ships plain Python UDFs to Python in Arrow batches by default (ArrowEvalPython; older versions: BatchEvalPython),
# but still calls your function once per row, in a separate Python process, invisible to the optimizer.
assert "EvalPython" in plan_udf, "Python UDFs show up as a *EvalPython node in the plan"
assert "EvalPython" not in physical_plan(emails.withColumn("b", builtin))

# %% [markdown]
# ## 4. Caching and recomputation

# %%
calls = spark.sparkContext.accumulator(0)


def counted(x):
    calls.add(1)
    return x


base = spark.sparkContext.parallelize(range(10_000), 4).map(counted)
base.count(); base.sum()
uncached_calls = calls.value
calls2 = spark.sparkContext.accumulator(0)
base2 = spark.sparkContext.parallelize(range(10_000), 4).map(lambda x: (calls2.add(1), x)[1]).cache()
base2.count(); base2.sum()
print(f"two actions: uncached map ran {uncached_calls:,} times, cached map ran {calls2.value:,} times")
assert uncached_calls == 20_000 and calls2.value == 10_000

# %% [markdown]
# ## 5. Output files and partitionBy

# %%
out = Path(tempfile.mkdtemp())
days = pd.date_range("2025-01-01", periods=30).date
ev = spark.createDataFrame(pd.DataFrame({"event_date": rng.choice(days, 60_000).astype(str), "v": rng.normal(size=60_000)})).repartition(8)
ev.write.partitionBy("event_date").parquet(str(out / "naive"))
ev.repartition("event_date").write.partitionBy("event_date").parquet(str(out / "repartitioned"))
n_naive = len(glob.glob(str(out / "naive" / "*" / "*.parquet")))
n_good = len(glob.glob(str(out / "repartitioned" / "*" / "*.parquet")))
print(f"partitionBy on randomly distributed data: {n_naive} files; after repartition(event_date): {n_good} files")
assert n_naive >= 8 * 25 and n_good == 30

shutil.rmtree(out)
spark.stop()
print("\nAll checks passed.")
