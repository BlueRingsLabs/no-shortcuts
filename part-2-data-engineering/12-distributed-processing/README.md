# Module 12: Distributed Processing

*Part II: Data Engineering · about 25 hours*

When one machine is not enough, and how to tell whether it really isn't.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What is a shuffle and why is it expensive?

## When you finish it, you can

- Explain the MapReduce model, partitioning, shuffles and the Spark execution model (DAGs, stages, tasks).
- Write PySpark DataFrame jobs and diagnose skew, spills and bad joins.
- Decide when distributed processing is justified and when DuckDB on one box wins.

## Lessons

12.1. [From MapReduce to Spark: the distributed model](12.1-mapreduce-spark-model.md)

12.2. [Spark in practice: partitions, shuffles, skew](12.2-spark-in-practice.md)

## Labs

Run them from the repository root, for example:

```bash
python part-2-data-engineering/12-distributed-processing/labs/lab_12_1_distributed_model.py
```

- [`lab_12_1_distributed_model.py`](labs/lab_12_1_distributed_model.py)
- [`lab_12_2_spark_practice.py`](labs/lab_12_2_spark_practice.py)

Back to [Part II](../) · [Syllabus](../../SYLLABUS.md)
