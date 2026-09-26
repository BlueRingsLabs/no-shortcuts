# Module 13: Pipelines and Orchestration

*Part II: Data Engineering · about 30 hours*

A pipeline that only works when you run it by hand is a hobby, not a pipeline.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What does it mean for a pipeline task to be idempotent, and why do you care at 3 AM?

## When you finish it, you can

- Design ETL/ELT pipelines that are idempotent, incremental and backfillable.
- Orchestrate pipelines with Airflow or Dagster, including retries, sensors, SLAs and alerting.
- Build tested transformation layers with dbt.

## Lessons

13.1. [ETL, ELT, idempotency and backfills](13.1-etl-elt-idempotency.md)

13.2. [Orchestration with Airflow and Dagster](13.2-orchestration.md)

13.3. [Transformations and tests with dbt](13.3-dbt.md)

## Labs

Run them from the repository root, for example:

```bash
python part-2-data-engineering/13-pipelines-orchestration/labs/lab_13_1_incremental_loads.py
```

- [`lab_13_1_incremental_loads.py`](labs/lab_13_1_incremental_loads.py)
- [`lab_13_2_orchestrator.py`](labs/lab_13_2_orchestrator.py)
- [`lab_13_3_dbt.py`](labs/lab_13_3_dbt.py)

Back to [Part II](../) · [Syllabus](../../SYLLABUS.md)
