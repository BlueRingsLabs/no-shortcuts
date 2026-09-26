# Module 11: Storage, Formats and Architecture

*Part II: Data Engineering · about 20 hours*

Where the bytes live, what shape they have, and why that decides your cloud bill.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why is Parquet so much faster than CSV for `SELECT avg(price) FROM sales`?

## When you finish it, you can

- Choose between row and columnar formats (CSV, JSON, Avro, Parquet, Arrow) and explain encodings, compression and predicate pushdown.
- Design object storage layouts with partitioning, and avoid the small files problem.
- Explain data lakes, warehouses and lakehouses, table formats like Iceberg and Delta, and the medallion architecture.

## Lessons

11.1. [File formats: CSV, JSON, Parquet, Avro, Arrow](11.1-file-formats.md)

11.2. [Lakes, warehouses and lakehouses](11.2-lakes-warehouses-lakehouses.md)

## Labs

Run them from the repository root, for example:

```bash
python part-2-data-engineering/11-storage-formats/labs/lab_11_1_file_formats.py
```

- [`lab_11_1_file_formats.py`](labs/lab_11_1_file_formats.py)
- [`lab_11_2_lakehouse.py`](labs/lab_11_2_lakehouse.py)

Back to [Part II](../) · [Syllabus](../../SYLLABUS.md)
