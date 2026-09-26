# Module 10: Data Wrangling and Analysis

*Part II: Data Engineering · about 40 hours*

Eighty percent of the job, zero percent of the glamour.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What did copy-on-write change in pandas 3, and why does `df[mask]['col'] = 0` no longer modify anything?
- How would you query a 20 GB Parquet file on a laptop with 16 GB of RAM?

## When you finish it, you can

- Use pandas efficiently: indexing, groupby, merge, reshape, dtypes, and avoiding the classic traps.
- Use Polars and DuckDB for fast single-node analytics on data bigger than RAM.
- Profile and clean data, and encode data quality expectations as tests.
- Run exploratory analyses and build honest visualizations that communicate a result.

## Lessons

10.1. [pandas without the pain](10.1-pandas.md)

10.2. [Polars and DuckDB: the modern single-node stack](10.2-polars-duckdb.md)

10.3. [Data cleaning and data quality](10.3-cleaning-quality.md)

10.4. [Exploratory analysis and visualization that says something](10.4-eda-visualization.md)

## Labs

Run them from the repository root, for example:

```bash
python part-2-data-engineering/10-data-wrangling/labs/lab_10_1_pandas.py
```

- [`lab_10_1_pandas.py`](labs/lab_10_1_pandas.py)
- [`lab_10_2_polars_duckdb.py`](labs/lab_10_2_polars_duckdb.py)
- [`lab_10_3_data_quality.py`](labs/lab_10_3_data_quality.py)
- [`lab_10_4_eda.py`](labs/lab_10_4_eda.py)

Back to [Part II](../) · [Syllabus](../../SYLLABUS.md)
