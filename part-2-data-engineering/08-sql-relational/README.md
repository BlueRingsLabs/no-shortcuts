# Module 08: SQL and Relational Databases

*Part II: Data Engineering · about 45 hours*

SQL is fifty years old and it will outlive most of the frameworks you are excited about today.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What does `COUNT(col)` return that `COUNT(*)` doesn't?
- Write a query that keeps only the latest row per user.

## When you finish it, you can

- Write correct SQL for filtering, grouping, joins, subqueries and CTEs, including NULL semantics.
- Use window functions for rankings, running totals, sessionization and deduplication.
- Read query plans, design indexes and fix slow queries.
- Explain transactions, isolation levels, and OLTP vs OLAP workloads.

## Lessons

08.1. [The relational model and SQL fundamentals](08.1-relational-model-sql.md)

08.2. [Joins, aggregation and window functions](08.2-joins-windows.md)

08.3. [Query plans, indexes and performance](08.3-query-plans-indexes.md)

08.4. [Transactions, isolation and OLTP vs OLAP](08.4-transactions-oltp-olap.md)

## Labs

Run them from the repository root, for example:

```bash
python part-2-data-engineering/08-sql-relational/labs/lab_08_1_sql_fundamentals.py
```

- [`lab_08_1_sql_fundamentals.py`](labs/lab_08_1_sql_fundamentals.py)
- [`lab_08_2_joins_windows.py`](labs/lab_08_2_joins_windows.py)
- [`lab_08_3_query_plans.py`](labs/lab_08_3_query_plans.py)
- [`lab_08_4_transactions.py`](labs/lab_08_4_transactions.py)

Back to [Part II](../) · [Syllabus](../../SYLLABUS.md)
