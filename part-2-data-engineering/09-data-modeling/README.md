# Module 09: Data Modeling

*Part II: Data Engineering · about 20 hours*

The schema is the API of your data. Design it badly and every downstream team pays interest forever.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What is the grain of a fact table and why do you declare it first?

## When you finish it, you can

- Normalize a schema to 3NF and explain when to denormalize.
- Design star schemas with facts, dimensions, grain and slowly changing dimensions.
- Choose between dimensional models, wide tables and data vault for a given use case.

## Lessons

09.1. [Normalization and entity modeling](09.1-normalization.md)

09.2. [Dimensional modeling: facts, dimensions and SCDs](09.2-dimensional-modeling.md)

## Labs

Run them from the repository root, for example:

```bash
python part-2-data-engineering/09-data-modeling/labs/lab_09_1_normalization.py
```

- [`lab_09_1_normalization.py`](labs/lab_09_1_normalization.py)
- [`lab_09_2_dimensional.py`](labs/lab_09_2_dimensional.py)

Back to [Part II](../) · [Syllabus](../../SYLLABUS.md)
