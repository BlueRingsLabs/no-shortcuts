# Module 01: Python for Engineers

*Part I: Foundations · about 40 hours*

Not "Python for beginners". Python for people who will write code that other people run at 3 AM.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What does `yield` do, and when is a generator better than a list?
- Why is `a[:, None] - b[None, :]` fast, and how much memory does it use?
- When does the GIL actually hurt you?

## When you finish it, you can

- Use Python's data model (dunder methods, iterators, generators, context managers) to write idiomatic code.
- Write typed, tested, packaged code with pytest and a pyproject.toml.
- Think in arrays: replace Python loops with NumPy vectorized operations and understand broadcasting and memory layout.
- Profile code, find the real bottleneck, and choose between threads, processes and asyncio.

## Lessons

01.1. [Python's data model and the idioms that matter](01.1-data-model-and-idioms.md)

01.2. [Types, tests and packages: code other people can trust](01.2-types-tests-packages.md)

01.3. [NumPy and vectorization: thinking in arrays](01.3-numpy-vectorization.md)

01.4. [Performance, profiling and concurrency](01.4-performance-concurrency.md)

## Labs

Run them from the repository root, for example:

```bash
python part-1-foundations/01-python-for-engineers/labs/lab_01_1_data_model.py
```

- [`lab_01_1_data_model.py`](labs/lab_01_1_data_model.py)
- [`lab_01_2_tested_package.py`](labs/lab_01_2_tested_package.py)
- [`lab_01_3_vectorization.py`](labs/lab_01_3_vectorization.py)
- [`lab_01_4_profiling_concurrency.py`](labs/lab_01_4_profiling_concurrency.py)

Back to [Part I](../) · [Syllabus](../../SYLLABUS.md)
