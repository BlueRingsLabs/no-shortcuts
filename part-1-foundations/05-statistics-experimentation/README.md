# Module 05: Statistics and Experimentation

*Part I: Foundations · about 40 hours*

The part of data science that decides whether your "improvement" is real or just noise with a nice chart.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What does a 95% confidence interval actually claim?
- Why is checking an A/B test every day and stopping when p < 0.05 a bad idea?

## When you finish it, you can

- Build estimators and confidence intervals, analytically and with the bootstrap.
- Run and interpret hypothesis tests without committing the classic p-value crimes.
- Design, size and analyze A/B tests, including power analysis, peeking and multiple comparisons.
- Reason about causality: confounders, randomization, DAGs, and basic observational methods.

## Lessons

05.1. [Estimation, confidence intervals and the bootstrap](05.1-estimation-bootstrap.md)

05.2. [Hypothesis testing and how people abuse it](05.2-hypothesis-testing.md)

05.3. [A/B testing: power, peeking and other crimes](05.3-ab-testing.md)

05.4. [Causal inference for engineers](05.4-causal-inference.md)

## Labs

Run them from the repository root, for example:

```bash
python part-1-foundations/05-statistics-experimentation/labs/lab_05_1_bootstrap.py
```

- [`lab_05_1_bootstrap.py`](labs/lab_05_1_bootstrap.py)
- [`lab_05_2_testing.py`](labs/lab_05_2_testing.py)
- [`lab_05_3_ab_testing.py`](labs/lab_05_3_ab_testing.py)
- [`lab_05_4_causal.py`](labs/lab_05_4_causal.py)

Back to [Part I](../) · [Syllabus](../../SYLLABUS.md)
