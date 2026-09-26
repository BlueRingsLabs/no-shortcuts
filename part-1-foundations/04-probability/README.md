# Module 04: Probability

*Part I: Foundations · about 40 hours*

Machine learning is applied probability wearing a hoodie. You need to speak it fluently.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- A test is 99% accurate and the disease affects 1 in 10,000 people. You test positive. Should you panic?
- Why is minimizing mean squared error the same as maximum likelihood under Gaussian noise?

## When you finish it, you can

- Work with discrete and continuous random variables, PMFs, PDFs and CDFs.
- Use the core distributions (Bernoulli, binomial, Poisson, categorical, Gaussian, exponential, beta, multivariate Gaussian) and know where each shows up.
- Apply conditional probability, independence and Bayes' rule, including base rate reasoning.
- Compute expectations, variances and covariances, and explain the law of large numbers and the CLT.
- Derive maximum likelihood and MAP estimators and connect them to loss functions and regularization.

## Lessons

04.1. [Random variables and distributions](04.1-random-variables.md)

04.2. [Joint, conditional and Bayes](04.2-conditional-bayes.md)

04.3. [Expectation, variance and the limit theorems](04.3-expectation-limit-theorems.md)

04.4. [Maximum likelihood and MAP](04.4-mle-map.md)

## Labs

Run them from the repository root, for example:

```bash
python part-1-foundations/04-probability/labs/lab_04_1_distributions.py
```

- [`lab_04_1_distributions.py`](labs/lab_04_1_distributions.py)
- [`lab_04_2_bayes.py`](labs/lab_04_2_bayes.py)
- [`lab_04_3_limit_theorems.py`](labs/lab_04_3_limit_theorems.py)
- [`lab_04_4_mle_map.py`](labs/lab_04_4_mle_map.py)

Back to [Part I](../) · [Syllabus](../../SYLLABUS.md)
