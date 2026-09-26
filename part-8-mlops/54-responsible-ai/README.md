# Module 54: Responsible AI

*Part VIII: MLOps and ML Systems · about 25 hours*

Not a compliance checkbox. The part where your model meets actual human beings.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why can't a classifier generally satisfy calibration and equal error rates across groups at the same time?

## When you finish it, you can

- Measure and mitigate unfairness with group metrics, and explain the impossibility results.
- Apply differential privacy (DP-SGD) and federated learning, and know their costs.
- Document models and data (model cards, datasheets) and map systems to regulation (EU AI Act, NIST AI RMF).

## Lessons

54.1. [Fairness](54.1-fairness.md)

54.2. [Privacy: differential privacy and federated learning](54.2-privacy-dp-federated.md)

54.3. [Governance, regulation and documentation](54.3-governance-regulation.md)

## Labs

Run them from the repository root, for example:

```bash
python part-8-mlops/54-responsible-ai/labs/lab_54_1_fairness.py
```

- [`lab_54_1_fairness.py`](labs/lab_54_1_fairness.py)
- [`lab_54_2_privacy.py`](labs/lab_54_2_privacy.py)
- [`lab_54_3_governance.py`](labs/lab_54_3_governance.py)

Back to [Part VIII](../) · [Syllabus](../../SYLLABUS.md)
