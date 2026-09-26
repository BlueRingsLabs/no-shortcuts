# Module 18: Evaluation

*Part III: Classical Machine Learning · about 30 hours*

The module that protects you from your own optimism.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why can ROC-AUC look great on a dataset with 0.1% positives while the model is useless?

## When you finish it, you can

- Design train/validation/test splits and cross-validation for i.i.d., grouped and temporal data.
- Choose and compute metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, MAE, RMSE, MAPE, R².
- Calibrate probabilities, choose thresholds from business costs, and read reliability diagrams.
- Detect and prevent leakage, and deal with class imbalance without fooling yourself.

## Lessons

18.1. [Splits and cross-validation that don't lie](18.1-splits-cross-validation.md)

18.2. [Classification and regression metrics](18.2-metrics.md)

18.3. [Calibration, thresholds and costs](18.3-calibration-thresholds.md)

18.4. [Leakage, imbalance and other ways to fool yourself](18.4-leakage-imbalance.md)

## Labs

Run them from the repository root, for example:

```bash
python part-3-classical-ml/18-evaluation/labs/lab_18_1_splits_cv.py
```

- [`lab_18_1_splits_cv.py`](labs/lab_18_1_splits_cv.py)
- [`lab_18_2_metrics.py`](labs/lab_18_2_metrics.py)
- [`lab_18_3_calibration.py`](labs/lab_18_3_calibration.py)
- [`lab_18_4_leakage_imbalance.py`](labs/lab_18_4_leakage_imbalance.py)

Back to [Part III](../) · [Syllabus](../../SYLLABUS.md)
