# Module 20: Trees and Ensembles

*Part III: Classical Machine Learning · about 35 hours*

If your data lives in a table, this is probably the module that pays your salary.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why do random forests not overfit as you add trees, but boosting can?

## When you finish it, you can

- Implement a decision tree with Gini/entropy splits and explain pruning.
- Explain why bagging reduces variance and how random forests decorrelate trees.
- Derive gradient boosting as gradient descent in function space and implement it.
- Tune XGBoost, LightGBM and CatBoost, and handle categorical features, missing values and early stopping.

## Lessons

20.1. [Decision trees](20.1-decision-trees.md)

20.2. [Bagging and random forests](20.2-bagging-random-forests.md)

20.3. [Boosting: from AdaBoost to gradient boosting](20.3-boosting.md)

20.4. [XGBoost, LightGBM and CatBoost in the real world](20.4-gbdt-in-practice.md)

## Labs

Run them from the repository root, for example:

```bash
python part-3-classical-ml/20-trees-ensembles/labs/lab_20_1_decision_trees.py
```

- [`lab_20_1_decision_trees.py`](labs/lab_20_1_decision_trees.py)
- [`lab_20_2_random_forests.py`](labs/lab_20_2_random_forests.py)
- [`lab_20_3_boosting.py`](labs/lab_20_3_boosting.py)
- [`lab_20_4_gbdt_practice.py`](labs/lab_20_4_gbdt_practice.py)

Back to [Part III](../) · [Syllabus](../../SYLLABUS.md)
