# %% [markdown]
# # Lab 22.2: Hyperparameter optimization
#
# 1. Grid vs random search when only 2 of 6 hyperparameters matter (a simulated objective, many repeats).
# 2. Random search vs Optuna TPE on a real model, same budget, scored on a held-out test set.
# 3. Pruning: a median pruner kills bad trials early. How many, and how much time it saves.
# 4. The winner's curse: searching over hyperparameters that do nothing.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")                   # HistGradientBoosting oversubscribes small machines otherwise
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
rng = np.random.default_rng(222)

# %% [markdown]
# ## 1. Grid vs random, when few parameters matter
#
# Objective on [0, 1]^6: a sharp peak in dimensions 0 and 1 at an unknown location; dimensions 2-5 barely matter.
# Budget: 64 evaluations. The grid is 2^6 (2 values per parameter); random draws 64 points.

# %%
def objective(P, peak):
    important = np.exp(-((P[:, 0] - peak[0]) ** 2 + (P[:, 1] - peak[1]) ** 2) / 0.01)
    return important + 0.02 * P[:, 2:].sum(1)


grid_vals = np.array([0.25, 0.75])
grid = np.array(np.meshgrid(*[grid_vals] * 6)).reshape(6, -1).T
res_grid, res_rand = [], []
for rep in range(500):
    peak = rng.uniform(0.1, 0.9, 2)
    res_grid.append(objective(grid, peak).max())
    res_rand.append(objective(rng.uniform(0, 1, (64, 6)), peak).max())
print(f"64 evaluations, best value found (mean over 500 problems): grid {np.mean(res_grid):.3f}, random {np.mean(res_rand):.3f}")
print(f"distinct values tried in the important dimension: grid {len(np.unique(grid[:, 0]))}, random 64")
assert np.mean(res_rand) > np.mean(res_grid) + 0.1

# %% [markdown]
# ## 2. Random search vs TPE on a real model
#
# Search space for HistGradientBoosting; 3-fold CV AUC as the objective; 30 trials each; final check on a held-out test set.

# %%
X, y = make_classification(n_samples=3000, n_features=30, n_informative=10, n_redundant=5, flip_y=0.05, class_sep=0.7, random_state=5)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
cv = StratifiedKFold(3, shuffle=True, random_state=0)


def make_model(trial):
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 4, 128, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 200, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
        max_iter=100, random_state=0)


def cv_objective(trial):
    return cross_val_score(make_model(trial), X_dev, y_dev, cv=cv, scoring="roc_auc").mean()


studies = {}
for name, sampler in [("random", optuna.samplers.RandomSampler(seed=0)), ("TPE", optuna.samplers.TPESampler(seed=0))]:
    t0 = time.perf_counter()
    st = optuna.create_study(direction="maximize", sampler=sampler)
    st.optimize(cv_objective, n_trials=30)
    best = make_model(optuna.trial.FixedTrial(st.best_params)).fit(X_dev, y_dev)
    test_auc = roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])
    curve = np.maximum.accumulate([t.value for t in st.trials])
    studies[name] = (st, test_auc)
    print(f"{name:6s}: best CV AUC {st.best_value:.4f} (after 10/20/30 trials: {curve[9]:.4f}/{curve[19]:.4f}/{curve[29]:.4f}), "
          f"held-out test AUC {test_auc:.4f}, {time.perf_counter() - t0:.0f}s")
default = HistGradientBoostingClassifier(random_state=0).fit(X_dev, y_dev)
print(f"untuned defaults: held-out test AUC {roc_auc_score(y_test, default.predict_proba(X_test)[:, 1]):.4f}")
print("read the test column, not the CV column: 30 trials of tuning bought a few thousandths of AUC over the defaults, or nothing.")
print("the CV 'improvement' was partly selection noise (section 4). That's normal for a decent default on clean features.")
print("best parameters (TPE):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in studies["TPE"][0].best_params.items()})
for name, (st, test_auc) in studies.items():
    assert st.best_value >= test_auc - 0.02, "the search's best CV score is not a promise"

# %% [markdown]
# ## 3. Pruning
#
# Each trial trains in steps of 20 boosting rounds and reports validation AUC; the median pruner stops trials that
# are below the median of earlier trials at the same step.

# %%
X_tr, X_val, y_tr, y_val = train_test_split(X_dev, y_dev, test_size=0.3, random_state=1, stratify=y_dev)


def stepped(trial, prune):
    m = HistGradientBoostingClassifier(learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                                       max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 4, 128, log=True),
                                       min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 200, log=True),
                                       warm_start=True, early_stopping=False, random_state=0)
    score = 0.0
    for step, n_iter in enumerate(range(20, 201, 20)):
        m.set_params(max_iter=n_iter).fit(X_tr, y_tr)
        score = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        trial.report(score, step)
        if prune and trial.should_prune():
            raise optuna.TrialPruned()
    return score


out = {}
for prune in (False, True):
    t0 = time.perf_counter()
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    st.optimize(lambda tr: stepped(tr, prune), n_trials=30)
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in st.trials)
    out[prune] = (time.perf_counter() - t0, st.best_value, n_pruned)
    print(f"pruning={prune!s:5s}: {n_pruned:2d} of 30 trials pruned, best validation AUC {st.best_value:.4f}, {out[prune][0]:.0f}s")
assert out[True][2] >= 8 and out[True][0] < 0.8 * out[False][0] and out[True][1] > out[False][1] - 0.01

# %% [markdown]
# ## 4. The winner's curse
#
# A search over "hyperparameters" that do nothing: the random seed of a small random forest. Every configuration has
# the same true performance; the best CV score still climbs with the number of trials.

# %%
Xw, yw = make_classification(n_samples=600, n_features=20, n_informative=5, flip_y=0.1, random_state=9)
Xw_dev, Xw_test, yw_dev, yw_test = train_test_split(Xw, yw, test_size=0.5, random_state=0, stratify=yw)
cvw = StratifiedKFold(3, shuffle=True, random_state=0)
scores, tests = [], []
for seed in range(200):
    m = RandomForestClassifier(n_estimators=15, max_features=0.3, random_state=seed)
    scores.append(cross_val_score(m, Xw_dev, yw_dev, cv=cvw).mean())
    tests.append(m.fit(Xw_dev, yw_dev).score(Xw_test, yw_test))
scores, tests = np.array(scores), np.array(tests)
for n in (1, 10, 50, 200):
    i = int(np.argmax(scores[:n]))
    print(f"best of {n:3d} seeds: CV accuracy {scores[i]:.3f}, the same model on held-out data {tests[i]:.3f} (average of all seeds {tests.mean():.3f})")
i200 = int(np.argmax(scores))
print(f"correlation between a seed's CV score and its held-out score: {np.corrcoef(scores, tests)[0, 1]:.2f}")
assert scores[i200] - tests[i200] > 0.02 and scores[i200] > scores[0]

print("\nAll checks passed.")
