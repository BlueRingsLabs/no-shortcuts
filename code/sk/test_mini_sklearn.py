"""Check the scikit-learn miniatures: scikit-learn's own estimator checks, then the real estimators side by side."""
import warnings

import numpy as np
from sklearn import __version__
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from mini_sklearn import MiniGridSearch, MiniLinearRegression, MiniPipeline, MiniStandardScaler

warnings.simplefilter("ignore")
checks = 0

# 1. scikit-learn's own test suite: nothing may fail (check_array_api_input is skipped without SCIPY_ARRAY_API)
for est, expected in [(MiniLinearRegression(), 51), (MiniStandardScaler(), 46)]:
    results = check_estimator(est, on_fail=None)
    failed = [r["check_name"] for r in results if r["status"] == "failed"]
    passed = sum(r["status"] == "passed" for r in results)
    assert not failed and passed == expected, (type(est).__name__, passed, failed)
    checks += passed

# 2. same answers as the real estimators on random data
rng = np.random.default_rng(0)
for _ in range(50):
    n, d = rng.integers(20, 200), rng.integers(1, 8)
    X = rng.normal(size=(n, d)) * rng.uniform(0.1, 100, size=d) + rng.normal(size=d)
    y = X @ rng.normal(size=d) + rng.normal()
    for fit_intercept in (True, False):
        m, s = MiniLinearRegression(fit_intercept=fit_intercept).fit(X, y), LinearRegression(fit_intercept=fit_intercept).fit(X, y)
        assert np.allclose(m.coef_, s.coef_) and np.allclose(m.intercept_, s.intercept_)
        assert np.allclose(m.predict(X), s.predict(X))
    assert np.allclose(MiniStandardScaler().fit_transform(X), StandardScaler().fit_transform(X))
    checks += 1

# 3. pipelines: same predictions and the same cross-validation scores
X, y = load_diabetes(return_X_y=True)
mini = MiniPipeline([("scale", MiniStandardScaler()), ("ridge", Ridge())])
real = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
assert np.allclose(mini.fit(X, y).predict(X), real.fit(X, y).predict(X))
assert np.allclose(cross_val_score(mini, X, y, cv=5), cross_val_score(real, X, y, cv=5))
checks += 1

# 4. grid search: same winner and the same mean score for every combination, on both pipelines
grid = {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0], "scale__with_std": [True, False]}
reference = GridSearchCV(real, grid, cv=5).fit(X, y)
for pipe in (real, mini):
    search = MiniGridSearch(pipe, grid).fit(X, y)
    assert search.best_params_ == reference.best_params_
    assert np.allclose([score for _, score in search.results_], reference.cv_results_["mean_test_score"])
    assert np.allclose(search.best_estimator_.predict(X), reference.best_estimator_.predict(X))
    checks += 1

print(f"{checks} checks against scikit-learn {__version__}: all passed")
