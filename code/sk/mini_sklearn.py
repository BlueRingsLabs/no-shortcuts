"""Miniature scikit-learn estimators that follow the estimator contract and pass check_estimator."""
from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted, validate_data


class MiniLinearRegression(RegressorMixin, BaseEstimator):

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept          # stored exactly as given, nothing else

    def fit(self, X, y):
        X, y = validate_data(self, X, y, y_numeric=True)   # checks, and sets n_features_in_
        A = np.column_stack([np.ones(len(X)), X]) if self.fit_intercept else X
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        if self.fit_intercept:
            self.intercept_, self.coef_ = w[0], w[1:]
        else:
            self.intercept_, self.coef_ = 0.0, w
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return X @ self.coef_ + self.intercept_


class MiniStandardScaler(TransformerMixin, BaseEstimator):
    """Subtract the mean and divide by the standard deviation, both learned in fit."""

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = validate_data(self, X)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        scale = X.std(axis=0) if self.with_std else np.ones(X.shape[1])
        self.scale_ = np.where(scale == 0, 1.0, scale)   # a constant column stays as it is
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return (X - self.mean_) / self.scale_


class MiniPipeline(BaseEstimator):
    """Chain transformers and a final estimator. Every step is fitted on the data that reaches it."""

    def __init__(self, steps):
        self.steps = steps                           # a list of (name, estimator)

    def fit(self, X, y=None):
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt, y)           # fit, then pass the transformed data on
        self.steps[-1][1].fit(Xt, y)
        return self

    def _transform(self, X):
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)                  # transform only: nothing is learned here
        return Xt

    def predict(self, X):
        return self.steps[-1][1].predict(self._transform(X))

    def score(self, X, y):
        return self.steps[-1][1].score(self._transform(X), y)

    def get_params(self, deep=True):
        params = {"steps": self.steps}
        if deep:
            for name, step in self.steps:
                params[name] = step
                for key, value in step.get_params(deep=True).items():
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params):
        for key, value in params.items():
            name, _, rest = key.partition("__")
            if key == "steps":
                self.steps = value
            elif rest:
                dict(self.steps)[name].set_params(**{rest: value})
            else:
                self.steps = [(n, value if n == name else s) for n, s in self.steps]
        return self


def param_combinations(grid):
    names = sorted(grid)
    for values in product(*(grid[n] for n in names)):
        yield dict(zip(names, values))


class MiniGridSearch:
    def __init__(self, estimator, param_grid, cv=5):
        self.estimator, self.param_grid, self.cv = estimator, param_grid, cv

    def fit(self, X, y):
        self.results_ = []
        for params in param_combinations(self.param_grid):
            scores = []
            for train, test in KFold(self.cv).split(X):
                model = clone(self.estimator).set_params(**params)   # a fresh model per fold
                model.fit(X[train], y[train])
                scores.append(model.score(X[test], y[test]))
            self.results_.append((params, np.mean(scores)))
        self.best_params_, self.best_score_ = max(self.results_, key=lambda r: r[1])
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)
        return self
