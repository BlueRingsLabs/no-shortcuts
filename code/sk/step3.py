import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class LinearRegressionV3(RegressorMixin, BaseEstimator):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        if self.fit_intercept:
            self.intercept_, self.coef_ = w[0], w[1:]
        else:
            self.intercept_, self.coef_ = 0.0, w
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
