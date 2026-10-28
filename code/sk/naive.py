import numpy as np


class NaiveLinearRegression:
    """The way most people would write it the first time."""

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        if self.fit_intercept:
            self.intercept, self.coef = w[0], w[1:]
        else:
            self.intercept, self.coef = 0.0, w
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef + self.intercept
