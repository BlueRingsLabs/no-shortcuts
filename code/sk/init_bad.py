from sklearn.base import BaseEstimator, RegressorMixin
from mini_sklearn import MiniLinearRegression


class RidgeBadInit(MiniLinearRegression):
    """Breaks the rule: it 'fixes' a hyperparameter inside __init__."""

    def __init__(self, fit_intercept=True, alpha=1.0):
        self.fit_intercept = fit_intercept
        self.alpha = max(float(alpha), 1e-3)        # looks harmless
