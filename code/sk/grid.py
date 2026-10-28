"""A grid search in miniature, checked against GridSearchCV."""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mini_sklearn import MiniGridSearch, MiniPipeline, MiniStandardScaler

X, y = load_diabetes(return_X_y=True)
grid = {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0], "scale__with_std": [True, False]}

pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
print(sorted(k for k in pipe.get_params() if "__" in k)[:6])

mini = MiniGridSearch(pipe, grid).fit(X, y)
real = GridSearchCV(pipe, grid, cv=5).fit(X, y)
print(mini.best_params_, round(mini.best_score_, 6))
print(real.best_params_, round(real.best_score_, 6))
ours = np.array([s for _, s in mini.results_])
print(np.allclose(ours, real.cv_results_["mean_test_score"]))
print(np.allclose(mini.best_estimator_.predict(X), real.best_estimator_.predict(X)))
print(hasattr(pipe.named_steps["ridge"], "coef_"))

mpipe = MiniPipeline([("scale", MiniStandardScaler()), ("ridge", Ridge())])
print(sorted(mpipe.get_params()))
print([k for k in mpipe.get_params() if "__" in k][:4])
mini2 = MiniGridSearch(mpipe, grid).fit(X, y)
print(mini2.best_params_, round(mini2.best_score_, 6))
print(np.allclose([s for _, s in mini2.results_], real.cv_results_["mean_test_score"]))
