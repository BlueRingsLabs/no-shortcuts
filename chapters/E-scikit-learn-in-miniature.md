# E. scikit-learn in miniature

scikit-learn is the clearest example in this addendum of a library that's mostly a **contract**. It has hundreds of models, and they're all different inside, but from the outside they all behave the same way: you create one with its settings, you call `fit` with data, and then you call `predict` or `transform`. That uniformity is what lets you swap a linear model for a random forest by changing one line, and it's what lets pipelines, cross-validation and grid search work with every model in the library, including ones you write yourself.

The authors described the design in a paper that I'd recommend to anyone who uses the library, because it explains in a few pages what the documentation spreads over hundreds ([Buitinck et al., "API design for machine learning software: experiences from the scikit-learn project", 2013](https://arxiv.org/abs/1309.0238)<sup>\[12\]</sup>). This chapter takes the contract from that design, writes estimators that follow it, and then does something you can only do with a library this disciplined: runs scikit-learn's *own* test suite on them.

## 1. Three kinds of objects, one interface

Everything in scikit-learn is an **estimator**: an object that learns something from data in `fit`. On top of that, there are two roles, and many objects have both:

- **Predictors** add `predict` (and often `predict_proba`, `decision_function` and `score`). `LinearRegression`, `RandomForestClassifier`, `KMeans`.
- **Transformers** add `transform`, which returns a modified version of the data. `StandardScaler`, `PCA`, `OneHotEncoder`, `SelectKBest`.

And a handful of rules hold for all of them:

1. **The constructor takes hyperparameters, and only hyperparameters.** No data, no computation, no validation. Each argument is stored on the object under its own name, unchanged.
2. **`fit` learns.** It takes the data (`X`, and `y` when there is one), and stores everything it learned in attributes whose names **end with an underscore**: `coef_`, `mean_`, `classes_`, `n_features_in_`.
3. **`fit` returns `self`**, so you can chain: `StandardScaler().fit(X).transform(X)`.
4. **`predict` and `transform` only use what was learned.** They never learn anything new from the data they receive.
5. **Data is a 2D array of shape `(n_samples, n_features)`.** Rows are samples, columns are features, always.

That's it. Rule 4 in particular is what lets scikit-learn protect you from data leakage, as [section E.8](#8-pipelines-and-why-fittransform-on-the-test-set-leaks) shows.

## 2. A first attempt at an estimator

Let's write a linear regression the way you'd probably write it the first time, knowing NumPy but not the contract:

```python
import numpy as np


class NaiveLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])   # a column of ones for the intercept
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        if self.fit_intercept:
            self.intercept, self.coef = w[0], w[1:]
        else:
            self.intercept, self.coef = 0.0, w
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef + self.intercept
```

It works. On simple data it gives the right coefficients. And it's wrong in more than a dozen ways, which we're going to let scikit-learn itself point out.

## 3. Let scikit-learn grade it

`sklearn.utils.estimator_checks.check_estimator` runs the battery of checks that scikit-learn's own estimators must pass. With `on_fail=None` it doesn't stop at the first failure; it returns a report of every check, which is perfect for learning:

```python
import warnings
from sklearn.utils.estimator_checks import check_estimator

warnings.simplefilter("ignore")
results = check_estimator(NaiveLinearRegression(), on_fail=None)
```

```text
AttributeError: The following error was raised: 'NaiveLinearRegression' object has no attribute '__sklearn_tags__'. It seems that there are no classes that implement `__sklearn_tags__` in the MRO and/or all classes in the MRO call `super().__sklearn_tags__()`. Make sure to inherit from `BaseEstimator` which implements `__sklearn_tags__` ...
```

It doesn't even start. The first requirement is to inherit from `BaseEstimator`, which provides the machinery every estimator needs (the tags that describe what it supports, `get_params`, `set_params`, its printed representation), and from the mixin for its role: `RegressorMixin` for a regressor, which adds `score`.

> [!NOTE]
> The order matters: mixins go on the left, `BaseEstimator` on the right, as in `class MyModel(RegressorMixin, BaseEstimator)`. Python looks up methods from left to right, and the mixins need to come first to adjust what `BaseEstimator` provides.

So let's change only the class line, to `class LinearRegressionV2(RegressorMixin, BaseEstimator)`, keep everything else as it was, and run the checks again. This time we get a full report. I've grouped it by what each failure is about (all 17 are real messages from scikit-learn 1.9.1, shortened):

```text
LinearRegressionV2 {'passed': 34, 'failed': 17, 'skipped': 1}
```

**Naming and state**

- `check_dont_overwrite_parameters`: "Estimator adds public attribute(s) during the fit method. Estimators are only allowed to add private attributes either started with _ or ended with _".
- `check_estimators_unfitted`: "Estimator should raise a NotFittedError when calling `predict` before fit."
- `check_fit_check_is_fitted`: "Estimator fails to pass `check_is_fitted` even though it has been fit."
- `check_n_features_in_after_fitting` and `check_n_features_in`: "`fit()` does not set the `n_features_in_` attribute."

**Input validation**

- `check_fit1d`: passing a 1D `X` should raise a `ValueError`, and ours accepted it.
- `check_fit2d_predict1d`: same, at prediction time.
- `check_estimators_nan_inf`: "doesn't check for NaN and inf in fit."
- `check_supervised_y_no_nan`: "should have raised error on fitting array y with inf value."
- `check_estimators_empty_data_messages`: "does not raise a ValueError when an empty data is used to train."
- `check_complex_data`: complex numbers should be rejected.
- `check_supervised_y_2d`: a column vector `y` of shape `(n, 1)` should produce a warning, not be silently accepted.
- `check_requires_y_none`: calling `fit(X, None)` raised an error, but not the standard one.
- `check_estimator_sparse_array`, `check_estimator_sparse_matrix`, `check_estimator_sparse_tag`: given sparse input, it should "fail gracefully" with a message saying sparse input isn't supported.

**Composition**

- `check_pipeline_consistency`: putting it inside a `Pipeline` broke, as a consequence of the problems above.

Read that list again: it's the contract, written as tests, by the library itself.

## 4. Fixing the naming: underscores and "is it fitted?"

The first group is about one rule: **learned things end with an underscore**. Our `fit` created `self.coef` and `self.intercept`, which look like hyperparameters, so scikit-learn can't tell them apart from settings. Renaming them to `coef_` and `intercept_` fixes it, and it also makes the second rule possible: `check_is_fitted(self)`, which looks for attributes ending with an underscore, can now tell a fitted estimator from an unfitted one.

```python
from sklearn.utils.validation import check_is_fitted


class LinearRegressionV3(RegressorMixin, BaseEstimator):
    ...
        if self.fit_intercept:
            self.intercept_, self.coef_ = w[0], w[1:]
        else:
            self.intercept_, self.coef_ = 0.0, w
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
```

```text
LinearRegressionV3 {'passed': 38, 'failed': 13, 'skipped': 1}
```

Four fewer failures (`n_features_in_` is still missing; it comes in the next section). The underscore convention does real work: it's how scikit-learn's tools know what was learned and what was set by you, and `clone` ([section E.7](#7-why-the-constructor-must-not-do-anything-getparams-and-clone)) depends on it.

## 5. Fixing the input: validate_data

The other 13 failures are all input validation, and here's the good news: you don't write those checks yourself. scikit-learn exposes the function its own estimators use, `validate_data`, which wraps the `check_array` we met in the traceback of [section B.5](B-a-method-for-learning-any-library.md#5-follow-the-traceback-into-the-library). It converts the input to a proper 2D float array, rejects NaN, infinity, complex numbers and empty arrays with standard messages, refuses sparse input unless you say you support it, and records `n_features_in_` (and `feature_names_in_`, when you pass a DataFrame). At prediction time, with `reset=False`, it checks that the new data has the same number of features as the training data.

The finished miniature:

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
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
```

```text
MiniLinearRegression {'passed': 51, 'skipped': 1}
```

Every check passes. (The one skipped is `check_array_api_input`, which only runs when the environment variable `SCIPY_ARRAY_API` is set, for estimators that support other array libraries. Ours doesn't claim to.)

It also gives the same answer as the real one:

```python
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5))
y = X @ np.array([1.5, -2.0, 0.0, 3.0, 0.5]) + 4.0 + rng.normal(scale=0.1, size=200)

m = MiniLinearRegression().fit(X, y)
s = LinearRegression().fit(X, y)
print(np.allclose(m.coef_, s.coef_), np.allclose(m.intercept_, s.intercept_))
print(m.coef_.round(3), round(m.intercept_, 3))
print(round(m.score(X, y), 6), round(s.score(X, y), 6))
```

```text
True True
[ 1.508e+00 -2.008e+00 -3.000e-03  3.005e+00  5.100e-01] 4.004
0.999396 0.999396
```

The true coefficients were 1.5, −2, 0, 3 and 0.5 with an intercept of 4; least squares recovers them up to the noise. And `score`, which we never wrote, comes from `RegressorMixin` (it's the R² of the predictions).

## 6. A transformer, and the method you get for free

The same contract gives us a scaler in a dozen lines:

```python
from sklearn.base import TransformerMixin


class MiniStandardScaler(TransformerMixin, BaseEstimator):
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
```

```text
MiniStandardScaler {'passed': 46, 'skipped': 1}
```

It produces the same output as `StandardScaler`, and it has a method we didn't write:

```python
print(MiniStandardScaler.fit_transform.__qualname__)
```

```text
TransformerMixin.fit_transform
```

`fit_transform` comes from the mixin: by default it's literally `fit(X, y).transform(X)`. Some transformers override it with a faster version that does both in one pass, but the meaning is always the same: **learn from this data, then transform this same data**. Keep that sentence in mind for [section E.8](#8-pipelines-and-why-fittransform-on-the-test-set-leaks).

Notice also that `y=None` in the scaler's `fit`. A transformer that doesn't use labels still accepts a `y`, because a pipeline passes `y` to every step, and the contract says every `fit` takes it.

## 7. Why the constructor must not do anything: get_params and clone

Rule 1 (the constructor only stores its arguments) is the easiest one to break, usually with good intentions. Here's an estimator that "fixes" a bad hyperparameter in its constructor:

```python
class RidgeBadInit(MiniLinearRegression):
    def __init__(self, fit_intercept=True, alpha=1.0):
        self.fit_intercept = fit_intercept
        self.alpha = max(float(alpha), 1e-3)        # looks harmless
```

```python
bad = RidgeBadInit(alpha=0.0)
print(bad.get_params())
```

```text
{'alpha': 0.001, 'fit_intercept': True}
```

You asked for `alpha=0.0`, and the object now reports `alpha=0.001`. That's a lie about its own configuration, and scikit-learn's tools trust that configuration completely. To see why, you need to know what `get_params` and `clone` do.

`BaseEstimator.get_params` doesn't keep a list of parameters anywhere. It reads the **signature of your `__init__`** (with `inspect.signature`, which we met in [section B.4](B-a-method-for-learning-any-library.md#4-ask-the-library-about-itself)) and returns the attribute with the same name for each argument. That's why every argument has to be stored under its own name.

`clone` builds a new, unfitted estimator with the same parameters, by calling your class with `get_params()`:

```python
from sklearn.base import clone

m = MiniLinearRegression().fit(X, y)
c = clone(m)
print(hasattr(c, "coef_"), c.get_params())
```

```text
False {'fit_intercept': True}
```

Same settings, nothing learned. Cross-validation, grid search and pipelines clone estimators constantly, because every fold must start from scratch. So if the constructor changes what it receives, every clone starts from a configuration you didn't ask for, and the results of a grid search report hyperparameters that were never used. And if the constructor *validates* its arguments, `set_params` (which grid search uses to try each combination) can blow up halfway through a search. The suite has a check for exactly that:

```text
check_do_not_raise_errors_in_init_or_set_params: ValueError: could not convert string to float: 'helloworld'
```

It passed a nonsense string as `alpha` and expected the constructor to accept it silently. Validation belongs at the start of `fit`, where the error points at the real problem.

> [!TIP]
> If you're writing an estimator and feel the need to do something in `__init__`, do it at the beginning of `fit` instead, and store the result in an attribute with a trailing underscore. That's what scikit-learn's own estimators do.

## 8. Pipelines, and why fit_transform on the test set leaks

A pipeline chains transformers and a final estimator into one object that is itself an estimator. The miniature makes its logic obvious:

```python
class MiniPipeline(BaseEstimator):
    def __init__(self, steps):
        self.steps = steps                           # a list of (name, estimator)

    def fit(self, X, y=None):
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt, y)           # learn from this data, transform it, pass it on
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
```

It gives the same predictions as scikit-learn's `Pipeline`, and because it follows the contract (it's a `BaseEstimator`, and `steps` is stored as given), `clone` and `cross_val_score` work on it with no extra code. On a train/test split and in 5-fold cross-validation, the two pipelines give identical numbers:

```text
True 0.925073 0.925073
[0.9009 0.9308 0.9319 0.8906 0.9345]
[0.9009 0.9308 0.9319 0.8906 0.9345]
```

Look at the difference between `fit` and `_transform`. During `fit`, each step **learns** from the data (`fit_transform`). During `predict`, each step only **applies** what it learned (`transform`). That difference is the entire point of pipelines, and it's exactly what a very common leakage bug violates.

The bug looks like this: you scale, or select features, or encode categories, using all of your data, and *then* you split into training and test, or you run cross-validation on the result. The transformation has already seen the test data. Its means, its selected features, its category encodings were all computed with information from the rows you're supposed to be testing on, so the score you get back is optimistic by an amount you don't know.

How much it matters depends on how much the step learns from the labels. Scaling learns very little, and nothing from `y`: on the regression data above, fitting the scaler on everything before cross-validating changes the score by 0.00012. Feature selection is a different story, because it chooses features by looking at `y`. Here's an experiment with nothing to learn at all:

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5000))          # 100 samples, 5,000 features of pure noise
y = rng.integers(0, 2, size=100)          # labels that have nothing to do with X

# wrong: choose the 20 "best" features using all the data, then cross-validate
X_selected = SelectKBest(f_classif, k=20).fit_transform(X, y)
wrong = cross_val_score(LogisticRegression(), X_selected, y, cv=5)

# right: the selection is part of the model, so it's refitted inside every fold
model = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
right = cross_val_score(model, X, y, cv=5)

print(f"selection outside CV: {wrong.mean():.2f}")
print(f"selection inside CV:  {right.mean():.2f}")
```

```text
selection outside CV: 0.85
selection inside CV:  0.48
```

The labels are coin flips. There's nothing to predict. And the wrong version reports **85% accuracy**, because out of 5,000 noise features, some are correlated with the coin flips *by chance in this sample*, and the selection found them using all 100 labels, including the ones the cross-validation later pretends not to know. The right version, with the selection inside the pipeline, gives 48%, which is what a coin flip deserves. Over ten different random seeds, the wrong way averages 0.87 (between 0.82 and 0.91) and the right way 0.51 (between 0.44 and 0.59).

This is a textbook example, literally: *The Elements of Statistical Learning* uses almost the same experiment (5,000 noise predictors, labels independent of them) in its section on "the wrong and right way to do cross-validation" ([Hastie, Tibshirani and Friedman, 2nd edition, section 7.10.2](https://hastie.su.domains/ElemStatLearn/)<sup>\[13\]</sup>). And it isn't only a beginner's mistake: a survey of research that uses ML found leakage in 294 published papers across 17 scientific fields, in some cases leading to wildly overoptimistic conclusions ([Kapoor and Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science", *Patterns*, 2023](https://doi.org/10.1016/j.patter.2023.100804)<sup>\[14\]</sup>).

> [!WARNING]
> The rule: **anything that learns from data is part of the model, and must be fitted only on training data.** Scalers, imputers, encoders, feature selection, PCA, oversampling, target encoding. Put them in a `Pipeline` and give the pipeline to `cross_val_score` or `GridSearchCV`, and scikit-learn will refit them inside every fold for you. Calling `fit_transform` on your test set, or on all the data before splitting, breaks the rule silently.

## 9. Grid search in miniature: clone, set_params and the double underscore

[Section E.7](#7-why-the-constructor-must-not-do-anything-getparams-and-clone) said grid search clones estimators constantly. Writing one makes it concrete, and it's shorter than you'd think:

```python
from itertools import product

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold


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
```

That's all of it: every combination of parameters, a clean copy of the model for every fold, the average score, and at the end one more copy, fitted on all the data with the winning combination. (I didn't inherit from `BaseEstimator` here, to keep it short; the real `GridSearchCV` is an estimator itself, so you can put it inside another cross-validation and get nested CV for free.)

Now the interesting part, which is the names in the grid. Let's search over a pipeline:

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
grid = {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0], "scale__with_std": [True, False]}

pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
mini = MiniGridSearch(pipe, grid).fit(X, y)
real = GridSearchCV(pipe, grid, cv=5).fit(X, y)

print(mini.best_params_, round(mini.best_score_, 6))
print(real.best_params_, round(real.best_score_, 6))
print(np.allclose([s for _, s in mini.results_], real.cv_results_["mean_test_score"]))
print(np.allclose(mini.best_estimator_.predict(X), real.best_estimator_.predict(X)))
print(hasattr(pipe.named_steps["ridge"], "coef_"))
```

```text
{'ridge__alpha': 0.1, 'scale__with_std': True} 0.482325
{'ridge__alpha': 0.1, 'scale__with_std': True} 0.482325
True
True
False
```

Same winner, the same score for each of the ten combinations, the same final model. And the last line is the one I like the most: after two complete grid searches, the `pipe` we passed in **was never fitted**. Everything happened on clones. That's the contract paying off again.

`ridge__alpha` means "the parameter `alpha` of the step called `ridge`". The double underscore is a naming convention, nothing more, and it lives in two methods. `get_params(deep=True)` flattens the parameters of every step into one dictionary with those prefixed names:

```python
print(sorted(k for k in pipe.get_params() if "__" in k)[:6])
```

```text
['ridge__alpha', 'ridge__copy_X', 'ridge__fit_intercept', 'ridge__max_iter', 'ridge__positive', 'ridge__random_state']
```

and `set_params` does the opposite: it splits the name at the first `__` and passes the rest down to that step. Since it only splits at the *first* one, it nests: a pipeline inside a column transformer inside another pipeline gives you names like `prep__num__impute__strategy`, which look horrible and work perfectly.

Now try the same search with the `MiniPipeline` of [section E.8](#8-pipelines-and-why-fittransform-on-the-test-set-leaks), the way it was written there, and scikit-learn says:

```text
ValueError: Invalid parameter 'ridge' for estimator MiniPipeline(steps=[('scale', MiniStandardScaler()), ('ridge', Ridge())]). Valid parameters are: ['steps'].
```

`BaseEstimator.get_params` only knows the arguments of `__init__`, and `MiniPipeline` has exactly one, `steps`. It has no idea that the list inside contains estimators with their own parameters. scikit-learn's `Pipeline` overrides both methods to look inside its steps, and our miniature needs the same thing:

```python
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
```

With those two methods added, `MiniGridSearch` over a `MiniPipeline` (with `MiniStandardScaler` inside) picks the same `{'ridge__alpha': 0.1, 'scale__with_std': True}` and gives the same ten mean scores as `GridSearchCV` on the real pipeline. Notice the `else` branch, too: `set_params(scale=StandardScaler())` replaces a whole step, which is how you can search over *which* scaler or *which* model to use, and not just over their settings.

> [!NOTE]
> Count the fits: 10 combinations × 5 folds, plus one final refit on all the data, is 51 fits. Grids grow multiplicatively, so three parameters with ten values each and 5 folds is 5,001 fits. When that hurts, `RandomizedSearchCV` samples a fixed number of combinations instead, and its docs explain when that's just as good.

## 10. Real tables: ColumnTransformer

Everything so far used a NumPy matrix of numbers, which is the data scikit-learn's contract is written for. Real data is a DataFrame with numbers, text categories and holes, and each kind needs different preprocessing. `ColumnTransformer` is the piece that connects the two worlds, and with the contract in your head it's easy to read: it's a transformer (so it has `fit` and `transform` and goes inside a pipeline) that holds other transformers, each one applied to a list of columns, and it concatenates their outputs side by side.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = pd.DataFrame({
    "age": [25, 32, np.nan, 51, 46, 38],
    "income": [32_000, 54_000, 61_000, np.nan, 88_000, 47_000],
    "city": ["Lima", "Oslo", "Lima", "Pune", "Oslo", "Pune"],
    "bought": [0, 1, 0, 1, 1, 0],
})
X, y = train.drop(columns="bought"), train["bought"]

prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age", "income"]),
    ("cat", OneHotEncoder(sparse_output=False), ["city"]),
]).set_output(transform="pandas")
model = Pipeline([("prep", prep), ("clf", LogisticRegression())]).fit(X, y)
print(model.named_steps["prep"].transform(X).round(2))
```

```text
   num__age  num__income  cat__city_Lima  cat__city_Oslo  cat__city_Pune
0     -1.56        -1.42             1.0             0.0             0.0
1     -0.74        -0.12             0.0             1.0             0.0
2     -0.04         0.30             1.0             0.0             0.0
3      1.48        -0.12             0.0             0.0             1.0
4      0.90         1.89             0.0             1.0             0.0
5     -0.04        -0.53             0.0             0.0             1.0
```

That's the matrix the logistic regression actually sees. The missing age in row 2 was filled with the median of the training ages (38) and then scaled, which is why it's −0.04 like row 5, whose age really is 38. The missing income in row 3 became the median income, 54,000, the same −0.12 as row 1. The city became three 0/1 columns, one per city seen in training. And because of `set_output(transform="pandas")` the result is a DataFrame with names that say where each column came from (`get_feature_names_out()` gives the same names without the DataFrame). Without it you get a bare NumPy array, and the names are the first thing you'll want when a model behaves strangely.

Everything that learns here learns in `fit`: the medians, the means and standard deviations, the list of cities. That's [section E.8](#8-pipelines-and-why-fittransform-on-the-test-set-leaks)'s rule applied to a whole table, and it's why the whole thing should live inside the pipeline that you give to cross-validation. It also means the list of cities is fixed at training time, which leads to the classic production error. A new customer from a city the model has never seen:

```python
new = pd.DataFrame({"age": [29], "income": [50_000], "city": ["Kyiv"]})
model.predict(new)
```

```text
ValueError: Found unknown categories ['Kyiv'] in column 0 during transform
```

That's the default (`handle_unknown="error"`), and it's a reasonable default, because silently doing something with a value you've never seen is a decision someone should make on purpose. With `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, the unknown city becomes all zeros (`[-1.09, -0.35, 0, 0, 0]`: "none of the cities I know") and the prediction goes through. Whether that's right depends on your problem; a new city that turns out to be your biggest market might deserve more than a vector of zeros. Either way, now it's a decision.

Two more things the column names buy you. The columns are selected by name, so a DataFrame with the same columns in a different order works and gives the same prediction. And a missing column fails loudly with an error that names it:

```text
ValueError: columns are missing: {'income'}
```

> [!NOTE]
> The step names compose into parameter names, as in [section E.9](#9-grid-search-in-miniature-clone-setparams-and-the-double-underscore). In this pipeline the imputer's strategy is `prep__num__simpleimputer__strategy` (`make_pipeline` names each step after its class, in lowercase), so a grid search can try `"median"` against `"mean"` without you rebuilding anything.

## 11. Checking the miniatures

Unlike the other chapters, most of the testing here was done by scikit-learn itself:

- `MiniLinearRegression` passes all 51 checks of `check_estimator` (one skipped, as explained);
- `MiniStandardScaler` passes all 46;
- both give the same results as `LinearRegression` and `StandardScaler` on random data (`np.allclose` on coefficients, intercepts and transformed data);
- `MiniPipeline` gives the same predictions and the same cross-validation scores as `Pipeline`;
- `MiniGridSearch` picks the same parameters as `GridSearchCV` and gives the same mean score for every combination, on both pipelines.

`test_mini_sklearn.py` in the companion code runs all of it (the estimator checks, 50 random datasets for the side-by-side comparisons, the pipelines and the grid searches):

```text
150 checks against scikit-learn 1.9.1: all passed
```

If you write estimators for your own projects (a custom transformer for your domain, a wrapper around a model from another library), run `check_estimator` on them. In a test suite, `sklearn.utils.estimator_checks.parametrize_with_checks` turns every check into a separate pytest test.

## 12. Reading the scikit-learn docs with this in your head

scikit-learn has some of the best documentation in the ecosystem, and it's organized in the four kinds from [section B.2](B-a-method-for-learning-any-library.md#2-documentation-comes-in-four-kinds-and-each-answers-a-different-question):

- The Glossary of Common Terms and API Elements, the most underrated page in the documentation. It defines precisely what `fit`, `transform`, `estimator`, `n_features_in_`, "transductive", "meta-estimator" and dozens of other terms mean in scikit-learn. When a docstring uses a word and you're not sure what it means *here*, it's in the glossary.
- "Developing scikit-learn estimators", the official version of this chapter: the full contract, the tags, and the rules for `__init__`. You don't need to be contributing to scikit-learn to read it; it's the best explanation of how the library thinks.
- The User Guide: for each family of models, what it does, the math behind it, and its practical caveats. Read the section of a model before using it in anything serious.
- The examples gallery: hundreds of complete, runnable scripts. When you want to know how to do something, search the gallery before searching the web.
- "Common pitfalls and recommended practices", a short page that covers leakage, inconsistent preprocessing and randomness. It's [section E.8](#8-pipelines-and-why-fittransform-on-the-test-set-leaks) in the authors' words.

## 13. Exercises

1. Write `MiniRidge`, ridge regression with an `alpha` hyperparameter (minimize squared error plus `alpha` times the squared norm of the coefficients, without penalizing the intercept). Make it pass `check_estimator`, and compare it with `sklearn.linear_model.Ridge`.
2. Write a `MiniKNNClassifier` with `n_neighbors`. A classifier must also set `classes_` in `fit` and inherit from `ClassifierMixin`. Run `check_estimator` and read what fails first.
3. Add `get_feature_names_out` to `MiniStandardScaler` (hint: `OneToOneFeatureMixin`, which we saw in the MRO of `StandardScaler` in [section B.4](B-a-method-for-learning-any-library.md#4-ask-the-library-about-itself)).
4. Repeat the leakage experiment of [section E.8](#8-pipelines-and-why-fittransform-on-the-test-set-leaks) with `k=5`, `k=100` and `k=1000`. How does the gap between the wrong and the right estimate change, and why?
5. Find an assistant-generated or tutorial notebook that calls `fit_transform` before `train_test_split`. Fix it with a pipeline, and measure how much the reported score changes.
6. Turn `MiniGridSearch` into an estimator (inherit from `BaseEstimator`, and add `predict` and `score` that use `best_estimator_`). Then `cross_val_score(MiniGridSearch(...), X, y)` is nested cross-validation: the search runs again inside every outer fold. Compare it with `cross_val_score(GridSearchCV(...), X, y)`.
7. In the data of [section E.10](#10-real-tables-columntransformer), replace one city with `np.nan`. What does `OneHotEncoder` do with it? Put a `SimpleImputer(strategy="constant", fill_value="missing")` before the encoder and explain the difference between "missing" and "a city I've never seen".

**Where this shows up in No Shortcuts:** all of Part III. Every classical model in the course is derived and implemented from scratch first and then used through scikit-learn, and lesson 18.4, "Leakage, imbalance and other ways to fool yourself", has eight versions of the bug from [section E.8](#8-pipelines-and-why-fittransform-on-the-test-set-leaks), measured.

---

[← D. pandas in miniature](D-pandas-in-miniature.md) · [Contents](../README.md#contents) · [F. PyTorch in miniature →](F-pytorch-in-miniature.md)
