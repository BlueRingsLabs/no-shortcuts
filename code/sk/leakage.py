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
