# %% [markdown]
# # Lab 17.4: Softmax regression, and one step past it
#
# 1. Stable softmax and cross-entropy from logits; temperature.
# 2. Softmax regression from scratch on the digits data, with a numerical gradient check.
# 3. Same model, scikit-learn: matching loss and predictions. OvR vs multinomial.
# 4. A harder, honest problem: the telecom customer segments.
# 5. XOR three ways: linear (fails), linear + product feature, a two-layer network with hand-written backprop.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

rng = np.random.default_rng(174)

# %% [markdown]
# ## 1. Softmax and cross-entropy, computed from logits

# %%
def log_softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def softmax(Z):
    return np.exp(log_softmax(Z))


def cross_entropy(Z, y):
    return -log_softmax(Z)[np.arange(len(y)), y].mean()


Z = np.array([[2.0, 1.0, 0.1], [1000.0, 999.0, -1000.0]])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
print("naive softmax :", naive[1], "(overflow)")
print("stable softmax:", softmax(Z)[1])
assert np.isnan(naive[1]).any() and np.allclose(softmax(Z)[1], softmax(Z - 999)[1])
assert np.isfinite(cross_entropy(Z, np.array([0, 2]))), "even the 1e-870 probability gives a finite loss from logits"

for T in (0.5, 1.0, 5.0):
    print(f"T={T}: {np.round(softmax(Z[:1] / T)[0], 3)}")
assert np.allclose(softmax(Z[:1])[0], [0.659, 0.242, 0.099], atol=1e-3)

# %% [markdown]
# ## 2. Softmax regression from scratch on digits

# %%
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16.0, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
n, d = X_tr.shape
K = 10
Y_tr = np.eye(K)[y_tr]
lam = 1e-3                                                      # L2 on W (not on b)


def loss_and_grad(W, b, X, Y, y):
    Z = X @ W + b
    P = softmax(Z)
    loss = cross_entropy(Z, y) + 0.5 * lam * np.sum(W**2)
    G = (P - Y) / len(X)
    return loss, X.T @ G + lam * W, G.sum(0)


# numerical gradient check on a few random coordinates (always do this before trusting a hand-derived gradient)
W0 = 0.01 * rng.normal(size=(d, K)); b0 = np.zeros(K)
_, gW, gb = loss_and_grad(W0, b0, X_tr, Y_tr, y_tr)
h = 1e-6
for _ in range(10):
    i, k = rng.integers(d), rng.integers(K)
    Wp, Wm = W0.copy(), W0.copy(); Wp[i, k] += h; Wm[i, k] -= h
    num = (loss_and_grad(Wp, b0, X_tr, Y_tr, y_tr)[0] - loss_and_grad(Wm, b0, X_tr, Y_tr, y_tr)[0]) / (2 * h)
    assert abs(num - gW[i, k]) < 1e-7 * max(1, abs(num)) + 1e-9, (num, gW[i, k])
print("gradient check passed")

W, b = np.zeros((d, K)), np.zeros(K)
for epoch in range(60):                                         # mini-batch SGD with momentum
    lr = 0.5 / (1 + epoch / 5)                                  # decay, or SGD noise keeps you off the optimum
    perm = rng.permutation(n)
    vW, vb = np.zeros_like(W), np.zeros_like(b)
    for start in range(0, n, 64):
        idx = perm[start:start + 64]
        _, gW, gb = loss_and_grad(W, b, X_tr[idx], Y_tr[idx], y_tr[idx])
        vW = 0.9 * vW - lr * gW; vb = 0.9 * vb - lr * gb
        W += vW; b += vb
train_loss = loss_and_grad(W, b, X_tr, Y_tr, y_tr)[0]
acc_mine = np.mean((X_te @ W + b).argmax(1) == y_te)
print(f"from scratch: training loss {train_loss:.4f}, test accuracy {acc_mine:.3f}")
assert acc_mine > 0.95

# identifiability: shifting every class's weights by the same vector changes nothing
v = rng.normal(size=(d, 1))
assert np.allclose(softmax(X_te @ (W + v) + b), softmax(X_te @ W + b))
# each row of P - Y sums to 0, so every gradient step (from W = 0) keeps the class weights summing to 0:
# gradient descent lands on the same representative L2 would pick (exercise 3)
print(f"class weight vectors sum to ~0: max |sum| = {np.abs(W.sum(1)).max():.2e}")
assert np.abs(W.sum(1)).max() < 1e-2

# %% [markdown]
# ## 3. The same model in scikit-learn, and OvR for comparison
#
# scikit-learn minimizes sum(log loss) + ||W||^2 / (2C), i.e. mean(log loss) + ||W||^2 / (2 C n). Match lam = 1 / (C n).

# %%
C = 1 / (lam * n)
sk = LogisticRegression(C=C, max_iter=5000, tol=1e-10).fit(X_tr, y_tr)
sk_loss = cross_entropy(X_tr @ sk.coef_.T + sk.intercept_, y_tr) + 0.5 * lam * np.sum(sk.coef_**2)
print(f"objective: scikit-learn (L-BFGS) {sk_loss:.5f}, mine (SGD) {train_loss:.5f}")
assert sk_loss <= train_loss + 1e-6 and train_loss - sk_loss < 0.01, "SGD gets close to the exact optimum"
agree = np.mean((X_te @ W + b).argmax(1) == sk.predict(X_te))
print(f"prediction agreement with scikit-learn: {agree:.3f}")
assert agree > 0.98

ovr = OneVsRestClassifier(LogisticRegression(C=C, max_iter=5000)).fit(X_tr, y_tr)
raw_ovr = np.column_stack([e.predict_proba(X_te)[:, 1] for e in ovr.estimators_])
print(f"OvR raw scores sum to between {raw_ovr.sum(1).min():.2f} and {raw_ovr.sum(1).max():.2f} (not a distribution; sklearn renormalizes)")
print(f"test log loss: multinomial {log_loss(y_te, sk.predict_proba(X_te)):.4f}, OvR {log_loss(y_te, ovr.predict_proba(X_te)):.4f}")
assert raw_ovr.sum(1).max() - raw_ovr.sum(1).min() > 0.1
assert log_loss(y_te, sk.predict_proba(X_te)) < log_loss(y_te, ovr.predict_proba(X_te))

# %% [markdown]
# ## 4. A harder, honest problem
#
# Four customer categories from demographics. Digits made linear models look brilliant; this dataset doesn't.

# %%
tele = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "telecust_1000.csv")
Xt, yt = tele.drop(columns="custcat"), tele["custcat"]
cat_cols = ["region", "marital", "ed", "retire", "gender"]
num_cols = [c for c in Xt.columns if c not in cat_cols]
pre = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cat_cols), (StandardScaler(), num_cols))
base = cross_val_score(DummyClassifier(strategy="most_frequent"), Xt, yt, cv=5).mean()
lin = cross_val_score(make_pipeline(pre, LogisticRegression(max_iter=2000)), Xt, yt, cv=5).mean()
ll = -cross_val_score(make_pipeline(pre, LogisticRegression(max_iter=2000)), Xt, yt, cv=5, scoring="neg_log_loss").mean()
print(f"customer category: majority class {base:.3f}, softmax regression {lin:.3f} accuracy; log loss {ll:.3f} vs uniform {np.log(4):.3f}")
print("better than guessing, far from good: the features just don't say much about the category. Not every problem is digits.")
assert base + 0.05 < lin < 0.6 and ll < np.log(4)

# %% [markdown]
# ## 5. XOR three ways

# %%
m = 400
Xx = rng.uniform(-1, 1, size=(m, 2))
yx = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int)
Xx_te = rng.uniform(-1, 1, size=(2000, 2))
yx_te = ((Xx_te[:, 0] > 0) ^ (Xx_te[:, 1] > 0)).astype(int)

acc_lin = LogisticRegression().fit(Xx, yx).score(Xx_te, yx_te)
prod = lambda A: np.column_stack([A, A[:, 0] * A[:, 1]])
acc_prod = LogisticRegression(C=100).fit(prod(Xx), yx).score(prod(Xx_te), yx_te)
print(f"linear: {acc_lin:.3f}; linear + x1*x2 feature: {acc_prod:.3f}")
assert acc_lin < 0.65 and acc_prod > 0.95

# a two-layer network: softmax regression on learned ReLU features, trained with hand-written backprop
H = 16
W1 = rng.normal(0, 1, size=(2, H)); b1 = np.zeros(H)
W2 = rng.normal(0, 0.1, size=(H, 2)); b2 = np.zeros(2)
Yx = np.eye(2)[yx]
lr = 0.3
for step in range(4000):
    A = Xx @ W1 + b1
    Hh = np.maximum(A, 0)                   # learned features
    Z = Hh @ W2 + b2                        # ... fed to softmax regression
    dZ = (softmax(Z) - Yx) / m              # the 17.4 gradient
    dW2, db2 = Hh.T @ dZ, dZ.sum(0)
    dA = (dZ @ W2.T) * (A > 0)              # chain rule back through the ReLU
    dW1, db1 = Xx.T @ dA, dA.sum(0)
    W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
forward = lambda A_: np.maximum(A_ @ W1 + b1, 0) @ W2 + b2
acc_mlp = np.mean(forward(Xx_te).argmax(1) == yx_te)
print(f"two-layer network with 16 hidden ReLUs: {acc_mlp:.3f} (it found its own features)")
assert acc_mlp > 0.95

# the hidden layer made XOR linearly separable: a linear model on its features does the job
Hfeat, Hfeat_te = np.maximum(Xx @ W1 + b1, 0), np.maximum(Xx_te @ W1 + b1, 0)
acc_on_h = LogisticRegression(C=100, max_iter=5000).fit(Hfeat, yx).score(Hfeat_te, yx_te)
print(f"plain logistic regression on the network's hidden features: {acc_on_h:.3f}")
assert acc_on_h > 0.95

print("\nAll checks passed.")
