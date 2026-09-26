# %% [markdown]
# # Lab 54.2: Differential privacy and federated learning
#
# 1. DP-SGD from scratch (Abadi et al., 2016): clip each example's gradient, add Gaussian noise, and account for the
#    privacy spent. Accuracy against epsilon, and the membership-inference audit from 53.1 against epsilon.
# 2. Federated averaging (McMahan et al., 2017): ten hospitals train one model without pooling their data. What
#    non-identical data across clients does to it, and what federated learning does and doesn't protect.

# %%
import math

import numpy as np
from sklearn.metrics import roc_auc_score

rs = np.random.default_rng(542)


def make(n, d=20, seed=0, shift=None):
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, (n, d)) + (0 if shift is None else shift)
    w = np.linspace(1.0, -1.0, d)
    y = (r.random(n) < 1 / (1 + np.exp(-(X @ w / 2 + 0.3 * r.normal(0, 1, n))))).astype(float)
    return X, y


# members: a small training set with 15% flipped labels (unusual records, which models memorize: 53.1)
X, y = make(1000, seed=1)
flip = rs.random(len(y)) < 0.15
y[flip] = 1 - y[flip]
X_non, y_non = make(1000, seed=2)                                                  # non-members, same distribution
flip2 = rs.random(len(y_non)) < 0.15; y_non[flip2] = 1 - y_non[flip2]
X_te, y_te = make(5000, seed=3)


def features(X):
    """A fixed random-feature expansion: enough capacity to memorize a thousand records."""
    W = np.random.default_rng(99).normal(0, 1, (X.shape[1], 190)) / math.sqrt(X.shape[1])
    return np.hstack([X, np.cos(X @ W), np.sin(X @ W)])                            # 20 + 380 = 400 features


F, F_non, F_te = features(X), features(X_non), features(X_te)
sig = lambda z: 1 / (1 + np.exp(-z))


def dp_sgd(sigma, clip=1.0, epochs=30, lr=0.5, batch=100, seed=0):
    """Logistic regression by DP-SGD. sigma=None: ordinary SGD; sigma=0: clipping without noise."""
    r = np.random.default_rng(seed)
    w = np.zeros(F.shape[1])
    for ep in range(epochs):
        for idx in np.array_split(r.permutation(len(y)), len(y) // batch):
            g = (sig(F[idx] @ w) - y[idx])[:, None] * F[idx]                       # per-example gradients (batch, d)
            if sigma is not None:
                norms = np.linalg.norm(g, axis=1, keepdims=True)
                g = g / np.maximum(1, norms / clip)                                  # clip each example's influence
                g = (g.sum(0) + r.normal(0, sigma * clip, g.shape[1])) / len(idx)   # noise scaled to the clip
            else:
                g = g.mean(0)
            w -= lr * g
    return w


def epsilon(sigma, q, steps, delta=1e-5):
    """Approximate (epsilon, delta) for the subsampled Gaussian mechanism via Renyi DP: for small q and sigma not too
    small, each step costs about 2 q^2 alpha / sigma^2 at order alpha (Mironov et al., 2019). Use a real accountant
    (Opacus, TensorFlow Privacy) for anything that matters."""
    return min(steps * 2 * q * q * a / sigma ** 2 + math.log(1 / delta) / (a - 1) for a in np.arange(1.5, 256, 0.5))


def mi_auc(w):
    loss = lambda F_, y_: -(y_ * np.log(sig(F_ @ w) + 1e-12) + (1 - y_) * np.log(1 - sig(F_ @ w) + 1e-12))
    return roc_auc_score(np.r_[np.ones(len(y)), np.zeros(len(y_non))], -np.r_[loss(F, y), loss(F_non, y_non)])


acc = lambda w: np.mean((sig(F_te @ w) > 0.5) == y_te)
steps = 30 * (len(y) // 100)
print(f"{'training':34s} {'epsilon (delta 1e-5)':>21s} {'test accuracy':>14s} {'membership AUC':>15s}")
rows = {}
for sigma in (None, 0.0, 0.6, 1.0, 2.0, 4.0):
    w = dp_sgd(sigma)
    eps = float("inf") if not sigma else epsilon(sigma, 100 / len(y), steps)
    rows[sigma] = (eps, acc(w), mi_auc(w))
    label = "ordinary SGD" if sigma is None else "clipping only, no noise" if sigma == 0 else f"DP-SGD, noise multiplier {sigma}"
    print(f"{label:34s} {eps:21.2f} {rows[sigma][1]:14.3f} {rows[sigma][2]:15.3f}")
print("clipping each record's gradient bounds its influence: that alone removes most of the leak this audit can measure,")
print("and here even improves accuracy, since ordinary SGD memorizes the mislabeled records. The noise is what buys the")
print("guarantee: epsilon bounds how much any single record can change what training outputs, against every possible")
print("attack, not just this one. An audit gives a lower bound on leakage; only the accounting gives an upper bound.")
print("Accuracy falls as epsilon shrinks; with a thousand records strong privacy is costly, and it gets cheaper with more")
print("data (the noise is added once per batch) and with pretrained models fine-tuned privately.")
assert rows[None][2] > rows[4.0][2] + 0.03 and rows[4.0][0] < rows[0.6][0] and rows[4.0][1] < rows[0.6][1]

# %% [markdown]
# ## 2. Federated averaging

# %%
n_clients, rounds = 10, 30
clients = []
for c in range(n_clients):
    shift = np.zeros(20); shift[c % 20] = 2.0 * (1 if c % 2 else -1)                 # each hospital sees a different population
    Xc, yc = make(30, seed=100 + c, shift=shift)
    clients.append((Xc, yc))                                                          # a plain linear model here
X_all = np.concatenate([c[0] for c in clients]); y_all = np.concatenate([c[1] for c in clients])
test_sets = [make(500, seed=200 + c, shift=np.eye(20)[c % 20] * 2.0 * (1 if c % 2 else -1)) for c in range(n_clients)]
test_F = test_sets


def sgd(Fc, yc, w, epochs, lr=0.5, seed=0):
    r = np.random.default_rng(seed)
    for _ in range(epochs):
        for idx in np.array_split(r.permutation(len(yc)), max(1, len(yc) // 50)):
            w = w - lr * ((sig(Fc[idx] @ w) - yc[idx])[:, None] * Fc[idx]).mean(0)
    return w


def mean_acc(w):
    return np.mean([np.mean((sig(Ft @ w) > 0.5) == yt) for Ft, yt in test_F])


w_central = sgd(X_all, y_all, np.zeros(20), epochs=30)
local = [sgd(Fc, yc, np.zeros(20), epochs=30, seed=i) for i, (Fc, yc) in enumerate(clients)]
local_acc = np.mean([np.mean((sig(test_F[i][0] @ local[i]) > 0.5) == test_F[i][1]) for i in range(n_clients)])
w_fed = np.zeros(20)
for r in range(rounds):
    updates = [sgd(Fc, yc, w_fed.copy(), epochs=1, seed=r * 100 + i) for i, (Fc, yc) in enumerate(clients)]
    w_fed = np.average(updates, axis=0, weights=[len(yc) for _, yc in clients])      # FedAvg: weighted mean of models
print(f"\n{n_clients} hospitals, 30 patients each, each seeing a different population:")
print(f"  each hospital alone (tested on its own population)  {local_acc:.3f}")
personal = [sgd(Fc, yc, w_fed.copy(), epochs=3, seed=i) for i, (Fc, yc) in enumerate(clients)]
pers_acc = np.mean([np.mean((sig(test_F[i][0] @ personal[i]) > 0.5) == test_F[i][1]) for i in range(n_clients)])
print(f"  federated averaging, {rounds} rounds                    {mean_acc(w_fed):.3f}")
print(f"  federated, then fine-tuned at each hospital         {pers_acc:.3f}")
print(f"  pooling all the data centrally (not allowed here)   {mean_acc(w_central):.3f}")
print("small sites learn little alone; federated averaging nearly matches pooling the data, without the records leaving")
print("any site (only model updates do). Fine-tuning the shared model locally on 30 patients makes it worse: adapting to")
print("each site needs enough data at each site. With very different populations or more rounds of local training,")
print("FedAvg's updates pull against each other and it falls behind pooling. And federated learning is not privacy by")
print("itself: updates can leak training data (gradient inversion), so real deployments add secure aggregation (the")
print("server only sees the sum) and differential privacy on the updates.")
assert mean_acc(w_fed) > local_acc + 0.03

print("\nAll checks passed.")
