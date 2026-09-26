# %% [markdown]
# # Lab 24.2: Backpropagation, by hand, checked three ways
#
# 1. A small layer library in NumPy: each layer has forward() and backward() (its vector-Jacobian product).
# 2. Gradient check of every layer and of a whole network (float64, central differences, relative error).
# 3. The same network in PyTorch: gradients match to machine precision.
# 4. Two classic bugs: the bias gradient without the sum, and the loss as a sum with a backward for the mean.
# 5. Reverse mode vs forward mode: one backward pass vs one pass per parameter.
# 6. Train on the digits with the hand-written library.

# %%
import time

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(242)
torch.set_default_dtype(torch.float64)

# %% [markdown]
# ## 1. The layer library

# %%
class Linear:
    def __init__(self, n_in, n_out, rng):
        self.W = rng.normal(size=(n_in, n_out)) * np.sqrt(2 / n_in)
        self.b = np.zeros(n_out)

    def forward(self, X):
        self.X = X                                               # stored for the backward pass
        return X @ self.W + self.b

    def backward(self, gY):
        self.gW = self.X.T @ gY                                  # same shape as W
        self.gb = gY.sum(0)                                      # b was broadcast over the batch: sum it back
        return gY @ self.W.T

    def params(self):
        return [(self.W, "gW"), (self.b, "gb")]


class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, gY):
        return gY * self.mask

    def params(self):
        return []


class Tanh:
    def forward(self, X):
        self.Y = np.tanh(X)
        return self.Y

    def backward(self, gY):
        return gY * (1 - self.Y**2)

    def params(self):
        return []


class SoftmaxCrossEntropy:                                       # fused, from logits (17.4)
    def forward(self, Z, y):
        Z = Z - Z.max(1, keepdims=True)
        self.P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
        self.y = y
        return -np.log(self.P[np.arange(len(y)), y]).mean()

    def backward(self):
        G = self.P.copy()
        G[np.arange(len(self.y)), self.y] -= 1
        return G / len(self.y)


class Net:
    def __init__(self, sizes, act=ReLU, seed=0):
        r = np.random.default_rng(seed)
        self.layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(a, b, r))
            if i < len(sizes) - 2:
                self.layers.append(act())
        self.loss = SoftmaxCrossEntropy()

    def forward(self, X, y):
        for layer in self.layers:
            X = layer.forward(X)
        return self.loss.forward(X, y)

    def backward(self):
        g = self.loss.backward()
        for layer in reversed(self.layers):                     # walk the tape backwards
            g = layer.backward(g)

    def params(self):
        return [(layer, p, g) for layer in self.layers for p, g in layer.params()]

# %% [markdown]
# ## 2. Gradient checks

# %%
def rel_err(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def grad_check(net, X, y, n_coords=12, eps=1e-6):
    net.forward(X, y); net.backward()
    worst = 0.0
    for layer, p, gname in net.params():
        g = getattr(layer, gname)
        for _ in range(n_coords):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps; lp = net.forward(X, y)
            p[idx] = old - eps; lm = net.forward(X, y)
            p[idx] = old
            worst = max(worst, rel_err((lp - lm) / (2 * eps), g[idx]))
    return worst


Xs = rng.normal(size=(8, 5)); ys = rng.integers(0, 3, 8)
for act in (Tanh, ReLU):
    err = grad_check(Net([5, 7, 6, 3], act=act), Xs, ys)
    print(f"{act.__name__:5s} network: worst relative error over sampled coordinates {err:.2e}")
    assert err < 1e-6

# %% [markdown]
# ## 3. Against PyTorch

# %%
net = Net([5, 7, 6, 3], act=Tanh, seed=3)
loss_np = net.forward(Xs, ys); net.backward()
tparams = []
Xt = torch.tensor(Xs)
h = Xt
for layer in net.layers:
    if isinstance(layer, Linear):
        W = torch.tensor(layer.W, requires_grad=True); b = torch.tensor(layer.b, requires_grad=True)
        tparams.append((layer, W, b)); h = h @ W + b
    else:
        h = torch.tanh(h)
loss_t = torch.nn.functional.cross_entropy(h, torch.tensor(ys))
loss_t.backward()
max_diff = max(max(np.abs(layer.gW - W.grad.numpy()).max(), np.abs(layer.gb - b.grad.numpy()).max()) for layer, W, b in tparams)
print(f"loss: mine {loss_np:.12f}, PyTorch {loss_t.item():.12f}; largest gradient difference {max_diff:.1e}")
assert abs(loss_np - loss_t.item()) < 1e-12 and max_diff < 1e-12

# %% [markdown]
# ## 4. Two classic bugs

# %%
class LinearNoSum(Linear):
    def backward(self, gY):
        self.gW = self.X.T @ gY
        self.gb = gY[0]                                          # "it's the same for every row, right?" It isn't.
        return gY @ self.W.T


buggy = Net([5, 7, 3], seed=1); buggy.layers[0].__class__ = LinearNoSum
buggy.forward(Xs, ys); buggy.backward()
ref = Net([5, 7, 3], seed=1); ref.forward(Xs, ys); ref.backward()
print(f"bias gradient without the sum: relative error vs correct {np.linalg.norm(buggy.layers[0].gb - ref.layers[0].gb) / np.linalg.norm(ref.layers[0].gb):.2f}"
      " (no shape error, just wrong)")
assert buggy.layers[0].gb.shape == ref.layers[0].gb.shape


class SumLoss(SoftmaxCrossEntropy):
    def forward(self, Z, y):
        return super().forward(Z, y) * len(y)                  # loss reported as a sum over the batch...
                                                                # ...while backward() still divides by the batch size
wrong = Net([5, 7, 3], seed=1); wrong.loss = SumLoss()
err_scale = grad_check(wrong, Xs, ys)
print(f"sum-loss with a mean-loss backward: gradient check relative error {err_scale:.2f} (off by exactly a factor of the batch size, 8)")
assert err_scale > 0.8

# %% [markdown]
# ## 5. Reverse mode vs forward mode

# %%
big = Net([64, 256, 256, 10], seed=0)
Xb, yb = rng.normal(size=(128, 64)), rng.integers(0, 10, 128)
n_params = sum(p.size for _, p, _ in big.params())
t0 = time.perf_counter()
for _ in range(20):
    big.forward(Xb, yb); big.backward()
t_rev = (time.perf_counter() - t0) / 20
t0 = time.perf_counter()
for _ in range(200):                                             # forward mode needs one pass per parameter direction;
    big.forward(Xb, yb)                                          # time 200 forward passes and extrapolate
t_fwd = (time.perf_counter() - t0) / 200
print(f"{n_params:,} parameters. One forward+backward: {t_rev * 1000:.1f} ms. One forward pass: {t_fwd * 1000:.2f} ms. "
      f"Forward-mode gradient ~ {n_params} passes ~ {n_params * t_fwd:.0f} s")
print(f"reverse mode is ~{n_params * t_fwd / t_rev:,.0f}x cheaper here; forward + backward costs {t_rev / t_fwd:.1f}x a forward pass")
assert t_rev / t_fwd < 6

stored = sum(layer.X.nbytes for layer in big.layers if isinstance(layer, Linear)) + sum(layer.mask.nbytes for layer in big.layers if isinstance(layer, ReLU))
print(f"activations stored for the backward pass at batch 128: {stored / 1e6:.2f} MB; parameters {n_params * 8 / 1e6:.2f} MB (float64)")

# %% [markdown]
# ## 6. Train with it

# %%
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
model = Net([64, 128, 64, 10], seed=0)
for epoch in range(40):
    perm = rng.permutation(len(X_tr))
    for start in range(0, len(X_tr), 32):
        idx = perm[start:start + 32]
        model.forward(X_tr[idx], y_tr[idx]); model.backward()
        for layer, p, gname in model.params():
            p -= 0.1 * getattr(layer, gname)                     # in-place SGD step
    if epoch in (0, 9, 39):
        h_ = X_te
        for layer in model.layers:
            h_ = layer.forward(h_)
        print(f"epoch {epoch + 1:2d}: training loss {model.forward(X_tr, y_tr):.4f}, test accuracy {np.mean(h_.argmax(1) == y_te):.3f}")
assert np.mean(h_.argmax(1) == y_te) > 0.96

print("\nAll checks passed.")
