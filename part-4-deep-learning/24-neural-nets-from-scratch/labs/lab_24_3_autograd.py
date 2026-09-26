# %% [markdown]
# # Lab 24.3: An autograd engine, scalar then tensor
#
# 1. `Value`: a scalar autograd engine. Checked against finite differences, including a node used twice.
# 2. `Tensor`: a NumPy-backed engine with broadcasting-aware gradients, matmul, reductions and a stable
#    log-softmax cross-entropy. Checked against PyTorch on random computations.
# 3. The reasons real engines have zero_grad and no_grad, demonstrated.
# 4. Train an MLP on the digits with the tensor engine. Then time scalar vs tensor.

# %%
import time
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(243)
torch.set_default_dtype(torch.float64)


def topo_order(root):
    order, seen, stack = [], set(), [(root, False)]
    while stack:                                                 # iterative DFS: no recursion limit on long graphs
        node, done = stack.pop()
        if done:
            order.append(node); continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        stack.extend((p, False) for p in node.parents)
    return order                                                 # parents before children

# %% [markdown]
# ## 1. The scalar engine

# %%
class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad, self.parents, self._backward = float(data), 0.0, parents, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad; other.grad += out.grad        # += : accumulate, never overwrite
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad; other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, n):
        out = Value(self.data**n, (self,))
        def _backward():
            self.grad += n * self.data ** (n - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,))
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(np.tanh(self.data), (self,))
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(self.data, 0.0), (self,))
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o)
    def __truediv__(self, o): return self * o**-1

    def backward(self):
        order = topo_order(self)
        self.grad = 1.0
        for node in reversed(order):
            node._backward()


def f_scalar(a, b, c):
    x = a * b + a                                                # a is used twice
    y = (x.tanh() + (c * c).exp()).log()
    return y * a / (b ** 2 + 1) - c.relu()


vals = [0.7, -1.3, 0.4]
a, b, c = (Value(v) for v in vals)
out = f_scalar(a, b, c); out.backward()
eps = 1e-6
for i, node in enumerate((a, b, c)):
    plus = list(vals); minus = list(vals); plus[i] += eps; minus[i] -= eps
    num = (f_scalar(*map(Value, plus)).data - f_scalar(*map(Value, minus)).data) / (2 * eps)
    print(f"d out / d {'abc'[i]}: engine {node.grad:+.8f}, finite difference {num:+.8f}")
    assert abs(node.grad - num) < 1e-6

# %% [markdown]
# ## 2. The tensor engine

# %%
_GRAD_ENABLED = [True]


@contextmanager
def no_grad():
    _GRAD_ENABLED.append(False)
    try:
        yield
    finally:
        _GRAD_ENABLED.pop()


def unbroadcast(g, shape):
    while g.ndim > len(shape):                                   # leading axes added by broadcasting
        g = g.sum(0)
    for i, s in enumerate(shape):                                # axes stretched from size 1
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class Tensor:
    def __init__(self, data, parents=(), requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)
        self.parents = parents if (_GRAD_ENABLED[-1] and self.requires_grad) else ()
        self.grad = None
        self._backward = lambda: None

    def _make(self, data, parents, backward):
        out = Tensor(data, parents)
        if out.parents:
            out._backward = backward(out)
        return out

    @staticmethod
    def _acc(t, g):
        if t.requires_grad:
            g = unbroadcast(g, t.data.shape)
            t.grad = g if t.grad is None else t.grad + g

    def __add__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        return self._make(self.data + o.data, (self, o), lambda out: lambda: (Tensor._acc(self, out.grad), Tensor._acc(o, out.grad)))

    def __mul__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        return self._make(self.data * o.data, (self, o),
                          lambda out: lambda: (Tensor._acc(self, out.grad * o.data), Tensor._acc(o, out.grad * self.data)))

    def __matmul__(self, o):
        return self._make(self.data @ o.data, (self, o),
                          lambda out: lambda: (Tensor._acc(self, out.grad @ o.data.T), Tensor._acc(o, self.data.T @ out.grad)))

    def relu(self):
        return self._make(np.maximum(self.data, 0), (self,), lambda out: lambda: Tensor._acc(self, out.grad * (self.data > 0)))

    def tanh(self):
        y = np.tanh(self.data)
        return self._make(y, (self,), lambda out: lambda: Tensor._acc(self, out.grad * (1 - y**2)))

    def exp(self):
        y = np.exp(self.data)
        return self._make(y, (self,), lambda out: lambda: Tensor._acc(self, out.grad * y))

    def sum(self, axis=None, keepdims=False):
        def backward(out):
            def _b():
                g = out.grad if (keepdims or axis is None) else np.expand_dims(out.grad, axis)
                Tensor._acc(self, np.broadcast_to(g, self.data.shape))
            return _b
        return self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), backward)

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    def cross_entropy(self, y):                                  # fused, stable: from logits, like the real thing
        Z = self.data - self.data.max(1, keepdims=True)
        logp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        n = len(y)
        def backward(out):
            def _b():
                G = np.exp(logp); G[np.arange(n), y] -= 1
                Tensor._acc(self, out.grad * G / n)
            return _b
        return self._make(-logp[np.arange(n), y].mean(), (self,), backward)

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self): return self * -1.0
    def __sub__(self, o): return self + (-o if isinstance(o, Tensor) else -np.asarray(o))

    def backward(self):
        for node in topo_order(self):
            node.grad = None if node is not self and node.parents else node.grad
        self.grad = np.ones_like(self.data)
        for node in reversed(topo_order(self)):
            node._backward()


# random computations with broadcasting, against PyTorch
for trial in range(5):
    A, B, bias = rng.normal(size=(6, 4)), rng.normal(size=(4, 5)), rng.normal(size=(5,))
    scale, y = rng.normal(size=(6, 1)), rng.integers(0, 5, 6)
    tA, tB, tbias, tscale = (Tensor(v, requires_grad=True) for v in (A, B, bias, scale))
    h = ((tA @ tB + tbias).tanh() * tscale + tbias).relu()
    loss = h.cross_entropy(y) + (h * h).sum() * 0.01 + tbias.exp().mean()
    loss.backward()
    pA, pB, pbias, pscale = (torch.tensor(v, requires_grad=True) for v in (A, B, bias, scale))
    ph = (torch.tanh(pA @ pB + pbias) * pscale + pbias).relu()
    ploss = torch.nn.functional.cross_entropy(ph, torch.tensor(y)) + (ph * ph).sum() * 0.01 + pbias.exp().mean()
    ploss.backward()
    diff = max(np.abs(t.grad - p.grad.numpy()).max() for t, p in [(tA, pA), (tB, pB), (tbias, pbias), (tscale, pscale)])
    assert abs(loss.data - ploss.item()) < 1e-12 and diff < 1e-12, diff
print("5 random computations with broadcasting: loss and all gradients match PyTorch to 1e-12")
print(f"example: bias shape {bias.shape}, its gradient shape {tbias.grad.shape} (summed back from {h.data.shape})")

# %% [markdown]
# ## 3. Why zero_grad and no_grad exist

# %%
w = Tensor(rng.normal(size=(3,)), requires_grad=True)
x = Tensor(rng.normal(size=(3,)))
for _ in range(2):
    (w * x).sum().backward()
print(f"after two backward() calls without zeroing, w.grad = {np.round(w.grad, 4)} = 2 x x = {np.round(2 * x.data, 4)}")
assert np.allclose(w.grad, 2 * x.data)
w.grad = None                                                    # zero_grad

with no_grad():
    y_eval = (w * x).sum()
print(f"inside no_grad: result has parents? {bool(y_eval.parents)} (no graph, nothing kept alive for a backward pass)")
assert not y_eval.parents

# %% [markdown]
# ## 4. Train an MLP on the digits with the tensor engine

# %%
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
sizes = [64, 64, 10]
params = []
for a_, b_ in zip(sizes[:-1], sizes[1:]):
    params += [Tensor(rng.normal(size=(a_, b_)) * np.sqrt(2 / a_), requires_grad=True), Tensor(np.zeros(b_), requires_grad=True)]


def forward(X):
    h_ = Tensor(X)
    for i in range(0, len(params), 2):
        h_ = h_ @ params[i] + params[i + 1]
        if i < len(params) - 2:
            h_ = h_.relu()
    return h_


t0 = time.perf_counter()
for epoch in range(30):
    perm = rng.permutation(len(X_tr))
    for s in range(0, len(X_tr), 32):
        idx = perm[s:s + 32]
        for p in params:
            p.grad = None
        loss = forward(X_tr[idx]).cross_entropy(y_tr[idx])
        loss.backward()
        for p in params:
            p.data -= 0.1 * p.grad
t_tensor = time.perf_counter() - t0
with no_grad():
    acc = np.mean(forward(X_te).data.argmax(1) == y_te)
print(f"tensor engine: 30 epochs in {t_tensor:.1f}s, test accuracy {acc:.3f}")
assert acc > 0.95

# the same forward+backward for ONE batch of 32 with the scalar engine
W1 = [[Value(v) for v in row] for row in params[0].data]; b1 = [Value(v) for v in params[1].data]
t0 = time.perf_counter()
xb = X_tr[:32]
total = Value(0.0)
for row in xb:
    hidden = [sum((W1[i][j] * float(row[i]) for i in range(64)), b1[j]).relu() for j in range(64)]
    total = total + sum(hidden, Value(0.0))
total.backward()
t_scalar_batch = time.perf_counter() - t0
n_batches = 30 * int(np.ceil(len(X_tr) / 32))
print(f"scalar engine, first layer only, one batch: {t_scalar_batch:.2f}s -> the full training run would take ~{t_scalar_batch * n_batches / 60:.0f}+ minutes")
print("same algorithm; the difference is one Python object per number vs one per array. That gap is why frameworks exist.")

print("\nAll checks passed.")
