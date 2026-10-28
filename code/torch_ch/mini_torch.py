"""A miniature PyTorch: a tensor that records how it was made, so gradients can flow backwards.

Values are NumPy arrays (chapter C already covered what an array is). Each operation returns a new
MiniTensor that remembers its inputs and a small function that turns the gradient of the output into
gradients of the inputs. backward() replays those functions in reverse order: that's the tape.
"""
import numpy as np


def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):                    # axes added on the left
        grad = grad.sum(axis=0)
    for axis, n in enumerate(shape):                 # axes that were size 1
        if n == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class MiniTensor:
    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = _parents                     # the tensors this one was computed from
        self._backward = _backward                   # grad of self -> grads of the parents

    @property
    def shape(self):
        return self.data.shape

    def _make(self, data, parents, backward):
        needs = any(p.requires_grad for p in parents)
        return MiniTensor(data, needs, parents if needs else (), backward if needs else None)

    # -- operations: forward value + how to send the gradient back -----------------------------
    def __add__(self, other):
        other = other if isinstance(other, MiniTensor) else MiniTensor(other)
        return self._make(self.data + other.data, (self, other),
                          lambda g: (_unbroadcast(g, self.shape), _unbroadcast(g, other.shape)))

    def __mul__(self, other):
        other = other if isinstance(other, MiniTensor) else MiniTensor(other)
        return self._make(self.data * other.data, (self, other),
                          lambda g: (_unbroadcast(g * other.data, self.shape),
                                     _unbroadcast(g * self.data, other.shape)))

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, MiniTensor) else MiniTensor(-np.asarray(other)))

    def __matmul__(self, other):
        return self._make(self.data @ other.data, (self, other),
                          lambda g: (g @ other.data.T, self.data.T @ g))

    def __pow__(self, k):
        return self._make(self.data ** k, (self,), lambda g: (g * k * self.data ** (k - 1),))

    def sum(self):
        return self._make(self.data.sum(), (self,), lambda g: (np.ones_like(self.data) * g,))

    def mean(self):
        n = self.data.size
        return self._make(self.data.mean(), (self,), lambda g: (np.ones_like(self.data) * g / n,))

    def relu(self):
        return self._make(np.maximum(self.data, 0), (self,), lambda g: (g * (self.data > 0),))

    def exp(self):
        out = np.exp(self.data)
        return self._make(out, (self,), lambda g: (g * out,))

    def log(self):
        return self._make(np.log(self.data), (self,), lambda g: (g / self.data,))

    # -- the tape, replayed backwards ----------------------------------------------------------
    def backward(self):
        order, seen = [], set()

        def visit(t):                                # topological order: inputs before outputs
            if id(t) not in seen:
                seen.add(id(t))
                for p in t._parents:
                    visit(p)
                order.append(t)

        visit(self)
        grads = {id(self): np.ones_like(self.data)}  # d(self)/d(self) = 1
        for t in reversed(order):
            g = grads.pop(id(t), None)
            if g is None:
                continue
            if t._backward is None:                  # a leaf: accumulate, like PyTorch
                if t.requires_grad:
                    t.grad = g if t.grad is None else t.grad + g
                continue
            for p, pg in zip(t._parents, t._backward(g)):
                if p.requires_grad:
                    grads[id(p)] = grads.get(id(p), 0) + pg


# -- nn.Module in miniature: parameters are found by watching attribute assignment ------------
class Parameter(MiniTensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class MiniModule:
    def __init__(self):
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name, value):              # the whole trick of nn.Module
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, MiniModule):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def named_parameters(self, prefix=""):
        for name, p in self._params.items():
            yield prefix + name, p
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix + name + ".")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args):
        return self.forward(*args)


class MiniLinear(MiniModule):
    def __init__(self, n_in, n_out, rng):
        super().__init__()
        bound = 1 / np.sqrt(n_in)                     # PyTorch's default init range
        self.weight = Parameter(rng.uniform(-bound, bound, size=(n_in, n_out)))
        self.bias = Parameter(rng.uniform(-bound, bound, size=(1, n_out)))

    def forward(self, x):
        return x @ self.weight + self.bias


# -- optimizers: an update rule applied to a list of parameters --------------------------------
class MiniSGD:
    def __init__(self, params, lr, momentum=0.0):
        self.params, self.lr, self.momentum = list(params), lr, momentum
        self.velocity = [None] * len(self.params)

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.momentum:
                self.velocity[i] = g if self.velocity[i] is None else self.momentum * self.velocity[i] + g
                g = self.velocity[i]
            p.data = p.data - self.lr * g


class MiniAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, (self.b1, self.b2), self.eps = list(params), lr, betas, eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)          # bias correction
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# -- Dataset and DataLoader in miniature --------------------------------------------------------
class MiniDataLoader:
    """A dataset is anything with __len__ and __getitem__. The loader shuffles indices and stacks batches."""

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, generator=None):
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.drop_last, self.generator = shuffle, drop_last, generator

    def _indices(self):
        n = len(self.dataset)
        if not self.shuffle:
            return list(range(n))
        import torch                                   # same permutation as torch's RandomSampler
        return torch.randperm(n, generator=self.generator).tolist()

    def __iter__(self):
        if self.generator is not None:
            import torch                               # torch draws a base seed for its worker
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)   # processes first
        idx = self._indices()
        for start in range(0, len(idx), self.batch_size):
            batch = idx[start:start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                return
            samples = [self.dataset[i] for i in batch]
            yield tuple(np.stack(field) for field in zip(*samples))   # the default "collate"

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)
