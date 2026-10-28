# F. PyTorch in miniature

PyTorch looks like a big library, and it is, but the part you use every day rests on three ideas, and you've already met the first one:

1. **A tensor is NumPy's array** (buffer, dtype, shape, strides, offset), with the option of living on a GPU.
2. **Autograd is a tape.** Every operation on a tensor that requires gradients is written down as it happens, and `backward()` replays the tape in reverse to compute every gradient.
3. **`nn.Module` is a container that finds your parameters for you**, by watching what you assign to it. Optimizers are update rules applied to that list of parameters, and the training loop is five lines that connect all of it.

The PyTorch paper describes these choices (and why they made the library feel like plain Python) better than most tutorials ([Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", 2019](https://arxiv.org/abs/1912.01703)<sup>\[15\]</sup>). This chapter builds each idea in miniature: a tensor with autograd on top of NumPy, a module that registers parameters, SGD and Adam, and a data loader. Then it checks all of them against PyTorch, down to the order of the batches.

Everything here runs on a CPU. The examples use PyTorch 2.15.0.

## 1. A tensor is an array you already know

```python
import torch

t = torch.arange(12).reshape(3, 4)
print(t.shape, t.stride(), t.dtype, t.storage_offset())

v = t[:, 1:3]
print(v.stride(), v.storage_offset(), v.is_contiguous())
```

```text
torch.Size([3, 4]) (4, 1) torch.int64 0
(4, 1) 1 False
```

Strides, an offset, and a slice that is a view: this is [section C.3](C-numpy-in-miniature.md#3-slicing-is-arithmetic-on-strides) again. The one visible difference is that PyTorch counts strides in elements, not bytes, exactly like our MiniArray did.

PyTorch also makes the view-or-copy question explicit with two different methods. `view` *only* reinterprets memory and refuses when it can't; `reshape` gives you a view when possible and a copy otherwise:

```python
t.T.view(12)
```

```text
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

That's the same situation as `a.T.reshape(12, copy=False)` in NumPy. `t.T.contiguous()` makes a copy laid out in row-major order, after which `view` works. When you see `.contiguous().view(...)` in model code (it's common in older transformer implementations), it's exactly this: make the memory row-major, then reinterpret it.

And the two libraries can share memory directly:

```python
import numpy as np

a = np.arange(3.0)
b = torch.from_numpy(a)
a[0] = 100
print(b)
```

```text
tensor([100.,   1.,   2.], dtype=torch.float64)
```

`torch.from_numpy` doesn't copy. Changing the NumPy array changed the tensor. (`torch.tensor(a)` copies, if you need them independent.) Notice also the dtype: NumPy's default float is 64-bit, while PyTorch's default is 32-bit, so a tensor that comes from NumPy is usually `float64` while the weights of your model are `float32`, and mixing them gives you a dtype error. When that happens, `.float()` on the input is almost always what you want.

## 2. Autograd is a tape

Here's the second idea, shown by PyTorch itself:

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x * x).sum()
print(y, y.grad_fn, y.grad_fn.next_functions)
```

```text
tensor(13., grad_fn=<SumBackward0>) <SumBackward0 object at 0x7fc206f46d40> ((<MulBackward0 object at 0x7fc20701e2c0>, 0),)
```

(The memory addresses will be different on your machine.) `y` knows it was made by a sum (`SumBackward0`), and the sum knows its input was made by a multiplication (`MulBackward0`). Every tensor that requires gradients carries a pointer to the operation that produced it, and every operation points to the operations that produced *its* inputs. Together they form a graph of the whole computation, recorded as it ran. That's the tape.

`backward()` walks that graph from the end to the beginning, applying the chain rule at each step:

```python
y.backward()
print(x.grad)
```

```text
tensor([4., 6.])
```

The derivative of x₁² + x₂² is (2x₁, 2x₂) = (4, 6). The important word in "the tape is recorded as it runs" is *runs*: the graph is built fresh on every forward pass, by ordinary Python code. If your model has an `if` or a `for` loop that depends on the data, the graph for that input includes exactly the branch that ran. That's what "define-by-run" (or "dynamic graph") means, and it's the main reason PyTorch feels like Python.

The general technique is called **reverse-mode automatic differentiation**, and it's older than deep learning; this survey covers it well, including why reverse mode is the right choice when you have millions of inputs (the parameters) and one output (the loss) ([Baydin et al., "Automatic Differentiation in Machine Learning: a Survey", 2018](https://arxiv.org/abs/1502.05767)<sup>\[16\]</sup>).

## 3. Building MiniTensor

Our miniature holds a NumPy array and, for every operation, records two things: the tensors it came from, and a function that turns the gradient of the output into gradients of the inputs.

```python
import numpy as np


class MiniTensor:
    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = _parents        # the tensors this one was computed from
        self._backward = _backward      # grad of self -> grads of the parents

    @property
    def shape(self):
        return self.data.shape

    def _make(self, data, parents, backward):
        needs = any(p.requires_grad for p in parents)
        return MiniTensor(data, needs, parents if needs else (), backward if needs else None)
```

`_make` is how every operation builds its result. If none of the inputs needs a gradient, nothing is recorded, which is also what PyTorch does (and what `torch.no_grad()` forces, which is why inference inside `no_grad` uses less memory).

Each operation is then two lines: the forward value, and the backward rule. Multiplication, whose derivative with respect to each input is the other input:

```python
    def __mul__(self, other):
        other = other if isinstance(other, MiniTensor) else MiniTensor(other)
        return self._make(self.data * other.data, (self, other),
                          lambda g: (_unbroadcast(g * other.data, self.shape),
                                     _unbroadcast(g * self.data, other.shape)))
```

Matrix multiplication, whose rules you can derive in a few lines (or look up once and remember forever):

```python
    def __matmul__(self, other):
        return self._make(self.data @ other.data, (self, other),
                          lambda g: (g @ other.data.T, self.data.T @ g))
```

ReLU passes the gradient through where the input was positive and blocks it elsewhere:

```python
    def relu(self):
        return self._make(np.maximum(self.data, 0), (self,), lambda g: (g * (self.data > 0),))
```

(`__add__`, `__sub__`, `__pow__`, `exp`, `log`, `sum` and `mean` follow the same pattern; they're in the companion code.)

What's `_unbroadcast`? It's the one non-obvious piece, and it connects this chapter to [section C.6](C-numpy-in-miniature.md#6-broadcasting-is-a-stride-of-zero). When you add a bias of shape `(1, 4)` to a batch of shape `(32, 4)`, broadcasting reads the same bias row 32 times (stride 0). So the bias influenced all 32 rows of the output, and its gradient is the **sum** of the gradients of those 32 rows:

```python
def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):          # axes added on the left
        grad = grad.sum(axis=0)
    for axis, n in enumerate(shape):       # axes that were size 1
        if n == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad
```

Broadcasting forward is a stride of zero; broadcasting backward is a sum. Forgetting that is a classic bug in hand-written backpropagation, and the test in [section F.9](#9-checking-the-miniature-against-pytorch) uses shapes that broadcast against each other precisely to catch it.

Finally, `backward`: sort the recorded graph so that every tensor comes after the tensors it was computed from, then walk it from the end, handing each tensor's gradient to its parents:

```python
    def backward(self):
        order, seen = [], set()

        def visit(t):                       # topological order: inputs before outputs
            if id(t) not in seen:
                seen.add(id(t))
                for p in t._parents:
                    visit(p)
                order.append(t)

        visit(self)
        grads = {id(self): np.ones_like(self.data)}    # d(self)/d(self) = 1
        for t in reversed(order):
            g = grads.pop(id(t), None)
            if g is None:
                continue
            if t._backward is None:                     # a leaf: accumulate, like PyTorch
                if t.requires_grad:
                    t.grad = g if t.grad is None else t.grad + g
                continue
            for p, pg in zip(t._parents, t._backward(g)):
                if p.requires_grad:
                    grads[id(p)] = grads.get(id(p), 0) + pg
```

Two details in there are worth stopping at, because PyTorch behaves the same way and they explain real bugs.

First, when a tensor is used twice (like `x` in `x * x`), it gets a gradient from each use, and they're **added** (`grads.get(...) + pg`). That's the multivariable chain rule.

Second, a **leaf** (a tensor you created, not one computed from others) doesn't get its gradient replaced on each call to `backward`. It gets it **added** to whatever was there. PyTorch does exactly the same:

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
(x * x).sum().backward()
(x * x).sum().backward()
print(x.grad)
```

```text
tensor([ 8., 12.])
```

Twice (4, 6). That's deliberate (it lets you accumulate gradients over several small batches before an update, when a big batch doesn't fit in memory), and it's the reason every training loop starts by zeroing the gradients. [Section F.6](#6-the-training-loop-line-by-line) shows what happens when you forget.

## 4. nn.Module finds your parameters by watching assignments

When you write `self.fc = nn.Linear(4, 1)` inside a module, how does `model.parameters()` later know that `fc.weight` exists? There's no registry you call. The trick is that `nn.Module` overrides `__setattr__`, the method Python calls on every assignment to an attribute: when the value is a `Parameter` or another `Module`, it's stored in a dictionary, and `parameters()` walks those dictionaries recursively.

The miniature is the whole idea in a dozen lines:

```python
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
```

And a linear layer on top of it, initialized the way PyTorch does it by default (uniform in ±1/√fan_in, which PyTorch's source code explains in a comment in `nn.Linear.reset_parameters`):

```python
class MiniLinear(MiniModule):
    def __init__(self, n_in, n_out, rng):
        super().__init__()
        bound = 1 / np.sqrt(n_in)
        self.weight = Parameter(rng.uniform(-bound, bound, size=(n_in, n_out)))
        self.bias = Parameter(rng.uniform(-bound, bound, size=(1, n_out)))

    def forward(self, x):
        return x @ self.weight + self.bias
```

(One difference from the real one: PyTorch stores the weight with shape `(out_features, in_features)` and computes `x @ W.T + b`. The math is the same.)

The names you see in `named_parameters()` and in a saved `state_dict` (`"first.weight"`, `"encoder.layers.0.attn.q_proj.bias"`) are just the attribute names joined with dots along the path, which is what `prefix + name + "."` builds.

Now the consequence. Since registration happens on assignment of a Module or Parameter, anything stored in some other way is invisible. The classic case is a plain Python list of layers:

```python
import torch.nn as nn


class HalfBroken(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = [nn.Linear(4, 4)]          # a plain list: not registered
        self.head = nn.Linear(4, 1)              # registered

    def forward(self, x):
        return self.head(torch.relu(self.hidden[0](x)))


torch.manual_seed(0)
model = HalfBroken()
print([n for n, _ in model.named_parameters()])
```

```text
['head.weight', 'head.bias']
```

The hidden layer's parameters aren't there. Train it for 50 steps with SGD and check:

```python
before = model.hidden[0].weight.detach().clone()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x, y = torch.randn(64, 4), torch.randn(64, 1)
for _ in range(50):
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()

print(torch.equal(before, model.hidden[0].weight), model.hidden[0].weight.grad is not None)
```

```text
True True
```

The hidden layer's weights are exactly what they were before training, even though `backward()` computed gradients for them (the second `True`). The optimizer was given `model.parameters()`, which doesn't include them, so it never updated them. The model trains, the loss goes down a bit (from 1.10 to 0.94 here, because the head is learning), nothing crashes, and half the network is frozen at its random initialization. The same layers saved with `torch.save(model.state_dict())` wouldn't be saved either, and `model.to("cuda")` wouldn't move them.

If *every* layer is in the list, at least you get an error, because there's nothing left to optimize:

```text
ValueError: optimizer got an empty parameter list
```

The fix is to use the containers that register their contents: `nn.ModuleList` for a list of modules, `nn.ModuleDict` for a dict, `nn.Sequential` for a chain, and `nn.Parameter` (or `register_buffer`, for tensors that should be saved but not trained) for raw tensors.

> [!WARNING]
> If you build layers in a loop, store them in `nn.ModuleList`, never in a Python list or dict. Then check `sum(p.numel() for p in model.parameters())` against the number of parameters you expect. It takes one line, and it catches this bug.

## 5. An optimizer is an update rule over a list of parameters

`torch.optim.SGD(model.parameters(), lr=0.1)` receives the list of parameters once, keeps it, and on every `step()` reads each parameter's `.grad` and updates the parameter in place. That's all an optimizer is. Plain SGD with momentum, in the form PyTorch implements it:

```python
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
```

Note the exact form: the velocity is `momentum * velocity + gradient`, and the learning rate multiplies the velocity. Some textbooks put the learning rate inside the velocity instead; with a constant learning rate the two are equivalent, but they behave differently when the learning rate changes during training, and PyTorch's documentation for `SGD` has a note about exactly this.

Adam keeps two running averages per parameter, of the gradient and of its square, corrects them for starting at zero, and divides one by the square root of the other ([Kingma and Ba, "Adam: A Method for Stochastic Optimization", 2015](https://arxiv.org/abs/1412.6980)<sup>\[17\]</sup>):

```python
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
```

Running both miniatures and `torch.optim` side by side on the same least-squares problem, from the same starting point, for 100 steps: the parameters end up identical (within floating-point tolerance) for plain SGD, SGD with momentum, and Adam. Which means these fifteen-line classes *are* the algorithms, and everything else in `torch.optim.Adam` (weight decay, `amsgrad`, fused GPU kernels, parameter groups with different learning rates) is options around them.

## 6. The training loop, line by line

With all the pieces, the training loop is no longer a ritual. Here it is with the miniature, fitting a line:

```python
import numpy as np

rng = np.random.default_rng(0)
X = np.linspace(-1, 1, 50).reshape(-1, 1)
Y = 3 * X - 2                                    # the line we want to recover

model = MiniLinear(1, 1, rng)
opt = MiniSGD(model.parameters(), lr=0.5)

for step in range(60):
    model.zero_grad()                            # 1. forget the previous gradients
    pred = model(MiniTensor(X))                  # 2. forward: compute, and record the tape
    loss = ((pred - MiniTensor(Y)) ** 2).mean()  # 3. a single number to minimize
    loss.backward()                              # 4. replay the tape: fill every .grad
    opt.step()                                   # 5. update the parameters
    if step in (0, 9, 29, 59):
        print(f"step {step:2d}  loss {float(loss.data):.2e}")

print(model.weight.data.round(4), model.bias.data.round(4))
```

```text
step  0  loss 4.95e+00
step  9  loss 1.20e-03
step 29  loss 4.77e-11
step 59  loss 3.77e-22
[[3.]] [[-2.]]
```

It recovers the line exactly. Every one of the five numbered lines is now something you've built:

1. `zero_grad` exists because leaves accumulate gradients ([section F.3](#3-building-minitensor)).
2. The forward pass builds the tape, fresh, by running ordinary code.
3. The loss is a single number, because reverse mode computes the gradient of *one* output with respect to everything.
4. `backward` walks the tape and fills `.grad` on every parameter.
5. `step` reads those `.grad` and updates the parameters it was given ([section F.5](#5-an-optimizer-is-an-update-rule-over-a-list-of-parameters)), which are only the ones `nn.Module` registered ([section F.4](#4-nnmodule-finds-your-parameters-by-watching-assignments)).

What happens if you skip step 1? In real PyTorch, on the same line (with `nn.Linear(1, 1)`, plain SGD with a learning rate of 0.05, and the same seed), 30 steps with and without `zero_grad` (loss at steps 0, 9, 19 and 29, then the learned weight and bias; the true ones are 3 and −2):

```text
with zero_grad    ([9.5716, 2.6276, 0.9376, 0.4191], [1.957, -1.892])
without zero_grad ([9.5716, 6.6228, 8.9934, 8.1757], [0.497, -4.482])
```

Without zeroing, each step uses the sum of all the gradients so far, so the updates keep growing in the direction of old gradients, overshoot, and the loss goes nowhere. No error, just a model that doesn't learn, and a bug that looks like "bad hyperparameters".

Two more lines you'll see in real loops, explained by the same model: `model.train()` and `model.eval()` switch the behavior of layers that act differently while training (dropout, batch normalization); they don't stop gradients. `with torch.no_grad():` stops the recording of the tape, which saves memory and time during evaluation and inference. They do different things, and evaluation code usually needs both. The next section shows both, on real layers.

## 7. Four things that break real loops

The miniature is float64, single-device and has no layers that change behavior. Real PyTorch has all of that, and each one comes with an error message (or, worse, without one) that makes sense once you have the model of the previous sections in your head.

**dtype.** NumPy's default float is float64; PyTorch's is float32 ([section F.1](#1-a-tensor-is-an-array-you-already-know)). Feed a NumPy array straight into a layer and:

```python
X = np.random.default_rng(0).normal(size=(4, 3))
layer = nn.Linear(3, 1)
print(X.dtype, layer.weight.dtype)
layer(torch.from_numpy(X))
```

```text
float64 torch.float32
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
```

"Double" is float64 and "Float" is float32, in C names. `torch.from_numpy(X).float()` fixes it (that's a copy now, by the way: a float32 tensor can't share memory with a float64 array). Devices have their own version: a model on the GPU and a batch on the CPU give an error that starts with "Expected all tensors to be on the same device". The fix is the same idea, `.to(device)` on both, and the habit that prevents it is to move the batch inside the loop, right next to the forward pass, where you can see it.

**In-place operations.** Look at the `exp` of the miniature again:

```python
    def exp(self):
        out = np.exp(self.data)
        return self._make(out, (self,), lambda g: (g * out,))
```

The backward function uses `out`, the *result* of the forward pass, because the derivative of eˣ is eˣ. So what happens if somebody modifies that result before calling `backward`? In the miniature, nothing visible:

```python
m = MiniTensor([1.0, 2.0, 3.0], requires_grad=True)
e = m.exp()
e.data += 1          # in place, behind the tape's back
e.sum().backward()
print(m.grad.round(4), np.exp([1.0, 2.0, 3.0]).round(4))
```

```text
[ 3.7183  8.3891 21.0855] [ 2.7183  7.3891 20.0855]
```

Every gradient is off by exactly one, and nobody said anything. That's the worst kind of bug, the kind that trains a slightly wrong model for weeks. PyTorch protects you from this one: every tensor carries a version counter that goes up with each in-place operation, the tape remembers the version it saved, and `backward` checks it:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.exp()
print(y._version)     # 0
y.add_(1)             # in place (the trailing underscore)
print(y._version)     # 1
y.sum().backward()
```

```text
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [3]], which is output 0 of Exp, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True, check_nan=False).
```

Read it with the miniature in mind and it says exactly what happened: the output of `Exp` was saved for the backward pass at version 0, and it's now at version 1. Notice it's not "in-place operations are forbidden". With `y = x * 2` instead of `x.exp()`, the same `y.add_(1)` is fine and the gradient comes out as `tensor([2., 2., 2.])`, because the backward of multiplying by 2 doesn't need `y` at all. Whether an in-place operation breaks autograd depends on whether someone saved the tensor you're overwriting, which is something you can now reason about instead of memorizing.

The related one you'll meet when you write an optimizer by hand:

```text
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

That's `w -= 0.1` on a parameter. PyTorch refuses to record an update *of a leaf* on the tape, because the update isn't part of the computation you want to differentiate. Wrapping it in `with torch.no_grad():` says "this isn't part of the model, don't record it", and it works. That's what `optimizer.step()` does internally.

**train() and eval().** Dropout zeroes each value with probability `p` during training, and multiplies the survivors by 1/(1 − p) so the expected value stays the same. In evaluation it does nothing:

```python
torch.manual_seed(0)
drop = nn.Dropout(p=0.5)
ones = torch.ones(10)
print(drop.training, drop(ones))
drop.eval()
print(drop.training, drop(ones))
```

```text
True tensor([0., 0., 2., 0., 0., 0., 2., 2., 0., 2.])
False tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```

`training` is a plain boolean attribute on every module, and `model.eval()` walks the tree of submodules (the same tree `__setattr__` built in [section F.4](#4-nnmodule-finds-your-parameters-by-watching-assignments)) and sets it to `False` on all of them. Forget it, and evaluation is random. With a small network containing a dropout layer, the same input twice in training mode gave 0.0029 and 0.289; in eval mode, 0.1355 both times. If your validation score changes every time you compute it, this is the first thing to look at.

And `eval()` doesn't touch the tape. With the model in eval mode, `net(inp).requires_grad` is still `True`; inside `with torch.no_grad():` it's `False`. One switches layer behavior, the other switches recording, which is why evaluation code usually has both.

**state_dict.** Saving a model in PyTorch means saving a dictionary, and it's worth looking at once:

```python
model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 1))
for k, v in model.state_dict().items():
    print(k, tuple(v.shape))
```

```text
0.weight (4, 3)
0.bias (4,)
1.weight (4,)
1.bias (4,)
1.running_mean (4,)
1.running_var (4,)
1.num_batches_tracked ()
2.weight (1, 4)
2.bias (1,)
```

The keys are the same dotted names `named_parameters` gives (our `MiniModule` builds them the same way), plus three things that aren't parameters: batch normalization's running statistics. They're **buffers**: state that's saved and moved to the GPU with the model, but that the optimizer never sees. This model has 29 parameters and 38 numbers in its `state_dict`. If you save only the parameters, a model with batch normalization comes back with default running statistics and gives different answers in eval mode, which is why the standard recipe is `torch.save(model.state_dict(), path)` and, later, building the same architecture and calling `load_state_dict`:

```text
<All keys matched successfully>
```

After loading, the original and the copy give identical outputs (`torch.equal` returns `True`). Load it into a different architecture and the error is a list of every key that doesn't match, which reads like a diff of the two models:

```text
RuntimeError: Error(s) in loading state_dict for Sequential:
	Unexpected key(s) in state_dict: "2.weight", "2.bias", "1.running_mean", "1.running_var", "1.num_batches_tracked". 
	size mismatch for 1.weight: copying a param with shape torch.Size([4]) from checkpoint, the shape in current model is torch.Size([1, 4]).
	size mismatch for 1.bias: copying a param with shape torch.Size([4]) from checkpoint, the shape in current model is torch.Size([1]).
```

(That was a model without the batch norm layer, so its second `Linear` got the name `1`, and PyTorch tried to put the batch norm's weights into it.) To resume *training*, also save `optimizer.state_dict()`: for Adam it holds the `m` and `v` of every parameter, the same lists our `MiniAdam` keeps, and without them the first steps after resuming behave as if training had just started.

## 8. Dataset and DataLoader

The last piece that ends up in every training script. A **Dataset** is any object with `__len__` and `__getitem__`; that's the entire interface. A **DataLoader** takes a dataset and handles everything around it: choosing the order of the indices (shuffled or not), grouping them into batches, fetching each sample, and stacking the samples of a batch into tensors (that last step is called **collate**). Optionally, it does the fetching in parallel worker processes.

The miniature (without the workers):

```python
class MiniDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, generator=None):
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.drop_last, self.generator = shuffle, drop_last, generator

    def _indices(self):
        n = len(self.dataset)
        if not self.shuffle:
            return list(range(n))
        return torch.randperm(n, generator=self.generator).tolist()

    def __iter__(self):
        if self.generator is not None:
            # torch draws a base seed for its worker processes first (see below)
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
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
```

The test compares it batch by batch with `torch.utils.data.DataLoader`, using generators with the same seed: the same batches, with the same samples, in the same order, for several batch sizes, with and without `drop_last`.

That line in `__iter__` about the base seed has a story, and it's the method of [section B.7](B-a-method-for-learning-any-library.md#7-the-rebuild-and-check-loop) doing its job. The first version of the miniature didn't have it: it shuffled with `torch.randperm` using the generator, which is what the sampler does, and the test failed, with the same samples in a different order. Reading the source of the real loader (`inspect.getsource(torch.utils.data.dataloader._BaseDataLoaderIter.__init__)`) showed why: before the sampler shuffles anything, the loader's iterator draws one random number from the *same* generator, as the base seed for its worker processes. One draw shifts the generator, which changes the permutation. Adding that draw made the test pass.

It matters beyond the miniature, because it's a real fact about PyTorch that affects reproducibility: if you share one generator between a DataLoader and anything else, the order of your batches depends on how many times the other thing drew from it. Give the DataLoader its own generator (`torch.Generator().manual_seed(...)`), and the order is reproducible. PyTorch's notes on reproducibility, and on seeding workers, are written for exactly this.

## 9. Checking the miniature against PyTorch

```text
359 checks against PyTorch 2.15.0: all passed
```

What's in those checks:

- 300 random expressions combining addition, subtraction, multiplication, ReLU, `exp` and squaring, on inputs of different shapes that broadcast against each other: the gradients from `MiniTensor` match `torch.autograd` (`torch.testing.assert_close`);
- 50 small two-layer networks (matrix multiplication, bias, ReLU, mean squared error): same loss and same gradient for every parameter;
- gradient accumulation across two calls to `backward`;
- parameter registration: nested modules found with dotted names, layers in a plain list ignored, like PyTorch;
- SGD, SGD with momentum and Adam: the same parameters as `torch.optim` after 100 steps;
- data loading: the same batches as `DataLoader`, in the same order, for four configurations.

For your own custom layers in real PyTorch, the tool for the first two checks is `torch.autograd.gradcheck`, which compares your backward pass with numerical derivatives computed by finite differences (use `float64` inputs, because finite differences in `float32` are too noisy).

## 10. Reading the PyTorch docs with this in your head

PyTorch splits its documentation across tutorials (pytorch.org/tutorials) and reference docs, plus a set of **notes** that are the explanation pages of [section B.2](B-a-method-for-learning-any-library.md#2-documentation-comes-in-four-kinds-and-each-answers-a-different-question) and the most valuable part of the documentation:

- "Autograd mechanics": how the graph is recorded, leaves, `requires_grad`, in-place operations and why they can break backward, and `no_grad` versus `inference_mode`.
- "Broadcasting semantics": the same rules as NumPy's, stated for tensors, including in-place operations.
- `torch.nn.Module`: read the list of methods once (`register_buffer`, `state_dict`, `named_modules`, `apply`, hooks). A lot of what ends up hand-written in model code already exists there.
- `torch.utils.data`: map-style and iterable datasets, samplers, `collate_fn`, workers and pinned memory.
- "Reproducibility": seeds, deterministic algorithms, and DataLoader workers.

When something behaves strangely, the same habits from [chapter B](B-a-method-for-learning-any-library.md) apply: `t.shape`, `t.dtype`, `t.device`, `t.stride()`, `t.requires_grad`, `t.grad_fn`, and `inspect.getsource` on the module that surprised you. PyTorch's Python-level code (modules, optimizers, data loading) is very readable; the operations themselves are in C++ and CUDA, and you rarely need to go there.

## 11. Exercises

1. Add `__truediv__` and `tanh` to `MiniTensor`, and add them to the random gradient test.
2. Implement `MiniTensor.sum(axis, keepdims)`. What does its backward look like? (Hint: it's broadcasting, in reverse of `_unbroadcast`.)
3. Write a two-layer network with `MiniModule` and `MiniLinear`, train it on `y = sin(3x)` with `MiniAdam`, and compare the final loss with the same network in real PyTorch.
4. Add `weight_decay` to `MiniSGD` the way PyTorch does it (added to the gradient before momentum), and check it against `torch.optim.SGD(..., weight_decay=...)`.
5. Use `torch.autograd.gradcheck` on a custom `torch.autograd.Function` that implements `x * sigmoid(x)` with your own backward. Then break the backward on purpose and read the error.
6. Give `MiniModule` a `training` attribute and `train()`/`eval()` methods that walk the tree of submodules, and write a `MiniDropout`. Check that with `p=0.5` the average of many outputs is close to the input in training mode, and equal to it in eval mode.
7. Write `state_dict()` and `load_state_dict()` for `MiniModule`, using `named_parameters`. Save a trained `MiniLinear`, load it into a fresh one, and check that both give the same predictions.

**Where this shows up in No Shortcuts:** lesson 24.3, "Building an autograd engine", builds a fuller version of [section F.3](#3-building-minitensor) before the course touches PyTorch, and Module 25 covers PyTorch itself: tensors and autograd, modules and the training loop, data loading, GPUs, mixed precision and reproducibility.

---

[← E. scikit-learn in miniature](E-scikit-learn-in-miniature.md) · [Contents](../README.md#contents) · [G. sentence-transformers in miniature →](G-sentence-transformers-in-miniature.md)
