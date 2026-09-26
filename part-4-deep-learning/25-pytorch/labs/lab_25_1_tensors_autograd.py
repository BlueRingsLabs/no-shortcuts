# %% [markdown]
# # Lab 25.1: Tensors and autograd without surprises
#
# 1. dtypes and the NumPy boundary: float64 vs float32, from_numpy sharing memory.
# 2. Views, strides, contiguity; expand vs repeat.
# 3. In-place operations vs autograd: the version-counter error, provoked and understood.
# 4. .grad accumulates: the zero_grad self-check, and gradient accumulation used on purpose.
# 5. retain_graph, non-scalar backward, autograd.grad and higher-order derivatives.
# 6. A custom autograd.Function (stable softplus) checked with gradcheck; a straight-through estimator.
# 7. Per-example gradients: a loop vs torch.func.vmap(grad(...)).

# %%
import time

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

torch.manual_seed(251)

# %% [markdown]
# ## 1. dtypes and the NumPy boundary

# %%
a_np = np.random.default_rng(0).normal(size=(3, 3))
t = torch.from_numpy(a_np)
print(f"numpy default {a_np.dtype} -> torch.from_numpy gives {t.dtype}; torch.randn gives {torch.randn(1).dtype}")
layer = torch.nn.Linear(3, 2)
try:
    layer(t)
    print("no error?")
except RuntimeError as e:
    print("float64 input into a float32 layer:", str(e).splitlines()[0][:90])
t[0, 0] = 999.0
print(f"after writing to the tensor, the NumPy array's [0, 0] is {a_np[0, 0]} (shared memory)")
assert a_np[0, 0] == 999.0
safe = torch.tensor(a_np, dtype=torch.float32)                  # torch.tensor copies (and converts)
safe[0, 0] = -1.0
assert a_np[0, 0] == 999.0

# %% [markdown]
# ## 2. Views, strides, contiguity

# %%
x = torch.arange(12.0).reshape(3, 4)
views = {"x": x, "x.T": x.T, "x[:, 1:3]": x[:, 1:3], "expand": x[:, None, :].expand(3, 5, 4)}
for name, v in views.items():
    shares = v.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    print(f"{name:10s} shape {tuple(v.shape)!s:12s} strides {v.stride()!s:12s} contiguous {v.is_contiguous()!s:5s} shares memory {shares}")
try:
    x.T.view(12)
except RuntimeError as e:
    print("x.T.view(12):", str(e).splitlines()[0][:80], "...")
print("x.T.reshape(12) works (it copies):", x.T.reshape(12)[:5].tolist())

sl = x[:, 1:3]
sl *= 0                                                         # in-place on a view
print("after zeroing a slice view, x =\n", x)
assert x[0, 1] == 0

base = torch.randn(1, 1000)
expanded, repeated = base.expand(1000, 1000), base.repeat(1000, 1)
print(f"expand uses {expanded.untyped_storage().nbytes() / 1e6:.3f} MB, repeat uses {repeated.untyped_storage().nbytes() / 1e6:.1f} MB")
assert expanded.stride()[0] == 0

# %% [markdown]
# ## 3. In-place operations vs autograd

# %%
a = torch.randn(5, requires_grad=True)
b = a.exp()
b.add_(1)                                                       # overwrites exp's output, which exp's backward needs
try:
    b.sum().backward()
    print("no error?")
    raised = False
except RuntimeError as e:
    print("in-place after exp:", str(e).splitlines()[0][:110], "...")
    raised = True
assert raised
a.grad = None
c = a * 2
c.add_(1)                                                       # mul's backward only needs the constant 2: fine
c.sum().backward()
print("in-place after a * 2 works; a.grad =", a.grad.tolist())

# %% [markdown]
# ## 4. .grad accumulates

# %%
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
xin = torch.tensor([0.5, -1.0, 2.0])
for step in range(3):
    loss = (w * xin).sum()
    loss.backward()
    print(f"step {step}: w.grad = {w.grad.tolist()} (no zero_grad: gradients pile up)")
assert torch.allclose(w.grad, 3 * xin)

# gradient accumulation on purpose: 4 micro-batches of 8 == 1 batch of 32
model = torch.nn.Sequential(torch.nn.Linear(10, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
X, y = torch.randn(32, 10), torch.randn(32, 1)
model.zero_grad()
torch.nn.functional.mse_loss(model(X), y).backward()
full = [p.grad.clone() for p in model.parameters()]
model.zero_grad()
for chunk in range(4):
    sl_ = slice(8 * chunk, 8 * chunk + 8)
    (torch.nn.functional.mse_loss(model(X[sl_]), y[sl_]) / 4).backward()      # scale so the sum is the mean
same = all(torch.allclose(g, p.grad, atol=1e-6) for g, p in zip(full, model.parameters()))
print(f"4 accumulated micro-batches reproduce the full-batch gradient: {same}")
assert same

# %% [markdown]
# ## 5. retain_graph, non-scalar backward, higher order

# %%
p = torch.tensor(2.0, requires_grad=True)
shared = p**3
l1, l2 = shared * 2, shared + 1
l1.backward(retain_graph=True)                                  # the graph under `shared` is needed again
l2.backward()
print(f"two losses sharing a subgraph: p.grad = {p.grad.item()} (= 2*3p^2 + 3p^2 = {9 * 2.0**2})")
try:
    l1.backward()
except RuntimeError as e:
    print("third backward without retain_graph:", str(e).splitlines()[0][:80], "...")

v = torch.randn(3, requires_grad=True)
y3 = v**2
y3.backward(torch.tensor([1.0, 0.0, 10.0]))                     # vector-Jacobian product, J = diag(2v)
assert torch.allclose(v.grad, torch.tensor([1.0, 0.0, 10.0]) * 2 * v.detach())
print("non-scalar backward(gradient=u) computes u^T J:", v.grad.tolist())

q = torch.tensor(1.5, requires_grad=True)
f = q**4
(g1,) = torch.autograd.grad(f, q, create_graph=True)
(g2,) = torch.autograd.grad(g1, q)
print(f"d/dq q^4 = {g1.item()} (4q^3 = {4 * 1.5**3}); second derivative {g2.item()} (12q^2 = {12 * 1.5**2})")
assert np.isclose(g2.item(), 12 * 1.5**2)

# %% [markdown]
# ## 6. Custom autograd Functions

# %%
class StableSoftplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.clamp(x, min=0) + torch.log1p(torch.exp(-torch.abs(x)))   # log(1 + e^x) without overflow
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.sigmoid(x)


xs = torch.tensor([-800.0, -5.0, 0.0, 5.0, 800.0])
naive = torch.log(1 + torch.exp(xs))
print(f"naive softplus: {naive.tolist()}; stable: {StableSoftplus.apply(xs).tolist()}")
assert torch.isinf(naive[-1]) and torch.isfinite(StableSoftplus.apply(xs)).all()
assert torch.autograd.gradcheck(StableSoftplus.apply, (torch.randn(20, dtype=torch.float64, requires_grad=True),))
print("gradcheck(StableSoftplus) passed")


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g                                                # pretend round was the identity


xr = torch.randn(10, dtype=torch.float64, requires_grad=True)
try:
    ok = torch.autograd.gradcheck(RoundSTE.apply, (xr,), raise_exception=False)
except Exception:
    ok = False
print(f"gradcheck on the straight-through estimator: {ok} (expected: it's a deliberate lie that lets gradients through)")
assert not ok

# %% [markdown]
# ## 7. Per-example gradients

# %%
net = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
Xb, yb = torch.randn(256, 20), torch.randn(256, 1)
params = {k: v.detach() for k, v in net.named_parameters()}


def loss_one(prm, x, y):
    out = functional_call(net, prm, (x[None],))
    return torch.nn.functional.mse_loss(out, y[None])


t0 = time.perf_counter()
loop = []
for i in range(256):
    net.zero_grad()
    torch.nn.functional.mse_loss(net(Xb[i:i + 1]), yb[i:i + 1]).backward()
    loop.append(net[0].weight.grad.clone())
t_loop = time.perf_counter() - t0
per_example_grad = vmap(grad(loss_one), in_dims=(None, 0, 0))
per_example_grad(params, Xb, yb)                                # first call pays one-time setup costs
t0 = time.perf_counter()
per_ex = per_example_grad(params, Xb, yb)
t_vmap = time.perf_counter() - t0
match = torch.allclose(torch.stack(loop), per_ex["0.weight"], atol=1e-6)
print(f"256 per-example gradients: loop {t_loop * 1000:.0f} ms, vmap(grad) {t_vmap * 1000:.1f} ms (after warm-up); identical: {match}")
assert match

print("\nAll checks passed.")
