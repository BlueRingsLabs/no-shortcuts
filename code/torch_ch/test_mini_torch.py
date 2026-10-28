"""Check the PyTorch miniature against PyTorch: gradients, module parameters, optimizers, data loading."""
import random

import numpy as np
import torch

from mini_torch import (MiniAdam, MiniDataLoader, MiniLinear, MiniModule, MiniSGD, MiniTensor,
                        Parameter)

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)
pyrng = random.Random(0)
checks = 0

UNARY = ["relu", "exp", "square", "neg"]
BINARY = ["add", "mul", "sub"]


def apply_unary(op, a, mini):
    if op == "relu":
        return a.relu() if mini else torch.relu(a)
    if op == "exp":
        return (a * 0.1).exp() if mini else torch.exp(a * 0.1)
    if op == "square":
        return a ** 2
    return -a


def apply_binary(op, a, b):
    return {"add": lambda: a + b, "mul": lambda: a * b, "sub": lambda: a - b}[op]()


# 1. gradients of random expressions, with broadcasting, against torch.autograd
for _ in range(300):
    shapes = pyrng.choice([((3, 4), (3, 4)), ((3, 4), (1, 4)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 3), (3,))])
    xa, xb = rng.normal(size=shapes[0]), rng.normal(size=shapes[1])
    ma, mb = MiniTensor(xa, requires_grad=True), MiniTensor(xb, requires_grad=True)
    ta, tb = torch.tensor(xa, requires_grad=True), torch.tensor(xb, requires_grad=True)
    m_out, t_out = ma, ta
    ops = [(pyrng.choice(BINARY), pyrng.choice(UNARY)) for _ in range(pyrng.randint(1, 3))]
    for bop, uop in ops:
        m_out = apply_unary(uop, apply_binary(bop, m_out, mb), True)
        t_out = apply_unary(uop, apply_binary(bop, t_out, tb), False)
    m_out.sum().backward()
    t_out.sum().backward()
    torch.testing.assert_close(torch.tensor(ma.grad), ta.grad)
    torch.testing.assert_close(torch.tensor(mb.grad), tb.grad)
    checks += 1

# 2. a small MLP with matmul, relu, mean squared error: same loss and same gradients as torch
for _ in range(50):
    X, y = rng.normal(size=(16, 5)), rng.normal(size=(16, 1))
    W1, b1, W2, b2 = rng.normal(size=(5, 8)), rng.normal(size=(1, 8)), rng.normal(size=(8, 1)), rng.normal(size=(1, 1))
    mp = [MiniTensor(a, requires_grad=True) for a in (W1, b1, W2, b2)]
    tp = [torch.tensor(a, requires_grad=True) for a in (W1, b1, W2, b2)]
    m_loss = ((((MiniTensor(X) @ mp[0] + mp[1]).relu() @ mp[2] + mp[3]) - MiniTensor(y)) ** 2).mean()
    t_loss = ((((torch.tensor(X) @ tp[0] + tp[1]).relu() @ tp[2] + tp[3]) - torch.tensor(y)) ** 2).mean()
    m_loss.backward()
    t_loss.backward()
    torch.testing.assert_close(torch.tensor(m_loss.data), t_loss.detach())
    for m, t in zip(mp, tp):
        torch.testing.assert_close(torch.tensor(m.grad), t.grad)
    checks += 1

# 3. gradients accumulate across backward() calls, like torch
x = MiniTensor([2.0, 3.0], requires_grad=True)
(x * x).sum().backward()
(x * x).sum().backward()
assert np.allclose(x.grad, [8.0, 12.0])
checks += 1


# 4. module registration: nested modules are found, a plain list is not (like torch)
class Net(MiniModule):
    def __init__(self):
        super().__init__()
        self.first = MiniLinear(3, 4, rng)
        self.second = MiniLinear(4, 1, rng)
        self.hidden = [MiniLinear(4, 4, rng)]        # a plain list: invisible


assert [n for n, _ in Net().named_parameters()] == ["first.weight", "first.bias", "second.weight", "second.bias"]
checks += 1

# 5. optimizers: identical trajectories to torch.optim on the same problem
for opt_name in ["sgd", "momentum", "adam"]:
    X, y = rng.normal(size=(32, 3)), rng.normal(size=(32, 1))
    w0 = rng.normal(size=(3, 1))
    mw, tw = Parameter(w0.copy()), torch.tensor(w0.copy(), requires_grad=True)
    if opt_name == "sgd":
        mo, to = MiniSGD([mw], lr=0.1), torch.optim.SGD([tw], lr=0.1)
    elif opt_name == "momentum":
        mo, to = MiniSGD([mw], lr=0.05, momentum=0.9), torch.optim.SGD([tw], lr=0.05, momentum=0.9)
    else:
        mo, to = MiniAdam([mw], lr=0.05), torch.optim.Adam([tw], lr=0.05)
    for _ in range(100):
        mw.grad = None
        to.zero_grad()
        ((MiniTensor(X) @ mw - MiniTensor(y)) ** 2).mean().backward()
        ((torch.tensor(X) @ tw - torch.tensor(y)) ** 2).mean().backward()
        mo.step()
        to.step()
    torch.testing.assert_close(torch.tensor(mw.data), tw.detach())
    checks += 1


# 6. data loading: same batches, in the same order, as torch's DataLoader with the same seed
class Squares:
    def __len__(self):
        return 10

    def __getitem__(self, i):
        return np.array([float(i)]), np.array(float(i * i))


for batch_size, drop_last in [(3, False), (3, True), (4, False), (10, False)]:
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    mini = list(MiniDataLoader(Squares(), batch_size, shuffle=True, drop_last=drop_last, generator=g1))
    real = list(torch.utils.data.DataLoader(Squares(), batch_size=batch_size, shuffle=True,
                                            drop_last=drop_last, generator=g2))
    assert len(mini) == len(real)
    for (mx, my), (tx, ty) in zip(mini, real):
        torch.testing.assert_close(torch.tensor(mx), tx)
        torch.testing.assert_close(torch.tensor(my), ty)
    checks += 1

print(f"{checks} checks against PyTorch {torch.__version__.split('+')[0]}: all passed")
