# %% [markdown]
# # Lab 26.1: Optimizers, implemented and compared
#
# 1. SGD, momentum, Nesterov, RMSProp, Adam and AdamW in a few lines each, matched against torch.optim step for step.
# 2. An ill-conditioned quadratic (kappa = 100) and the Rosenbrock valley: steps to converge, each with its best learning rate.
# 3. Adam's bias correction: the first steps with and without it.
# 4. L2 in Adam vs decoupled weight decay (AdamW): which weights actually shrink.
# 5. A real network: each optimizer with a tuned learning rate.

# %%
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_default_dtype(torch.float64)
torch.manual_seed(261)

# %% [markdown]
# ## 1. The optimizers

# %%
class Opt:
    def __init__(self, params, lr, kind, mu=0.9, beta2=0.999, eps=1e-8, wd=0.0, nesterov=False, bias_correction=True):
        self.p, self.lr, self.kind, self.mu, self.b2, self.eps, self.wd = list(params), lr, kind, mu, beta2, eps, wd
        self.nesterov, self.bc, self.t = nesterov, bias_correction, 0
        self.m = [torch.zeros_like(q) for q in self.p]
        self.s = [torch.zeros_like(q) for q in self.p]

    @torch.no_grad()
    def step(self):
        self.t += 1
        for q, m, s in zip(self.p, self.m, self.s):
            g = q.grad
            if self.kind == "sgd":
                q -= self.lr * g
            elif self.kind == "momentum":
                m.mul_(self.mu).add_(g)                          # v = mu v + g
                q -= self.lr * (g + self.mu * m if self.nesterov else m)
            elif self.kind == "rmsprop":
                s.mul_(self.b2).add_((1 - self.b2) * g * g)
                q -= self.lr * g / (s.sqrt() + self.eps)
            elif self.kind in ("adam", "adam_l2", "adamw"):
                if self.kind == "adam_l2":
                    g = g + self.wd * q                          # L2: the penalty's gradient joins g
                if self.kind == "adamw":
                    q -= self.lr * self.wd * q                   # decoupled: shrink directly
                m.mul_(self.mu).add_((1 - self.mu) * g)
                s.mul_(self.b2).add_((1 - self.b2) * g * g)
                mh = m / (1 - self.mu**self.t) if self.bc else m
                sh = s / (1 - self.b2**self.t) if self.bc else s
                q -= self.lr * mh / (sh.sqrt() + self.eps)

    def zero_grad(self):
        for q in self.p:
            q.grad = None


def make_net(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.Tanh(), torch.nn.Linear(32, 3))


X, y = torch.randn(64, 10), torch.randint(0, 3, (64,))
pairs = [
    ("SGD", lambda p: Opt(p, 0.1, "sgd"), lambda p: torch.optim.SGD(p, lr=0.1)),
    ("momentum", lambda p: Opt(p, 0.05, "momentum"), lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    ("Nesterov", lambda p: Opt(p, 0.05, "momentum", nesterov=True), lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True)),
    ("RMSProp", lambda p: Opt(p, 0.01, "rmsprop", beta2=0.99), lambda p: torch.optim.RMSprop(p, lr=0.01, alpha=0.99, eps=1e-8)),
    ("Adam", lambda p: Opt(p, 0.01, "adam"), lambda p: torch.optim.Adam(p, lr=0.01)),
    ("Adam + L2", lambda p: Opt(p, 0.01, "adam_l2", wd=0.1), lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.1)),
    ("AdamW", lambda p: Opt(p, 0.01, "adamw", wd=0.1), lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1)),
]
for name, mine_f, ref_f in pairs:
    a, b = make_net(), make_net()
    oa, ob = mine_f(a.parameters()), ref_f(b.parameters())
    for _ in range(50):
        for net, o in ((a, oa), (b, ob)):
            o.zero_grad(); torch.nn.functional.cross_entropy(net(X), y).backward(); o.step()
    diff = max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
    print(f"{name:10s}: max parameter difference vs torch.optim after 50 steps {diff:.1e}")
    assert diff < 1e-10, name

# %% [markdown]
# ## 2. Landscapes you can see

# %%
th = np.pi / 6                                                    # condition number 100, rotated 30 degrees: the steep
R = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])  # direction isn't aligned with an axis (it rarely is)
H = R @ torch.diag(torch.tensor([100.0, 1.0])) @ R.T
quad = lambda w: 0.5 * w @ H @ w
rosen = lambda w: (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2


def steps_to_converge(f, start, target, make_opt, max_steps=5_000, tol=1e-6):
    w = torch.tensor(start, requires_grad=True)
    o = make_opt([w])
    for t in range(max_steps):
        o.zero_grad(); f(w).backward(); o.step()
        if not torch.isfinite(w).all():
            return max_steps
        if (w.detach() - torch.tensor(target)).norm() < tol:
            return t + 1
    return max_steps


lrs = np.logspace(-4, 0, 9)
for fname, f, start, target in (("quadratic, kappa=100", quad, [1.0, 0.3], [0.0, 0.0]), ("Rosenbrock", rosen, [-1.2, 1.0], [1.0, 1.0])):
    print(f"\n{fname}: fewest steps to within 1e-6 of the minimum (best learning rate from a grid)")
    best = {}
    for name, kind, kw in (("GD", "sgd", {}), ("momentum", "momentum", {}), ("Nesterov", "momentum", {"nesterov": True}), ("Adam", "adam", {})):
        results = [(steps_to_converge(f, start, target, lambda p: Opt(p, lr, kind, **kw)), lr) for lr in lrs]
        best[name] = min(results)
        shown = f"{best[name][0]:6d}" if best[name][0] < 5000 else " >5000"
        print(f"  {name:9s} {shown} steps (lr = {best[name][1]:.1e})")
    if fname.startswith("quadratic"):
        assert best["momentum"][0] < best["GD"][0] / 3, "momentum fixes ill-conditioning"
        print("  Adam's per-coordinate scaling is diagonal: it can't undo curvature that isn't aligned with the axes, so on this")
        print("  rotated valley it's no better than momentum (align the valley with an axis and Adam looks like magic)")

# %% [markdown]
# ## 3. Adam's bias correction

# %%
w_bc, w_nobc = torch.tensor([1.0], requires_grad=True), torch.tensor([1.0], requires_grad=True)
o_bc, o_nobc = Opt([w_bc], 0.01, "adam"), Opt([w_nobc], 0.01, "adam", bias_correction=False)
print("\nstep   update with correction   update without")
for t in range(1, 6):
    before = (w_bc.item(), w_nobc.item())
    for w, o in ((w_bc, o_bc), (w_nobc, o_nobc)):
        o.zero_grad(); (3.0 * w).sum().backward(); o.step()                  # constant gradient 3
    print(f"{t:4d}   {before[0] - w_bc.item():.5f}                  {before[1] - w_nobc.item():.5f}")
first_uncorrected = 0.01 * 0.1 * 3 / np.sqrt(0.001 * 9)
print(f"(without correction the first step is lr * 0.1 / sqrt(0.001) = {first_uncorrected / 0.01:.2f} x lr: too big, and it stays off for hundreds of steps)")

# %% [markdown]
# ## 4. L2 in Adam vs decoupled weight decay
#
# Two groups of weights that don't affect the loss much: one gets large, noisy gradients, the other tiny ones.
# Which ones does each method actually shrink?

# %%
def run_decay(kind, steps=2000, wd=0.1, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.ones(2000, requires_grad=True)
    noise_scale = torch.cat([torch.full((1000,), 10.0), torch.full((1000,), 0.01)])   # large-gradient vs small-gradient weights
    o = Opt([w], lr, kind, wd=wd)
    for _ in range(steps):
        o.zero_grad()
        w.grad = noise_scale * torch.randn(2000, generator=g)   # zero-mean gradient noise: the loss doesn't care about w
        o.step()
    return w.detach()[:1000].abs().mean().item(), w.detach()[1000:].abs().mean().item()


for kind in ("adam_l2", "adamw"):
    big, small = run_decay(kind)
    print(f"{kind:8s}: mean |w| after 2000 steps (started at 1): large-gradient weights {big:.3f}, small-gradient weights {small:.3f}")
l2_big, l2_small = run_decay("adam_l2")
w_big, w_small = run_decay("adamw")
print("Adam + L2 barely decays the weights with noisy gradients (the decay is divided by their large sqrt(s));")
print("AdamW shrinks every weight by the same factor, (1 - lr * wd) per step, whatever its gradients look like")
assert l2_big > 0.95 and l2_small < 0.5 and abs(w_big - w_small) < 0.05

# %% [markdown]
# ## 5. A real network, each optimizer with its own best learning rate

# %%
digits = load_digits()
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(digits.data / 16, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
Xd_tr, Xd_te, yd_tr, yd_te = (torch.tensor(a) for a in (Xd_tr, Xd_te, yd_tr, yd_te))


def train_digits(make_opt, epochs=8, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
    o = make_opt(net.parameters())
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xd_tr), generator=g)
        for s in range(0, len(Xd_tr), 64):
            idx = perm[s:s + 64]
            o.zero_grad(); torch.nn.functional.cross_entropy(net(Xd_tr[idx]), yd_tr[idx]).backward(); o.step()
    with torch.no_grad():
        return torch.nn.functional.cross_entropy(net(Xd_tr), yd_tr).item(), (net(Xd_te).argmax(1) == yd_te).double().mean().item()


print("\noptimizer   best lr    train loss   test acc    (worst lr in the grid: train loss)")
grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
summary = {}
for name, kind, kw in (("SGD", "sgd", {}), ("momentum", "momentum", {}), ("RMSProp", "rmsprop", {"beta2": 0.99}), ("Adam", "adam", {})):
    res = [(train_digits(lambda p: Opt(p, lr, kind, **kw)), lr) for lr in grid]
    (loss, acc), lr = min(res, key=lambda r: r[0][0])
    worst = max(r[0][0] for r in res)
    summary[name] = loss
    print(f"{name:10s}  {lr:7.0e}    {loss:.4f}       {acc:.3f}      ({worst:.3f})")
print("at this budget plain SGD lags; among momentum and the adaptive methods, the tuned learning rate matters more than the choice,")
print("and a bad learning rate (last column) wrecks every one of them")
assert summary["Adam"] < summary["SGD"] and summary["momentum"] < summary["SGD"]

print("\nAll checks passed.")
