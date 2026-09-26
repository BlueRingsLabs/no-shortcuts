# %% [markdown]
# # Lab 26.2: Initialization, learning rates and schedules
#
# 1. Symmetry: zero-initialized hidden units stay clones of each other.
# 2. Signal through 50 layers: too small, Xavier, He, too big; tanh and ReLU; forward and backward.
# 3. A 30-layer ReLU MLP: PyTorch default vs He init, same everything else.
# 4. The learning rate range test.
# 5. Schedules on a fixed budget: constant, step, cosine, warmup + cosine, one-cycle.
# 6. Batch size and learning rate: scaling the step with the batch.
# 7. The update-to-weight ratio as a health signal.

# %%
import math

import matplotlib

matplotlib.use("Agg")
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

torch.manual_seed(262)
OUT = Path(__file__).with_name("outputs"); OUT.mkdir(exist_ok=True)
Xn, yn = make_classification(n_samples=6000, n_features=40, n_informative=20, n_classes=5, n_clusters_per_class=2, class_sep=1.0, random_state=0)
X_tr, X_te, y_tr, y_te = (torch.tensor(a) for a in train_test_split(Xn, yn, test_size=0.25, random_state=0))
X_tr, X_te = X_tr.float(), X_te.float()

# %% [markdown]
# ## 1. Symmetry

# %%
net = nn.Sequential(nn.Linear(40, 8), nn.Tanh(), nn.Linear(8, 5))
for layer in (net[0], net[2]):
    nn.init.constant_(layer.weight, 0.1); nn.init.zeros_(layer.bias)
opt = torch.optim.SGD(net.parameters(), lr=0.1)
for _ in range(200):
    opt.zero_grad(); nn.functional.cross_entropy(net(X_tr[:512]), y_tr[:512]).backward(); opt.step()
W = net[0].weight.detach()
print(f"8 hidden units after 200 steps from identical init: max difference between any two units' weights {(W - W[0]).abs().max():.2e}")
assert (W - W[0]).abs().max() < 1e-6

# %% [markdown]
# ## 2. Signal through 50 layers

# %%
def propagate(act, gain, depth=50, width=256, n=512):
    x = torch.randn(n, width, requires_grad=True)
    h, stds = x, []
    for _ in range(depth):
        W = torch.randn(width, width) * math.sqrt(gain / width)
        h = act(h @ W)
        stds.append(h.std().item())
    h.sum().backward()
    return stds, x.grad.std().item()


fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
results = {}
for ax, (aname, act) in zip(axes, (("tanh", torch.tanh), ("ReLU", torch.relu))):
    for label, gain in (("0.5/n (too small)", 0.5), ("1/n (Xavier)", 1.0), ("2/n (He)", 2.0), ("4/n (too big)", 4.0)):
        stds, gstd = propagate(act, gain)
        results[(aname, gain)] = (stds[-1], gstd)
        ax.semilogy(stds, label=label)
        print(f"{aname:5s} Var(w) = {label:18s}: activation std at layer 50 {stds[-1]:9.2e}, gradient std at the input {gstd:9.2e}")
    ax.set_title(aname); ax.set_xlabel("layer"); ax.set_ylabel("activation std")
axes[0].legend(fontsize=8)
fig.savefig(OUT / "signal_propagation.png", dpi=110); plt.close(fig)
relu_he = results[("ReLU", 2.0)][0]; relu_x = results[("ReLU", 1.0)][0]
assert 0.1 < relu_he < 10 and relu_x < 1e-4, "ReLU needs He's factor 2; Xavier's 1/n loses half the signal per layer"
assert 0.05 < results[("tanh", 1.0)][0] < 1 and results[("tanh", 0.5)][0] < 1e-5

# %% [markdown]
# ## 3. A 30-layer ReLU network: default vs He init

# %%
def deep_mlp(init):
    torch.manual_seed(0)
    layers = []
    for i in range(30):
        lin = nn.Linear(40 if i == 0 else 128, 128)
        if init == "he":
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu"); nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(128, 5))


def train_steps(model, lr, steps=600, bs=128, sched=None, opt_name="sgd", seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) if opt_name == "sgd" else torch.optim.AdamW(model.parameters(), lr=lr)
    s = sched(opt) if sched else None
    losses = []
    for _ in range(steps):
        idx = torch.randint(0, len(X_tr), (bs,), generator=g)
        opt.zero_grad(); loss = nn.functional.cross_entropy(model(X_tr[idx]), y_tr[idx]); loss.backward(); opt.step()
        if s:
            s.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    return np.mean(losses[-50:]), acc, losses


for init in ("default", "he"):
    m = deep_mlp(init)
    with torch.no_grad():
        out_std = m(X_tr[:512]).std().item()
    loss, acc, _ = train_steps(m, 0.01)
    print(f"30-layer ReLU MLP, {init:7s} init: output std at init {out_std:.2e}; after 600 steps loss {loss:.3f}, test accuracy {acc:.3f}")
    if init == "default":
        default_loss = loss
    else:
        he_loss = loss
assert he_loss < default_loss - 0.2

# %% [markdown]
# ## 4. Learning rate range test

# %%
def make_mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(40, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 5))


m = make_mlp()
opt = torch.optim.SGD(m.parameters(), lr=1e-5, momentum=0.9)
lrs = np.logspace(-5, 1, 300)
g = torch.Generator().manual_seed(0)
range_losses = []
for lr in lrs:
    for pg in opt.param_groups:
        pg["lr"] = lr
    idx = torch.randint(0, len(X_tr), (128,), generator=g)
    opt.zero_grad(); loss = nn.functional.cross_entropy(m(X_tr[idx]), y_tr[idx]); loss.backward(); opt.step()
    range_losses.append(loss.item() if math.isfinite(loss.item()) else 1e3)
smooth = np.convolve(range_losses, np.ones(15) / 15, mode="same")
i_min = int(np.argmin(smooth[10:-10])) + 10
diverge = next((i for i in range(i_min, len(lrs)) if smooth[i] > 2 * smooth[i_min] + 0.5), len(lrs) - 1)
suggested = lrs[i_min] / 3
print(f"range test: loss lowest near lr = {lrs[i_min]:.2e}, blows up from lr ~ {lrs[diverge]:.2e}; suggested lr ~ {suggested:.2e}")
fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
ax.semilogx(lrs, np.clip(smooth, 0, 5)); ax.axvline(suggested, ls="--", color="0.5"); ax.set_xlabel("learning rate"); ax.set_ylabel("loss (smoothed)")
fig.savefig(OUT / "lr_range_test.png", dpi=110); plt.close(fig)
assert 1e-3 < suggested < 1

# %% [markdown]
# ## 5. Schedules on a fixed budget

# %%
STEPS, PEAK = 1500, 0.1                                       # near the loss minimum of the range test


def warmup_cosine(warm):
    return lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(t, STEPS) / STEPS)))


schedules = {
    "constant (peak)": (PEAK, None),
    "constant (peak / 10)": (PEAK / 10, None),
    "step decay /10 at 50%, 75%": (PEAK, lambda o: torch.optim.lr_scheduler.MultiStepLR(o, [STEPS // 2, 3 * STEPS // 4], 0.1)),
    "cosine": (PEAK, lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, STEPS)),
    "warmup 100 + cosine": (PEAK, warmup_cosine(100)),
    "one-cycle": (PEAK, lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=PEAK, total_steps=STEPS)),
}
sched_res = {}
for name, (lr, sch) in schedules.items():
    loss, acc, _ = train_steps(make_mlp(), lr, steps=STEPS, sched=sch)
    sched_res[name] = (loss, acc)
    print(f"{name:28s} final training loss {loss:.4f}, test accuracy {acc:.3f}")
decayed = min(sched_res[k][1] for k in ("step decay /10 at 50%, 75%", "cosine", "warmup 100 + cosine", "one-cycle"))
assert decayed > sched_res["constant (peak)"][1], "any decay beats sitting at the peak learning rate"

# %% [markdown]
# ## 6. Batch size and learning rate

# %%
EXAMPLES = 128 * 800                                            # a fixed budget of examples seen
for bs in (32, 128, 512):
    steps = EXAMPLES // bs
    same_lr = train_steps(make_mlp(), 0.02, steps=steps, bs=bs)[1]
    scaled = train_steps(make_mlp(), 0.02 * bs / 128, steps=steps, bs=bs)[1]
    print(f"batch {bs:4d} ({steps:5d} steps): lr fixed at 0.02 -> test acc {same_lr:.3f}; lr scaled linearly ({0.02 * bs / 128:.3f}) -> {scaled:.3f}")
    if bs == 512:
        assert scaled > same_lr

# %% [markdown]
# ## 7. Update-to-weight ratio

# %%
for lr in (1e-4, 1e-2, 1.0):
    m = make_mlp(); opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    ratios = []
    for step in range(100):
        idx = torch.randint(0, len(X_tr), (128,))
        before = [p.detach().clone() for p in m.parameters()]
        opt.zero_grad(); loss = nn.functional.cross_entropy(m(X_tr[idx]), y_tr[idx]); loss.backward(); opt.step()
        if step >= 50:
            ratios.append(np.mean([((p.detach() - b).norm() / b.norm()).item() for p, b in zip(m.parameters(), before) if p.dim() == 2]))
    print(f"lr {lr:6.0e}: mean update/weight ratio {np.mean(ratios):.1e}, loss {loss.item():.3f}")
print("around 1e-3 per step is healthy. lr=1 is the trap: it killed the ReLUs in the first steps (loss stuck at ln 5 = 1.609),")
print("and a dead network makes small updates. A small ratio can mean 'dead', not 'stable': read it together with the loss.")
print("figures saved to", OUT)

print("\nAll checks passed.")
