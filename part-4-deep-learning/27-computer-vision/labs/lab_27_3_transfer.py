# %% [markdown]
# # Lab 27.3: Transfer learning, including how to get it wrong
#
# Pretrained ImageNet weights are a download away in real life; here the labs run offline, so we pretrain our own
# backbone on a large source task (6,000 synthetic shape images) and transfer it.
#
# 1. Pretrain a small ResNet-style backbone on the source task.
# 2. Target: the same four shapes in a harder domain (smaller objects, much more clutter), with only 40 labels.
#    From scratch vs linear probe vs naive full fine-tuning vs careful fine-tuning (low backbone lr, frozen BatchNorm) vs LP-FT.
# 3. How the gap changes with more target labels.
# 4. Preprocessing mismatch: forget the pretraining normalization.
# 5. A narrow source doesn't transfer to a different domain (digits).

# %%
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shapes import make_classification  # noqa: E402

torch.manual_seed(273)
torch.set_num_threads(4)

# %% [markdown]
# ## 1. Pretraining

# %%
Xs, ys = (torch.tensor(a) for a in make_classification(6000, seed=0))
MEAN, STD = Xs.mean((0, 2, 3), keepdim=True), Xs.std((0, 2, 3), keepdim=True)    # the pretraining normalization
norm = lambda x: (x - MEAN) / STD


def block(a, b, s):
    return nn.Sequential(nn.Conv2d(a, b, 3, s, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(),
                         nn.Conv2d(b, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU())


def backbone():
    return nn.Sequential(block(3, 32, 1), block(32, 64, 2), block(64, 128, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())


def fit(model, X, y, epochs, groups, bs=64, bn_eval=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    model.train()
    for _ in range(epochs):
        if bn_eval:                                              # keep pretrained BatchNorm statistics
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        perm = torch.randperm(len(X), generator=g)
        for s in range(0, len(X), bs):
            i = perm[s:s + bs]
            opt.zero_grad(); F.cross_entropy(model(X[i]), y[i]).backward(); opt.step()


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


torch.manual_seed(0)
source_bb = backbone()
source = nn.Sequential(source_bb, nn.Linear(128, 4))
fit(source, norm(Xs), ys, epochs=4, groups=[{"params": source.parameters(), "lr": 2e-3}])
Xs_val, ys_val = (torch.tensor(a) for a in make_classification(1000, seed=9))
print(f"source task accuracy: {accuracy(source, norm(Xs_val), ys_val):.3f}")
PRETRAINED = copy.deepcopy(source_bb.state_dict())

# %% [markdown]
# ## 2. A harder target domain with 40 labels

# %%
Xt, yt = (torch.tensor(a) for a in make_classification(3040, seed=5, clutter=6, r_range=(0.1, 0.2)))
Xt = norm(Xt)
X_test, y_test = Xt[1040:], yt[1040:]


def new_model(pretrained):
    torch.manual_seed(1)
    bb = backbone()
    if pretrained:
        bb.load_state_dict(PRETRAINED)
    return nn.Sequential(bb, nn.Linear(128, 4))


def run(strategy, n_labels):
    Xtr, ytr = Xt[:n_labels], yt[:n_labels]
    ep = max(20, 4000 // n_labels)                               # roughly constant number of steps
    bs = min(20, n_labels)
    m = new_model(strategy != "scratch")
    bb, head = m[0], m[1]
    if strategy == "scratch":
        fit(m, Xtr, ytr, ep, [{"params": m.parameters(), "lr": 1e-3}], bs=bs)
    elif strategy == "linear probe":
        for p in bb.parameters():
            p.requires_grad = False
        fit(m, Xtr, ytr, ep, [{"params": head.parameters(), "lr": 3e-3}], bs=bs, bn_eval=True)
    elif strategy == "full fine-tune, naive":
        fit(m, Xtr, ytr, ep, [{"params": m.parameters(), "lr": 1e-3}], bs=bs)
    elif strategy == "fine-tune, low lr + frozen BN":
        fit(m, Xtr, ytr, ep, [{"params": bb.parameters(), "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-3}], bs=bs, bn_eval=True)
    elif strategy == "LP-FT":
        for p in bb.parameters():
            p.requires_grad = False
        fit(m, Xtr, ytr, ep, [{"params": head.parameters(), "lr": 3e-3}], bs=bs, bn_eval=True)
        for p in bb.parameters():
            p.requires_grad = True
        fit(m, Xtr, ytr, ep // 2, [{"params": bb.parameters(), "lr": 1e-4}, {"params": head.parameters(), "lr": 3e-4}], bs=bs, bn_eval=True)
    return accuracy(m, X_test, y_test)


STRATS = ["scratch", "linear probe", "full fine-tune, naive", "fine-tune, low lr + frozen BN", "LP-FT"]
res40 = {s: run(s, 40) for s in STRATS}
print("\n40 labeled target images (10 per class), test accuracy:")
for s, a in res40.items():
    print(f"  {s:31s} {a:.3f}")
print("naive full fine-tuning wrecks the features it came for; a gentle schedule, frozen BatchNorm statistics and LP-FT recover")
print("most of the damage. With 10 labels per class the probe is still the one to beat; fine-tuning pays off with more data.")
assert res40["linear probe"] > res40["scratch"] + 0.4
assert res40["full fine-tune, naive"] < res40["linear probe"] - 0.1
assert max(res40["fine-tune, low lr + frozen BN"], res40["LP-FT"]) > res40["full fine-tune, naive"] + 0.1

# %% [markdown]
# ## 3. More labels

# %%
print("\nlabels   scratch   linear probe   LP-FT")
for n in (40, 200, 1000):
    r = {s: (res40[s] if n == 40 else run(s, n)) for s in ("scratch", "linear probe", "LP-FT")}
    print(f"{n:6d}   {r['scratch']:7.3f}   {r['linear probe']:12.3f}   {r['LP-FT']:5.3f}")
    if n == 1000:
        assert r["scratch"] > res40["scratch"] + 0.2, "with enough data, from scratch catches up (partly)"

# %% [markdown]
# ## 4. Preprocessing mismatch

# %%
m = new_model(True)
for p in m[0].parameters():
    p.requires_grad = False
fit(m, Xt[:200], yt[:200], 20, [{"params": m[1].parameters(), "lr": 3e-3}], bs=20, bn_eval=True)
raw_test = X_test * STD + MEAN                                   # the same images, without the pretraining normalization
print(f"\nlinear probe evaluated with the pretraining normalization {accuracy(m, X_test, y_test):.3f}; "
      f"with raw [0, 1] pixels {accuracy(m, raw_test, y_test):.3f}")
assert accuracy(m, raw_test, y_test) < accuracy(m, X_test, y_test) - 0.1

# %% [markdown]
# ## 5. A narrow source, a different domain

# %%
d = load_digits()
Xd = F.interpolate(torch.tensor(d.images / 16, dtype=torch.float32)[:, None], size=32, mode="bilinear").repeat(1, 3, 1, 1)
Xd = (Xd - Xd.mean()) / Xd.std()
yd = torch.tensor(d.target)
perm = torch.randperm(len(yd), generator=torch.Generator().manual_seed(0))
tr, te = perm[:100], perm[500:]
res_d = {}
for strategy in ("scratch", "linear probe"):
    m = new_model(strategy != "scratch")
    m[1] = nn.Linear(128, 10)
    if strategy == "linear probe":
        for p in m[0].parameters():
            p.requires_grad = False
        fit(m, Xd[tr], yd[tr], 60, [{"params": m[1].parameters(), "lr": 3e-3}], bs=25, bn_eval=True)
    else:
        fit(m, Xd[tr], yd[tr], 60, [{"params": m.parameters(), "lr": 1e-3}], bs=25)
    res_d[strategy] = accuracy(m, Xd[te], yd[te])
print(f"\ndigits with 100 labels: from scratch {res_d['scratch']:.3f}, linear probe on shape features {res_d['linear probe']:.3f}")
print("features from a four-shape task know about four shapes. ImageNet transfers widely because ImageNet is broad.")
assert res_d["linear probe"] < res_d["scratch"] - 0.3

print("\nAll checks passed.")
