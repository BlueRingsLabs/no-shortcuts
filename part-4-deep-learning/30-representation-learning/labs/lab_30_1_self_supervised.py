# %% [markdown]
# # Lab 30.1: Self-supervised and contrastive learning
#
# 1. Augmentations, batched on tensors: crop, flip, color jitter, grayscale, noise.
# 2. InfoNCE (the SimCLR loss) from scratch, checked against its definition as a classification problem.
# 3. SimCLR pretraining on unlabeled shape images; linear probe with 40 and 400 labels vs a random encoder and vs
#    supervised training on the same labels.
# 4. Augmentations decide what's thrown away: color jitter makes the features color-blind.
# 5. Collapse: SimSiam with and without the stop-gradient.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shapes import make_classification  # noqa: E402

torch.manual_seed(301)
torch.set_num_threads(4)

X_np, y_np = make_classification(4000, seed=0, clutter=3)
X, y = torch.tensor(X_np), torch.tensor(y_np)
X_unlab = X[:3000]                                                    # pretraining: images only
X_lab, y_lab = X[3000:3400], y[3000:3400]                             # at most 400 labels for the probe
X_test, y_test = X[3400:], y[3400:]
MEAN, STD = X_unlab.mean((0, 2, 3), keepdim=True), X_unlab.std((0, 2, 3), keepdim=True)

# %% [markdown]
# ## 1. Augmentations
#
# Random resized crop and flip through one affine grid per image; color jitter as random per-channel gains and a
# brightness shift; random grayscale; a little noise. All batched: a Python loop over images would be the bottleneck.

# %%
def augment(x, color=True, g=None):
    B = len(x)
    scale = torch.empty(B).uniform_(0.6, 1.0, generator=g)           # crop keeps 60% to 100% of the side
    tx, ty = ((torch.rand(B, generator=g) * 2 - 1) * (1 - scale) for _ in range(2))
    flip = torch.where(torch.rand(B, generator=g) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = scale * flip, scale, tx, ty
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False, padding_mode="reflection")
    if color:
        x = x * torch.empty(B, 3, 1, 1).uniform_(0.5, 1.5, generator=g) + torch.empty(B, 1, 1, 1).uniform_(-0.2, 0.2, generator=g)
        gray = torch.rand(B, 1, 1, 1, generator=g) < 0.2
        x = torch.where(gray, x.mean(1, keepdim=True).expand_as(x), x)
    x = x + 0.03 * torch.randn(x.shape, generator=g)
    return (x.clamp(0, 1) - MEAN) / STD


# %% [markdown]
# ## 2. InfoNCE
#
# A batch of N images gives 2N views. For each view, its partner is the positive; the other 2N - 2 views are negatives.
# The loss is cross-entropy for "which of the 2N - 1 others is my partner?", on cosine similarities divided by a
# temperature.

# %%
def info_nce(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))                                 # a view is not its own positive
    N = len(z1)
    target = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)])  # view i's partner is i + N, and vice versa
    return F.cross_entropy(sim, target)


z1, z2 = torch.randn(8, 16), torch.randn(8, 16)
zz = F.normalize(torch.cat([z1, z2]), dim=1)
manual = 0.0
for i in range(16):
    j = (i + 8) % 16
    others = [k for k in range(16) if k != i]
    manual += -(zz[i] @ zz[j] / 0.2 - torch.logsumexp(torch.stack([zz[i] @ zz[k] / 0.2 for k in others]), 0))
assert torch.allclose(info_nce(z1, z2), manual / 16, atol=1e-5)
print(f"InfoNCE matches the definition. At chance (random embeddings, large batch) it's about log(2N - 1): "
      f"with N = 256, log(511) = {np.log(511):.2f}")

# %% [markdown]
# ## 3. SimCLR, then a linear probe

# %%
def encoder():
    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten())


def simclr(epochs=20, color=True, bs=256, seed=0, tau=0.2):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    enc = encoder()
    proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))   # the head is thrown away afterwards
    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    steps = epochs * (len(X_unlab) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(X_unlab), generator=g)
        for s in range(0, len(X_unlab) - bs + 1, bs):
            xb = X_unlab[perm[s:s + bs]]
            loss = info_nce(proj(enc(augment(xb, color, g))), proj(enc(augment(xb, color, g))), tau)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
    return enc, losses


@torch.no_grad()
def features(enc, x):
    enc.eval()
    return torch.cat([enc((x[s:s + 500] - MEAN) / STD) for s in range(0, len(x), 500)]).numpy()


def probe(enc, n_labels, Xtr=None, ytr=None, Xte=None, yte=None):
    Xtr, ytr = (X_lab if Xtr is None else Xtr)[:n_labels], (y_lab if ytr is None else ytr)[:n_labels].numpy()
    Xte, yte = (X_test if Xte is None else Xte), (y_test if yte is None else yte).numpy()
    Ftr, Fte = features(enc, Xtr), features(enc, Xte)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=3000).fit((Ftr - mu) / sd, ytr)
    return clf.score((Fte - mu) / sd, yte)


def supervised(n_labels, epochs=60, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = nn.Sequential(encoder(), nn.Linear(128, 4))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    Xs, ys = X_lab[:n_labels], y_lab[:n_labels]
    for _ in range(epochs):
        perm = torch.randperm(n_labels, generator=g)
        for s in range(0, n_labels, 40):
            i = perm[s:s + 40]
            loss = F.cross_entropy(m(augment(Xs[i], True, g)), ys[i])     # same augmentations, to be fair
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return (m((X_test - MEAN) / STD).argmax(1) == y_test).float().mean().item()


t0 = time.time()
enc_ssl, losses = simclr()
print(f"\nSimCLR: {len(losses)} steps in {time.time() - t0:.0f}s, InfoNCE from {losses[0]:.2f} to {np.mean(losses[-10:]):.2f}")
torch.manual_seed(0)
enc_rand = encoder()
res = {}
for n in (40, 400):
    res[n] = {"random encoder, linear probe": probe(enc_rand, n), "SimCLR encoder, linear probe": probe(enc_ssl, n),
              "supervised from scratch": supervised(n)}
print("\ntest accuracy (4 classes, chance 0.25)")
print("                                   40 labels   400 labels")
for k in res[40]:
    print(f"  {k:32s} {res[40][k]:8.3f}   {res[400][k]:9.3f}")
print(f"({time.time() - t0:.0f}s so far)")
assert res[40]["SimCLR encoder, linear probe"] > res[40]["random encoder, linear probe"] + 0.1
assert res[40]["SimCLR encoder, linear probe"] > res[40]["supervised from scratch"]

# %% [markdown]
# ## 4. What the augmentations throw away
#
# A second task: tint each image toward red, green or blue (that channel x1.3, the others x0.8) and ask which tint it
# got. SimCLR trained with color jitter was told that differently tinted views of an image are the same thing.

# %%
def tint(x, seed):
    c = torch.randint(0, 3, (len(x),), generator=torch.Generator().manual_seed(seed))
    gain = torch.full((len(x), 3), 0.8)
    gain[torch.arange(len(x)), c] = 1.3
    return (x * gain[:, :, None, None]).clamp(0, 1), c


Xt_lab, c_lab = tint(X_lab, 1)
Xt_test, c_test = tint(X_test, 2)
t1 = time.time()
enc_nocolor, _ = simclr(color=False)
print(f"\nSimCLR without color jitter trained in {time.time() - t1:.0f}s")
print("linear probe with 400 labels:        shape   tint (chance 0.33)")
tint_acc = {}
for name, enc in (("SimCLR with color jitter", enc_ssl), ("SimCLR without color jitter", enc_nocolor), ("random encoder", enc_rand)):
    tint_acc[name] = probe(enc, 400, Xt_lab, c_lab, Xt_test, c_test)
    print(f"  {name:32s} {probe(enc, 400):.3f}   {tint_acc[name]:.3f}")
col_with, col_without = tint_acc["SimCLR with color jitter"], tint_acc["SimCLR without color jitter"]
print("the invariances you ask for are the information you lose. And without color jitter the model learned no shape at")
print("all: matching two crops of an image by their color statistics was enough to win the contrastive game.")
assert col_without > col_with + 0.1

# %% [markdown]
# ## 5. Collapse
#
# SimSiam: two views, an encoder + projector f and a predictor h; loss = -cos(h(f(x1)), f(x2)), symmetrized. No
# negatives at all. The trivial solution (every image -> the same vector) has the lowest possible loss. What prevents it
# is the stop-gradient on the target branch. Remove it and watch.

# %%
def simsiam(stop_grad, epochs=6, bs=256, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    f = nn.Sequential(encoder(), nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 64))   # no BN on the output: see the lesson
    h = nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 64))
    opt = torch.optim.SGD(list(f.parameters()) + list(h.parameters()), lr=0.05, momentum=0.9, weight_decay=1e-4)
    D = lambda p, z: -F.cosine_similarity(p, z.detach() if stop_grad else z, dim=1).mean()
    for _ in range(epochs):
        perm = torch.randperm(len(X_unlab), generator=g)
        for s in range(0, len(X_unlab) - bs + 1, bs):
            xb = X_unlab[perm[s:s + bs]]
            z1, z2 = f(augment(xb, True, g)), f(augment(xb, True, g))
            loss = 0.5 * D(h(z1), z2) + 0.5 * D(h(z2), z1)
            opt.zero_grad(); loss.backward(); opt.step()
    f.eval()
    with torch.no_grad():
        z = F.normalize(f((X_test - MEAN) / STD), dim=1)
    return loss.item(), z.std(0).mean().item(), f[0]


t2 = time.time()
print("\nSimSiam, 6 epochs. Collapse indicator: std of the normalized outputs per dimension (1/sqrt(64) = 0.125 if spread")
print("out, 0 if every image maps to the same point)")
col = {}
for sg in (True, False):
    loss, spread, enc = simsiam(sg)
    col[sg] = (loss, spread, probe(enc, 400))
    print(f"  stop-gradient {'on ' if sg else 'off'}: final loss {loss:.3f} (minimum -1), output std {spread:.4f}, probe accuracy {col[sg][2]:.3f}")
print(f"({time.time() - t2:.0f}s) without the stop-gradient the loss reaches its minimum by mapping everything to one point.")
print("(SimSiam learns slowly: 6 epochs is enough to show collapse, not to learn good features.)")
assert col[False][1] < 0.3 * col[True][1]

print("\nAll checks passed.")
