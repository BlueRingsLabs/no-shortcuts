# %% [markdown]
# # Lab 53.1: Adversarial machine learning, from the defender's side
#
# A digit classifier (scikit-learn's 8x8 digits, 1,797 real images). Three audits a security review would run:
# 1. Robustness evaluation: how much worst-case pixel noise does it take to change predictions? FGSM and PGD are the
#    standard measuring instruments. Then adversarial training, and what robustness costs.
# 2. Data poisoning: a training set received from a vendor contains a small planted backdoor (the test fixture). Can
#    audits of the data, the labels or the trained model find it?
# 3. Privacy: does the model reveal which examples it was trained on? A membership-inference audit, and how
#    regularization changes the answer.

# %%
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from torch import nn

torch.set_num_threads(1)
torch.manual_seed(531)
d = load_digits()
X = torch.tensor(d.data / 16.0, dtype=torch.float32)                               # pixels in [0, 1]
y = torch.tensor(d.target)
perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
tr, te = perm[:1200], perm[1200:]


def mlp():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def pgd(model, x, y, eps, steps=10, step_size=None):
    """Projected gradient ascent on the loss within an L-infinity ball of radius eps (Madry et al., 2018).
    steps=1 with step_size=eps is FGSM (Goodfellow et al., 2015)."""
    step_size = step_size or 2.5 * eps / steps
    x_adv = x.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        g, = torch.autograd.grad(loss, x_adv)
        x_adv = (x_adv.detach() + step_size * g.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return x_adv.detach()


def train(Xtr, ytr, adv_eps=0.0, epochs=60, wd=0.0, seed=0):
    torch.manual_seed(seed)
    m = mlp()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=wd)
    for ep in range(epochs):
        for idx in torch.randperm(len(Xtr)).split(64):
            xb, yb = Xtr[idx], ytr[idx]
            if adv_eps:
                m.eval(); xb = pgd(m, xb, yb, adv_eps, steps=5); m.train()
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def acc(m, x, yy):
    with torch.no_grad():
        return (m(x).argmax(1) == yy).float().mean().item()


# %% [markdown]
# ## 1. Robustness evaluation, and adversarial training

# %%
std = train(X[tr], y[tr])
robust = train(X[tr], y[tr], adv_eps=0.1)
print(f"{'eps (max change per pixel)':28s} {'standard: FGSM':>15s} {'standard: PGD':>14s} {'adv.-trained: PGD':>18s}")
curve = {}
for eps in (0.0, 0.05, 0.1, 0.2):
    a_f = acc(std, pgd(std, X[te], y[te], eps, steps=1, step_size=eps) if eps else X[te], y[te])
    a_p = acc(std, pgd(std, X[te], y[te], eps) if eps else X[te], y[te])
    a_r = acc(robust, pgd(robust, X[te], y[te], eps) if eps else X[te], y[te])
    curve[eps] = (a_f, a_p, a_r)
    print(f"{eps:28.2f} {a_f:15.3f} {a_p:14.3f} {a_r:18.3f}")
print("changes of at most a tenth of the intensity range per pixel, chosen by gradient, cut the standard model's")
print("accuracy from 97% to about 40%; a fifth of the range takes it near zero. The multi-step search (PGD) is always")
print("at least as strong a test as the one-step one (FGSM), so evaluate with PGD, and with more steps and restarts for")
print("a real audit. Training on PGD examples holds 84% at eps 0.1. Here it didn't cost clean accuracy (on a problem")
print("this small it acts as regularization); on large image models it usually costs a few points, and several times")
print("the training compute. Robustness is measured against a threat model (which changes, how large): report the curve.")
assert curve[0.2][1] < 0.1 and curve[0.1][2] > curve[0.1][1] + 0.3
assert all(curve[e][1] <= curve[e][0] + 0.02 for e in curve)

# %% [markdown]
# ## 2. Auditing a training set for a backdoor
#
# The fixture: 8% of the "7" images in the received training set were relabeled "1" and stamped with a small bright
# patch in one corner. A model trained on it behaves normally, except on images with the patch.

# %%
def stamp(x):
    x = x.clone().view(-1, 8, 8)
    x[:, 6:8, 6:8] = 1.0
    return x.view(-1, 64)


Xp, yp = X[tr].clone(), y[tr].clone()
sevens = torch.nonzero(yp == 7)[:, 0]
poison_idx = sevens[torch.randperm(len(sevens), generator=torch.Generator().manual_seed(1))[:len(sevens) // 12]]
Xp[poison_idx] = stamp(Xp[poison_idx]); yp[poison_idx] = 1
is_poison = torch.zeros(len(yp), dtype=torch.bool); is_poison[poison_idx] = True
bd = train(Xp, yp)
test_sevens = X[te][y[te] == 7]
print(f"\nreceived training set: {len(yp)} images, {is_poison.sum().item()} poisoned ({is_poison.float().mean():.1%})")
print(f"model trained on it: clean test accuracy {acc(bd, X[te], y[te]):.3f}; sevens with the patch classified as 1: "
      f"{(bd(stamp(test_sevens)).argmax(1) == 1).float().mean():.2f}")

# audit 1, the data: spectral signatures. Within a class, poisoned examples may share a direction in the model's
# representation that clean ones don't (Tran et al., 2018). Flag the 10% of each class furthest along it.
with torch.no_grad():
    feats = bd[:4](Xp)                                                                # penultimate-layer activations
flag = torch.zeros(len(yp), dtype=torch.bool)
for c in range(10):
    idx = torch.nonzero(yp == c)[:, 0]
    f = feats[idx] - feats[idx].mean(0)
    score = (f @ torch.linalg.svd(f, full_matrices=False).Vh[0]) ** 2
    flag[idx[torch.argsort(-score)[:max(1, len(idx) // 10)]]] = True
caught_spectral = (flag & is_poison).sum().item()
# audit 2, the labels: flag examples whose label disagrees with an out-of-fold prediction
folds = torch.randperm(len(yp), generator=torch.Generator().manual_seed(2)) % 5
oof = torch.zeros(len(yp), dtype=torch.long)
for k in range(5):
    m = train(Xp[folds != k], yp[folds != k], epochs=40)
    with torch.no_grad():
        oof[folds == k] = m(Xp[folds == k]).argmax(1)
caught_cv = ((oof != yp) & is_poison).sum().item()
print(f"data audit, spectral signatures: flagged {flag.sum().item()} images, {caught_spectral} of {is_poison.sum().item()} poisoned")
print(f"label audit, out-of-fold disagreement: {(oof != yp).sum().item()} flagged, {caught_cv} of {is_poison.sum().item()} poisoned "
      "(models trained on the other folds learned the trigger too)")


# audit 3, the model: scan for triggers. Stamp candidate patches on clean test images of every class and measure how
# often the prediction flips to one single class. A clean model has no reason to do that.
def patch_share(model, r0, c0, target):
    x = X[te].clone().view(-1, 8, 8); x[:, r0:r0 + 2, c0:c0 + 2] = 1.0
    with torch.no_grad():
        pred = model(x.view(-1, 64)).argmax(1)
    return (pred[y[te] != target] == target).float().mean().item()


def trigger_scan(model):
    worst = (0.0, None)
    for r0, c0 in ((0, 0), (0, 6), (6, 0), (6, 6)):
        x = X[te].clone().view(-1, 8, 8); x[:, r0:r0 + 2, c0:c0 + 2] = 1.0
        with torch.no_grad():
            pred = model(x.view(-1, 64)).argmax(1)
        for target in range(10):
            others = y[te] != target
            share = (pred[others] == target).float().mean().item()
            if share > worst[0]:
                worst = (share, (r0, c0, target))
    return worst


for name, m in (("model trained on the received data", bd), ("model trained on clean data", std)):
    share, where = trigger_scan(m)
    print(f"trigger scan, {name}: worst 2x2 patch sends {share:.0%} of other classes' images to class {where[2]} "
          f"(patch at row {where[0]}, column {where[1]})")
found = trigger_scan(bd)[1]
print(f"the same patch on the clean model: {patch_share(std, *found):.0%} to class {found[2]}")
print("nine poisoned images out of 1,200 plant a backdoor that fires 94% of the time and leaves clean accuracy intact.")
print("Both data audits miss them: published detectors work in their papers' settings and are easy to fall outside of.")
print("The trigger scan points at the right patch and class, but a clean model has sensitive patches too (a bright top-right")
print("corner looks like part of a 7): scans produce candidates for a person to inspect, not verdicts. The defenses that")
print("don't depend on luck come first: provenance and access control for every training source, review of what enters")
print("training, and reproducible pipelines that record exactly which data trained which model (48.2).")
assert found == (6, 6, 1) and trigger_scan(bd)[0] > 2 * patch_share(std, *found)

# %% [markdown]
# ## 3. A membership-inference audit

# %%
# a small dataset with 20% of labels wrong (records that don't follow the pattern, which a model can only memorize),
# split into members (trained on) and non-members from the same pool
pool = torch.randperm(len(X), generator=torch.Generator().manual_seed(7))[:600]
Xm, ym = X[pool], y[pool].clone()
noisy = torch.rand(len(ym), generator=torch.Generator().manual_seed(8)) < 0.2
ym[noisy] = torch.randint(0, 10, (int(noisy.sum()),), generator=torch.Generator().manual_seed(9))
members, nonmembers = torch.arange(300), torch.arange(300, 600)


def mi_auc(m):
    """How well does the per-example loss separate training members from non-members? 0.5 = no leak."""
    with torch.no_grad():
        l_in = F.cross_entropy(m(Xm[members]), ym[members], reduction="none")
        l_out = F.cross_entropy(m(Xm[nonmembers]), ym[nonmembers], reduction="none")
    return roc_auc_score([1] * 300 + [0] * 300, (-torch.cat([l_in, l_out])).numpy())


audits = {}
for name, kw in (("300 epochs, no regularization", dict(epochs=300)), ("60 epochs", dict(epochs=60)),
                 ("15 epochs + weight decay", dict(epochs=15, wd=1e-2))):
    m = train(Xm[members], ym[members], **kw)
    audits[name] = mi_auc(m)
    print(f"\n{name:32s} accuracy on clean test images {acc(m, X[te], y[te]):.3f}; membership AUC {audits[name]:.3f}")
print("the more a model memorizes, the more its loss reveals which records it trained on: here, mostly the odd records")
print("that don't fit the pattern, which in personal data are often the most sensitive ones. Audit this before releasing")
print("a model trained on personal data (54.2). Regularization and early stopping reduce the leak; only differential")
print("privacy bounds it (54.2).")
assert audits["300 epochs, no regularization"] > audits["15 epochs + weight decay"] + 0.1

print("\nAll checks passed.")
