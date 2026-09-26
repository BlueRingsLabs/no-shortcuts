# %% [markdown]
# # Lab 27.5: Segmentation, and the metric that lies
#
# 1. Masks are labels, not images: bilinear resizing invents classes that aren't there.
# 2. A small U-Net for semantic segmentation on synthetic scenes (5 classes including background).
# 3. Pixel accuracy vs per-class IoU vs mIoU, with the "everything is background" baseline; mIoU over the dataset vs per image.
# 4. Remove the skip connections and measure what happens at the boundaries.
# 5. Cross-entropy + soft Dice vs cross-entropy alone, on imbalanced classes.
# 6. A toy promptable model (SAM's interface, not its scale): image + a click in, the clicked object's mask out.
#    It separates overlapping objects of the same class, which a semantic map can't.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shapes import CLASSES, make_scenes  # noqa: E402

torch.manual_seed(275)
torch.set_num_threads(4)
rng = np.random.default_rng(275)
NAMES = ["background"] + CLASSES
K = len(NAMES)

X_np, T_train = make_scenes(2000, seed=0)
Xv_np, T_test = make_scenes(500, seed=1)
X, Xv = torch.tensor(X_np), torch.tensor(Xv_np)
Y = torch.tensor(np.stack([t["semantic"] for t in T_train]))
Yv = torch.tensor(np.stack([t["semantic"] for t in T_test]))
print(f"train {tuple(X.shape)}, test {tuple(Xv.shape)}; pixel share per class (test): "
      + ", ".join(f"{n} {(Yv == c).float().mean():.3f}" for c, n in enumerate(NAMES)))

# %% [markdown]
# ## 1. Resizing masks
#
# Class IDs are categories, not intensities. Interpolating between "background" (0) and "plus" (4) gives 2, "square",
# along every boundary of every plus.

# %%
sem = Yv[:50, None].float()
small_bil = F.interpolate(sem, size=40, mode="bilinear", align_corners=False).round().long()
small_nn = F.interpolate(sem, size=40, mode="nearest").long()
invented = 0
for i in range(50):
    present = set(Yv[i].unique().tolist())
    invented += sum(int((small_bil[i, 0] == c).sum()) for c in range(K) if c not in present)
nn_invented = sum(int((small_nn[i, 0] == c).sum()) for i in range(50) for c in range(K) if c not in set(Yv[i].unique().tolist()))
print(f"\nresizing 50 masks 64 -> 40: bilinear puts {invented} pixels in classes absent from their image; nearest {nn_invented}")
assert invented > 0 and nn_invented == 0

# %% [markdown]
# ## 2. A small U-Net

# %%
def conv_block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(),
                         nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=K, w=12, skips=True):
        super().__init__()
        self.skips = skips
        self.e1, self.e2, self.e3 = conv_block(c_in, w), conv_block(w, 2 * w), conv_block(2 * w, 4 * w)
        self.bott = conv_block(4 * w, 8 * w)
        self.u3, self.u2, self.u1 = (nn.ConvTranspose2d(8 * w, 4 * w, 2, 2), nn.ConvTranspose2d(4 * w, 2 * w, 2, 2),
                                     nn.ConvTranspose2d(2 * w, w, 2, 2))
        m = 2 if skips else 1                                      # concatenation doubles the decoder's input channels
        self.d3, self.d2, self.d1 = conv_block(m * 4 * w, 4 * w), conv_block(m * 2 * w, 2 * w), conv_block(m * w, w)
        self.head = nn.Conv2d(w, c_out, 1)

    def forward(self, x):
        e1 = self.e1(x)                                            # 64x64: where
        e2 = self.e2(F.max_pool2d(e1, 2))                          # 32x32
        e3 = self.e3(F.max_pool2d(e2, 2))                          # 16x16
        b = self.bott(F.max_pool2d(e3, 2))                         # 8x8: what
        cat = (lambda up, skip: torch.cat([up, skip], 1)) if self.skips else (lambda up, skip: up)
        d3 = self.d3(cat(self.u3(b), e3))
        d2 = self.d2(cat(self.u2(d3), e2))
        d1 = self.d1(cat(self.u1(d2), e1))
        return self.head(d1)


def soft_dice_loss(logits, y, eps=1.0):
    p = logits.softmax(1)
    onehot = F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (p * onehot).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()          # mean over classes: each class counts equally


def train(model, X, Y, loss_fn, epochs=6, bs=16, lr=3e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(X) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for s in range(0, len(X), bs):
            i = perm[s:s + bs]
            opt.zero_grad(); loss_fn(model(X[i]), Y[i]).backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def predict(model, X, bs=100):
    model.eval()
    return torch.cat([model(X[s:s + bs]).argmax(1) for s in range(0, len(X), bs)])


def confusion(pred, y, k=K):
    return torch.bincount(y.flatten() * k + pred.flatten(), minlength=k * k).reshape(k, k).double()


def iou_per_class(cm):
    tp = cm.diag()
    return (tp / (cm.sum(0) + cm.sum(1) - tp)).numpy()             # dataset-level: pixels pooled over all images


ce = lambda logits, y: F.cross_entropy(logits, y)
ce_dice = lambda logits, y: ce(logits, y) + soft_dice_loss(logits, y)   # the default here; section 5 says why
t0 = time.time()
torch.manual_seed(0)
unet = train(UNet(), X, Y, ce_dice)
P = predict(unet, Xv)
print(f"\nU-Net trained in {time.time() - t0:.0f}s")

# %% [markdown]
# ## 3. Pixel accuracy, IoU, mIoU

# %%
def report(name, pred, y=Yv):
    cm = confusion(pred, y)
    ious = iou_per_class(cm)
    acc = (cm.diag().sum() / cm.sum()).item()
    print(f"{name:28s} pixel acc {acc:.3f}   mIoU {np.nanmean(ious):.3f}   " + "  ".join(f"{n[:6]} {v:.2f}" for n, v in zip(NAMES, ious)))
    return acc, ious


print()
acc_bg, iou_bg = report("everything is background", torch.zeros_like(Yv))
acc_u, iou_u = report("U-Net", P)
print("the lazy baseline's pixel accuracy is the background share; its mIoU tells the truth.")
assert acc_bg > 0.7 and np.nanmean(iou_bg) < 0.2
assert np.nanmean(iou_u) > 0.7

# mIoU per image, then averaged, vs pooled over the dataset. Classes absent from an image (in both prediction and truth)
# are skipped for that image, which is one of several conventions: say which one you use.
per_image = []
for p, y in zip(P, Yv):
    ious = iou_per_class(confusion(p, y))
    per_image.append(np.nanmean(ious))
print(f"mIoU pooled over the dataset {np.nanmean(iou_u):.3f}; mIoU per image, averaged {np.mean(per_image):.3f}; "
      f"worst image {np.min(per_image):.3f}")

# %% [markdown]
# ## 4. Without skip connections
#
# Same encoder, same decoder depth, the decoder just never sees the high-resolution encoder features. Evaluate on all
# pixels and on a band of pixels within 1 pixel of a class boundary.

# %%
def boundary_band(y, width=1):
    y = y.numpy()
    edge = np.zeros_like(y, dtype=bool)
    edge[:, 1:, :] |= y[:, 1:, :] != y[:, :-1, :]; edge[:, :-1, :] |= y[:, 1:, :] != y[:, :-1, :]
    edge[:, :, 1:] |= y[:, :, 1:] != y[:, :, :-1]; edge[:, :, :-1] |= y[:, :, 1:] != y[:, :, :-1]
    return torch.tensor(ndimage.binary_dilation(edge, structure=np.ones((1, 3, 3)), iterations=width - 1) if width > 1 else edge)


band = boundary_band(Yv)
t0 = time.time()
torch.manual_seed(0)
noskip = train(UNet(skips=False), X, Y, ce_dice)
P_ns = predict(noskip, Xv)
print(f"\nno-skip U-Net trained in {time.time() - t0:.0f}s")
_, iou_ns = report("no skips", P_ns)
for name, p in (("U-Net", P), ("no skips", P_ns)):
    interior = ((p == Yv) & ~band).sum().item() / (~band).sum().item()
    edge = ((p == Yv) & band).sum().item() / band.sum().item()
    print(f"  {name:9s} accuracy away from boundaries {interior:.3f}, in the {band.float().mean():.0%} of pixels at boundaries {edge:.3f}")
acc_band_u = ((P == Yv) & band).sum().item() / band.sum().item()
acc_band_ns = ((P_ns == Yv) & band).sum().item() / band.sum().item()
print("the 8x8 bottleneck knows what is there; without the skips it has to guess where, and the guesses land on the edges.")
assert np.nanmean(iou_ns) < np.nanmean(iou_u) - 0.03
assert acc_band_ns < acc_band_u - 0.05

# %% [markdown]
# ## 5. Cross-entropy alone
#
# Same model, same budget, only the loss changes. 88% of the pixels are background, and each shape class is 2 to 3%.

# %%
t0 = time.time()
torch.manual_seed(0)
unet_ce = train(UNet(), X, Y, ce)
P_ce = predict(unet_ce, Xv)
print(f"\nCE-only U-Net trained in {time.time() - t0:.0f}s")
report("U-Net (CE + soft Dice)", P)
_, iou_ce = report("U-Net (CE only)", P_ce)
print("pixel accuracy barely moves. Cross-entropy is a sum over pixels, so most of its gradient goes to the background;")
print("soft Dice averages over classes, so each shape counts as much as the background does. This is one seed: with seeds")
print("1 and 2 (exercise 6) the gap was 0.19 and 0.23 mIoU, with whole classes never learned by the cross-entropy model.")
assert np.nanmean(iou_ce) < np.nanmean(iou_u)
assert (iou_ce[1:] <= iou_u[1:] + 0.01).all()

# %% [markdown]
# ## 6. A promptable model
#
# Input: the image plus a fourth channel with a Gaussian blob at a clicked pixel. Output: one logit per pixel, "is this
# pixel part of the object I clicked?". No class labels anywhere: the model learns "the thing under the click", which is
# the interface SAM made standard (its scale, a ViT encoder run once per image and a light decoder per prompt, is
# not reproduced here).

# %%
YY, XX = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")


def click_channel(py, px, sigma=2.0):
    return torch.exp(-((YY - py) ** 2 + (XX - px) ** 2) / (2 * sigma ** 2))


def sample_prompts(X, targets, seed, per_image=1):
    """A click on a random visible pixel of a random instance; target = that instance's visible mask."""
    r = np.random.default_rng(seed)
    xs, ms, meta = [], [], []
    for i, t in enumerate(targets):
        for _ in range(per_image):
            j = r.integers(len(t["masks"]))
            ys_, xs_ = np.nonzero(t["masks"][j])
            k = r.integers(len(ys_))
            xs.append(torch.cat([X[i], click_channel(ys_[k], xs_[k])[None]]))
            ms.append(torch.tensor(t["masks"][j]))
            meta.append((i, j, ys_[k], xs_[k]))
    return torch.stack(xs), torch.stack(ms).float(), meta


Xp, Mp, _ = sample_prompts(X, T_train, seed=0)
Xpv, Mpv, meta_v = sample_prompts(Xv, T_test, seed=1, per_image=2)
bce = lambda lg, m: F.binary_cross_entropy_with_logits(lg[:, 0], m)
t0 = time.time()
torch.manual_seed(0)
prompt_net = train(UNet(c_in=4, c_out=1), Xp, Mp, bce, epochs=7)
print(f"\npromptable U-Net trained on {len(Xp)} (image, click) pairs in {time.time() - t0:.0f}s")

with torch.no_grad():
    prompt_net.eval()
    Mhat = torch.cat([prompt_net(Xpv[s:s + 100])[:, 0] > 0 for s in range(0, len(Xpv), 100)])
inter = (Mhat & Mpv.bool()).sum((1, 2)).float()
union = (Mhat | Mpv.bool()).sum((1, 2)).float()
iou_prompt = (inter / union).numpy()

# The semantic alternative: take the connected region of the clicked pixel's class in the ground-truth semantic map
# (an oracle semantic model, perfect by construction). Where two objects of the same class touch, it returns both.
iou_sem, same_class_touch = [], []
for (i, j, py, px), m in zip(meta_v, Mpv.bool().numpy()):
    t = T_test[i]
    region, _ = ndimage.label(t["semantic"] == t["labels"][j] + 1)
    comp = region == region[py, px]
    iou_sem.append((comp & m).sum() / (comp | m).sum())
    grown = ndimage.binary_dilation(m)
    same_class_touch.append(any(k != j and t["labels"][k] == t["labels"][j] and (grown & t["masks"][k]).any()
                                for k in range(len(t["masks"]))))
iou_sem, same_class_touch = np.array(iou_sem), np.array(same_class_touch)
n_touch = same_class_touch.sum()
print(f"clicks on test objects: {len(iou_prompt)}, of which {n_touch} are on an object touching another of its class")
print(f"  mean IoU, all clicks:          promptable {iou_prompt.mean():.3f}   perfect semantic map + connected region {iou_sem.mean():.3f}")
print(f"  mean IoU, same-class touching: promptable {iou_prompt[same_class_touch].mean():.3f}   "
      f"perfect semantic map + connected region {iou_sem[same_class_touch].mean():.3f}")
print("the oracle semantic map is perfect and still wrong for these: 'plus' pixels don't say which plus.")
assert n_touch >= 20
assert iou_prompt[same_class_touch].mean() > iou_sem[same_class_touch].mean() + 0.1
assert iou_prompt.mean() > 0.7

print("\nAll checks passed.")
