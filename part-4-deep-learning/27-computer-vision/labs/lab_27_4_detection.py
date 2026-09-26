# %% [markdown]
# # Lab 27.4: Object detection, evaluation first
#
# 1. IoU and greedy NMS from scratch, checked against torchvision.ops.
# 2. Average precision from scratch, checked on cases with known answers (including exercise 3).
# 3. A small CenterNet-style detector (center heatmaps + sizes + offsets) trained on synthetic scenes.
# 4. mAP@0.5 and COCO-style mAP@[.5:.95], per class and by object size.
# 5. What NMS buys (max-pool suppression on/off), and the confidence threshold as a precision/recall choice.

# %%
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_iou, nms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shapes import CLASSES, make_scenes  # noqa: E402

torch.manual_seed(274)
torch.set_num_threads(4)
rng = np.random.default_rng(274)

# %% [markdown]
# ## 1. IoU and NMS

# %%
def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def greedy_nms(boxes, scores, thr):
    order, keep = np.argsort(-scores), []
    while len(order):
        i = order[0]
        keep.append(i)
        order = order[1:][iou(boxes[i:i + 1], boxes[order[1:]])[0] <= thr]
    return np.array(keep)


print(f"IoU of (0,0,10,10) and (5,5,15,15): {iou(np.array([[0, 0, 10, 10.]]), np.array([[5, 5, 15, 15.]]))[0, 0]:.4f} (exercise 1: 25/175 = 0.1429)")
B = rng.uniform(0, 50, (200, 2)); B = np.hstack([B, B + rng.uniform(5, 25, (200, 2))]).astype(np.float32)
S_ = rng.random(200).astype(np.float32)
assert np.allclose(iou(B, B), box_iou(torch.tensor(B), torch.tensor(B)).numpy(), atol=1e-5)
assert set(greedy_nms(B, S_, 0.5)) == set(nms(torch.tensor(B), torch.tensor(S_), 0.5).numpy())
print("IoU and NMS match torchvision on 200 random boxes")

# %% [markdown]
# ## 2. Average precision

# %%
def average_precision(dets, gts, cls, thr=0.5):
    """dets: list per image of (boxes, scores, labels); gts: list per image of dicts with boxes, labels."""
    records, n_gt = [], 0
    for (b, s, l), t in zip(dets, gts):
        g = t["boxes"][t["labels"] == cls]
        n_gt += len(g)
        used = np.zeros(len(g), bool)
        sel = l == cls
        bb, ss = b[sel], s[sel]
        for j in np.argsort(-ss):
            hit = False
            if len(g):
                ov = iou(bb[j:j + 1], g)[0]
                ov[used] = -1                                    # each ground truth can be matched once: duplicates are FPs
                k = int(ov.argmax())
                if ov[k] >= thr:
                    used[k] = hit = True
            records.append((ss[j], hit))
    if n_gt == 0:
        return np.nan
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records]); fp = np.cumsum([not r[1] for r in records])
    recall, precision = tp / n_gt, tp / np.maximum(tp + fp, 1)
    mrec, mpre = np.r_[0, recall, 1], np.r_[0, precision, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])                      # precision envelope
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


gt = [{"boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], np.float32), "labels": np.array([0, 0, 0])}]
perfect = [(gt[0]["boxes"], np.array([0.9, 0.8, 0.7]), np.zeros(3, int))]
assert average_precision(perfect, gt, 0) == 1.0
dup = [(np.vstack([gt[0]["boxes"], gt[0]["boxes"][:1]]), np.array([0.9, 0.8, 0.7, 0.95]), np.zeros(4, int))]
print(f"perfect detections: AP 1.0; plus a higher-scoring duplicate of one object: AP {average_precision(dup, gt, 0):.3f}")
# exercise 3: TP, FP, TP, FP, FP, TP with 3 ground truths
far = np.array([[60, 60, 70, 70]], np.float32)
seq = [gt[0]["boxes"][0], far[0], gt[0]["boxes"][1], far[0] + 5, far[0] + 10, gt[0]["boxes"][2]]
ex3 = [(np.array(seq, np.float32), np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]), np.zeros(6, int))]
print(f"exercise 3: AP {average_precision(ex3, gt, 0):.4f} (by hand: (1 + 2/3 + 1/2) / 3 = {(1 + 2 / 3 + 0.5) / 3:.4f})")
assert abs(average_precision(ex3, gt, 0) - (1 + 2 / 3 + 0.5) / 3) < 1e-9

# %% [markdown]
# ## 3. A CenterNet-style detector
#
# Output stride 4: a 64x64 image gives 16x16 maps. Per class, a heatmap with Gaussian peaks at object centers;
# at each center, the box width and height and the sub-cell offset of the true center.

# %%
SIZE, STRIDE, C = 64, 4, len(CLASSES)
G = SIZE // STRIDE
X_tr, T_tr = make_scenes(2500, seed=0)
X_te, T_te = make_scenes(500, seed=1)


def encode(targets):
    n = len(targets)
    hm = np.zeros((n, C, G, G), np.float32); wh = np.zeros((n, 2, G, G), np.float32)
    off = np.zeros((n, 2, G, G), np.float32); mask = np.zeros((n, 1, G, G), np.float32)
    yy, xx = np.mgrid[0:G, 0:G]
    for i, t in enumerate(targets):
        for b, lab in zip(t["boxes"], t["labels"]):
            cx, cy = (b[0] + b[2]) / 2 / STRIDE, (b[1] + b[3]) / 2 / STRIDE
            ix, iy = min(int(cx), G - 1), min(int(cy), G - 1)
            sigma = max(0.6, min(b[2] - b[0], b[3] - b[1]) / STRIDE / 6)
            hm[i, lab] = np.maximum(hm[i, lab], np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sigma**2)))
            wh[i, :, iy, ix] = [(b[2] - b[0]) / SIZE, (b[3] - b[1]) / SIZE]
            off[i, :, iy, ix] = [cx - ix, cy - iy]
            mask[i, 0, iy, ix] = 1
    return [torch.tensor(a) for a in (hm, wh, off, mask)]


HM, WH, OFF, MASK = encode(T_tr)


def cbr(a, b, s=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, s, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU())


class CenterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(cbr(3, 24), cbr(24, 48, 2), cbr(48, 48), cbr(48, 96, 2), cbr(96, 96), cbr(96, 96))
        self.hm, self.wh, self.off = nn.Conv2d(96, C, 1), nn.Conv2d(96, 2, 1), nn.Conv2d(96, 2, 1)
        nn.init.constant_(self.hm.bias, -2.2)                    # start predicting "background" with p ~ 0.1 (26.5's init check)

    def forward(self, x):
        f = self.body(x)
        return self.hm(f), self.wh(f), self.off(f)


def focal_loss(logits, gt, alpha=2, beta=4):                     # CenterNet's penalty-reduced focal loss
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    pos_loss = -(torch.log(p) * (1 - p) ** alpha * pos).sum()
    neg_loss = -(torch.log(1 - p) * p**alpha * (1 - gt) ** beta * (1 - pos)).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1)


model = CenterNet()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
EPOCHS = 10
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * ((len(X_tr) + 31) // 32))
Xt = torch.tensor(X_tr)
for epoch in range(EPOCHS):
    perm = torch.randperm(len(Xt))
    for s in range(0, len(Xt), 32):
        i = perm[s:s + 32]
        h, w, o = model(Xt[i])
        m = MASK[i]
        loss = focal_loss(h, HM[i]) + (5 * F.l1_loss(w * m, WH[i] * m, reduction="sum") + F.l1_loss(o * m, OFF[i] * m, reduction="sum")) / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if epoch in (0, EPOCHS - 1):
        print(f"epoch {epoch + 1}: loss {loss.item():.3f}")

# %% [markdown]
# ## 4. Evaluation

# %%
def decode(h, w, o, k=20, suppress=True):
    p = h.sigmoid()
    if suppress:
        p = p * (F.max_pool2d(p, 3, 1, 1) == p)                   # keep only local maxima: NMS on a heatmap
    out = []
    for i in range(p.shape[0]):
        sc, idx = p[i].flatten().topk(k)
        cl, rem = idx // (G * G), idx % (G * G)
        ys, xs = rem // G, rem % G
        cx, cy = (xs + o[i, 0, ys, xs]) * STRIDE, (ys + o[i, 1, ys, xs]) * STRIDE
        bw, bh = w[i, 0, ys, xs] * SIZE, w[i, 1, ys, xs] * SIZE
        out.append((torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).numpy(), sc.numpy(), cl.numpy()))
    return out


model.eval()
with torch.no_grad():
    h, w, o = model(torch.tensor(X_te))
dets = decode(h, w, o)
ap50 = {c: average_precision(dets, T_te, c, 0.5) for c in range(C)}
map_coco = np.mean([np.mean([average_precision(dets, T_te, c, t) for c in range(C)]) for t in np.arange(0.5, 0.96, 0.05)])
print("AP@0.5 per class:", {CLASSES[c]: round(v, 3) for c, v in ap50.items()})
print(f"mAP@0.5 {np.mean(list(ap50.values())):.3f}; COCO-style mAP@[.5:.95] {map_coco:.3f} (much stricter on box precision)")
assert np.mean(list(ap50.values())) > 0.75 and map_coco < np.mean(list(ap50.values()))


def recall_by_size(targets, dets_, thr=0.3):                    # per-object view: was each ground truth found?
    rows = []
    for t, (bx, sc, lb) in zip(targets, dets_):
        keep = sc >= thr
        for gb, gl in zip(t["boxes"], t["labels"]):
            ov = iou(gb[None], bx[keep])[0] if keep.any() else np.array([0.0])
            found = bool(np.any((ov >= 0.5) & (lb[keep] == gl))) if keep.any() else False
            rows.append(((gb[2] - gb[0]) * (gb[3] - gb[1]), found))
    area, found = np.array(rows).T
    return area, found.astype(bool)


area, found = recall_by_size(T_te, dets)
for lo, hi, name in ((0, 150, "small (< 150 px^2)"), (150, 400, "medium"), (400, 1e9, "large (>= 400 px^2)")):
    sel = (area >= lo) & (area < hi)
    print(f"recall at score >= 0.3 for {name:20s}: {found[sel].mean():.3f} ({sel.sum()} objects)")
small_r, large_r = found[area < 150].mean(), found[area >= 400].mean()
assert small_r < large_r, "small objects are the hard ones"

# %% [markdown]
# ## 5. What suppression buys, and choosing a threshold

# %%
raw = decode(h, w, o, suppress=False)
map_raw = np.mean([average_precision(raw, T_te, c) for c in range(C)])
print(f"\nmAP@0.5 without max-pool suppression {map_raw:.3f} vs with {np.mean(list(ap50.values())):.3f}: neighbors of each peak become duplicate false positives")
assert map_raw < np.mean(list(ap50.values()))

n_gt = sum(len(t["labels"]) for t in T_te)
print("threshold  detections  precision  recall")
for thr in (0.1, 0.3, 0.5, 0.7):
    tp = fp = 0
    for (b, s, l), t in zip(dets, T_te):
        keep = s >= thr
        used = np.zeros(len(t["labels"]), bool)
        for bi, li in zip(b[keep], l[keep]):
            ov = iou(bi[None], t["boxes"])[0] if len(t["boxes"]) else np.array([])
            ov = np.where((t["labels"] == li) & ~used, ov, -1) if len(ov) else ov
            if len(ov) and ov.max() >= 0.5:
                used[ov.argmax()] = True; tp += 1
            else:
                fp += 1
    print(f"{thr:9.1f}  {tp + fp:10d}  {tp / max(tp + fp, 1):9.3f}  {tp / n_gt:6.3f}")
print("mAP summarizes all thresholds; deployment needs one, chosen from the cost of a miss vs a false alarm (18.3)")

print("\nAll checks passed.")
