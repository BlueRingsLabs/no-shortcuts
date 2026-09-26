# %% [markdown]
# # Lab 26.3: Regularizing a network that wants to memorize
#
# A 512-wide, 3-layer MLP (about 300,000 parameters) trained on only 100 digit images, evaluated on the normal test
# set and on a test set of digits shifted by one pixel (the kind of variation real data has).
#
# 1. Inverted dropout by hand: expectation preserved, train vs eval behavior.
# 2. Baseline: memorization. Then weight decay, dropout, augmentation, mixup, label smoothing, early stopping,
#    and a combination, each averaged over 3 seeds, with the seed spread shown (it matters).
# 3. Label smoothing and confidence: logit sizes and calibration error.
# 4. Memorizing random labels: the same network fits noise.

# %%
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(263)
digits = load_digits()
X = torch.tensor(digits.images / 16, dtype=torch.float32)       # (n, 8, 8): keep the image shape for augmentation
y = torch.tensor(digits.target)
idx_tr, idx_rest = train_test_split(np.arange(len(y)), train_size=100, random_state=0, stratify=digits.target)
idx_val, idx_te = train_test_split(idx_rest, test_size=0.5, random_state=0, stratify=digits.target[idx_rest])
X_tr, y_tr, X_val, y_val, X_te, y_te = X[idx_tr], y[idx_tr], X[idx_val], y[idx_val], X[idx_te], y[idx_te]
print(f"train {len(y_tr)}, validation {len(y_val)}, test {len(y_te)}")


def shift_images(xb, dx, dy):                                  # shift with zero padding (no wrap-around)
    out = torch.zeros_like(xb)
    ys, yd = slice(max(0, -dy), 8 - max(0, dy)), slice(max(0, dy), 8 - max(0, -dy))
    xs, xd = slice(max(0, -dx), 8 - max(0, dx)), slice(max(0, dx), 8 - max(0, -dx))
    out[:, yd, xd] = xb[:, ys, xs]
    return out


gshift = torch.Generator().manual_seed(99)
X_te_shift = torch.stack([shift_images(x[None], *torch.randint(-1, 2, (2,), generator=gshift).tolist())[0] for x in X_te])

# %% [markdown]
# ## 1. Inverted dropout

# %%
def inverted_dropout(h, p, training):
    if not training or p == 0:
        return h
    mask = (torch.rand_like(h) > p).float()
    return h * mask / (1 - p)


h = torch.ones(1_000_000) * 3.0
out = inverted_dropout(h, 0.3, True)
print(f"dropout p=0.3 on a constant 3.0: mean after {out.mean():.4f}, fraction zeroed {(out == 0).float().mean():.3f}, "
      f"variance {out.var():.3f} (theory 9 * 0.3/0.7 = {9 * 0.3 / 0.7:.3f}); at eval: unchanged {torch.equal(inverted_dropout(h, 0.3, False), h)}")
assert abs(out.mean().item() - 3.0) < 0.01

# %% [markdown]
# ## 2. What each regularizer buys

# %%
class MLP(nn.Module):
    def __init__(self, p_drop=0.0):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(64, 512), nn.Linear(512, 512), nn.Linear(512, 10)
        self.p = p_drop

    def forward(self, x):
        x = x.flatten(1)
        x = inverted_dropout(torch.relu(self.l1(x)), self.p, self.training)
        x = inverted_dropout(torch.relu(self.l2(x)), self.p, self.training)
        return self.l3(x)


def shift_augment(xb, g):
    dx, dy = torch.randint(-1, 2, (2,), generator=g).tolist()   # shift the whole batch by up to one pixel
    return shift_images(xb, dx, dy)


def evaluate(model, Xe, ye):
    model.eval()
    with torch.no_grad():
        logits = model(Xe)
    model.train()
    return (logits.argmax(1) == ye).float().mean().item(), nn.functional.cross_entropy(logits, ye).item(), logits


def train(p_drop=0.0, wd=0.0, augment=False, mixup=0.0, smoothing=0.0, early_stop=False, epochs=150, labels=None, seed=0):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    labels = y_tr if labels is None else labels
    model = MLP(p_drop)
    decay = [p for n, p in model.named_parameters() if p.dim() > 1]
    no_decay = [p for n, p in model.named_parameters() if p.dim() == 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
    best = (-1, None, 0)
    for epoch in range(epochs):
        perm = torch.randperm(len(labels), generator=g)
        for s in range(0, len(labels), 25):
            i = perm[s:s + 25]
            xb, yb = X_tr[i], labels[i]
            if augment:
                xb = shift_augment(xb, g)
            if mixup > 0:
                lam = float(np.random.default_rng(int(torch.randint(0, 2**31, (1,), generator=g))).beta(mixup, mixup))
                j = torch.randperm(len(i), generator=g)
                xb = lam * xb + (1 - lam) * xb[j]
                out = model(xb)
                loss = lam * nn.functional.cross_entropy(out, yb, label_smoothing=smoothing) + \
                    (1 - lam) * nn.functional.cross_entropy(out, yb[j], label_smoothing=smoothing)
            else:
                loss = nn.functional.cross_entropy(model(xb), yb, label_smoothing=smoothing)
            opt.zero_grad(); loss.backward(); opt.step()
        if early_stop:
            val_acc = evaluate(model, X_val, y_val)[0]
            if val_acc > best[0]:
                best = (val_acc, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
            elif epoch - best[2] >= 25:
                break
    if early_stop:
        model.load_state_dict(best[1])
    return model, (best[2] if early_stop else epochs)


configs = {
    "baseline": {},
    "weight decay 0.5": {"wd": 0.5},
    "dropout 0.5": {"p_drop": 0.5},
    "shift augmentation": {"augment": True},
    "mixup (alpha=0.4)": {"mixup": 0.4},
    "label smoothing 0.1": {"smoothing": 0.1},
    "early stopping": {"early_stop": True},
    "augment + dropout + wd": {"augment": True, "p_drop": 0.3, "wd": 0.1},
}
res = {}
print(f"{'mean of 3 seeds':26s} {'train acc':>9s} {'test acc':>14s} {'shifted test acc':>17s}")
for name, kw in configs.items():
    runs = []
    for seed in range(3):
        model, ep = train(seed=seed, **kw)
        runs.append((evaluate(model, X_tr, y_tr)[0], evaluate(model, X_te, y_te)[0], evaluate(model, X_te_shift, y_te)[0]))
    runs = np.array(runs)
    res[name] = runs.mean(0)
    print(f"{name:26s} {runs[:, 0].mean():9.3f} {runs[:, 1].mean():8.3f} ± {runs[:, 1].std():.3f} {runs[:, 2].mean():10.3f} ± {runs[:, 2].std():.3f}")
print("on the normal test set most regularizers move accuracy by about the seed-to-seed spread: measure before you believe.")
print("on shifted digits, only the model that was *shown* shifts during training copes: augmentation teaches an invariance.")
assert res["baseline"][0] == 1.0, "the baseline memorizes"
assert res["shift augmentation"][2] > res["baseline"][2] + 0.08 and res["augment + dropout + wd"][2] > res["baseline"][2] + 0.08
assert res["label smoothing 0.1"][1] >= res["baseline"][1] - 0.005

# %% [markdown]
# ## 3. Label smoothing and confidence

# %%
def ece(probs, labels, bins=10):
    conf, pred = probs.max(1)
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * ((pred[m] == labels[m]).float().mean() - conf[m].mean()).abs()
    return float(total)


for eps in (0.0, 0.1, 0.3):
    model, _ = train(smoothing=eps)
    _, _, logits = evaluate(model, X_te, y_te)
    top2 = logits.topk(2, dim=1).values
    probs = logits.softmax(1)
    print(f"label smoothing {eps}: mean gap between the top two logits {(top2[:, 0] - top2[:, 1]).mean():6.2f}, "
          f"mean max probability {probs.max(1).values.mean():.3f}, ECE {ece(probs, y_te):.3f}")
    if eps == 0.0:
        gap0 = (top2[:, 0] - top2[:, 1]).mean()
    if eps == 0.1:
        gap1 = (top2[:, 0] - top2[:, 1]).mean()
print(f"(with eps = 0.1 and 10 classes the loss-optimal logit gap is ln(0.91/0.01) = {np.log(91):.2f})")
print("this small model wasn't overconfident to begin with, so smoothing made it *underconfident*: calibration got worse.")
print("label smoothing fixes overconfidence; it doesn't know whether you had any.")
assert gap1 < gap0

# %% [markdown]
# ## 4. Random labels

# %%
g = torch.Generator().manual_seed(1)
random_labels = y_tr[torch.randperm(len(y_tr), generator=g)]
for name, lbl in (("true labels", None), ("shuffled labels", random_labels)):
    for epochs in (20, 150):
        model, _ = train(labels=lbl, epochs=epochs)
        train_acc = (model(X_tr).argmax(1) == (y_tr if lbl is None else lbl)).float().mean().item()
        print(f"{name:15s} after {epochs:3d} epochs: training accuracy {train_acc:.3f}, test accuracy {evaluate(model, X_te, y_te)[0]:.3f}")
print("the same network memorizes pure noise: capacity alone can't explain why it generalizes on the real labels (Zhang et al., 2017)")

print("\nAll checks passed.")
