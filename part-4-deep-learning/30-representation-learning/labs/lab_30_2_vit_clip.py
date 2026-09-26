# %% [markdown]
# # Lab 30.2: Vision transformers and CLIP
#
# 1. Patch embedding: a strided convolution is "cut into patches, flatten, multiply by a matrix".
# 2. A small ViT vs a small CNN, at 500 and 4,000 training images: what an inductive bias is worth.
# 3. A small CLIP: an image encoder and a text encoder trained with a symmetric contrastive loss on captioned images.
# 4. Zero-shot classification with prompts, including color-shape combinations never seen in training.
# 5. Text-to-image retrieval, including combinations never seen in training.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shapes import CLASSES, COLORS, make_classification, make_colored  # noqa: E402

torch.manual_seed(302)
torch.set_num_threads(4)

# %% [markdown]
# ## 1. Patches

# %%
d, P = 48, 4
conv = nn.Conv2d(3, d, kernel_size=P, stride=P)
x = torch.randn(2, 3, 32, 32)
patches = x.unfold(2, P, P).unfold(3, P, P)                           # (B, 3, 8, 8, P, P)
patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, 64, 3 * P * P)  # 64 patches of 48 numbers, row-major
linear = patches @ conv.weight.reshape(d, -1).T + conv.bias
assert torch.allclose(conv(x).flatten(2).transpose(1, 2), linear, atol=1e-5)
print("patch embedding: Conv2d(kernel = stride = 4) == unfold into 4x4 patches + one linear layer. A 32x32 image is 64 tokens.")

# %% [markdown]
# ## 2. ViT vs CNN
#
# The same shapes task as Module 27, with augmentation for both models and the same number of optimizer steps.

# %%
class ViT(nn.Module):
    def __init__(self, n_classes=4, d=64, L=4, h=4, P=4, img=32):
        super().__init__()
        self.patch = nn.Conv2d(3, d, P, P)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))                  # a learned [CLS] token, read out at the end
        self.pos = nn.Parameter(torch.randn(1, (img // P) ** 2 + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)   # 29.2's block, bidirectional
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, n_classes)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.blocks(t))[:, 0])


def cnn(n_classes=4):
    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n_classes))


def augment(x, g):
    flip = torch.rand(len(x), generator=g) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    pad = F.pad(x, (3, 3, 3, 3), mode="reflect")
    i, j = torch.randint(0, 7, (2,), generator=g)
    return pad[:, :, i:i + 32, j:j + 32]


Xc, yc = (torch.tensor(a) for a in make_classification(5000, seed=3, clutter=3))
Xc_test, yc_test = Xc[4000:], yc[4000:]
MEAN, STD = Xc[:4000].mean((0, 2, 3), keepdim=True), Xc[:4000].std((0, 2, 3), keepdim=True)


def fit_eval(make, n, steps=600, lr=3e-3):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    m = make()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for _ in range(steps):
        i = torch.randint(0, n, (64,), generator=g)
        loss = F.cross_entropy(m((augment(Xc[i], g) - MEAN) / STD), yc[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        return (m((Xc_test - MEAN) / STD).argmax(1) == yc_test).float().mean().item(), sum(p.numel() for p in m.parameters())


t0 = time.time()
arch = {}
print("\ntest accuracy after 600 steps (chance 0.25)")
for n in (500, 4000):
    for name, make in (("CNN", cnn), ("ViT", ViT)):
        arch[(name, n)], n_params = fit_eval(make, n)
        print(f"  {name}  {n_params:7,} params, {n:5d} training images: {arch[(name, n)]:.3f}")
print(f"({time.time() - t0:.0f}s) the CNN assumes locality and translation equivariance; the ViT has to learn them from data.")
assert arch[("CNN", 500)] > arch[("ViT", 500)] + 0.1
assert arch[("ViT", 4000)] > arch[("ViT", 500)] + 0.1

# %% [markdown]
# ## 3. A small CLIP
#
# Images: one shape in one of six named colors. Captions: "a red circle", "a photo of a red circle", "the circle is red",
# and so on. Four color-shape combinations never appear in training.

# %%
COLOR_NAMES = list(COLORS)
HELD_OUT = {("circle", "red"), ("square", "blue"), ("triangle", "yellow"), ("plus", "cyan")}
TEMPLATES = ["a {c} {s}", "a photo of a {c} {s}", "the {s} is {c}", "{c} {s}", "a picture of the {c} {s}", "a {s} that is {c}"]

Xi_np, s_np, c_np = make_colored(7000, seed=0)
combo = lambda s, c: (CLASSES[s], COLOR_NAMES[c])
seen = np.array([combo(s, c) not in HELD_OUT for s, c in zip(s_np, c_np)])
tr = np.flatnonzero(seen[:6000])
te_seen = 6000 + np.flatnonzero(seen[6000:])
te_unseen = 6000 + np.flatnonzero(~seen[6000:])
Xi = torch.tensor(Xi_np)
s_t, c_t = torch.tensor(s_np), torch.tensor(c_np)
IMEAN, ISTD = Xi[tr].mean((0, 2, 3), keepdim=True), Xi[tr].std((0, 2, 3), keepdim=True)
print(f"\n{len(tr)} training pairs (unseen combinations removed), {len(te_seen)} test images of seen and "
      f"{len(te_unseen)} of unseen combinations")

vocab = sorted({w for t in TEMPLATES for w in t.split() if "{" not in w} | set(COLOR_NAMES) | set(CLASSES))
wtoi = {w: i + 1 for i, w in enumerate(vocab)}                          # 0 = padding
MAXLEN = 7


def tokenize(texts):
    out = torch.zeros(len(texts), MAXLEN, dtype=torch.long)
    for i, t in enumerate(texts):
        ids = [wtoi[w] for w in t.split()]
        out[i, :len(ids)] = torch.tensor(ids)
    return out


class TextEncoder(nn.Module):
    def __init__(self, d=64, out=64):
        super().__init__()
        self.emb = nn.Embedding(len(vocab) + 1, d, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, MAXLEN, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.proj = nn.Linear(d, out)

    def forward(self, tok):
        pad = tok == 0
        h = self.tf(self.emb(tok) + self.pos, src_key_padding_mask=pad)
        h = (h * ~pad[..., None]).sum(1) / (~pad).sum(1, keepdim=True)   # mean over real tokens
        return F.normalize(self.proj(h), dim=-1)


class ImageEncoder(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        self.net = cnn(out)

    def forward(self, x):
        return F.normalize(self.net((x - IMEAN) / ISTD), dim=-1)


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.image, self.text = ImageEncoder(), TextEncoder()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))   # a learned temperature


def clip_loss(model, xb, tok, keys):
    zi, zt = model.image(xb), model.text(tok)
    logits = model.logit_scale.exp().clamp(max=100) * zi @ zt.T
    # CLIP's targets are the diagonal: image i goes with caption i. Web captions are almost never identical; ours
    # repeat constantly (24 combinations), so every pair with the same color and shape counts as a positive.
    same = (keys[:, None] == keys[None, :]).float()
    target = same / same.sum(1, keepdim=True)
    return 0.5 * (-(target * logits.log_softmax(1)).sum(1).mean() - (target * logits.log_softmax(0)).sum(0).mean())


def train_clip(steps=1000, bs=128, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    r = np.random.default_rng(seed)
    model = CLIP()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for _ in range(steps):
        i = torch.tensor(r.choice(tr, bs, replace=False))
        texts = [TEMPLATES[r.integers(len(TEMPLATES))].format(c=COLOR_NAMES[c_t[k]], s=CLASSES[s_t[k]]) for k in i]
        loss = clip_loss(model, augment(Xi[i], g), tokenize(texts), s_t[i] * 10 + c_t[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


t0 = time.time()
clip = train_clip().eval()
print(f"CLIP trained in {time.time() - t0:.0f}s; learned temperature 1/{clip.logit_scale.exp().item():.1f}")

# %% [markdown]
# ## 4. Zero-shot classification
#
# No classifier is trained. Write a caption for each of the 24 combinations, embed them, and label each image with the
# caption it's closest to. Prompt ensembling: average the embeddings of all templates for a class.

# %%
@torch.no_grad()
def class_embeddings(names, templates):
    embs = torch.stack([clip.text(tokenize([t.format(**nm) for t in templates])).mean(0) for nm in names])
    return F.normalize(embs, dim=-1)


with torch.no_grad():
    img_emb = torch.cat([clip.image(Xi[s:s + 500]) for s in range(6000, 7000, 500)])
all_combos = [{"s": s, "c": c} for s in CLASSES for c in COLOR_NAMES]
true_combo = (s_t * len(COLOR_NAMES) + c_t)[6000:]


def zero_shot(templates, idx):
    pred = (img_emb[idx - 6000] @ class_embeddings(all_combos, templates).T).argmax(1)
    return (pred == true_combo[idx - 6000]).float().mean().item()


print("\nzero-shot accuracy over all 24 color-shape combinations (chance 0.042):")
zs = {}
for name, tmpl in (("one prompt: 'a {c} {s}'", ["a {c} {s}"]), ("ensemble of all 6 templates", TEMPLATES)):
    zs[name] = (zero_shot(tmpl, te_seen), zero_shot(tmpl, te_unseen))
    print(f"  {name:30s} seen combinations {zs[name][0]:.3f}   never-seen combinations {zs[name][1]:.3f}")
print("'red circle' never appeared in training, as a caption or an image; 'red' and 'circle' did, separately.")
assert zs["ensemble of all 6 templates"][0] > 0.9 and zs["ensemble of all 6 templates"][1] > 0.5

# shape only, with prompts the model never saw in that form
shape_names = [{"s": s, "c": ""} for s in CLASSES]
pred_shape = (img_emb @ class_embeddings(shape_names, ["a {s}"]).T).argmax(1)
ens = F.normalize(torch.stack([class_embeddings([{"s": s, "c": c} for c in COLOR_NAMES], TEMPLATES).mean(0) for s in CLASSES]), dim=-1)
pred_shape_ens = (img_emb @ ens.T).argmax(1)
acc_plain = (pred_shape == s_t[6000:]).float().mean().item()
acc_ens = (pred_shape_ens == s_t[6000:]).float().mean().item()
print(f"\nzero-shot shape (4 classes): prompt 'a {{shape}}' (a form never seen in training) {acc_plain:.3f}; "
      f"averaging all colors and templates {acc_ens:.3f}")
print("this toy is easy enough that one prompt already works, so ensembling has nothing to fix here. On ImageNet, CLIP's")
print("authors got about 5 points from prompt engineering and ensembling together: real captions are messier than mine.")
assert acc_ens > 0.9

# %% [markdown]
# ## 5. Retrieval

# %%
queries = [("a green triangle", "triangle", "green"), ("the plus is magenta", "plus", "magenta"),
           ("a photo of a red circle", "circle", "red"), ("a blue square", "square", "blue")]   # the last two: never seen
with torch.no_grad():
    q = clip.text(tokenize([t for t, _, _ in queries]))
top = (q @ img_emb.T).topk(5, dim=1).indices
print("\ntext -> image retrieval over the 1,000 test images (top 5 results):")
prec = []
for (text, shape, color), idx in zip(queries, top):
    got = [combo(int(s_t[6000 + k]), int(c_t[6000 + k])) for k in idx]
    prec.append(sum(g == (shape, color) for g in got) / 5)
    print(f"  {text!r:28s} precision@5 {prec[-1]:.1f}   top result: {got[0][1]} {got[0][0]}")
assert np.mean(prec) > 0.8

print("\nAll checks passed.")
