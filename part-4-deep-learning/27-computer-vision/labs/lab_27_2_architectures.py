# %% [markdown]
# # Lab 27.2: A small zoo of CNN architectures
#
# Each design from the lesson, shrunk to 32x32 synthetic shapes (circle, square, triangle, plus on cluttered
# backgrounds), trained with the same recipe and budget.
#
# 1. Parameter counts of the blocks in exercise 1, checked.
# 2. The zoo: MLP, LeNet, VGG-style, ResNet-style, MobileNetV2-style, ConvNeXt-style.
# 3. Parameters, FLOPs (counted with hooks), measured CPU latency, and test accuracy side by side.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from shapes import CLASSES, make_classification  # noqa: E402

torch.manual_seed(272)
torch.set_num_threads(4)
X_tr, y_tr = make_classification(4000, seed=0)
X_te, y_te = make_classification(1000, seed=1)
X_tr, y_tr, X_te, y_te = torch.tensor(X_tr), torch.tensor(y_tr), torch.tensor(X_te), torch.tensor(y_te)
mean, std = X_tr.mean((0, 2, 3), keepdim=True), X_tr.std((0, 2, 3), keepdim=True)
X_tr, X_te = (X_tr - mean) / std, (X_te - mean) / std
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"{len(X_tr)} training images of {CLASSES}, {len(X_te)} test images")

# %% [markdown]
# ## 1. Block parameter counts

# %%
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out)
        self.c2, self.b2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out)
        self.short = nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.short(x))


class Bottleneck(nn.Module):
    def __init__(self, c, mid):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(),
                                  nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(),
                                  nn.Conv2d(mid, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(self.body(x) + x)


print(f"basic block, 64 channels: {count(BasicBlock(64, 64)):,} params (exercise 1: 73,984)")
print(f"bottleneck block, 256 channels (64 inside): {count(Bottleneck(256, 64)):,} params (exercise 1: 70,400)")
assert count(BasicBlock(64, 64)) == 73_984 and count(Bottleneck(256, 64)) == 70_400

# %% [markdown]
# ## 2. The zoo

# %%
class InvertedResidual(nn.Module):                               # MobileNetV2
    def __init__(self, c_in, c_out, stride, expand=4):
        super().__init__()
        h = c_in * expand
        self.use_res = stride == 1 and c_in == c_out
        self.body = nn.Sequential(nn.Conv2d(c_in, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU6(),
                                  nn.Conv2d(h, h, 3, stride, 1, groups=h, bias=False), nn.BatchNorm2d(h), nn.ReLU6(),
                                  nn.Conv2d(h, c_out, 1, bias=False), nn.BatchNorm2d(c_out))      # linear projection: no activation

    def forward(self, x):
        return x + self.body(x) if self.use_res else self.body(x)


class LayerNorm2d(nn.LayerNorm):                                 # LayerNorm over channels, for (N, C, H, W)
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)        # large depthwise kernel
        self.norm = nn.LayerNorm(c)
        self.pw1, self.pw2 = nn.Linear(c, 4 * c), nn.Linear(4 * c, c)   # inverted bottleneck as per-pixel linear layers
        self.gamma = nn.Parameter(torch.ones(c))                 # layer scale (ConvNeXt starts it at 1e-6 for deep nets; at this
                                                                  # depth and budget that leaves the blocks as near-identities)

    def forward(self, x):
        h = self.dw(x).permute(0, 2, 3, 1)
        h = self.pw2(F.gelu(self.pw1(self.norm(h)))) * self.gamma
        return x + h.permute(0, 3, 1, 2)


def conv_bn_relu(a, b, **kw):
    return [nn.Conv2d(a, b, 3, padding=1, bias=False, **kw), nn.BatchNorm2d(b), nn.ReLU()]


def zoo():
    return {
        "MLP": nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4)),
        "LeNet": nn.Sequential(nn.Conv2d(3, 6, 5), nn.Tanh(), nn.AvgPool2d(2), nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
                               nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 4)),
        "VGG-style": nn.Sequential(*conv_bn_relu(3, 32), *conv_bn_relu(32, 32), nn.MaxPool2d(2),
                                   *conv_bn_relu(32, 64), *conv_bn_relu(64, 64), nn.MaxPool2d(2),
                                   *conv_bn_relu(64, 128), *conv_bn_relu(128, 128), nn.MaxPool2d(2),
                                   nn.Flatten(), nn.Linear(128 * 16, 256), nn.ReLU(), nn.Linear(256, 4)),   # the big dense head
        "ResNet-style": nn.Sequential(*conv_bn_relu(3, 32), BasicBlock(32, 32), BasicBlock(32, 64, 2), BasicBlock(64, 64),
                                      BasicBlock(64, 128, 2), BasicBlock(128, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 4)),
        "MobileNetV2-style": nn.Sequential(*conv_bn_relu(3, 16), InvertedResidual(16, 24, 2), InvertedResidual(24, 24, 1),
                                           InvertedResidual(24, 48, 2), InvertedResidual(48, 48, 1), InvertedResidual(48, 96, 2),
                                           InvertedResidual(96, 96, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 4)),
        "ConvNeXt-style": nn.Sequential(nn.Conv2d(3, 48, 2, 2), LayerNorm2d(48),              # patchify stem: 32 -> 16 (4x4 on 32 px loses too much)
                                        ConvNeXtBlock(48), ConvNeXtBlock(48),
                                        LayerNorm2d(48), nn.Conv2d(48, 96, 2, 2),              # separate downsampling: 16 -> 8
                                        ConvNeXtBlock(96), ConvNeXtBlock(96),
                                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(96), nn.Linear(96, 4)),
    }

# %% [markdown]
# ## 3. Cost and accuracy

# %%
def count_flops(model, x):
    flops = []

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            flops.append(2 * k * out.numel() // out.shape[0])
        elif isinstance(m, nn.Linear):
            flops.append(2 * m.in_features * out.numel() // out.shape[0])

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model.eval(); model(x[:1])
    for h in hs:
        h.remove()
    return sum(flops)


def latency_ms(model, x, reps=20):
    model.eval()
    with torch.no_grad():
        model(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
    return (time.perf_counter() - t0) / reps * 1000


def train(model, epochs=5, lr=2e-3):
    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * (len(X_tr) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_tr), generator=g)
        for s in range(0, len(X_tr) - 63, 64):
            idx = perm[s:s + 64]
            xb = X_tr[idx]
            if torch.rand(1, generator=g) < 0.5:
                xb = xb.flip(3)                                   # horizontal flip: all four shapes are left-right symmetric
            opt.zero_grad(); F.cross_entropy(model(xb), y_tr[idx]).backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        return (model(X_te).argmax(1) == y_te).float().mean().item()


print(f"{'model':18s} {'params':>9s} {'MFLOPs/img':>11s} {'CPU ms/batch of 256':>20s} {'test acc':>9s}")
results = {}
for name, model in zoo().items():
    params, flops = count(model), count_flops(model, X_te)
    lat = latency_ms(model, X_te[:256])
    acc = train(model)
    results[name] = (params, flops, lat, acc)
    print(f"{name:18s} {params:9,} {flops / 1e6:11.1f} {lat:20.1f} {acc:9.3f}")

print("\nnotes: the VGG-style model spends most of its parameters in the dense head; the MobileNet-style one has the fewest")
print("FLOPs but not the lowest latency per FLOP (depthwise convolutions are memory-bound); the MLP, with no convolutional")
print("prior, is far behind at the same budget. ConvNeXt's design was tuned with a long schedule and heavy augmentation;")
print("on this short budget it trails the plainer designs, which is the paper's own point about recipes, seen from the other side.")
vgg_head = 128 * 16 * 256 + 256 + 256 * 4 + 4
print(f"VGG-style dense head: {vgg_head:,} of {results['VGG-style'][0]:,} parameters ({vgg_head / results['VGG-style'][0]:.0%})")
best_conv = max(v[3] for k, v in results.items() if k not in ("MLP", "LeNet"))
assert best_conv > results["MLP"][3] + 0.15
assert results["MobileNetV2-style"][1] < results["ResNet-style"][1]
lat_per_mflop = {k: v[2] / (v[1] / 1e6) for k, v in results.items()}
assert lat_per_mflop["MobileNetV2-style"] > lat_per_mflop["ResNet-style"], "fewer FLOPs, but not proportionally faster"

print("\nAll checks passed.")
