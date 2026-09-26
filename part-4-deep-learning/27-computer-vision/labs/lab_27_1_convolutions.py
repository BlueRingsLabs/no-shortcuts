# %% [markdown]
# # Lab 27.1: Convolutions from scratch
#
# 1. conv2d as nested loops and as im2col + one matmul, matched against F.conv2d (stride, padding, dilation, groups).
# 2. Output shapes and parameter counts: formula vs PyTorch, including the self-check.
# 3. Receptive fields: the formula, and the effective receptive field measured with gradients.
# 4. Translation equivariance (and where striding breaks it).
# 5. The transposed convolution is the adjoint of the convolution.
# 6. CNN vs MLP on digits and on shifted digits.

# %%
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(271)
torch.set_default_dtype(torch.float64)

# %% [markdown]
# ## 1. conv2d two ways

# %%
def out_size(n, k, s=1, p=0, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def conv2d_loops(x, w, b, s=1, p=0, d=1, groups=1):
    N, C, H, W = x.shape
    O, Cg, k, _ = w.shape
    xp = F.pad(x, (p, p, p, p))
    Ho, Wo = out_size(H, k, s, p, d), out_size(W, k, s, p, d)
    y = torch.zeros(N, O, Ho, Wo)
    og = O // groups
    for o in range(O):
        g = o // og
        for i in range(Ho):
            for j in range(Wo):
                patch = xp[:, g * Cg:(g + 1) * Cg, i * s: i * s + d * (k - 1) + 1: d, j * s: j * s + d * (k - 1) + 1: d]
                y[:, o, i, j] = (patch * w[o]).sum((1, 2, 3)) + b[o]
    return y


def conv2d_im2col(x, w, b, s=1, p=0, d=1):
    N, C, H, W = x.shape
    O, _, k, _ = w.shape
    cols = F.unfold(x, kernel_size=k, dilation=d, padding=p, stride=s)       # (N, C*k*k, L): every patch as a column
    y = w.view(O, -1) @ cols + b[:, None]                                   # one matrix multiplication
    return y.view(N, O, out_size(H, k, s, p, d), out_size(W, k, s, p, d))


x = torch.randn(2, 4, 11, 13)
for (s, p, d, groups) in [(1, 0, 1, 1), (2, 1, 1, 1), (1, 2, 2, 1), (2, 1, 1, 2), (1, 1, 1, 4)]:
    w = torch.randn(8, 4 // groups, 3, 3); b = torch.randn(8)
    ref = F.conv2d(x, w, b, stride=s, padding=p, dilation=d, groups=groups)
    assert torch.allclose(conv2d_loops(x, w, b, s, p, d, groups), ref, atol=1e-10)
    if groups == 1:
        assert torch.allclose(conv2d_im2col(x, w, b, s, p, d), ref, atol=1e-10)
print("loops and im2col match F.conv2d for stride, padding, dilation and groups")

xb = torch.randn(8, 16, 32, 32); wb = torch.randn(32, 16, 3, 3); bb = torch.randn(32)
t0 = time.perf_counter(); conv2d_loops(xb[:1], wb, bb, p=1); t_loop = time.perf_counter() - t0
conv2d_im2col(xb[:1], wb, bb, p=1)                              # warm-up (first call pays one-time costs)
t0 = time.perf_counter()
for _ in range(20):
    conv2d_im2col(xb[:1], wb, bb, p=1)
t_col = (time.perf_counter() - t0) / 20
print(f"one 32x32 image, 16->32 channels: Python loops {t_loop * 1000:.0f} ms, im2col {t_col * 1000:.2f} ms ({t_loop / t_col:.0f}x)")

# %% [markdown]
# ## 2. Shapes and parameter counts

# %%
print(f"self-check: 3x3, stride 2, padding 1 on 64x64 -> {out_size(64, 3, 2, 1)}x{out_size(64, 3, 2, 1)}; "
      f"PyTorch: {tuple(nn.Conv2d(1, 1, 3, 2, 1)(torch.randn(1, 1, 64, 64)).shape[-2:])}")
assert out_size(64, 3, 2, 1) == 32
stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3), nn.MaxPool2d(3, 2, 1))
print("ResNet stem on 224x224:", tuple(stem(torch.randn(1, 3, 224, 224)).shape), "(formula: 56x56)")
count = lambda m: sum(p.numel() for p in m.parameters())
full = nn.Conv2d(256, 256, 3, padding=1)
sep = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, groups=256), nn.Conv2d(256, 256, 1))
print(f"3x3 conv 256->256: {count(full):,} params; depthwise separable: {count(sep):,} ({count(full) / count(sep):.1f}x fewer)")
assert count(full) == 256 * (256 * 9 + 1) and count(sep) == 256 * 10 + 256 * 257
mlp_first = nn.Linear(3 * 224 * 224, 1000)
print(f"for comparison, one dense layer from a 224x224 RGB image to 1,000 units: {count(mlp_first):,} params")

# %% [markdown]
# ## 3. Receptive fields: theoretical and effective

# %%
def receptive_field(layers):
    r, jump = 1, 1
    for k, s in layers:
        r += (k - 1) * jump
        jump *= s
    return r


layers = [(3, 1), (3, 2), (3, 1), (3, 2), (3, 1)]
print(f"receptive field of conv3 s1, s2, s1, s2, s1: {receptive_field(layers)} (exercise 3)")
assert receptive_field(layers) == 21

net = nn.Sequential(*[m for _ in range(8) for m in (nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())])
for m in net:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
inp = torch.randn(64, 8, 41, 41, requires_grad=True)
net(inp)[:, :, 20, 20].sum().backward()                        # gradient of the center output unit
g = inp.grad.abs().sum((0, 1))
nonzero = (g > 0).nonzero()
extent = (nonzero[:, 0].max() - nonzero[:, 0].min() + 1).item()
mass = g / g.sum()
central = mass[20 - 3: 20 + 4, 20 - 3: 20 + 4].sum().item()
print(f"8 layers of 3x3: theoretical receptive field {receptive_field([(3, 1)] * 8)}, nonzero gradient over {extent}x{extent}; "
      f"but {central:.0%} of the gradient mass sits in the central 7x7 (the effective receptive field is much smaller)")
assert extent == 17 and central > 0.5

# %% [markdown]
# ## 4. Equivariance

# %%
conv = nn.Conv2d(1, 4, 3, padding=1, padding_mode="circular")
img = torch.randn(1, 1, 16, 16)
shifted = torch.roll(img, shifts=(3, 5), dims=(2, 3))
assert torch.allclose(conv(shifted), torch.roll(conv(img), shifts=(3, 5), dims=(2, 3)))
print("stride-1 convolution with circular padding: shift then convolve == convolve then shift")
strided = nn.Conv2d(1, 4, 3, stride=2, padding=1, padding_mode="circular")
for sh in (1, 2):
    a = strided(torch.roll(img, sh, dims=3)); b_ = torch.roll(strided(img), sh // 2, dims=3)
    print(f"stride 2, input shifted by {sh}: output equals a shifted output? {torch.allclose(a, b_)}")
print("(only shifts that are multiples of the stride survive downsampling exactly)")

# %% [markdown]
# ## 5. Transposed convolution = adjoint

# %%
w = torch.randn(6, 3, 3, 3)
xa = torch.randn(1, 3, 10, 10); yb = torch.randn(1, 6, 5, 5)
lhs = (F.conv2d(xa, w, stride=2, padding=1) * yb).sum()          # <conv(x), y>
rhs = (xa * F.conv_transpose2d(yb, w, stride=2, padding=1, output_padding=1)).sum()   # <x, conv^T(y)>
print(f"<conv(x), y> = {lhs.item():.6f}, <x, conv_transpose(y)> = {rhs.item():.6f}")
assert torch.allclose(lhs, rhs)
xg = xa.clone().requires_grad_(True)
(F.conv2d(xg, w, stride=2, padding=1) * yb).sum().backward()
assert torch.allclose(xg.grad, F.conv_transpose2d(yb, w, stride=2, padding=1, output_padding=1))
print("and the gradient of a convolution with respect to its input is exactly that transposed convolution")

# %% [markdown]
# ## 6. CNN vs MLP, on digits and on shifted digits

# %%
torch.set_default_dtype(torch.float32)
digits = load_digits()
X = torch.tensor(digits.images / 16, dtype=torch.float32)[:, None]   # (n, 1, 8, 8)
y = torch.tensor(digits.target)
i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.4, random_state=0, stratify=digits.target)
X_tr, y_tr, X_te, y_te = X[i_tr], y[i_tr], X[i_te], y[i_te]


def shift(xb, dx, dy):
    out = torch.zeros_like(xb)
    ys, yd = slice(max(0, -dy), 8 - max(0, dy)), slice(max(0, dy), 8 - max(0, -dy))
    xs, xd = slice(max(0, -dx), 8 - max(0, dx)), slice(max(0, dx), 8 - max(0, -dx))
    out[..., yd, xd] = xb[..., ys, xs]
    return out


g = torch.Generator().manual_seed(7)
X_te_shift = torch.stack([shift(xi, *torch.randint(-1, 2, (2,), generator=g).tolist()) for xi in X_te])

cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(128, 10))
mlp = nn.Sequential(nn.Flatten(), nn.Linear(64, 150), nn.ReLU(), nn.Linear(150, 10))


def train(model, epochs=40):
    torch.manual_seed(0)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X_tr))
        for s in range(0, len(X_tr), 64):
            idx = perm[s:s + 64]
            opt.zero_grad(); F.cross_entropy(model(X_tr[idx]), y_tr[idx]).backward(); opt.step()
    with torch.no_grad():
        return (model(X_te).argmax(1) == y_te).float().mean().item(), (model(X_te_shift).argmax(1) == y_te).float().mean().item()


for name, model in (("CNN", cnn), ("MLP", mlp)):
    acc, acc_shift = train(model)
    print(f"{name}: {count(model):6,} params, test accuracy {acc:.3f}, on digits shifted by up to one pixel {acc_shift:.3f}")
    if name == "CNN":
        cnn_shift = acc_shift
    else:
        mlp_shift = acc_shift
print("no augmentation for either: the CNN degrades much less, though a one-pixel shift of an 8x8 digit is 12% of its width,")
print("and strided/pooled layers aren't exactly shift-invariant. Augmentation (26.3) finishes the job.")
assert cnn_shift > mlp_shift + 0.15

print("\nAll checks passed.")
