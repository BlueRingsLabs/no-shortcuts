# %% [markdown]
# # Lab 31.4: Flow matching, guidance and latent generation
#
# 1. Flow matching on lab 31.2's ring: regress a velocity field, sample by integrating an ODE. Quality vs Euler steps.
# 2. How curved are the paths? And reflow: retrain on the model's own (noise, sample) pairs to straighten them.
# 3. Classifier-free guidance on digits: one conditional model, a guidance scale, and the fidelity/diversity tradeoff.
# 4. Latent generation: an autoencoder squeezes digits to 8 numbers; the flow model works there.

# %%
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

torch.manual_seed(314)
torch.set_num_threads(4)


class VelocityNet(nn.Module):
    def __init__(self, dim, h=256, n_classes=0):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, h) if n_classes else None          # the extra index is "no label"
        self.inp, self.temb = nn.Linear(dim, h), nn.Sequential(nn.Linear(32, h), nn.SiLU(), nn.Linear(h, h))
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))

    def forward(self, x, t, y=None):
        freqs = torch.exp(-math.log(1000) * torch.arange(16) / 16)
        ang = 1000 * t[:, None] * freqs                                                # t in [0, 1]
        h = self.inp(x) + self.temb(torch.cat([ang.sin(), ang.cos()], 1))
        if self.emb is not None:
            h = h + self.emb(y)
        return self.net(h)


def train_fm(model, data_fn, steps, bs=256, lr=1e-3, seed=0, p_uncond=0.0, n_classes=0, pairs=None):
    """Conditional flow matching with straight paths: x_t = (1 - t) x0 + t x1, x0 ~ N(0, I), target velocity x1 - x0."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for _ in range(steps):
        if pairs is None:
            x1, y = data_fn(bs, g)
            x0 = torch.randn(x1.shape, generator=g)                                    # independent noise
        else:
            i = torch.randint(0, len(pairs[0]), (bs,), generator=g)
            x0, x1, y = pairs[0][i], pairs[1][i], None                                 # coupled pairs (reflow)
        if n_classes:
            drop = torch.rand(bs, generator=g) < p_uncond
            y = torch.where(drop, torch.full_like(y, n_classes), y)                    # sometimes train without the label
        t = torch.rand(bs, generator=g)
        xt = (1 - t)[:, None] * x0 + t[:, None] * x1
        loss = F.mse_loss(model(xt, t, y), x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def euler(model, x0, steps, y=None, guidance=None, n_classes=0, return_path=False):
    x, path = x0.clone(), [x0.clone()]
    for i in range(steps):
        t = torch.full((len(x),), i / steps)
        if guidance is None:
            v = model(x, t, y)
        else:                                                                          # classifier-free guidance
            v_c, v_u = model(x, t, y), model(x, t, torch.full_like(y, n_classes))
            v = v_u + guidance * (v_c - v_u)
        x = x + v / steps
        path.append(x.clone())
    return (x, torch.stack(path)) if return_path else x


# %% [markdown]
# ## 1. Flow matching on the ring

# %%
CENTERS = torch.tensor([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 9)[:-1]], dtype=torch.float32) * 2
SD = 0.1


def ring(n, g):
    return CENTERS[torch.randint(0, 8, (n,), generator=g)] + SD * torch.randn(n, 2, generator=g), None


def mode_stats(s):
    near, which = torch.cdist(s, CENTERS).min(1)
    good = near < 3 * SD
    return int((torch.bincount(which[good], minlength=8) >= 0.02 * len(s)).sum()), good.float().mean().item()


t0 = time.time()
fm = train_fm(VelocityNet(2), ring, 6000)
print(f"flow matching model trained in {time.time() - t0:.0f}s (same network and budget as lab 31.3's denoisers)")
z = torch.randn(2000, 2, generator=torch.Generator().manual_seed(9))
print("\nEuler steps   modes   quality")
fm_res = {}
for steps in (1, 2, 5, 10, 25, 100):
    fm_res[steps] = mode_stats(euler(fm, z, steps))
    print(f"{steps:11d}   {fm_res[steps][0]:5d}   {fm_res[steps][1]:.2f}")
print("lab 31.3, cosine DDPM: DDIM 5 steps 0.79, 10 steps 0.91, 25 steps 0.96; DDPM 1,000 steps 0.98")
assert fm_res[100][0] == 8 and fm_res[100][1] > 0.9 and fm_res[1][1] < 0.5

# %% [markdown]
# ## 2. Curved paths, and reflow
#
# Each training pair (noise, data point) is joined by a straight line, but the pairs are random, so the lines cross,
# and the learned velocity at a crossing is their average: the model's own paths are curved. Reflow: sample
# (z, ODE(z)) pairs from the trained model and train again on those. The new pairs don't cross, so the new paths are
# straighter, and few-step sampling works.

# %%
def straightness(model, z, steps=100):
    _, path = euler(model, z, steps, return_path=True)
    length = (path[1:] - path[:-1]).norm(dim=-1).sum(0)
    return ((path[-1] - path[0]).norm(dim=-1) / length).mean().item()          # 1 = a straight line


zr = torch.randn(20000, 2, generator=torch.Generator().manual_seed(1))
pairs = (zr, euler(fm, zr, 100))
t0 = time.time()
fm2 = train_fm(VelocityNet(2), None, 6000, pairs=pairs)
print(f"\nreflow model trained in {time.time() - t0:.0f}s on 20,000 (noise, sample) pairs from the first model")
print(f"straightness (distance / path length, 1 = straight): original {straightness(fm, z[:500]):.3f}, reflow {straightness(fm2, z[:500]):.3f}")
print("Euler steps   original      reflow")
rf = {}
for steps in (1, 2, 5):
    rf[steps] = mode_stats(euler(fm2, z, steps))
    print(f"{steps:11d}   {fm_res[steps][0]}/{fm_res[steps][1]:.2f}      {rf[steps][0]}/{rf[steps][1]:.2f}")
print("reflow buys few-step quality with a second training run, and inherits the first model's mistakes.")
assert rf[1][1] > fm_res[1][1] + 0.2 and straightness(fm2, z[:500]) > straightness(fm, z[:500])

# %% [markdown]
# ## 3. Classifier-free guidance
#
# A digit model conditioned on the label, trained with the label replaced by "no label" 10% of the time, so the same
# network is also an unconditional model. At sampling time: v = v_uncond + w (v_cond - v_uncond). w = 1 is the plain
# conditional model; w > 1 pushes further in the direction that makes the sample more like its label.

# %%
digits = load_digits()
Xd = torch.tensor(digits.data / 16.0, dtype=torch.float32) * 2 - 1
yd = torch.tensor(digits.target)
judge = LogisticRegression(C=10, max_iter=5000).fit(digits.data / 16.0, digits.target)


def digit_batch(n, g):
    i = torch.randint(0, len(Xd), (n,), generator=g)
    return Xd[i], yd[i]


t0 = time.time()
cfm = train_fm(VelocityNet(64, h=512, n_classes=10), digit_batch, 8000, bs=128, p_uncond=0.1, n_classes=10)
print(f"\nconditional digit model trained in {time.time() - t0:.0f}s")
labels = torch.arange(10).repeat_interleave(100)
z64 = torch.randn(1000, 64, generator=torch.Generator().manual_seed(4))
X01 = (Xd + 1) / 2
print("guidance w   on-label (judge)   pixels past the data range   distance to nearest real digit of that class")
cfg = {}
for w in (0.0, 1.0, 2.0, 4.0, 6.0):
    raw = euler(cfm, z64, 50, labels, guidance=w, n_classes=10)
    s01 = ((raw + 1) / 2).clamp(0, 1)
    agree = (judge.predict_proba(s01.numpy()).argmax(1) == labels.numpy()).mean()
    over = (raw.abs() > 1.05).float().mean().item()
    nn_dist = np.mean([torch.cdist(s01[labels == d], X01[yd == d]).min(1).values.mean().item() for d in range(10)])
    cfg[w] = (agree, over, nn_dist)
    print(f"{w:10.1f}   {agree:16.3f}   {over:26.3f}   {nn_dist:44.3f}")
print("w = 0 ignores the label. On this easy task w = 1 is already on-label; w = 2 makes samples a little more typical")
print("(closer to real digits); beyond that, guidance mostly shoves pixels past the range of the data, the")
print("oversaturated look of high-guidance images, and samples drift away from real digits again.")
assert cfg[0.0][0] < 0.25 and cfg[1.0][0] > 0.9
assert cfg[2.0][1] < cfg[4.0][1] < cfg[6.0][1] and cfg[2.0][2] < cfg[6.0][2]

# %% [markdown]
# ## 4. Latent flow matching
#
# Stable Diffusion's move: don't generate pixels; generate the latent code of a pretrained autoencoder, then decode. The
# autoencoder does perception (edges, strokes); the generative model only has to model 8 numbers instead of 64. Here
# the savings are small; at 512 x 512 x 3 -> 64 x 64 x 4, they're a factor of 48.

# %%
class AE(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, k))
        self.dec = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, 64))


t0 = time.time()
torch.manual_seed(0)
ae = AE()
opt = torch.optim.Adam(ae.parameters(), lr=2e-3)
g = torch.Generator().manual_seed(0)
for step in range(4000):
    xb = Xd[torch.randint(0, len(Xd), (128,), generator=g)]
    code = ae.enc(xb)
    loss = F.mse_loss(ae.dec(code + 0.05 * torch.randn(code.shape, generator=g)), xb) + 1e-3 * code.pow(2).mean()   # a little noise and a little weight on the codes keep the latent space smooth and bounded
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    codes = ae.enc(Xd)
    mu_c, sd_c = codes.mean(0), codes.std(0)
    rec = F.mse_loss(ae.dec(codes), Xd).item()
print(f"\nautoencoder (64 -> 8 -> 64) trained in {time.time() - t0:.0f}s, reconstruction MSE {rec:.4f} on the [-1, 1] scale")
latent_data = (codes - mu_c) / sd_c                                                   # standardized, like SD's scale factor


def latent_batch(n, g):
    i = torch.randint(0, len(Xd), (n,), generator=g)
    return latent_data[i], yd[i]


t0 = time.time()
lfm = train_fm(VelocityNet(8, h=256, n_classes=10), latent_batch, 8000, bs=128, p_uncond=0.1, n_classes=10)
t_latent = time.time() - t0
z8 = torch.randn(1000, 8, generator=torch.Generator().manual_seed(4))
with torch.no_grad():
    lat = euler(lfm, z8, 50, labels, guidance=2.0, n_classes=10)
    s_lat = ((ae.dec(lat * sd_c + mu_c) + 1) / 2).clamp(0, 1)
p = judge.predict_proba(s_lat.numpy())
nn_lat = np.mean([torch.cdist(s_lat[labels == d], X01[yd == d]).min(1).values.mean().item() for d in range(10)])
print(f"latent model (8 dims, hidden 256) trained in {t_latent:.0f}s. Guidance 2: on-label {(p.argmax(1) == labels.numpy()).mean():.3f}, "
      f"distance to nearest real digit {nn_lat:.3f}")
print(f"pixel model (64 dims, hidden 512), guidance 2: on-label {cfg[2.0][0]:.3f}, distance to nearest real digit {cfg[2.0][2]:.3f}")
print("a smaller generative model in a smaller space, as good or better: the autoencoder already knows what strokes are.")
chars = " .:-=+*#%@"
row = s_lat[::100].numpy()
print("one latent sample per digit:\n" + "\n".join("  ".join("".join(chars[min(9, int(v * 10))] for v in s[r * 8:(r + 1) * 8]) for s in row) for r in range(8)))
assert (p.argmax(1) == labels.numpy()).mean() > 0.8 and nn_lat < cfg[2.0][2] + 0.1

print("\nAll checks passed.")
