# %% [markdown]
# # Lab 31.3: Diffusion models (DDPM and DDIM)
#
# 1. The forward process: iterating the noising steps and the closed form q(x_t | x_0) agree.
# 2. Noise schedules: linear vs cosine, seen through the signal-to-noise ratio.
# 3. The denoiser learns the score: on a 1D mixture of Gaussians, compare the trained network with the exact answer.
# 4. DDPM on lab 31.2's ring of 8 Gaussians: mode coverage and quality, against the GANs.
# 5. DDIM: the same trained model, sampled deterministically in far fewer steps.
# 6. Digits: does diffusion avoid the GAN's partial mode collapse?

# %%
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

torch.manual_seed(313)
torch.set_num_threads(4)

T = 1000
betas_lin = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)                   # Ho et al.'s linear schedule


def cosine_alpha_bar(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0])[1:].clamp(1e-5, 1.0)                                        # Nichol & Dhariwal's cosine schedule


alpha_bar = torch.cumprod(1 - betas_lin, 0)

# %% [markdown]
# ## 1. The forward process

# %%
x0 = torch.tensor([1.5, -0.5], dtype=torch.float64).repeat(100_000, 1)
x = x0.clone()
g = torch.Generator().manual_seed(0)
for t in range(300):                                                              # x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) eps
    x = torch.sqrt(1 - betas_lin[t]) * x + torch.sqrt(betas_lin[t]) * torch.randn(x.shape, generator=g, dtype=torch.float64)
ab = alpha_bar[299]
print(f"after 300 steps: empirical mean {x.mean(0).numpy().round(4)}, var {x.var(0).numpy().round(4)}")
print(f"closed form:     mean {(ab.sqrt() * x0[0]).numpy().round(4)}, var {1 - ab.item():.4f}  (sqrt(alpha_bar) x_0, 1 - alpha_bar)")
assert torch.allclose(x.mean(0), ab.sqrt() * x0[0], atol=0.01) and torch.allclose(x.var(0), (1 - ab).expand(2), atol=0.01)

# %% [markdown]
# ## 2. Schedules

# %%
ab_cos = cosine_alpha_bar(T)
print("\n  t      alpha_bar linear   cosine    log SNR linear   cosine")
for t in (0, 100, 250, 500, 750, 900, 999):
    snr = lambda a: math.log(a / (1 - a))
    print(f"{t:4d}   {alpha_bar[t]:14.4f}   {ab_cos[t]:7.4f}   {snr(alpha_bar[t]):13.2f}   {snr(ab_cos[t]):6.2f}")
print("the linear schedule keeps 8% of the signal variance at t = 500 and 0.3% at t = 750: its last quarter of steps is")
print("spent on what is nearly pure noise. The cosine schedule spends them more evenly (Nichol & Dhariwal's motivation).")
assert alpha_bar[750] < 0.01 and ab_cos[750] > 0.1

# %% [markdown]
# ## 3. The denoiser is a score estimator
#
# Data: a 1D mixture 0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2). For Gaussian noising, the best possible noise prediction at a
# point x_t is eps*(x_t, t) = -sqrt(1 - alpha_bar_t) * d/dx log p_t(x_t), where p_t is the noised data density, known
# exactly here (another mixture). Train a small network and compare.

# %%
class Denoiser(nn.Module):
    def __init__(self, dim, h=256, n_classes=0):
        super().__init__()
        self.emb = nn.Embedding(n_classes, h) if n_classes else None
        self.inp, self.temb = nn.Linear(dim, h), nn.Sequential(nn.Linear(32, h), nn.SiLU(), nn.Linear(h, h))
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))

    def forward(self, x, t, y=None):
        freqs = torch.exp(-math.log(1000) * torch.arange(16) / 16)
        ang = t[:, None].float() * freqs                                           # sinusoidal time embedding (29.3)
        h = self.inp(x) + self.temb(torch.cat([ang.sin(), ang.cos()], 1))
        if self.emb is not None and y is not None:
            h = h + self.emb(y)
        return self.net(h)


def train_ddpm(model, data_fn, steps, ab, bs=256, lr=1e-3, seed=0, labels=None):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    abf = ab.float()
    for _ in range(steps):
        x0, y = data_fn(bs, g)
        t = torch.randint(0, T, (bs,), generator=g)
        eps = torch.randn(x0.shape, generator=g)
        xt = abf[t].sqrt()[:, None] * x0 + (1 - abf[t]).sqrt()[:, None] * eps
        loss = F.mse_loss(model(xt, t, y), eps)                                    # the "simple" loss: predict the noise
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def mix1d(n, g):
    c = torch.where(torch.rand(n, generator=g) < 0.5, -2.0, 2.0)
    return (c + 0.3 * torch.randn(n, generator=g))[:, None], None


t0 = time.time()
den1d = train_ddpm(Denoiser(1), mix1d, 3000, alpha_bar)
print(f"\n1D denoiser trained in {time.time() - t0:.0f}s")
print("   t     x_t    network eps   exact eps*")
errs = []
for t in (50, 200, 500):
    a = alpha_bar[t].item()
    xs = torch.linspace(-3, 3, 7)
    mus, var = torch.tensor([-2.0, 2.0]) * math.sqrt(a), 0.09 * a + (1 - a)       # the noised mixture p_t
    logw = -(xs[:, None] - mus) ** 2 / (2 * var)
    w = logw.softmax(1)
    score = (w * (mus - xs[:, None]) / var).sum(1)                                 # d/dx log p_t(x)
    exact = -math.sqrt(1 - a) * score
    with torch.no_grad():
        net = den1d(xs[:, None], torch.full((7,), t))[:, 0]
    errs.append((net - exact).abs())
    for xi, n_, e_ in zip(xs[::2], net[::2], exact[::2]):
        print(f"{t:4d}   {xi:5.1f}   {n_:11.3f}   {e_:10.3f}")
errs = torch.stack(errs)
worst_t, worst_x = divmod(int(errs.argmax()), 7)
print(f"mean |error| {errs.mean():.3f}; the largest ({errs.max():.2f}) is at t = {(50, 200, 500)[worst_t]}, x = {xs[worst_x]:.0f}, in the gap")
print("between the modes, where lightly noised training samples rarely land. The network was only asked to guess the")
print("noise; what it learned, wherever it saw data, is the gradient of the log density: the score.")
assert errs.mean() < 0.1 and errs.max() < 0.4

# %% [markdown]
# ## 4. DDPM on the ring

# %%
CENTERS = torch.tensor([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 9)[:-1]], dtype=torch.float32) * 2
SD = 0.1


def ring(n, g):
    return CENTERS[torch.randint(0, 8, (n,), generator=g)] + SD * torch.randn(n, 2, generator=g), None


def mode_stats(s):
    near, which = torch.cdist(s, CENTERS).min(1)
    good = near < 3 * SD
    return int((torch.bincount(which[good], minlength=8) >= 0.02 * len(s)).sum()), good.float().mean().item()


@torch.no_grad()
def ddpm_sample(model, n, dim, ab, betas, seed=0, y=None, guidance=0.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    abf, bf = ab.float(), betas.float()
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps = model(x, tt, y)
        mean = (x - bf[t] / (1 - abf[t]).sqrt() * eps) / (1 - bf[t]).sqrt()      # the posterior mean, from predicted noise
        x = mean + (bf[t].sqrt() * torch.randn(x.shape, generator=g) if t > 0 else 0)
    return x


@torch.no_grad()
def ddim_sample(model, n, dim, ab, steps, seed=0, y=None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    abf = ab.float()
    ts = torch.linspace(T - 1, 0, steps).long()
    for i, t in enumerate(ts):
        eps = model(x, torch.full((n,), int(t)), y)
        x0_hat = (x - (1 - abf[t]).sqrt() * eps) / abf[t].sqrt()                  # predict the clean sample...
        a_prev = abf[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = a_prev.sqrt() * x0_hat + (1 - a_prev).sqrt() * eps                     # ...and jump straight to the next level
    return x


def betas_from(ab):
    return (1 - ab / torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])).clamp(max=0.999)


print("\nDDPM on the ring, 6,000 training steps, 1,000 sampling steps:")
t0 = time.time()
ring_res = {}
for name, ab_, bt_ in (("linear", alpha_bar, betas_lin), ("cosine", ab_cos, betas_from(ab_cos))):
    den = train_ddpm(Denoiser(2), ring, 6000, ab_)
    ring_res[name] = mode_stats(ddpm_sample(den, 2000, 2, ab_, bt_))
    print(f"  {name} schedule: modes covered {ring_res[name][0]}/8, samples within 3 sd of a mode {ring_res[name][1]:.2f}")
    if name == "cosine":
        den2d = den
print(f"({time.time() - t0:.0f}s) same network, same budget; the schedule decides how much of it goes to useful noise levels.")
print("lab 31.2 for comparison: vanilla GAN seeds (8, 0.68), (8, 0.89), (1, 0.38); WGAN-GP (8, 0.48), (8, 0.43)")
m, q = ring_res["cosine"]
assert m == 8 and q > 0.9 and ring_res["linear"][1] < q - 0.1

# %% [markdown]
# ## 5. DDIM: fewer steps, no noise

# %%
print("\nsampler          steps   modes   quality")
ddim_res = {}
for steps in (5, 10, 25, 100):
    s = ddim_sample(den2d, 2000, 2, ab_cos, steps)
    ddim_res[steps] = mode_stats(s)
    print(f"DDIM           {steps:6d}   {ddim_res[steps][0]:5d}   {ddim_res[steps][1]:.2f}")
print(f"DDPM           {T:6d}   {m:5d}   {q:.2f}")
print("the same network, no retraining. DDIM follows a deterministic path from noise to data and tolerates big jumps.")
assert ddim_res[25][0] == 8 and ddim_res[25][1] > 0.9 and ddim_res[5][1] < ddim_res[100][1]

# %% [markdown]
# ## 6. Digits

# %%
digits = load_digits()
Xd = torch.tensor(digits.data / 16.0, dtype=torch.float32) * 2 - 1
yd = torch.tensor(digits.target)
judge = LogisticRegression(C=10, max_iter=5000).fit(digits.data / 16.0, digits.target)


def digit_batch(n, g):
    i = torch.randint(0, len(Xd), (n,), generator=g)
    return Xd[i], yd[i]


t0 = time.time()
ab_d = cosine_alpha_bar(T)
den_d = train_ddpm(Denoiser(64, h=512), lambda n, g: (digit_batch(n, g)[0], None), 8000, ab_d, bs=128)
print(f"\ndigit denoiser trained in {time.time() - t0:.0f}s")
samples = ((ddim_sample(den_d, 2000, 64, ab_d, 100) + 1) / 2).clamp(0, 1).numpy()
proba = judge.predict_proba(samples)
counts = np.bincount(proba.argmax(1), minlength=10)
print(f"judge's mean confidence {proba.max(1).mean():.3f} (real digits {judge.predict_proba(digits.data / 16.0).max(1).mean():.3f}; "
      f"lab 31.2's GAN 0.849)")
print(f"which digits the 2,000 samples look like: {counts.tolist()} (the GAN: [57, 194, 21, 223, 145, 345, 302, 35, 311, 367])")
chars = " .:-=+*#%@"
print("\n".join("  ".join("".join(chars[min(9, int(v * 10))] for v in s[r * 8:(r + 1) * 8]) for s in samples[:10]) for r in range(8)))
print("diffusion is trained on every example at every noise level with a plain regression loss: nothing rewards dropping")
print("a mode. That, more than image quality, is why it replaced GANs.")
assert counts.min() >= 100

print("\nAll checks passed.")
