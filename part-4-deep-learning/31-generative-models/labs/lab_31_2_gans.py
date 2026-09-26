# %% [markdown]
# # Lab 31.2: GANs, and the ways they fail
#
# 1. The saturating generator loss has no gradient exactly when the generator needs one most.
# 2. The optimal discriminator is p_data / (p_data + p_g): check it on 1D data.
# 3. A ring of 8 Gaussians: vanilla (non-saturating) GAN vs WGAN-GP. Mode coverage and sample quality over training.
# 4. A GAN on 8x8 digits, with a classifier counting which digits it makes.

# %%
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

torch.manual_seed(312)
torch.set_num_threads(4)

# %% [markdown]
# ## 1. Saturating vs non-saturating
#
# The minimax game has the generator minimize log(1 - D(G(z))). Early on D easily spots fakes: D(G(z)) ~ 0. With
# D = sigmoid(l), look at the gradient of each generator loss with respect to the discriminator's logit l.

# %%
l = torch.linspace(-8, 2, 6, requires_grad=True)
sat = torch.log(1 - torch.sigmoid(l)).sum()            # minimize this (the original minimax loss)
ns = -F.logsigmoid(l).sum()                             # minimize this instead (non-saturating: maximize log D(G(z)))
g_sat, = torch.autograd.grad(sat, l)
g_ns, = torch.autograd.grad(ns, l)
print("D(G(z))     gradient of log(1 - D)   gradient of -log D")
for li, a, b in zip(l.detach(), g_sat, g_ns):
    print(f"{torch.sigmoid(li):9.4f}   {a:22.4f}   {b:18.4f}")
print("when the discriminator is winning (D ~ 0), the saturating loss gives the generator almost nothing to follow.")
assert abs(g_sat[0]) < 1e-3 and abs(g_ns[0]) > 0.99

# %% [markdown]
# ## 2. The optimal discriminator

# %%
real = torch.randn(20000, 1) * 0.5 + 1.0                                     # p_data = N(1, 0.5^2)
fake = torch.randn(20000, 1) * 1.0 - 0.5                                     # a fixed "generator": N(-0.5, 1)
D = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
opt = torch.optim.Adam(D.parameters(), lr=3e-3)
for step in range(1500):
    i = torch.randint(0, 20000, (512,))
    loss = F.binary_cross_entropy_with_logits(D(real[i]), torch.ones(512, 1)) + \
        F.binary_cross_entropy_with_logits(D(fake[i]), torch.zeros(512, 1))
    opt.zero_grad(); loss.backward(); opt.step()
xs = torch.linspace(-2, 3, 6)[:, None]
pd = torch.distributions.Normal(1.0, 0.5).log_prob(xs).exp()
pg = torch.distributions.Normal(-0.5, 1.0).log_prob(xs).exp()
with torch.no_grad():
    learned = torch.sigmoid(D(xs))
print("\n   x     learned D(x)   p_data / (p_data + p_g)")
for x_, a, b in zip(xs[:, 0], learned[:, 0], (pd / (pd + pg))[:, 0]):
    print(f"{x_:5.1f}   {a:12.3f}   {b:12.3f}")
assert (learned[:5, 0] - (pd / (pd + pg))[:5, 0]).abs().max() < 0.08
print("matches wherever there's data. At x = 3 both densities are tiny (4 and 3.5 standard deviations out), no samples")
print("land there, and the network's value is whatever its extrapolation happens to be. A discriminator only knows the")
print("regions it has seen, which matters when the generator wanders somewhere new.")
print("plugging D* into the minimax objective gives 2 JSD(p_data || p_g) - log 4: the generator minimizes a Jensen-Shannon divergence")

# %% [markdown]
# ## 3. Eight Gaussians on a ring
#
# A sample is "high quality" if it lies within 3 standard deviations of some mode; a mode is "covered" if it receives at
# least 2% of 2,000 samples (a uniform generator gives each 12.5%).

# %%
CENTERS = torch.tensor([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 9)[:-1]], dtype=torch.float32) * 2
SD = 0.1


def sample_ring(n, g=None):
    k = torch.randint(0, 8, (n,), generator=g)
    return CENTERS[k] + SD * torch.randn(n, 2, generator=g)


def mode_stats(s):
    d = torch.cdist(s, CENTERS)
    near, which = d.min(1)
    good = near < 3 * SD
    counts = torch.bincount(which[good], minlength=8)
    return int((counts >= 0.02 * len(s)).sum()), good.float().mean().item()


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


GP_WEIGHT = 0.1   # lambda = 10 is the paper's setting for images. On this 2D ring it over-constrains the critic: training
                  # stalls for 1,000 to 2,000+ steps, how long depends on the seed and even on the CPU's rounding, and some
                  # runs never cover every mode. The paper's code uses 0.1 for its toy datasets, for this reason.


def train_gan(kind, steps=3000, seed=0, log_every=500):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    G, D = mlp(2, 2), mlp(2, 1)
    if kind == "vanilla":
        optG, optD = (torch.optim.Adam(m.parameters(), lr=2e-4, betas=(0.5, 0.999)) for m in (G, D))
    else:                                                                             # the settings of Gulrajani et al.'s own toy
        optG, optD = (torch.optim.Adam(m.parameters(), lr=1e-4, betas=(0.5, 0.9)) for m in (G, D))   # experiments
    history = []
    for step in range(steps):
        for _ in range(1 if kind == "vanilla" else 5):                        # WGAN: several critic steps per generator step
            x = sample_ring(256, g)
            fake = G(torch.randn(256, 2, generator=g)).detach()
            if kind == "vanilla":
                lossD = F.binary_cross_entropy_with_logits(D(x), torch.ones(256, 1)) + \
                    F.binary_cross_entropy_with_logits(D(fake), torch.zeros(256, 1))
            else:
                eps = torch.rand(256, 1, generator=g)
                mid = (eps * x + (1 - eps) * fake).requires_grad_()
                grad, = torch.autograd.grad(D(mid).sum(), mid, create_graph=True)
                gp = ((grad.norm(dim=1) - 1) ** 2).mean()                     # gradient penalty: the critic stays 1-Lipschitz
                lossD = D(fake).mean() - D(x).mean() + GP_WEIGHT * gp
            optD.zero_grad(); lossD.backward(); optD.step()
        fake = G(torch.randn(256, 2, generator=g))
        lossG = -F.logsigmoid(D(fake)).mean() if kind == "vanilla" else -D(fake).mean()
        optG.zero_grad(); lossG.backward(); optG.step()
        if (step + 1) % log_every == 0:
            with torch.no_grad():
                history.append(mode_stats(G(torch.randn(2000, 2, generator=torch.Generator().manual_seed(9)))))
    return G, history


t0 = time.time()
ring = {}
print("\nmodes covered / high-quality fraction, every 500 generator steps")
for kind, seeds, steps in (("vanilla", (0, 1, 2), 3000), ("wgan-gp", (0, 1), 1500)):
    for seed in seeds:
        _, hist = train_gan(kind, steps=steps, seed=seed)
        ring[(kind, seed)] = hist
        print(f"  {kind:8s} seed {seed}: " + "  ".join(f"{m}/{q:.2f}" for m, q in hist))
print(f"({time.time() - t0:.0f}s) same code, same data, different seeds: the vanilla GAN is excellent, late, or lost.")
print("WGAN-GP covers every mode every time, and puts more samples between the modes: reliable, less sharp.")
final = {k: v[-1] for k, v in ring.items()}
assert all(final[("wgan-gp", s)][0] == 8 for s in (0, 1))
assert min(final[("vanilla", s)][0] for s in (0, 1, 2)) < 8
assert max(final[("vanilla", s)][1] for s in (0, 1, 2)) > max(final[("wgan-gp", s)][1] for s in (0, 1)) + 0.1

# %% [markdown]
# ## 4. Digits
#
# An MLP GAN on 8x8 digits. The judge (a logistic regression trained on real digits) tells us which digit each sample
# looks like. A generator that only makes a few kinds of digit has collapsed, however good each one looks.

# %%
digits = load_digits()
Xd = torch.tensor(digits.data / 16.0, dtype=torch.float32) * 2 - 1            # [-1, 1], for a tanh output
judge = LogisticRegression(C=10, max_iter=5000).fit(digits.data / 16.0, digits.target)


def train_digit_gan(steps=4000, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    G = nn.Sequential(nn.Linear(16, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64), nn.Tanh())
    D = nn.Sequential(nn.Linear(64, 256), nn.LeakyReLU(0.2), nn.Linear(256, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))
    optG, optD = (torch.optim.Adam(m.parameters(), lr=2e-4, betas=(0.5, 0.999)) for m in (G, D))
    for step in range(steps):
        x = Xd[torch.randint(0, len(Xd), (128,), generator=g)]
        fake = G(torch.randn(128, 16, generator=g))
        lossD = F.binary_cross_entropy_with_logits(D(x), torch.full((128, 1), 0.9)) + \
            F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(128, 1))   # one-sided label smoothing
        optD.zero_grad(); lossD.backward(); optD.step()
        lossG = -F.logsigmoid(D(fake)).mean()
        optG.zero_grad(); lossG.backward(); optG.step()
    return G


t0 = time.time()
Gd = train_digit_gan()
with torch.no_grad():
    samples = ((Gd(torch.randn(2000, 16, generator=torch.Generator().manual_seed(5))) + 1) / 2).clamp(0, 1).numpy()
proba = judge.predict_proba(samples)
counts = np.bincount(proba.argmax(1), minlength=10)
print(f"\ndigit GAN trained in {time.time() - t0:.0f}s. Judge's mean confidence on samples {proba.max(1).mean():.3f} "
      f"(real digits: {judge.predict_proba(digits.data / 16.0).max(1).mean():.3f})")
print(f"which digits the 2,000 samples look like: {counts.tolist()} (a uniform generator: 200 each)")
chars = " .:-=+*#%@"
print("\n".join("  ".join("".join(chars[min(9, int(v * 10))] for v in s[r * 8:(r + 1) * 8]) for s in samples[:10]) for r in range(8)))
rare = [d for d in range(10) if counts[d] < 100]
print(f"digits the generator almost never makes: {rare}. Each sample looks fine; the distribution doesn't. Partial mode")
print("collapse is invisible if you look at samples one at a time, which is how GAN papers used to be reviewed.")
assert proba.max(1).mean() > 0.7 and len(rare) >= 2

print("\nAll checks passed.")
