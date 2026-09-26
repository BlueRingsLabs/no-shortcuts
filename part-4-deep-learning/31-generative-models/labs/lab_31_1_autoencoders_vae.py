# %% [markdown]
# # Lab 31.1: Autoencoders and VAEs
#
# 1. A linear autoencoder learns PCA's subspace (and not PCA's axes). A nonlinear one does better per latent dimension.
# 2. Why the reparameterization trick: gradient variance of the score-function estimator vs the pathwise one.
# 3. A VAE on 8x8 digits: the closed-form KL checked by Monte Carlo, the ELBO, and its two terms.
# 4. Samples from the prior: plain autoencoder vs VAE.
# 5. beta: trading reconstruction for a latent space you can sample from (rate vs distortion).

# %%
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

torch.manual_seed(311)
torch.set_num_threads(4)

digits = load_digits()
X_all = torch.tensor(digits.data / 16.0, dtype=torch.float32)             # 1,797 images of 64 pixels in [0, 1]
y_all = torch.tensor(digits.target)
perm = torch.randperm(len(X_all), generator=torch.Generator().manual_seed(0))
X, y, Xte, yte = X_all[perm[:1500]], y_all[perm[:1500]], X_all[perm[1500:]], y_all[perm[1500:]]

# a digit classifier, used only as a judge of generated samples
judge = LogisticRegression(C=10, max_iter=5000).fit(X.numpy(), y.numpy())
print(f"judge (logistic regression) test accuracy {judge.score(Xte.numpy(), yte.numpy()):.3f}")


def train(model, loss_fn, epochs=150, lr=2e-3, bs=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        p = torch.randperm(len(X), generator=g)
        for s in range(0, len(X), bs):
            loss = loss_fn(model, X[p[s:s + bs]])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# %% [markdown]
# ## 1. Autoencoders and PCA

# %%
class AE(nn.Module):
    def __init__(self, k, linear=False):
        super().__init__()
        if linear:
            self.enc, self.dec = nn.Linear(64, k), nn.Linear(k, 64)
        else:
            self.enc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, k))
            self.dec = nn.Sequential(nn.Linear(k, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, x):
        return self.dec(self.enc(x))


mse = lambda m, xb: F.mse_loss(m(xb), xb)
print("\ntest reconstruction MSE per pixel by latent size")
print("   k     PCA     linear AE   nonlinear AE")
rec = {}
t0 = time.time()
for k in (2, 8, 16):
    pca = PCA(k).fit(X.numpy())
    pca_err = ((pca.inverse_transform(pca.transform(Xte.numpy())) - Xte.numpy()) ** 2).mean()
    torch.manual_seed(0); lin = train(AE(k, linear=True), mse, epochs=600, lr=1e-2)   # linear: cheap, and slow to converge
    torch.manual_seed(0); nl = train(AE(k), mse)
    with torch.no_grad():
        rec[k] = (pca_err, mse(lin, Xte).item(), mse(nl, Xte).item())
    print(f"{k:4d}   {rec[k][0]:.4f}   {rec[k][1]:.4f}      {rec[k][2]:.4f}")
    if k == 8:
        lin8, pca8 = lin, pca
print(f"({time.time() - t0:.0f}s)")

# the linear AE's decoder spans the same subspace as the top-8 principal components, but with different axes
W = lin8.dec.weight.detach().numpy()                                          # 64 x 8
Q, _ = np.linalg.qr(W)
cosines = np.linalg.svd(Q.T @ pca8.components_.T, compute_uv=False)           # cosines of the principal angles
print(f"linear AE vs PCA, k = 8: principal angles between the subspaces have cosines from {cosines.min():.3f} to {cosines.max():.3f}")
col_cos = np.abs((W / np.linalg.norm(W, axis=0)).T @ pca8.components_.T).max(1)
print(f"but individual decoder directions match individual components poorly: best |cosine| per direction {np.round(col_cos, 2)}")
assert cosines.min() > 0.99 and abs(rec[8][0] - rec[8][1]) < 0.001 and col_cos.min() < 0.9
assert rec[2][2] < rec[2][0] - 0.005

# %% [markdown]
# ## 2. Why reparameterize
#
# Goal: the gradient with respect to mu of E_{z ~ N(mu, 1)}[f(z)], with f(z) = z^2 (true value 2 mu).
# Score function (REINFORCE): f(z) * d/dmu log p(z) = f(z) (z - mu). Pathwise (reparameterization): z = mu + eps,
# gradient f'(mu + eps) = 2 (mu + eps). Both unbiased. Compare their variance.

# %%
mu, n = 1.5, 100_000
eps = torch.randn(n, generator=torch.Generator().manual_seed(1))
z = mu + eps
score = z ** 2 * (z - mu)
path = 2 * z
print(f"\ntrue gradient {2 * mu:.3f}")
print(f"  score-function estimator: mean {score.mean():.3f}, variance per sample {score.var():.2f}")
print(f"  pathwise estimator:       mean {path.mean():.3f}, variance per sample {path.var():.2f}")
print(f"the score-function estimator needs about {score.var() / path.var():.0f} times as many samples for the same precision")
assert abs(path.mean() - 3) < 0.05 and abs(score.mean() - 3) < 0.2 and score.var() > 5 * path.var()

# %% [markdown]
# ## 3. The VAE
#
# Encoder q(z|x) = N(mu(x), diag sigma(x)^2), prior p(z) = N(0, I), decoder p(x|z) = Bernoulli per pixel (pixels are
# in [0, 1]; binary cross-entropy is the continuous-valued stand-in everyone uses). ELBO = E_q[log p(x|z)] - KL(q || p).

# %%
class VAE(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 2 * k))
        self.dec = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, 64))
        self.k = k

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)                 # the reparameterization trick
        return self.dec(z), mu, logvar


def kl_closed(mu, logvar):
    return 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(-1)                  # KL(N(mu, sigma^2) || N(0, 1)), summed over z


def vae_terms(m, xb):
    logits, mu, logvar = m(xb)
    rec = F.binary_cross_entropy_with_logits(logits, xb, reduction="none").sum(-1)   # -log p(x|z), one sample
    return rec, kl_closed(mu, logvar)


def vae_loss(beta):
    def f(m, xb):
        rec, kl = vae_terms(m, xb)
        return (rec + beta * kl).mean()
    return f


# the closed-form KL against a Monte Carlo estimate
mu_, logvar_ = torch.randn(3, 8, dtype=torch.float64), torch.randn(3, 8, dtype=torch.float64) * 0.5
zs = mu_ + torch.exp(0.5 * logvar_) * torch.randn(200_000, 3, 8, dtype=torch.float64)
log_q = torch.distributions.Normal(mu_, torch.exp(0.5 * logvar_)).log_prob(zs).sum(-1)
log_p = torch.distributions.Normal(0.0, 1.0).log_prob(zs).sum(-1)
print(f"\nKL closed form {np.round(kl_closed(mu_, logvar_).numpy(), 3)} vs Monte Carlo {np.round((log_q - log_p).mean(0).numpy(), 3)}")
assert torch.allclose(kl_closed(mu_, logvar_), (log_q - log_p).mean(0), atol=0.05)

t0 = time.time()
torch.manual_seed(0)
vae = train(VAE(), vae_loss(1.0), epochs=200)
with torch.no_grad():
    rec_te, kl_te = vae_terms(vae, Xte)
print(f"VAE (8 latents) trained in {time.time() - t0:.0f}s. Test: reconstruction term {rec_te.mean():.1f} nats, "
      f"KL {kl_te.mean():.1f} nats, negative ELBO {(rec_te + kl_te).mean():.1f} nats per image")

# %% [markdown]
# ## 4. Sampling from the prior
#
# Draw z ~ N(0, I) and decode. For the plain autoencoder there is no prior: its latent space was never asked to look
# like anything, so we sample from a Gaussian with the codes' own mean and covariance, which is the most we can do.
# The judge classifier scores the samples: how confident is it that each one is some digit?

# %%
def judge_confidence(imgs):
    p = judge.predict_proba(imgs.clamp(0, 1).numpy())
    return p.max(1).mean(), np.bincount(p.argmax(1), minlength=10)


torch.manual_seed(0)
ae8 = train(AE(8), mse)
with torch.no_grad():
    codes = ae8.enc(X)
    gauss = torch.distributions.MultivariateNormal(codes.mean(0), torch.cov(codes.T) + 1e-4 * torch.eye(8))
    ae_samples = ae8.dec(gauss.sample((1000,)))
    vae_samples = torch.sigmoid(vae.dec(torch.randn(1000, 8, generator=torch.Generator().manual_seed(3))))
conf_real = judge.predict_proba(Xte.numpy()).max(1).mean()
conf_ae, counts_ae = judge_confidence(ae_samples)
conf_vae, counts_vae = judge_confidence(vae_samples)
print(f"\njudge's mean confidence: real test digits {conf_real:.3f}, VAE samples {conf_vae:.3f}, "
      f"autoencoder + fitted Gaussian {conf_ae:.3f}")
print(f"digit classes among 1,000 VAE samples: {counts_vae.tolist()}")


def show(imgs, n=10):
    chars = " .:-=+*#%@"
    rows = []
    for r in range(8):
        rows.append("  ".join("".join(chars[min(9, int(v * 10))] for v in img[r * 8:(r + 1) * 8]) for img in imgs[:n]))
    return "\n".join(rows)


print("\nVAE samples:\n" + show(vae_samples.numpy()))
with torch.no_grad():
    a, b = vae.encode(X[y == 0][:1])[0], vae.encode(X[y == 1][:1])[0]
    path = torch.sigmoid(vae.dec(torch.stack([a[0] + t * (b[0] - a[0]) for t in np.linspace(0, 1, 8)])))
print("\ninterpolating between the codes of a 0 and a 1:\n" + show(path.numpy(), 8))
assert conf_vae > conf_ae

# %% [markdown]
# ## 5. beta

# %%
print("\n beta   reconstruction   KL (rate)   active latents   judge confidence on prior samples")
beta_res = {}
t0 = time.time()
for beta in (0.1, 1.0, 4.0):
    torch.manual_seed(0)
    m = vae if beta == 1.0 else train(VAE(), vae_loss(beta), epochs=200)
    with torch.no_grad():
        r, kl = vae_terms(m, Xte)
        mu_te, logvar_te = m.encode(Xte)
        per_dim_kl = kl_closed(mu_te[..., None], logvar_te[..., None]).mean(0)       # KL of each latent separately
        samples = torch.sigmoid(m.dec(torch.randn(1000, 8, generator=torch.Generator().manual_seed(3))))
    active = int((per_dim_kl > 0.1).sum())
    beta_res[beta] = (r.mean().item(), kl.mean().item(), active, judge_confidence(samples)[0])
    print(f"{beta:5.1f}   {beta_res[beta][0]:14.1f}   {beta_res[beta][1]:9.1f}   {active:14d}   {beta_res[beta][3]:.3f}")
print(f"({time.time() - t0:.0f}s) a latent is 'active' if its KL is above 0.1 nats: the others carry no information, their")
print("posterior is the prior. Small beta: more information in z, sharp reconstructions, prior samples that fall into")
print("holes. beta = 4: KL 0, posterior collapse. The decoder ignores z and draws the same average digit every time.")
assert beta_res[0.1][0] < beta_res[1.0][0] < beta_res[4.0][0]
assert beta_res[0.1][1] > beta_res[1.0][1] > beta_res[4.0][1]
assert beta_res[4.0][1] < 0.5 and beta_res[4.0][2] == 0 and beta_res[0.1][2] > beta_res[1.0][2]

print("\nAll checks passed.")
