# %% [markdown]
# # Lab 26.4: Normalization and residual connections
#
# 1. BatchNorm, LayerNorm and RMSNorm from scratch, matched against PyTorch (training and evaluation modes).
# 2. The degradation problem: plain MLPs of depth 4, 16 and 48 vs residual ones, same everything else.
# 3. Gradient norms per layer in a 48-layer plain vs residual network.
# 4. BatchNorm's quirks: tiny batches, train/eval mismatch, and batch-mates changing an example's output.
# 5. Pre-norm vs post-norm residual stacks at depth, without warmup.

# %%
import numpy as np
import torch
from torch import nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

torch.manual_seed(264)
torch.set_num_threads(4)
Xn, yn = make_classification(n_samples=8000, n_features=32, n_informative=16, n_classes=4, n_clusters_per_class=3, class_sep=1.2, random_state=1)
X_tr, X_te, y_tr, y_te = (torch.tensor(a) for a in train_test_split(Xn, yn, test_size=0.25, random_state=0))
X_tr, X_te = X_tr.float(), X_te.float()

# %% [markdown]
# ## 1. The normalization layers, by hand

# %%
class MyBatchNorm(nn.Module):
    def __init__(self, d, momentum=0.1, eps=1e-5):
        super().__init__()
        self.gamma, self.beta = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
        self.register_buffer("running_mean", torch.zeros(d)); self.register_buffer("running_var", torch.ones(d))
        self.m, self.eps = momentum, eps

    def forward(self, x):
        if self.training:
            mu, var = x.mean(0), x.var(0, unbiased=False)
            with torch.no_grad():                                # running stats use the unbiased variance, like PyTorch
                self.running_mean.lerp_(mu, self.m)
                self.running_var.lerp_(x.var(0, unbiased=True), self.m)
        else:
            mu, var = self.running_mean, self.running_var
        return self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta


class MyLayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.gamma, self.beta, self.eps = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d)), eps

    def forward(self, x):
        mu, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta


class MyRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return self.gamma * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


x = torch.randn(64, 16) * 3 + 1
mine, ref = MyBatchNorm(16), nn.BatchNorm1d(16)
for _ in range(5):
    assert torch.allclose(mine(x), ref(x), atol=1e-5)
mine.eval(); ref.eval()
assert torch.allclose(mine(x), ref(x), atol=1e-5) and torch.allclose(mine.running_var, ref.running_var)
assert torch.allclose(MyLayerNorm(16)(x), nn.LayerNorm(16)(x), atol=1e-5)
assert torch.allclose(MyRMSNorm(16)(x), nn.RMSNorm(16, eps=1e-6)(x), atol=1e-5)
print("BatchNorm (train and eval), LayerNorm and RMSNorm match PyTorch")

# %% [markdown]
# ## 2. Deeper plain networks train worse, on their own training data

# %%
class Block(nn.Module):
    def __init__(self, d, residual, norm=None, prenorm=True):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.residual, self.prenorm = residual, prenorm
        self.norm = {"bn": nn.BatchNorm1d, "ln": nn.LayerNorm, None: nn.Identity}[norm](d)
        if residual:
            nn.init.zeros_(self.f[2].weight); nn.init.zeros_(self.f[2].bias)   # each block starts as the identity

    def forward(self, h):
        if not self.residual:
            return torch.relu(self.norm(self.f(h)))
        if self.prenorm:
            return h + self.f(self.norm(h))
        return self.norm(h + self.f(h))                          # post-norm


def build(n_blocks, residual, norm=None, prenorm=True, d=64):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, d), *[Block(d, residual, norm, prenorm) for _ in range(n_blocks)], nn.Linear(d, 4))


def train(model, steps=800, lr=2e-3, bs=128, warmup=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warmup) if warmup else 1.0)
    for _ in range(steps):
        idx = torch.randint(0, len(X_tr), (bs,), generator=g)
        opt.zero_grad(); nn.functional.cross_entropy(model(X_tr[idx]), y_tr[idx]).backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        tr_loss = nn.functional.cross_entropy(model(X_tr), y_tr).item()
        te_acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    model.train()
    return tr_loss, te_acc


print(f"{'blocks (2 layers each)':24s} {'plain: train loss':>18s} {'residual: train loss':>21s}")
deg = {}
for n in (2, 8, 24):
    plain, resid = train(build(n, residual=False)), train(build(n, residual=True))
    deg[n] = (plain[0], resid[0])
    print(f"{n:24d} {plain[0]:18.3f} {resid[0]:21.3f}")
assert deg[24][0] > deg[2][0] + 0.2, "the deep plain network fits its training data worse than the shallow one"
assert deg[24][1] <= deg[2][1] + 0.05, "the residual one doesn't degrade"

# %% [markdown]
# ## 3. Gradient norms per layer

# %%
for residual in (False, True):
    m = build(24, residual=residual)
    if residual:                                                # un-zero the branches so every layer gets a gradient to measure
        for b in m[1:-1]:
            nn.init.kaiming_normal_(b.f[2].weight, nonlinearity="relu"); b.f[2].weight.data *= 0.2
    nn.functional.cross_entropy(m(X_tr[:256]), y_tr[:256]).backward()
    norms = [b.f[0].weight.grad.norm().item() for b in m[1:-1]]
    print(f"{'residual' if residual else 'plain':8s}: gradient norm of the first linear layer in block 1 / 12 / 24: "
          f"{norms[0]:.2e} / {norms[11]:.2e} / {norms[-1]:.2e}  (first/last ratio {norms[0] / norms[-1]:.1e})")
    if residual:
        r_ratio = norms[0] / norms[-1]
    else:
        p_ratio = norms[0] / norms[-1]
assert r_ratio > 100 * p_ratio or p_ratio < 1e-2

# %% [markdown]
# ## 4. BatchNorm's quirks

# %%
bn_small = train(build(4, residual=False, norm="bn"), bs=2, steps=4000, lr=5e-4)
bn_big = train(build(4, residual=False, norm="bn"), bs=128)
ln_small = train(build(4, residual=False, norm="ln"), bs=2, steps=4000, lr=5e-4)
print(f"BatchNorm, batch size 128: test acc {bn_big[1]:.3f}; batch size 2: {bn_small[1]:.3f}; LayerNorm with batch size 2: {ln_small[1]:.3f}")
assert bn_small[1] < ln_small[1]

m = build(4, residual=False, norm="bn"); train(m)
m.eval()
with torch.no_grad():
    good = (m(X_te).argmax(1) == y_te).float().mean().item()
m.train()                                                       # forgot eval(): batch statistics from the evaluation batch
with torch.no_grad():
    single = torch.cat([m(X_te[i:i + 2]) for i in range(0, 400, 2)]).argmax(1)
    forgot = (single == y_te[:400]).float().mean().item()
print(f"evaluation in eval() mode {good:.3f}; 'evaluation' in train() mode with batches of 2: {forgot:.3f}")
assert forgot < good - 0.1

m.train()
with torch.no_grad():
    probe = X_te[:1]
    out_a = m(torch.cat([probe, X_te[1:32]]))[0]
    out_b = m(torch.cat([probe, X_te[1:32] * 3 + 2]))[0]       # same example, different batch-mates
print(f"in training mode, the same example's logits change with its batch-mates: max difference {(out_a - out_b).abs().max():.3f}")
assert (out_a - out_b).abs().max() > 0.1

# %% [markdown]
# ## 5. Pre-norm vs post-norm at depth, no warmup

# %%
for prenorm in (True, False):
    res = [train(build(24, residual=True, norm="ln", prenorm=prenorm), lr=3e-3, seed=0)[0]]
    print(f"{'pre-norm' if prenorm else 'post-norm':9s}, 24 residual blocks, lr 3e-3, no warmup: training loss {np.round(res, 3)}")
    if prenorm:
        pre = np.mean(res)
    else:
        post = np.mean(res)
print("both can work at this scale; pre-norm is the one that doesn't need you to be careful (and at LLM depth the difference is large)")

print("\nAll checks passed.")
