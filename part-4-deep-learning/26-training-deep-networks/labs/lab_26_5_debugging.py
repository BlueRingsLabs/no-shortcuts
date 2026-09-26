# %% [markdown]
# # Lab 26.5: Planted bugs, and the checks that catch them
#
# Each check from the recipe runs on a correct model and on a model with one silent bug. The correct model must pass,
# the buggy one must fail. None of the bugs raises an exception.
#
# 1. Loss at initialization          vs  an output layer initialized far too large.
# 2. Does the model use its input?   vs  a forward pass that detaches and zeroes the input path.
# 3. Overfit one batch               vs  zero_grad() called between backward() and step().
#    (and a stale optimizer after replacing a layer: caught by counting parameters, not by overfitting)
# 4. Batch mixing (gradient check)   vs  view() used where permute() was needed.
# 5. Logits vs probabilities         vs  softmax applied before cross_entropy.
# 6. NaN hunting with anomaly mode   vs  a log of a probability that underflowed to 0.
# 7. Train/eval preprocessing        vs  validation data normalized with its own statistics.

# %%
import math
import warnings

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(265)
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
mean, std = X_tr.mean(0), X_tr.std(0) + 1e-6
X_tr_n = torch.tensor((X_tr - mean) / std, dtype=torch.float32)
X_te_n = torch.tensor((X_te - mean) / std, dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)
K = 10
results = []


def report(check, ok_correct, ok_buggy, detail):
    status = "OK " if (ok_correct and not ok_buggy) else "?? "
    results.append(ok_correct and not ok_buggy)
    print(f"[{status}] {check:32s} correct model passes: {ok_correct!s:5s} buggy model passes: {ok_buggy!s:5s}  {detail}")


class Net(nn.Module):
    def __init__(self, bug=None):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Linear(128, K)
        self.bug = bug
        if bug == "big_init":
            nn.init.normal_(self.head.weight, std=5.0)

    def forward(self, x):
        if self.bug == "detached_input":
            x = x.detach() * 0 + x.mean().detach()              # "normalization" gone wrong: every row becomes a constant
        if self.bug == "view_mixing":
            b = x.shape[0]
            x = x.view(8, 8, b).permute(2, 0, 1).reshape(b, 64)  # meant to reshape (b, 64) to images and back: mixes examples
        h = self.head(self.body(x))
        if self.bug == "softmax_twice":
            h = h.softmax(-1)                                     # probabilities fed to a loss that expects logits
        return h


def fit(model, X, y, steps=300, lr=1e-3, params=None, bs=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (min(bs, len(X)),), generator=g)
        opt.zero_grad(); loss = nn.functional.cross_entropy(model(X[idx]), y[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        return nn.functional.cross_entropy(model(X), y).item()

# %% [markdown]
# ## 1. Loss at initialization

# %%
def init_loss_ok(model):
    with torch.no_grad():
        loss = nn.functional.cross_entropy(model(X_tr_n), y_tr).item()
    return abs(loss - math.log(K)) < 0.5, loss


ok_c, l_c = init_loss_ok(Net()); ok_b, l_b = init_loss_ok(Net("big_init"))
report("1. loss at init ~ ln K", ok_c, ok_b, f"(ln 10 = {math.log(K):.2f}; correct {l_c:.2f}, buggy {l_b:.2f})")

# %% [markdown]
# ## 2. Does the model use its input?

# %%
def uses_input(bug=None):
    real = fit(Net(bug), X_tr_n, y_tr)
    zero = fit(Net(bug), torch.zeros_like(X_tr_n), y_tr)
    return real < zero - 0.5, real, zero


ok_c, r_c, z_c = uses_input(); ok_b, r_b, z_b = uses_input("detached_input")
report("2. real input beats zero input", ok_c, ok_b, f"(correct {r_c:.2f} vs {z_c:.2f}; buggy {r_b:.2f} vs {z_b:.2f})")

# %% [markdown]
# ## 3. Overfit one batch

# %%
Xb, yb = X_tr_n[:16], y_tr[:16]


def overfits(zero_grad_bug, steps=300, lr=3e-3):
    m = Net()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(Xb), yb)
        loss.backward()
        if zero_grad_bug:
            opt.zero_grad()                                     # "cleaning up" in the wrong place: step() sees zero gradients
        opt.step()
    with torch.no_grad():
        final = nn.functional.cross_entropy(m(Xb), yb).item()
    return final < 0.01, final


ok_c, lc = overfits(False); ok_b, lb = overfits(True)
report("3. overfit a batch of 16", ok_c, ok_b, f"(final loss correct {lc:.4f}, buggy {lb:.4f})")

m = Net()
opt = torch.optim.Adam(m.parameters(), lr=3e-3)
m.head = nn.Linear(128, K)                                      # replaced after the optimizer collected the parameters
in_opt = {id(p) for g_ in opt.param_groups for p in g_["params"]}
missing = sum(p.numel() for p in m.parameters() if id(p) not in in_opt)
loss_stale = fit(m, Xb, yb, steps=300, lr=3e-3, params=[p for g_ in opt.param_groups for p in g_["params"]], bs=16)
print(f"     stale optimizer after replacing the head: it *still* overfits the batch (loss {loss_stale:.4f}), because the body")
print(f"     compensates for a random head. What catches it: {missing} model parameters are not in the optimizer.")
assert missing == 128 * K + K

# %% [markdown]
# ## 4. Batch mixing

# %%
def no_mixing(model):
    model.eval()
    x = torch.randn(8, 64, requires_grad=True)
    model(x)[3].sum().backward()
    others = torch.ones(8, dtype=torch.bool); others[3] = False
    leak = x.grad[others].abs().max().item()
    return leak == 0.0, leak


ok_c, lk_c = no_mixing(Net()); ok_b, lk_b = no_mixing(Net("view_mixing"))
report("4. no gradient across examples", ok_c, ok_b, f"(max gradient reaching other examples: correct {lk_c}, buggy {lk_b:.2e})")
acc_mix = (Net("view_mixing").eval()(X_te_n).argmax(1) == y_te).float().mean()     # untrained, just to show it runs silently
print(f"     the view_mixing model trains without any error: final training loss {fit(Net('view_mixing'), X_tr_n, y_tr):.3f} "
      f"(correct model {fit(Net(), X_tr_n, y_tr):.3f})")

# %% [markdown]
# ## 5. Softmax applied twice

# %%
def grad_norm_first_step(model):
    loss = nn.functional.cross_entropy(model(X_tr_n[:64]), y_tr[:64])
    loss.backward()
    return sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None).sqrt().item(), loss.item()


gc, lc0 = grad_norm_first_step(Net()); gb, lb0 = grad_norm_first_step(Net("softmax_twice"))
final_c, final_b = fit(Net(), X_tr_n, y_tr), fit(Net("softmax_twice"), X_tr_n, y_tr)
report("5. healthy gradient + learning", gc > 3 * gb and final_c < 0.3, final_b < 0.3,
       f"(initial loss {lc0:.2f} vs {lb0:.2f}, looks fine! gradient norm {gc:.3f} vs {gb:.3f}; final loss {final_c:.3f} vs {final_b:.3f})")

# %% [markdown]
# ## 6. NaN hunting

# %%
def custom_loss(logits, y, bug):
    if bug:
        p = logits.softmax(-1)
        return -torch.log(p[torch.arange(len(y)), y]).mean()     # log of a probability: -inf when it underflows to 0
    return nn.functional.cross_entropy(logits, y)                # log-softmax computed stably (07.2)


def first_nan_step(bug, steps=200):
    torch.manual_seed(0)
    m = Net(); opt = torch.optim.SGD(m.parameters(), lr=2.0)     # an aggressive learning rate makes logits large fast
    for step in range(steps):
        idx = torch.randint(0, len(X_tr_n), (64,))
        loss = custom_loss(m(X_tr_n[idx]), y_tr[idx], bug)
        if not torch.isfinite(loss):
            return step
        opt.zero_grad(); loss.backward(); opt.step()
        if not all(torch.isfinite(p).all() for p in m.parameters()):
            return step
    return None


nan_c, nan_b = first_nan_step(False), first_nan_step(True)
report("6. no NaN in 200 steps", nan_c is None, nan_b is None, f"(stable loss: {nan_c}; log(softmax): first non-finite at step {nan_b})")

torch.manual_seed(0)
m = Net()
logits = m(X_tr_n[:8]) * 1000                                    # force a probability to underflow
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with torch.autograd.set_detect_anomaly(True):
        try:
            custom_loss(logits, y_tr[:8], bug=True).backward()
            where = "no error raised (the loss was already inf in the forward pass)"
        except RuntimeError as e:
            where = str(e).splitlines()[0][:110]
print(f"     anomaly mode on the buggy loss: {where}")

# %% [markdown]
# ## 7. Train/eval preprocessing mismatch

# %%
model = Net(); fit(model, X_tr_n, y_tr, steps=600)
model.eval()
bad_te = torch.tensor((X_te - X_te.mean(0)) / (X_te.std(0) + 1e-6), dtype=torch.float32)   # normalized with its own stats
shifted_te = torch.tensor(((X_te + 2.0) - (X_te + 2.0).mean(0)) / (X_te.std(0) + 1e-6), dtype=torch.float32)
raw_shift_ok = torch.tensor(((X_te + 2.0) - mean) / std, dtype=torch.float32)             # production data brighter by 2
with torch.no_grad():
    acc = lambda X_: (model(X_).argmax(1) == y_te).float().mean().item()
    tr_through_eval = (model(torch.tensor((X_tr - mean) / std, dtype=torch.float32)).argmax(1) == y_tr).float().mean().item()
print(f"     same test set: normalized with training stats {acc(X_te_n):.3f}, with its own stats {acc(bad_te):.3f}")
print(f"     the self-normalized pipeline also hides real drift: brightened test data scores {acc(shifted_te):.3f} "
      f"(looks fine) while the correct pipeline shows {acc(raw_shift_ok):.3f}")
report("7. eval pipeline = train pipeline", abs(tr_through_eval - 1.0) < 0.05, abs(acc(raw_shift_ok) - acc(shifted_te)) < 0.02,
       "(a pipeline that renormalizes each batch can't see drift, 51.1)")

print(f"\n{sum(results)} of {len(results)} checks separated the correct model from the buggy one")
assert all(results)
print("All checks passed.")
