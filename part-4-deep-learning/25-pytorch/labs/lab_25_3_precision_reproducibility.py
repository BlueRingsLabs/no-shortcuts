# %% [markdown]
# # Lab 25.3: Precision, compilation and reproducibility
#
# Runs on a CPU; GPU-only details are checked for and skipped.
#
# 1. The three formats: range and precision of float32, float16, bfloat16.
# 2. Gradients vanishing in float16, rescued by loss scaling (manual, then GradScaler's logic).
# 3. Accumulating a long sum in 16 bits.
# 4. Training with bfloat16 autocast vs float32: same accuracy.
# 5. Memory arithmetic for Adam.
# 6. torch.compile on a chain of elementwise ops (if a C++ compiler is available).
# 7. Reproducibility: seeds, summation order, deterministic algorithms.

# %%
import time

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(253)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device} (the numerics below are the same on every device)")

# %% [markdown]
# ## 1. Formats

# %%
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"{str(dt):15s} max {fi.max:10.3e}  smallest normal {fi.tiny:9.2e}  eps (relative precision) {fi.eps:8.1e}")
print("65504 + 32 in float16 =", (torch.tensor(65504.0, dtype=torch.float16) + 32).item(), "| 70000 in float16 =", torch.tensor(70000.0).half().item())
print("1 + 0.004 in bfloat16 =", (torch.tensor(1.0, dtype=torch.bfloat16) + 0.004).item(), "(the 0.004 is below bf16's precision at 1.0)")
assert torch.isinf(torch.tensor(70000.0).half())

# %% [markdown]
# ## 2. Vanishing gradients in float16, and loss scaling

# %%
grads = torch.logspace(-10, -3, 8)
print("gradient    float16      bfloat16     float16 x 2^16 then unscaled in float32")
for g in grads:
    g16, gbf = g.half().item(), g.bfloat16().item()
    scaled = (g * 2**16).half().float() / 2**16
    print(f"{g.item():.1e}   {g16:.3e}   {gbf:.3e}   {scaled.item():.3e}")
assert grads[0].half().item() == 0.0 and ((grads[0] * 2**16).half().float() / 2**16).item() > 0

# a tiny model whose true gradients are small: with a float16 backward they underflow
lin = torch.nn.Linear(256, 1).half()
x = (torch.randn(64, 256) * 1e-3).half()
y = torch.zeros(64, 1).half()


def grad_fraction_zero(scale):
    lin.zero_grad()
    loss = ((lin(x) - y) ** 2).mean() * 1e-3                    # a small loss, as late in training
    (loss * scale).backward()
    g = lin.weight.grad.float() / scale
    return (g == 0).float().mean().item()


print(f"weight gradients that are exactly zero: no scaling {grad_fraction_zero(1.0):.0%}, loss scale 2^16 {grad_fraction_zero(2.0**16):.0%}")
assert grad_fraction_zero(1.0) > 0.5 > grad_fraction_zero(2.0**16)


def dynamic_scale(overflow_steps, scale=2.0**16, growth_interval=5):  # the GradScaler policy, in five lines
    history, good = [], 0
    for step in range(20):
        if step in overflow_steps:
            scale /= 2; good = 0                               # inf/nan in the grads: skip the step, halve the scale
        else:
            good += 1
            if good == growth_interval:
                scale *= 2; good = 0                           # a run of clean steps: try a larger scale
        history.append(scale)
    return history


print("dynamic loss scale with overflows at steps 3 and 4:", [f"2^{int(np.log2(s))}" for s in dynamic_scale({3, 4})][:12])

# %% [markdown]
# ## 3. Accumulation in 16 bits

# %%
vals = torch.full((100_000,), 0.01)
acc16 = torch.tensor(0.0, dtype=torch.float16)
for v in vals[:20_000].half():
    acc16 = acc16 + v
print(f"adding 0.01 twenty thousand times: float16 accumulator {acc16.item():.2f}, true {20_000 * 0.01:.2f}, "
      f"float32 accumulator {vals[:20_000].sum().item():.2f}")
print("the float16 sum stalls once the total is large enough that +0.01 rounds away; hardware accumulates products in float32 for this reason")
assert acc16.item() < 50

# %% [markdown]
# ## 4. Training with bfloat16 autocast

# %%
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
X_tr, X_te = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_te, dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)


def train(use_bf16, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 10))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)       # master weights and optimizer state stay float32
    g = torch.Generator().manual_seed(seed)
    for epoch in range(15):
        perm = torch.randperm(len(X_tr), generator=g)
        for s in range(0, len(X_tr), 64):
            idx = perm[s:s + 64]
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(X_tr[idx])                       # matmuls in bf16
                loss = torch.nn.functional.cross_entropy(logits, y_tr[idx])   # autocast keeps the loss in float32
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
        out = model(X_te)
    return (out.argmax(1) == y_te).float().mean().item(), logits.dtype, loss.dtype


acc32, _, _ = train(False)
acc16, logit_dtype, loss_dtype = train(True)
print(f"test accuracy: float32 {acc32:.3f}, bfloat16 autocast {acc16:.3f} (logits computed in {logit_dtype}, loss in {loss_dtype})")
assert abs(acc32 - acc16) < 0.02 and logit_dtype == torch.bfloat16 and loss_dtype == torch.float32

# %% [markdown]
# ## 5. Memory arithmetic

# %%
for n_params in (125e6, 7e9, 70e9):
    fp32_adam = 16 * n_params
    mixed = (4 + 8 + 2 + 2) * n_params
    print(f"{n_params / 1e9:6.2f}B params: fp32 + Adam {fp32_adam / 1e9:8.1f} GB, mixed precision + Adam {mixed / 1e9:8.1f} GB (before activations)")
print("the same: mixed precision keeps fp32 master weights and moments, so it saves activation memory and time, not optimizer state")
model = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(), torch.nn.Linear(4096, 1024))
opt = torch.optim.Adam(model.parameters())
model(torch.randn(8, 1024)).sum().backward(); opt.step()
state_bytes = sum(t.numel() * t.element_size() for st in opt.state.values() for t in st.values() if torch.is_tensor(t) and t.dim() > 0)
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"measured: parameters {param_bytes / 1e6:.1f} MB, Adam state {state_bytes / 1e6:.1f} MB (= 2x the parameters)")
assert abs(state_bytes / param_bytes - 2) < 0.01

# %% [markdown]
# ## 6. torch.compile

# %%
def gelu_chain(x):
    for _ in range(8):
        x = torch.nn.functional.gelu(x) * 0.9 + torch.sin(x) * 0.1
    return x


xc = torch.randn(4_000_000)
try:
    compiled = torch.compile(gelu_chain)
    t0 = time.perf_counter(); compiled(xc); t_compile = time.perf_counter() - t0
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        gelu_chain(xc)
    t_eager = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        compiled(xc)
    t_comp = (time.perf_counter() - t0) / reps
    same = torch.allclose(gelu_chain(xc), compiled(xc), atol=1e-5)
    print(f"torch.compile: first call {t_compile:.1f}s (compilation); then eager {t_eager * 1000:.0f} ms vs compiled {t_comp * 1000:.0f} ms "
          f"({t_eager / t_comp:.1f}x), same results {same}")
    print("16 elementwise ops fused into one kernel: the data is read once instead of 16 times")
    assert same
except Exception as e:                                          # no C++ toolchain, unsupported platform...
    print(f"torch.compile unavailable here ({type(e).__name__}); skipping")

# %% [markdown]
# ## 7. Reproducibility

# %%
def one_run(seed):
    torch.manual_seed(seed)
    m = torch.nn.Linear(64, 10)
    o = torch.optim.SGD(m.parameters(), lr=0.1)
    for s in range(0, 512, 64):
        o.zero_grad(); torch.nn.functional.cross_entropy(m(X_tr[s:s + 64]), y_tr[s:s + 64]).backward(); o.step()
    return m.weight.detach().clone()


print(f"same seed twice, bit-identical: {torch.equal(one_run(0), one_run(0))}; different seeds identical: {torch.equal(one_run(0), one_run(1))}")
assert torch.equal(one_run(0), one_run(0))

r = np.random.default_rng(0)
numbers = torch.tensor(r.normal(size=1_000_000) * 10.0 ** r.integers(-3, 4, 1_000_000), dtype=torch.float32)
sums = {tuple(p[:3].tolist()): numbers[p].sum().item() for p in (torch.randperm(len(numbers)) for _ in range(5))}
print(f"the same million float32 numbers summed in 5 different orders: {len(set(sums.values()))} distinct results, "
      f"spread {max(sums.values()) - min(sums.values()):.3e}")
print(f"in float64: {len({numbers.double()[torch.randperm(len(numbers))].sum().item() for _ in range(5)})} distinct results (smaller differences, not none in general)")
assert len(set(sums.values())) > 1

torch.use_deterministic_algorithms(True)
print("torch.use_deterministic_algorithms(True) set: nondeterministic kernels (mostly on CUDA) now raise instead of running")
torch.use_deterministic_algorithms(False)

print("\nAll checks passed.")
