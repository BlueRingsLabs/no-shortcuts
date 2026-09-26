# %% [markdown]
# # Lab 33.3: Profiling, and making it fast
#
# A training loop with the usual sins, fixed one at a time, measuring throughput after each fix. On a CPU, so the
# absolute numbers are small; the method is the same on a GPU, and the lesson says where the ranking of fixes changes.
#
# 0. Benchmarking properly: warm-up, repeats, and what the first iteration hides.
# 1. The slow loop, and the profiler's view of it.
# 2. Fix the data pipeline: per-sample Python work -> batched tensor operations.
# 3. Fix the dtype: an accidental float64 model.
# 4. Fix the batch size: per-step overheads.
# 5. Fuse: hand-written layers vs built-in fused ones, and torch.compile.
# 6. Overlap loading with compute: DataLoader workers.

# %%
import os
import time

os.environ.setdefault("KINETO_LOG_LEVEL", "5")                                  # quiet the profiler's startup chatter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

torch.manual_seed(333)
torch.set_num_threads(4)

N_SAMPLES, D_IN, N_CLASSES = 4096, 256, 10
RAW = torch.randint(0, 256, (N_SAMPLES, D_IN), dtype=torch.uint8)                   # "raw bytes", like decoded images
LABELS = torch.randint(0, N_CLASSES, (N_SAMPLES,))


class MyLayerNorm(nn.Module):                                                          # correct, and five kernels instead of one
    def __init__(self, d):
        super().__init__()
        self.g, self.b = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-5) * self.g + self.b


def my_gelu(x):                                                                        # the tanh approximation, by hand
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class Net(nn.Module):
    def __init__(self, fused=False, d=512, depth=4):
        super().__init__()
        self.inp = nn.Linear(D_IN, d)
        self.norms = nn.ModuleList((nn.LayerNorm(d) if fused else MyLayerNorm(d)) for _ in range(depth))
        self.lins = nn.ModuleList(nn.Linear(d, d) for _ in range(depth))
        self.out = nn.Linear(d, N_CLASSES)
        self.act = (lambda x: F.gelu(x, approximate="tanh")) if fused else my_gelu

    def forward(self, x):
        h = self.inp(x)
        for ln, lin in zip(self.norms, self.lins):
            h = h + lin(self.act(ln(h)))
        return self.out(h)


def slow_batch(idx):
    """Per-sample preprocessing in Python: convert, normalize, augment (random flip of the feature order), stack."""
    out = []
    for i in idx.tolist():
        x = RAW[i].double() / 255.0                                                    # float64: nobody asked for it
        x = (x - 0.5) / 0.25
        if torch.rand(1).item() < 0.5:
            x = x.flip(0)
        out.append(x)
    return torch.stack(out), LABELS[idx]


def fast_batch(idx, dtype=torch.float32):
    x = RAW[idx].to(dtype) / 255.0
    x = (x - 0.5) / 0.25
    flip = torch.rand(len(idx)) < 0.5
    return torch.where(flip[:, None], x.flip(1), x), LABELS[idx]


def train_steps(model, make_batch, bs, steps, dtype):
    model = model.to(dtype)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for s in range(steps):
        idx = torch.randint(0, N_SAMPLES, (bs,))
        x, y = make_batch(idx)
        loss = F.cross_entropy(model(x.to(dtype)), y)
        opt.zero_grad(); loss.backward(); opt.step()


def throughput(model_fn, make_batch, bs, dtype=torch.float32, samples=4096, warmup=2, repeats=3):
    """Samples per second: the median of several timed windows, after a warm-up (section 0's advice, applied)."""
    model = model_fn()
    train_steps(model, make_batch, bs, warmup, dtype)                                  # warm-up: allocations, caches, compilation
    steps = max(1, samples // bs)
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        train_steps(model, make_batch, bs, steps, dtype)
        rates.append(steps * bs / (time.perf_counter() - t0))
    return sorted(rates)[len(rates) // 2]


# %% [markdown]
# ## 0. Benchmarking properly

# %%
m = Net(fused=True)
x_b = torch.randn(256, D_IN)
times = []
for i in range(6):
    t0 = time.perf_counter()
    m(x_b).sum().backward()
    times.append((time.perf_counter() - t0) * 1e3)
print("forward + backward, the same batch, six times (ms):", [f"{t:.1f}" for t in times])
print("the first call pays for allocation and one-time setup. Time after a warm-up, repeat, and report the median.")
print("on a GPU, also synchronize (torch.cuda.synchronize()) before reading the clock: kernels run asynchronously.")

# %% [markdown]
# ## 1. The slow loop, and the profiler

# %%
results = {}
results["0. baseline: per-sample Python, float64, batch 16, hand-written layers"] = throughput(
    lambda: Net(), slow_batch, 16, dtype=torch.float64, samples=1024)

model = Net().double()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(10):
        idx = torch.randint(0, N_SAMPLES, (16,))
        with record_function("data preparation"):
            x, y = slow_batch(idx)
        with record_function("forward + backward + step"):
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8)
print("\nprofiler, top operations by self CPU time over 10 slow steps:")
print("\n".join(line[:110] for line in table.splitlines()[:12]))
regions = {e.key: e.cpu_time_total for e in prof.key_averages() if e.key in ("data preparation", "forward + backward + step")}
share = regions["data preparation"] / sum(regions.values())
print(f"labeled regions: data preparation {regions['data preparation'] / 1e3:.1f} ms, "
      f"forward + backward + step {regions['forward + backward + step'] / 1e3:.1f} ms ({share:.0%} data)")
print("I wrote this loop expecting the per-sample Python preprocessing to be the villain. The profiler says the time is")
print("in the matrix multiplies, in float64. Fixing the preprocessing first would have been a waste of an afternoon.")
assert share < 0.35

# %% [markdown]
# ## 2 to 5. One fix at a time

# %%
results["1. + batched preprocessing"] = throughput(
    lambda: Net(), lambda i: fast_batch(i, torch.float64), 16, dtype=torch.float64, samples=2048)
results["2. + float32 instead of float64"] = throughput(lambda: Net(), fast_batch, 16)
results["3. + batch 256 instead of 16"] = throughput(lambda: Net(), fast_batch, 256)
results["4. + built-in fused LayerNorm and GELU"] = throughput(lambda: Net(fused=True), fast_batch, 256)

t0 = time.time()
compiled_unfused = torch.compile(Net())
results["5. hand-written layers + torch.compile"] = throughput(lambda: compiled_unfused, fast_batch, 256, warmup=3)
compile_time = time.time() - t0

print("\nthroughput, samples per second:")
base = results["0. baseline: per-sample Python, float64, batch 16, hand-written layers"]
for k, v in results.items():
    print(f"  {k:72s} {v:9,.0f}   {v / base:5.1f}x")
print(f"(torch.compile's first steps include compilation: about {compile_time:.0f}s in total here, paid once per shape)")
r = list(results.values())
# How much each change should buy depends on the machine: batch 16 is limited by per-step Python overhead, which runs on
# one core, batch 256 by matrix multiplies, which use every core. The gain from batching measured 4.8x on a 4-core
# machine, 2.8x on a 2-core cloud runner and 2.0x to 2.7x on a single thread. Run-to-run noise here is about 15% (fix 1
# measured anywhere from 0.92x to 1.16x). So the checks ask for effects well outside the noise, not for one machine's sizes.
print(f"fix 1 changed throughput by {r[1] / r[0]:.2f}x: within run-to-run noise, as the profiler predicted. The dtype, the")
if r[5] < 0.95 * r[4]:
    print(f"batch size and fusion did the work. Compiling the hand-written layers closed {(r[5] - r[3]) / (r[4] - r[3]):.0%} of the gap to the")
    print("built-in fused ones: a compiler finds some fusions for you, and hand-picked fused kernels are still hard to beat.")
else:
    print("batch size and fusion did the work. Compiling the hand-written layers matched the built-in fused ones here: the")
    print("compiler found the same fusions by itself. On other machines and models it closes only part of that gap.")
assert r[1] < 1.3 * r[0] and r[2] > 1.3 * r[1] and r[3] > 1.5 * r[2]
assert r[4] > r[3] and r[5] > r[3]

# %% [markdown]
# ## 6. Overlapping data loading with compute
#
# Pretend each sample costs 1 ms to load (disk, decoding). With workers=0 that time adds to every step; with workers,
# the next batches are prepared while the model trains.

# %%
class SlowDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1024

    def __getitem__(self, i):
        time.sleep(0.001)                                                              # I/O or decoding
        return fast_batch(torch.tensor([i]))[0][0], LABELS[i]


def epoch_time(workers):
    model = Net(fused=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    dl = torch.utils.data.DataLoader(SlowDataset(), batch_size=128, num_workers=workers, persistent_workers=False)
    t0 = time.perf_counter()
    for x, y in dl:
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return time.perf_counter() - t0


t_w0, t_w3 = epoch_time(0), epoch_time(3)
print(f"\none epoch with 1 ms per sample of loading: num_workers=0 {t_w0:.2f}s, num_workers=3 {t_w3:.2f}s")
print("the GPU (here: the main process) should never wait for data. If it does, the fix isn't in the model.")
assert t_w3 < 0.7 * t_w0

print("\nAll checks passed.")
