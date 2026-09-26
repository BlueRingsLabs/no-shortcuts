# %% [markdown]
# # Lab 07.3: Measuring your own machine
#
# You'll estimate your machine's memory bandwidth and matmul FLOP/s, then use them to predict (roughly) how long other
# operations should take: a tiny roofline model built from measurements.
#
# 1. Memory bandwidth from a large array copy.
# 2. Peak-ish FLOP/s from a large float32 matmul.
# 3. Elementwise ops are memory-bound: their time tracks bytes, not FLOPs.
# 4. Access patterns: contiguous vs strided traversal; row vs column sums.
# 5. Fusion: one pass vs several passes over the same data.
#
# Numbers vary a lot between machines (and CI runners), so the assertions check ratios and orders of magnitude, not absolute speeds.

# %%
import time

import numpy as np
import torch

torch.set_num_threads(max(1, torch.get_num_threads()))


def best_time(fn, reps=5):
    fn()                                         # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)

# %% [markdown]
# ## 1. Memory bandwidth

# %%
n = 50_000_000                                    # 200 MB in float32
src = np.ones(n, dtype=np.float32)
dst = np.empty_like(src)
t_copy = best_time(lambda: np.copyto(dst, src))
bw = 2 * src.nbytes / t_copy                      # read + write
print(f"copy of {src.nbytes / 1e6:.0f} MB: {t_copy * 1000:.1f} ms -> ~{bw / 1e9:.1f} GB/s effective bandwidth")

# %% [markdown]
# ## 2. Matmul FLOP/s

# %%
m = 2048
A = torch.randn(m, m)
B = torch.randn(m, m)
t_mm = best_time(lambda: A @ B, reps=3)
flops = 2 * m**3 / t_mm
print(f"{m}x{m} float32 matmul: {t_mm * 1000:.1f} ms -> ~{flops / 1e9:.0f} GFLOP/s")
balance = flops / bw
print(f"machine balance ~{balance:.0f} FLOPs per byte")

# %% [markdown]
# ## 3. Elementwise operations are memory-bound
#
# y = x * 2 moves 8 bytes per element (read + write) and does 1 FLOP. Predicted time = bytes / bandwidth.
# y = sin(x) * 2 + 1 does more arithmetic per element but (for numpy) creates temporaries; we compare measured
# FLOP/s of the elementwise op with the matmul's.

# %%
x = np.ones(n, dtype=np.float32)
y = np.empty_like(x)
t_scale = best_time(lambda: np.multiply(x, 2.0, out=y))
predicted = 2 * x.nbytes / bw
elem_flops = n / t_scale
print(f"x*2 on {n:,} floats: {t_scale * 1000:.1f} ms (bandwidth prediction {predicted * 1000:.1f} ms), "
      f"{elem_flops / 1e9:.2f} GFLOP/s vs matmul {flops / 1e9:.0f}")
assert 0.3 < t_scale / predicted < 3, "an elementwise op should take about bytes / bandwidth"
assert flops > 10 * elem_flops, "matmul should achieve far more FLOP/s than an elementwise op"

# %% [markdown]
# ## 4. Access patterns

# %%
big = np.random.default_rng(0).random((4000, 4000), dtype=np.float32)
# Whole-array reductions: numpy is smart. For axis=0 it adds entire rows together (contiguous) instead of walking
# down each column, so both directions are fast. Don't take my word for which is faster on your machine: measure.
t_rows = best_time(lambda: big.sum(axis=1))
t_cols = best_time(lambda: big.sum(axis=0))
print(f"big.sum(axis=1) {t_rows * 1000:.1f} ms   big.sum(axis=0) {t_cols * 1000:.1f} ms  (numpy reorders the work)")

# But when YOUR code walks the array one column at a time, you get the strided pattern and pay for it.
t_row_loop = best_time(lambda: [big[i, :].sum() for i in range(big.shape[0])], reps=3)
t_col_loop = best_time(lambda: [big[:, j].sum() for j in range(big.shape[1])], reps=3)
print(f"loop over rows {t_row_loop * 1000:.0f} ms   loop over columns {t_col_loop * 1000:.0f} ms")
assert t_col_loop > 1.5 * t_row_loop, "column slices of a C-ordered array are strided"

flat = np.ones(40_000_000, dtype=np.float32)
t_contig = best_time(lambda: flat[:2_500_000].sum())
t_strided = best_time(lambda: flat[::16].sum())            # same number of elements, one per 64-byte cache line
print(f"sum of 2.5M contiguous floats {t_contig * 1000:.2f} ms   2.5M floats at stride 16 {t_strided * 1000:.2f} ms")
assert t_strided > 2 * t_contig, "strided access wastes most of every cache line"

# %% [markdown]
# ## 5. Fusion: fewer passes over memory
#
# Computing a*x + b, then relu, then *c as three numpy passes vs one torch-compiled... we keep it simple and portable:
# three separate passes (each reading and writing the full array) vs doing it in place, which halves temporaries.
# The point is the traffic count, which we predict before measuring.

# %%
xt = torch.ones(n)


def unfused(x):
    t1 = x * 1.5            # pass 1: read x, write t1
    t2 = t1 + 0.5           # pass 2: read t1, write t2
    t3 = torch.relu(t2)     # pass 3: read t2, write t3
    return t3 * 2.0         # pass 4: read t3, write out


def fewer_passes(x):
    out = torch.addcmul(torch.full_like(x, 0.5), x, torch.tensor(1.5))   # pass 1 (+ fill): a*x + b
    out.relu_().mul_(2.0)                                                 # passes 2-3, in place (no new allocations)
    return out


t_unfused = best_time(lambda: unfused(xt), reps=3)
t_fewer = best_time(lambda: fewer_passes(xt), reps=3)
assert torch.allclose(unfused(xt[:1000]), fewer_passes(xt[:1000]))
print(f"4 separate passes {t_unfused * 1000:.0f} ms   fewer allocations/passes {t_fewer * 1000:.0f} ms")
print("a real fused kernel (torch.compile, 33.3) does all of it in ONE read and ONE write")
assert t_fewer < t_unfused

# %%
print("\nAll checks passed.")
