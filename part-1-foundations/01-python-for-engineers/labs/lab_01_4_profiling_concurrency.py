# %% [markdown]
# # Lab 01.4: Measure, then choose the right tool
#
# 1. Profile an O(n^2) deduplication, find the hotspot with cProfile, fix it, measure the speedup.
# 2. I/O-bound work (simulated with sleeps): sequential vs threads vs asyncio with a concurrency cap.
# 3. CPU-bound pure-Python work: threads don't help (the GIL), processes do.

# %%
import asyncio
import cProfile
import io
import os
import pstats
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# %% [markdown]
# ## 1. Profile first

# %%
def dedupe_slow(records):
    seen, out = [], []
    for r in records:
        if r["id"] not in seen:          # O(len(seen)) per lookup
            seen.append(r["id"])
            out.append(r)
    return out


def dedupe_fast(records):
    seen, out = set(), []
    for r in records:
        if r["id"] not in seen:          # O(1) average
            seen.add(r["id"])
            out.append(r)
    return out


rng = random.Random(0)
records = [{"id": rng.randrange(6000), "v": i} for i in range(12000)]

prof = cProfile.Profile()
prof.enable()
slow = dedupe_slow(records)
prof.disable()
buf = io.StringIO()
pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(3)
print(buf.getvalue()[:900])
assert "dedupe_slow" in buf.getvalue(), "the profiler should point at the hot function"

t0 = time.perf_counter(); slow = dedupe_slow(records); t_slow = time.perf_counter() - t0
t0 = time.perf_counter(); fast = dedupe_fast(records); t_fast = time.perf_counter() - t0
assert slow == fast, "same result, first occurrence kept"
print(f"slow {t_slow * 1000:.1f} ms, fast {t_fast * 1000:.2f} ms, speedup {t_slow / t_fast:.0f}x")
assert t_slow / t_fast > 20

# %% [markdown]
# ## 2. I/O-bound: threads and asyncio
#
# `fake_io` stands in for a network call: it just waits. 60 calls of 50-100 ms each.

# %%
N_CALLS, CAP = 60, 15
durations = [0.05 + 0.05 * rng.random() for _ in range(N_CALLS)]


def fake_io(d: float) -> float:
    time.sleep(d)      # releases the GIL, like a real socket read
    return d


t0 = time.perf_counter()
seq = [fake_io(d) for d in durations]
t_seq = time.perf_counter() - t0

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=CAP) as pool:
    thr = list(pool.map(fake_io, durations))
t_thr = time.perf_counter() - t0


async def fake_io_async(d: float, sem: asyncio.Semaphore, in_flight: list[int]) -> float:
    async with sem:
        in_flight[0] += 1
        in_flight[1] = max(in_flight[1], in_flight[0])   # track the peak concurrency
        await asyncio.sleep(d)                            # NOT time.sleep: that would block the loop
        in_flight[0] -= 1
        return d


async def run_async() -> tuple[list[float], int]:
    sem = asyncio.Semaphore(CAP)
    in_flight = [0, 0]
    res = await asyncio.gather(*(fake_io_async(d, sem, in_flight) for d in durations))
    return res, in_flight[1]


t0 = time.perf_counter()
asy, peak = asyncio.run(run_async())
t_asy = time.perf_counter() - t0

print(f"sequential {t_seq:.2f}s   threads {t_thr:.2f}s   asyncio {t_asy:.2f}s   (peak in flight: {peak})")
assert seq == thr == asy, "results must come back in input order"
assert peak <= CAP, "the semaphore must cap concurrency"
assert t_thr < t_seq / 4 and t_asy < t_seq / 4

# %% [markdown]
# ## 3. CPU-bound pure Python: threads vs processes

# %%
def busy(n: int) -> int:
    # Deliberately pure Python, so the GIL is held the whole time.
    s = 0
    for i in range(n):
        s += i * i % 7
    return s


WORK = [400_000] * 8

if __name__ == "__main__":
    t0 = time.perf_counter(); r_seq = [busy(n) for n in WORK]; t_cpu_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        r_thr = list(pool.map(busy, WORK))
    t_cpu_thr = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        r_proc = list(pool.map(busy, WORK))
    t_cpu_proc = time.perf_counter() - t0

    cores = os.cpu_count() or 1
    print(f"cores={cores}  sequential {t_cpu_seq:.2f}s  threads {t_cpu_thr:.2f}s  processes {t_cpu_proc:.2f}s")
    assert r_seq == r_thr == r_proc
    # Threads can't beat sequential by much on pure-Python work (the GIL). On a free-threaded Python they could.
    assert t_cpu_thr > 0.6 * t_cpu_seq, "threads unexpectedly sped up GIL-bound work (free-threaded build?)"
    if cores >= 2:
        assert t_cpu_proc < 0.9 * t_cpu_seq, "processes should beat sequential on a multi-core machine"
    print("\nAll checks passed.")
