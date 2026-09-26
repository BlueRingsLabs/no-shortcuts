# %% [markdown]
# # Lab 07.1: Data structures and streaming algorithms
#
# 1. List vs set membership: O(n m) vs O(n + m), measured.
# 2. Top-k: sort vs heap vs argpartition.
# 3. Reservoir sampling: every item equally likely.
# 4. Bloom filter: measured false positive rate vs theory.
# 5. Count-min sketch: never underestimates, bounded overestimate.
# 6. HyperLogLog: distinct counts in a few KB.
# 7. Edit distance by dynamic programming.

# %%
import hashlib
import heapq
import time

import numpy as np

rng = np.random.default_rng(0)


def h64(item, seed: int) -> int:
    """Stable 64-bit hash (unlike Python's hash(), identical across runs and machines)."""
    return int.from_bytes(hashlib.blake2b(f"{seed}:{item}".encode(), digest_size=8).digest(), "little")

# %% [markdown]
# ## 1. Membership

# %%
events = rng.integers(0, 1_000_000, 20_000).tolist()
banned_list = rng.integers(0, 1_000_000, 5_000).tolist()
banned_set = set(banned_list)
t0 = time.perf_counter(); a = [e for e in events if e in banned_list]; t_list = time.perf_counter() - t0
t0 = time.perf_counter(); b = [e for e in events if e in banned_set]; t_set = time.perf_counter() - t0
print(f"list: {t_list:.3f}s   set: {t_set * 1000:.2f} ms   ({t_list / t_set:.0f}x)")
assert a == b and t_list > 50 * t_set

# %% [markdown]
# ## 2. Top-k

# %%
scores = rng.normal(size=2_000_000)
k = 10
def best_of(fn, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t0)
    return out, min(times)


top_sort, t_sort = best_of(lambda: np.sort(scores)[-k:][::-1])
top_part, t_part = best_of(lambda: np.sort(scores[np.argpartition(scores, -k)[-k:]])[::-1])
top_heap = np.array(heapq.nlargest(k, scores[:200_000]))     # the streaming version, on a slice (it's pure Python)
assert np.allclose(top_sort, top_part)
assert np.allclose(top_heap, np.sort(scores[:200_000])[-k:][::-1])
print(f"top-{k} of 2M: full sort {t_sort * 1000:.0f} ms, argpartition {t_part * 1000:.0f} ms")
assert t_part < t_sort

# %% [markdown]
# ## 3. Reservoir sampling

# %%
def reservoir(stream, k, rng):
    sample = []
    for i, x in enumerate(stream, start=1):
        if i <= k:
            sample.append(x)
        else:
            j = rng.integers(0, i)          # uniform in [0, i)
            if j < k:                       # happens with probability k / i
                sample[j] = x
    return sample


n, k = 50, 5
counts = np.zeros(n)
for _ in range(20_000):
    for x in reservoir(range(n), k, rng):
        counts[x] += 1
freq = counts / 20_000
print(f"inclusion probability: min {freq.min():.3f} max {freq.max():.3f} (theory {k / n:.3f})")
assert np.all(np.abs(freq - k / n) < 0.012)

# %% [markdown]
# ## 4. Bloom filter

# %%
class Bloom:
    def __init__(self, n_items: int, fp_rate: float):
        self.m = int(np.ceil(-n_items * np.log(fp_rate) / np.log(2) ** 2))
        self.h = max(1, round(self.m / n_items * np.log(2)))
        self.bits = np.zeros(self.m, dtype=bool)

    def _positions(self, item):
        a, b = h64(item, 1), h64(item, 2)                 # double hashing: h_i = a + i*b
        return [(a + i * b) % self.m for i in range(self.h)]

    def add(self, item):
        self.bits[self._positions(item)] = True

    def __contains__(self, item):
        return bool(self.bits[self._positions(item)].all())


n_items = 20_000
bf = Bloom(n_items, 0.01)
for i in range(n_items):
    bf.add(f"url-{i}")
assert all(f"url-{i}" in bf for i in range(0, n_items, 7)), "no false negatives, ever"
fp = np.mean([f"other-{i}" in bf for i in range(50_000)])
theory = (1 - np.exp(-bf.h * n_items / bf.m)) ** bf.h
print(f"Bloom: {bf.m / n_items:.1f} bits/item, {bf.h} hashes, false positives {fp:.4f} (theory {theory:.4f})")
assert abs(fp - theory) < 0.004

# %% [markdown]
# ## 5. Count-min sketch on a Zipf-distributed stream (a few items are very frequent)

# %%
class CountMin:
    def __init__(self, eps: float, delta: float):
        self.w = int(np.ceil(np.e / eps))
        self.d = int(np.ceil(np.log(1 / delta)))
        self.table = np.zeros((self.d, self.w), dtype=np.int64)

    def add(self, item, count=1):
        for r in range(self.d):
            self.table[r, h64(item, r) % self.w] += count

    def estimate(self, item):
        return min(self.table[r, h64(item, r) % self.w] for r in range(self.d))


stream = rng.zipf(1.3, 100_000)
stream = stream[stream < 10_000]
cm = CountMin(eps=0.001, delta=0.01)
true_counts = {}
for x in stream.tolist():
    cm.add(x)
    true_counts[x] = true_counts.get(x, 0) + 1
N = len(stream)
errors = np.array([cm.estimate(x) - c for x, c in true_counts.items()])
print(f"count-min: {cm.d}x{cm.w} counters for {len(true_counts):,} distinct items; "
      f"max overestimate {errors.max()} (bound eps*N = {0.001 * N:.0f})")
assert errors.min() >= 0, "never underestimates"
assert np.mean(errors <= 0.001 * N) >= 0.99

# %% [markdown]
# ## 6. HyperLogLog

# %%
class HLL:
    def __init__(self, p: int = 12):
        self.p, self.m = p, 1 << p
        self.reg = np.zeros(self.m, dtype=np.int8)

    def add(self, item):
        x = h64(item, 0)
        j = x >> (64 - self.p)                            # first p bits pick the register
        w = x & ((1 << (64 - self.p)) - 1)
        rank = (64 - self.p) - w.bit_length() + 1         # position of the leftmost 1-bit
        self.reg[j] = max(self.reg[j], rank)

    def count(self):
        alpha = 0.7213 / (1 + 1.079 / self.m)
        est = alpha * self.m**2 / np.sum(2.0 ** (-self.reg.astype(float)))
        zeros = np.sum(self.reg == 0)
        if est <= 2.5 * self.m and zeros:                 # small-range correction
            est = self.m * np.log(self.m / zeros)
        return est


hll2 = HLL(p=12)                                          # 4096 one-byte registers = 4 KB
for i in range(100_000):
    hll2.add(f"user-{i}")
    hll2.add(f"user-{i}")                                 # duplicates don't change anything
rel = abs(hll2.count() - 100_000) / 100_000
print(f"100,000 distinct users (each seen twice): estimate {hll2.count():,.0f}  ({rel:.2%} error, "
      f"expected ~{1.04 / np.sqrt(hll2.m):.1%})")
assert rel < 0.05

# %% [markdown]
# ## 7. Edit distance

# %%
def edit_distance(a: str, b: str) -> int:
    D = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    D[:, 0] = np.arange(len(a) + 1)
    D[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = min(D[i - 1, j] + 1,                            # deletion
                          D[i, j - 1] + 1,                            # insertion
                          D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))   # substitution (or match)
    return int(D[-1, -1])


assert edit_distance("kitten", "sitting") == 3
assert edit_distance("", "abc") == 3 and edit_distance("same", "same") == 0
print("edit distance: ok")

# %%
print("\nAll checks passed.")
