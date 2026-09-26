# %% [markdown]
# # Lab 14.2: Event time, watermarks, state
#
# 1. An out-of-order event stream (network delays, an offline device, a consumer outage).
# 2. Processing-time vs event-time tumbling windows against the truth.
# 3. Watermark delay vs lateness: the latency/completeness trade-off, measured.
# 4. Session windows that merge when a late event fills a gap.
# 5. Stateful dedup with a TTL, and bounded state.
# 6. Applying a CDC stream (with duplicates) to a mirror table.

# %%
from collections import defaultdict

import numpy as np
import pandas as pd

rng = np.random.default_rng(15)

# %% [markdown]
# ## 1. The stream

# %%
n = 20_000
event_time = np.sort(rng.uniform(0, 3600, n))                    # one hour of events, in seconds
delay = rng.exponential(2.0, n)                                   # normal network delay
offline = rng.uniform(size=n) < 0.02                              # 2% of devices were offline for a while
delay[offline] += rng.uniform(60, 900, offline.sum())
arrival = event_time + delay
outage = (arrival > 1800) & (arrival < 2100)                      # the consumer was down 5 minutes...
arrival[outage] = 2100 + (arrival[outage] - 1800) * 0.05          # ...then caught up in a burst
stream = pd.DataFrame({"event_id": np.arange(n), "event_time": event_time, "arrival": arrival}).sort_values("arrival")
print(f"{n:,} events; out of order: {np.mean(np.diff(stream['event_time'].to_numpy()) < 0):.0%} of consecutive pairs")

WINDOW = 60
truth = np.bincount((event_time // WINDOW).astype(int), minlength=60)

# %% [markdown]
# ## 2. Processing time vs event time

# %%
by_processing = np.bincount(np.minimum(stream["arrival"] // WINDOW, 59).astype(int), minlength=60)
err_proc = np.abs(by_processing - truth).max()
print(f"processing-time windows: worst window off by {err_proc} events "
      f"(minute 35 shows {by_processing[35]} vs true {truth[35]}: the catch-up spike that never happened)")
assert by_processing[35] > 2 * truth[35]

# %% [markdown]
# ## 3. Event-time windows with a watermark

# %%
def event_time_windows(stream, allowed_delay, window=WINDOW):
    open_windows = defaultdict(int)
    results, late, max_et = {}, 0, -np.inf
    emit_latency = []
    for et, arr in zip(stream["event_time"].to_numpy(), stream["arrival"].to_numpy()):
        max_et = max(max_et, et)
        watermark = max_et - allowed_delay
        w = int(et // window)
        if (w + 1) * window <= watermark and w not in open_windows:
            late += 1                                             # its window was already emitted: drop (and count)
        else:
            open_windows[w] += 1
        for w_open in [w_ for w_ in open_windows if (w_ + 1) * window <= watermark]:
            results[w_open] = open_windows.pop(w_open)            # finalize and emit
            emit_latency.append(arr - (w_open + 1) * window)      # how long after the window's end we emitted
    results.update(open_windows)                                  # end of stream: flush
    counts = np.array([results.get(w, 0) for w in range(60)])
    return counts, late, float(np.median(emit_latency))


rows = []
for d in (5, 30, 120, 600, 1200):
    counts, late, lat = event_time_windows(stream, d)
    rows.append((d, late, np.abs(counts - truth).max(), lat))
res = pd.DataFrame(rows, columns=["allowed_delay_s", "late_dropped", "max_window_error", "median_emit_delay_s"])
print(res.to_string(index=False))
assert res["late_dropped"].is_monotonic_decreasing and res["median_emit_delay_s"].is_monotonic_increasing
assert res.iloc[-1]["late_dropped"] < res.iloc[0]["late_dropped"] / 10
counts_5, _, _ = event_time_windows(stream, 5)
assert np.abs(counts_5 - truth).max() < err_proc, "even an aggressive event-time watermark beats processing time"

# %% [markdown]
# ## 4. Session windows (gap = 30 min) and merging

# %%
def sessions(times, gap=1800):
    out = []
    for t in sorted(times):
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


user_events = [0, 600, 1200, 4000, 4400]                         # two sessions: 0-1200 and 4000-4400
assert len(sessions(user_events)) == 2
late_event = 2600                                                # arrives late, filling the gap
merged = sessions(user_events + [late_event])
print(f"sessions before the late event: {sessions(user_events)}; after: {merged}")
assert len(merged) == 1, "a late event can merge two sessions into one: downstream must handle retractions"

# %% [markdown]
# ## 5. Stateful dedup with a TTL

# %%
redelivered = stream.sample(frac=0.05, random_state=1).assign(arrival=lambda d: d["arrival"] + rng.uniform(1, 120, len(d)))
with_dupes = pd.concat([stream, redelivered]).sort_values("arrival")


def dedup(events, ttl):
    seen, out, peak = {}, [], 0
    for eid, arr in zip(events["event_id"].to_numpy(), events["arrival"].to_numpy()):
        if eid in seen and arr - seen[eid] <= ttl:
            continue
        seen[eid] = arr
        out.append(eid)
        if len(out) % 500 == 0:                                   # periodically expire old ids
            seen = {k: v for k, v in seen.items() if arr - v <= ttl}
        peak = max(peak, len(seen))
    return out, peak


kept_long, peak_long = dedup(with_dupes, ttl=10**9)
kept_ttl, peak_ttl = dedup(with_dupes, ttl=300)
print(f"dedup: {len(with_dupes):,} deliveries -> {len(kept_ttl):,} events; state size peak: no TTL {peak_long:,}, 5-min TTL {peak_ttl:,}")
assert len(set(kept_ttl)) == len(kept_ttl) == n and peak_ttl < peak_long / 5

# %% [markdown]
# ## 6. CDC stream -> mirror table

# %%
source, changes, lsn = {}, [], 0
for step in range(5000):
    lsn += 1
    k = int(rng.integers(0, 500))
    op = rng.choice(["upsert", "delete"], p=[0.85, 0.15])
    if op == "upsert":
        source[k] = {"id": k, "value": int(rng.integers(0, 1000)), "lsn": lsn}
        changes.append(("u", k, dict(source[k]), lsn))
    elif k in source:
        del source[k]
        changes.append(("d", k, None, lsn))
dupes = [changes[i] for i in rng.choice(len(changes), 300)]       # at-least-once redelivery, out of order
cdc_stream = changes + dupes

mirror, last_lsn = {}, {}
for op, k, row, change_lsn in cdc_stream:
    if change_lsn <= last_lsn.get(k, 0):
        continue                                                  # stale or duplicate change for this key: ignore
    last_lsn[k] = change_lsn
    if op == "u":
        mirror[k] = row
    else:
        mirror.pop(k, None)
assert mirror == source
print(f"CDC: {len(cdc_stream):,} change events (incl. {len(dupes)} redeliveries) -> mirror identical to source ({len(source)} rows)")

print("\nAll checks passed.")
