# %% [markdown]
# # Lab 15.3: Monitoring the data, not just the jobs
#
# Twelve weeks of daily loads of an events table, with weekly seasonality and three planted incidents:
#   - a truncated upstream export (volume drop),
#   - a column that silently starts arriving mostly NULL (distribution change),
#   - a late load (freshness).
#
# 1. Compute per-load metrics (volume, freshness, null rates, schema).
# 2. A naive static threshold vs a weekday-aware robust baseline: count false alarms and misses.
# 3. Schema drift detection.
# 4. An alert routed with severity and a runbook link.

# %%
from dataclasses import dataclass

import numpy as np
import pandas as pd

rng = np.random.default_rng(17)
days = pd.date_range("2025-03-03", periods=84, freq="D")
INCIDENT_VOLUME = pd.Timestamp("2025-05-06")
INCIDENT_NULLS = pd.Timestamp("2025-05-13")
INCIDENT_LATE = pd.Timestamp("2025-05-20")

# %% [markdown]
# ## 1. Per-load metrics

# %%
rows = []
for d in days:
    weekend = d.dayofweek >= 5
    base = 400_000 if weekend else 1_200_000
    volume = int(base * rng.normal(1.0 + 0.002 * (d - days[0]).days, 0.04))     # slow growth + noise
    null_rate = abs(rng.normal(0.001, 0.0003))
    loaded_hour = rng.normal(5.5, 0.3)                                        # usually loaded ~05:30 UTC
    columns = ("event_id", "customer_id", "ts", "event_type", "product_id")
    if d == INCIDENT_VOLUME:
        volume = int(volume * 0.70)                                          # 30% of the data missing
    if d == INCIDENT_NULLS:
        null_rate = 0.12
    if d == INCIDENT_LATE:
        loaded_hour = 11.0
    if d >= pd.Timestamp("2025-05-25"):
        columns = ("event_id", "customer_id", "ts", "event_type", "product_id", "campaign")   # upstream added a field
    rows.append({"day": d, "volume": volume, "null_rate_customer_id": null_rate, "loaded_hour": loaded_hour, "columns": columns})
metrics = pd.DataFrame(rows)
print(metrics.head(8)[["day", "volume", "null_rate_customer_id", "loaded_hour"]].round(4).to_string(index=False))

# %% [markdown]
# ## 2. Static threshold vs weekday-aware robust baseline

# %%
static_alerts = set(metrics.loc[metrics["volume"] < 800_000, "day"])


def robust_alerts(metrics, col, weeks=6, z=5.0, min_rel=0.15):
    alerts = set()
    for i, r in metrics.iterrows():
        hist = metrics[(metrics["day"] < r["day"]) & (metrics["day"].dt.dayofweek == r["day"].dayofweek)].tail(weeks)
        if len(hist) < 3:
            continue
        med = hist[col].median()
        mad = (hist[col] - med).abs().median() * 1.4826 or 1e-9
        if abs(r[col] - med) / mad > z and abs(r[col] - med) / med > min_rel:
            alerts.add(r["day"])
    return alerts


vol_alerts = robust_alerts(metrics, "volume")
print(f"static threshold (< 800k): {len(static_alerts)} alerts, incident caught: {INCIDENT_VOLUME in static_alerts}")
print(f"weekday robust baseline:   {len(vol_alerts)} alerts, incident caught: {INCIDENT_VOLUME in vol_alerts}")
false_vol = vol_alerts - {INCIDENT_VOLUME}
print(f"  false alarms: static {len(static_alerts - {INCIDENT_VOLUME})}, robust {len(false_vol)} "
      f"{sorted(d.date() for d in false_vol)}  (fewer, not zero: that's what tuning is for)")
assert INCIDENT_VOLUME in vol_alerts and len(false_vol) <= 1
assert len(static_alerts) >= 20, "the static rule fires every weekend..."
assert INCIDENT_VOLUME not in static_alerts, "...and misses the real incident on a weekday (70% of ~1.3M is still above 800k)"

null_alerts = {r["day"] for _, r in metrics.iterrows() if r["null_rate_customer_id"] > 10 * metrics["null_rate_customer_id"].median()}
assert null_alerts == {INCIDENT_NULLS}
late = set(metrics.loc[metrics["loaded_hour"] > 7.0, "day"])                  # SLO: loaded by 07:00 UTC
assert late == {INCIDENT_LATE}
print("null-rate monitor caught", sorted(d.date() for d in null_alerts), "| freshness SLO caught", sorted(d.date() for d in late))

# %% [markdown]
# ## 3. Schema drift

# %%
drift = []
for prev, cur in zip(metrics.itertuples(), metrics.iloc[1:].itertuples()):
    added, removed = set(cur.columns) - set(prev.columns), set(prev.columns) - set(cur.columns)
    if added or removed:
        drift.append((cur.day.date(), sorted(added), sorted(removed)))
print("schema changes:", drift)
assert drift == [(pd.Timestamp("2025-05-25").date(), ["campaign"], [])]

# %% [markdown]
# ## 4. Alerts with severity and a runbook

# %%
@dataclass
class Alert:
    table: str
    check: str
    day: pd.Timestamp
    severity: str
    runbook: str

    def render(self):
        return f"[{self.severity.upper()}] {self.table}: {self.check} on {self.day.date()} -> {self.runbook}"


SEVERITY = {"volume": "page", "null_rate": "page", "freshness": "ticket", "schema_added_column": "log"}
RUNBOOKS = "https://wiki.example.internal/runbooks/events#"
alerts = ([Alert("events", "volume", d, SEVERITY["volume"], RUNBOOKS + "volume") for d in vol_alerts]
          + [Alert("events", "null_rate customer_id", d, SEVERITY["null_rate"], RUNBOOKS + "nulls") for d in null_alerts]
          + [Alert("events", "freshness SLO 07:00", d, SEVERITY["freshness"], RUNBOOKS + "late") for d in late]
          + [Alert("events", f"schema: added {a}", pd.Timestamp(day), SEVERITY["schema_added_column"], RUNBOOKS + "schema")
             for day, a, _ in drift])
for a in sorted(alerts, key=lambda a: a.day):
    print(a.render())
pages = [a for a in alerts if a.severity == "page"]
paged_days = {a.day for a in pages}
assert {INCIDENT_VOLUME, INCIDENT_NULLS} <= paged_days, "both real data incidents page someone"
assert len(pages) <= 3, "and at most one page in twelve weeks is noise: track that ratio and keep tuning"

print("\nAll checks passed.")
