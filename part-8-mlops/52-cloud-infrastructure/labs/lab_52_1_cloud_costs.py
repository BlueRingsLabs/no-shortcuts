# %% [markdown]
# # Lab 52.1: Cloud costs you can calculate before the bill does
#
# Prices are illustrative (they vary by provider, region and month); the arithmetic is what carries over.
# 1. A nightly training job of 4 GPU-hours: on-demand, or spot instances that get preempted. Checkpointing, and the
#    interval that minimizes wasted work (Young and Daly's formula), checked by simulation.
# 2. Which GPU? Price per hour vs price per job.
# 3. The quiet costs: idle instances and data egress.

# %%
import math

import numpy as np

ON_DEMAND, SPOT = 4.00, 1.40                                                        # $/GPU-hour, illustrative
JOB_HOURS = 4.0

# %% [markdown]
# ## 1. Spot instances and checkpoints

# %%
def run_spot(ckpt_every_h, mtbf_h, ckpt_cost_h=0.05, restart_h=0.15, seed=0, n=2000):
    """Simulate n nights. Preemptions arrive as a Poisson process (mean time between them: mtbf_h). Work since the last
    checkpoint is lost; each restart costs time to get a new instance and reload."""
    r = np.random.default_rng(seed)
    wall = np.zeros(n)
    for k in range(n):
        done, t = 0.0, 0.0
        while done < JOB_HOURS:
            segment = min(ckpt_every_h, JOB_HOURS - done)
            need = segment + (ckpt_cost_h if done + segment < JOB_HOURS else 0)
            fail_at = r.exponential(mtbf_h)
            if fail_at >= need:
                t += need; done += segment
            else:
                t += fail_at + restart_h                                             # lose the segment in progress
        wall[k] = t
    return wall


print(f"on-demand: {JOB_HOURS:.0f} h x ${ON_DEMAND:.2f} = ${JOB_HOURS * ON_DEMAND:.2f} a night, done in {JOB_HOURS:.1f} h")
mtbf = 3.0
print(f"\nspot at ${SPOT:.2f}/h, preempted on average every {mtbf:.0f} h; checkpoint takes 3 minutes, a restart 9 minutes:")
print(f"  {'checkpoint every':>16s} {'mean wall time':>15s} {'95th percentile':>16s} {'mean cost':>10s}")
res = {}
for every in (4.0, 2.0, 1.0, 0.5, 0.25, 0.1):
    w = run_spot(every, mtbf)
    res[every] = w
    print(f"  {every * 60:13.0f} min {w.mean():14.2f}h {np.percentile(w, 95):15.2f}h {w.mean() * SPOT:9.2f}$")
young = math.sqrt(2 * 0.05 * mtbf)
best = min(res, key=lambda e: res[e].mean())
print(f"Young/Daly optimum: checkpoint every sqrt(2 x checkpoint cost x MTBF) = {young * 60:.0f} minutes; best simulated: {best * 60:.0f} min")
print("without checkpoints, a 4-hour job on instances that die every 3 hours keeps starting over: cheaper than on-demand")
print("on average, but one night in twenty it takes over 20 hours, and the 'nightly' job isn't ready in the morning.")
print("With a checkpoint every half hour it costs less than half of on-demand and reliably finishes in about 5 to 6")
print("hours. Spot needs code that can resume: save model, optimizer, scheduler, data position and RNG state.")
assert res[best].mean() * SPOT < 0.5 * JOB_HOURS * ON_DEMAND and np.percentile(res[4.0], 95) > 2 * np.percentile(res[best], 95)

# %% [markdown]
# ## 2. Which GPU?

# %%
GPUS = {"small (L4-class)": (0.80, 1.0), "mid (A100-class)": (3.00, 4.5), "large (H100-class)": (6.00, 9.0)}
# ($/h, throughput relative to the small one on this job; illustrative)
job_small_hours = 30.0
print(f"\na job that takes {job_small_hours:.0f} h on the small GPU:")
for name, (price, speed) in GPUS.items():
    hours = job_small_hours / speed
    print(f"  {name:20s} ${price:5.2f}/h   {hours:5.1f} h   ${price * hours:6.2f} per job")
print("the most expensive GPU per hour can cost less per job, and finishes 9 times sooner, if the job uses it (large batches,")
print("mixed precision, no data-loading bottleneck, 33.3). Measure throughput on your job; don't pick by the hourly price.")

# %% [markdown]
# ## 3. The quiet costs

# %%
dev_boxes, hours_used = 6, 3 * 5 * 4                                                # 6 people, ~3 h a day of real use
always_on = dev_boxes * 24 * 30 * ON_DEMAND
used = dev_boxes * hours_used * ON_DEMAND
print(f"\n6 GPU dev instances left running all month: ${always_on:,.0f}; the hours actually used: ${used:,.0f} "
      f"({used / always_on:.0%}). Auto-stop idle instances; it's the single most common cloud-bill surprise.")
egress_per_gb = 0.09
for tb in (1, 20):
    print(f"moving {tb} TB out of the cloud (or across regions) at ${egress_per_gb}/GB: ${tb * 1000 * egress_per_gb:,.0f}, every time")
print("keep compute next to the data, don't train in one region on data stored in another, and set budget alerts")
print("with per-team tags so the surprise is a Slack message on day 3, not an invoice on day 31.")

print("\nAll checks passed.")
