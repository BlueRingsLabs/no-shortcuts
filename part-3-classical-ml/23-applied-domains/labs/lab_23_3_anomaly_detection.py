# %% [markdown]
# # Lab 23.3: Anomaly detection
#
# 1. Masking: classical vs robust z-scores on contaminated data.
# 2. Correlation breakers: per-feature z vs Mahalanobis (classical vs robust MCD covariance).
# 3. Six detectors against three kinds of planted anomalies. Nobody wins everything.
# 4. A small persistent shift in a stream: 3-sigma chart vs CUSUM, detection delay and false alarms.
# 5. Contextual anomalies: global threshold vs residual from a seasonal baseline.
# 6. The alert budget: precision among the top k vs ROC-AUC.

# %%
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(233)

# %% [markdown]
# ## 1. Masking

# %%
normal = rng.normal(100, 10, 900)
outliers = rng.normal(150, 3, 100)                              # 10% contamination, clustered at 5 sd (a batch of bad readings)
x = np.r_[normal, outliers]
is_out = np.r_[np.zeros(900, bool), np.ones(100, bool)]
z = (x - x.mean()) / x.std()
mad = 1.4826 * np.median(np.abs(x - np.median(x)))
rz = (x - np.median(x)) / mad
print(f"classical: mean {x.mean():.1f}, sd {x.std():.1f}; caught {np.sum((np.abs(z) > 3) & is_out)} of 100 outliers")
print(f"robust:    median {np.median(x):.1f}, MAD-sd {mad:.1f}; caught {np.sum((np.abs(rz) > 3) & is_out)} of 100, "
      f"false alarms {np.sum((np.abs(rz) > 3) & ~is_out)}")
assert np.sum((np.abs(rz) > 3) & is_out) >= 95 and np.sum((np.abs(z) > 3) & is_out) < 10

# %% [markdown]
# ## 2. Correlation breakers

# %%
cov = np.array([[1.0, 0.9], [0.9, 1.0]])
Xn = rng.multivariate_normal([0, 0], cov, 2000)
b1 = rng.uniform(1.0, 1.5, 40) * rng.choice([-1, 1], 40)
breakers = np.column_stack([b1, -b1 + rng.normal(0, 0.1, 40)])  # each coordinate within 1.5 sd, the combination absurd
contam = rng.multivariate_normal([4, -4], 0.1 * np.eye(2), 100)   # a contaminating cluster that distorts the classical estimate
X2 = np.vstack([Xn, contam, breakers]); lab2 = np.r_[np.zeros(2100), np.ones(40)]
per_feature = np.abs(StandardScaler().fit_transform(X2)).max(1)
maha_classic = EmpiricalCovariance().fit(X2).mahalanobis(X2)
maha_robust = MinCovDet(random_state=0).fit(X2).mahalanobis(X2)
sel = np.r_[np.ones(2000, bool), np.zeros(100, bool), np.ones(40, bool)]   # score breakers against the clean normals
for name, s in [("max per-feature |z|", per_feature), ("Mahalanobis, classical cov", maha_classic), ("Mahalanobis, robust MCD", maha_robust)]:
    print(f"{name:28s} ROC-AUC for correlation breakers {roc_auc_score(lab2[sel], s[sel]):.3f}")
assert roc_auc_score(lab2[sel], maha_robust[sel]) > 0.99 > roc_auc_score(lab2[sel], per_feature[sel])
assert roc_auc_score(lab2[sel], maha_robust[sel]) > roc_auc_score(lab2[sel], maha_classic[sel])

# %% [markdown]
# ## 3. Six detectors, three kinds of anomaly
#
# Normal data lies near a 2-D plane inside 6-D space, in two clusters: a tight one and a loose one.
# Anomalies: global (far from everything), local (just beside the tight cluster, closer to the data than many loose-cluster
# points are to each other), and off-plane (inside the data's range, but a small step off the plane the data lives on).

# %%
d = 6
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
B, c = Q[:, :2], Q[:, 2]                                         # the plane, and a direction orthogonal to it
embed = lambda L, noise: L @ B.T + rng.normal(0, noise, (len(L), d))
normal_all = np.vstack([embed(rng.normal([0, 0], 0.3, (1500, 2)), 0.05), embed(rng.normal([6, 6], 2.0, (1500, 2)), 0.05)])
glob = rng.uniform(-8, 14, (30, d))
ang = rng.uniform(0, 2 * np.pi, 30)
local = embed(np.column_stack([np.cos(ang), np.sin(ang)]) * rng.uniform(1.2, 1.6, (30, 1)), 0.05)   # scattered around the tight cluster
offp = embed(rng.normal([6, 6], 1.5, (30, 2)), 0.05) + 0.8 * c
X3 = np.vstack([normal_all, glob, local, offp])
kind = np.r_[np.zeros(3000), np.ones(30), 2 * np.ones(30), 3 * np.ones(30)]
Z3 = StandardScaler().fit(normal_all).transform(X3)
pca = PCA(2).fit(Z3)                                            # fitted on the contaminated data, like every other detector here
pca_clean = PCA(2).fit(Z3[kind == 0])                           # fitted on known-clean reference data (novelty detection)
scores = {
    "isolation forest": -IsolationForest(n_estimators=300, random_state=0).fit(Z3).score_samples(Z3),
    "LOF (k=20)": -LocalOutlierFactor(n_neighbors=20).fit(Z3).negative_outlier_factor_,
    "k-NN distance": NearestNeighbors(n_neighbors=11).fit(Z3).kneighbors(Z3)[0][:, -1],
    "robust Mahalanobis": MinCovDet(random_state=0).fit(Z3).mahalanobis(Z3),
    "GMM density (2 comp.)": -GaussianMixture(2, random_state=0).fit(Z3).score_samples(Z3),
    "PCA reconstruction": ((Z3 - pca.inverse_transform(pca.transform(Z3))) ** 2).sum(1),
    "PCA recon. (clean fit)": ((Z3 - pca_clean.inverse_transform(pca_clean.transform(Z3))) ** 2).sum(1),
}
names = {1: "global", 2: "local", 3: "off-plane"}
print(f"{'average precision':24s} " + "  ".join(f"{names[k]:>10s}" for k in (1, 2, 3)))
table = {}
for m_name, s in scores.items():
    row = []
    for k in (1, 2, 3):
        mask = (kind == 0) | (kind == k)
        row.append(average_precision_score(kind[mask] == k, s[mask]))
    table[m_name] = row
    print(f"{m_name:24s} " + "  ".join(f"{v:10.3f}" for v in row))
winners = {names[k]: max(table, key=lambda m_: table[m_][i]) for i, k in enumerate((1, 2, 3))}
print("best per anomaly type:", winners)
assert len(set(winners.values())) >= 2, "no single detector wins every type"
assert table["LOF (k=20)"][1] > table["robust Mahalanobis"][1]
assert table["PCA recon. (clean fit)"][2] > 0.9 > table["PCA reconstruction"][2], "30 gross outliers tilt a PCA fitted on contaminated data"

# %% [markdown]
# ## 4. A small persistent shift: 3-sigma vs CUSUM

# %%
def run_stream(shift_at=500, n=1500, shift=0.5, h=5.0, k=0.25, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(0, 1, n); x[shift_at:] += shift
    first_3s = next((t for t in range(n) if abs(x[t]) > 3), None)
    S, first_cusum = 0.0, None
    for t in range(n):
        S = max(0.0, S + x[t] - k)
        if S > h and first_cusum is None:
            first_cusum = t
    return first_3s, first_cusum


def summarize(h):
    d3, dc, f3, fc = [], [], 0, 0
    for seed in range(200):
        a_, b_ = run_stream(seed=seed, h=h)
        f3 += a_ is not None and a_ < 500; fc += b_ is not None and b_ < 500
        if a_ is not None and a_ >= 500: d3.append(a_ - 500)
        if b_ is not None and b_ >= 500: dc.append(b_ - 500)
    return f3 / 200, fc / 200, np.median(d3), np.median(dc)


# a fair comparison needs equal false-alarm rates: raise CUSUM's threshold h until it false-alarms no more than the chart
fa_3s = summarize(5.0)[0]
h = next(h for h in np.arange(5.0, 20.0, 0.5) if summarize(h)[1] <= fa_3s)
f3, fc, d3, dc = summarize(h)
print(f"shift of 0.5 sd at t=500, 200 simulated streams. False alarm before the shift: 3-sigma {f3:.0%}, CUSUM (h={h}) {fc:.0%}")
print(f"median detection delay after the shift: 3-sigma {d3:.0f} points, CUSUM {dc:.0f} points")
assert dc < d3 / 3

# %% [markdown]
# ## 5. Contextual anomalies: the 3 a.m. problem

# %%
hours = np.arange(24 * 7 * 8)                                   # 8 weeks, hourly
daily = 100 + 80 * np.sin(2 * np.pi * (hours % 24 - 8) / 24).clip(0) + 20 * (hours % (24 * 7) < 24 * 5)
traffic = rng.poisson(daily).astype(float)
anom_idx = rng.choice(np.flatnonzero((hours % 24 >= 1) & (hours % 24 <= 5) & (hours > 24 * 14)), 12, replace=False)
traffic[anom_idx] = rng.poisson(170, 12)                        # daytime-level traffic at night
truth = np.zeros(len(hours), bool); truth[anom_idx] = True
glob_score = traffic
baseline = np.array([np.median(traffic[max(0, t - 24 * 7 * 4):t][(hours[max(0, t - 24 * 7 * 4):t] % (24 * 7)) == t % (24 * 7)])
                     if t >= 24 * 7 else np.nan for t in range(len(hours))])
resid = np.abs(traffic - baseline) / np.sqrt(np.maximum(baseline, 1))
ok = hours >= 24 * 14
print(f"night-time spikes to daytime levels: AP with a global threshold {average_precision_score(truth[ok], glob_score[ok]):.3f}, "
      f"with the residual from a same-hour-of-week baseline {average_precision_score(truth[ok], resid[ok]):.3f}")
assert average_precision_score(truth[ok], resid[ok]) > 0.7 and average_precision_score(truth[ok], resid[ok]) > 10 * average_precision_score(truth[ok], glob_score[ok])

# %% [markdown]
# ## 6. The alert budget

# %%
n_ev = 200_000
lab6 = np.zeros(n_ev, bool); lab6[rng.choice(n_ev, 40, replace=False)] = True   # 0.02% true incidents
score6 = rng.normal(0, 1, n_ev) + 3.2 * lab6                   # a decent detector
k = 40                                                          # the team can review 40 alerts a day
top = np.argsort(-score6)[:k]
print(f"ROC-AUC {roc_auc_score(lab6, score6):.3f} (sounds great); precision among the top {k} alerts {lab6[top].mean():.2f}; "
      f"recall at that budget {lab6[top].sum() / lab6.sum():.2f}")
print("with 0.02% prevalence, even an excellent ranking hands the analysts mostly false alarms; context (section 5) is how you do better")
assert roc_auc_score(lab6, score6) > 0.98 and lab6[top].mean() < 0.6

print("\nAll checks passed.")
