# %% [markdown]
# # Lab 05.2: Tests, and how to break them
#
# 1. Under H0, p-values are uniform: 5% false positives at alpha = 0.05, by construction.
# 2. Welch t-test vs permutation test on skewed data.
# 3. McNemar for two classifiers on the same test set.
# 4. Multiple comparisons: 20 null metrics, and what Holm and BH do.
# 5. Peeking: stop-when-significant inflates the false positive rate.
# 6. Mann-Whitney U / (n_a n_b) is the ROC AUC.

# %%
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. p-values under the null

# %%
# 20,000 experiments at once: scipy's tests are vectorized along an axis
pvals = stats.ttest_ind(rng.normal(0, 1, (20000, 50)), rng.normal(0, 1, (20000, 50)), axis=1, equal_var=False).pvalue
print(f"fraction p < 0.05 under H0: {np.mean(pvals < 0.05):.4f}")
hist = np.histogram(pvals, bins=10, range=(0, 1))[0] / len(pvals)
print("p-value histogram (should be flat ~0.1):", np.round(hist, 3))
assert abs(np.mean(pvals < 0.05) - 0.05) < 0.006 and np.all(np.abs(hist - 0.1) < 0.01)

# %% [markdown]
# ## 2. Welch vs permutation test on revenue-like data

# %%
def permutation_test(a, b, n_perm=10_000, rng=rng):
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    diffs = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(pooled)
        diffs[k] = p[:len(a)].mean() - p[len(a):].mean()
    return (np.sum(np.abs(diffs) >= abs(observed)) + 1) / (n_perm + 1)


a = rng.lognormal(3.0, 1.0, 400)
b = rng.lognormal(3.15, 1.0, 400)          # ~16% higher mean
p_welch = stats.ttest_ind(a, b, equal_var=False).pvalue
p_perm = permutation_test(a, b)
print(f"means {a.mean():.1f} vs {b.mean():.1f}: Welch p={p_welch:.4f}, permutation p={p_perm:.4f}")
assert abs(np.log10(p_welch) - np.log10(p_perm)) < 0.5, "with n=400 per group, they roughly agree"

# %% [markdown]
# ## 3. McNemar

# %%
b_cnt, c_cnt = 22, 38                       # A right & B wrong, A wrong & B right
z = (c_cnt - b_cnt) / np.sqrt(b_cnt + c_cnt)
p_normal = 2 * stats.norm.sf(z)
z_cc = (abs(c_cnt - b_cnt) - 1) / np.sqrt(b_cnt + c_cnt)
p_cc = 2 * stats.norm.sf(z_cc)
p_exact = stats.binomtest(c_cnt, b_cnt + c_cnt, 0.5).pvalue
print(f"McNemar: normal p={p_normal:.3f}, continuity-corrected p={p_cc:.3f}, exact p={p_exact:.3f}")
assert 0.03 < p_normal < 0.045 and 0.045 < p_exact < 0.06

# %% [markdown]
# ## 4. Twenty metrics, nothing real

# %%
reps = 2000
all_p = stats.ttest_ind(rng.normal(size=(reps, 20, 200)), rng.normal(size=(reps, 20, 200)), axis=2).pvalue
fwer_raw = np.sum(np.any(all_p < 0.05, axis=1))
# Holm rejects at least one hypothesis iff the smallest p-value is <= alpha / m (its first step), so the
# family-wise error rate can be computed without running the full procedure 2000 times.
fwer_holm = np.sum(all_p.min(axis=1) <= 0.05 / 20)
print(f"P(at least one 'significant' metric): uncorrected {fwer_raw / reps:.2f} (theory {1 - 0.95**20:.2f}), "
      f"Holm {fwer_holm / reps:.3f}")
assert fwer_raw / reps > 0.55 and fwer_holm / reps < 0.065

pv = np.array([0.0004, 0.002, 0.009, 0.012, 0.03] + list(np.linspace(0.06, 0.99, 45)))
holm = multipletests(pv, alpha=0.05, method="holm")[0]
bh = multipletests(pv, alpha=0.05, method="fdr_bh")[0]
print(f"exercise 3: Holm rejects {holm.sum()}, BH rejects {bh.sum()}")
assert holm.sum() == 1 and bh.sum() == 2

# %% [markdown]
# ## 5. Peeking
#
# An A/A test (no real difference). 30 "days", 200 users per arm per day. The honest analyst tests once at the end;
# the impatient one tests every day and stops at the first p < 0.05.

# %%
def aa_tests(rng, reps=3000, days=30, per_day=200, p=0.1):
    """Vectorized over reps: cumulative conversions per day for two identical arms."""
    ca = np.cumsum(rng.binomial(per_day, p, (reps, days)), axis=1)
    cb = np.cumsum(rng.binomial(per_day, p, (reps, days)), axis=1)
    n = per_day * np.arange(1, days + 1)
    pa, pb = ca / n, cb / n
    pool = (pa + pb) / 2
    z = np.abs(pa - pb) / np.sqrt(2 * pool * (1 - pool) / n)
    significant = 2 * stats.norm.sf(z) < 0.05           # (reps, days): significant on day d?
    return significant.any(axis=1), significant[:, -1]  # peeker stops at the first hit; honest analyst looks once


peeked, final = aa_tests(rng)
print(f"false positive rate: test once at the end {final.mean():.3f}   peek daily {peeked.mean():.3f}")
assert final.mean() < 0.065 and peeked.mean() > 0.15

# %% [markdown]
# ## 6. Mann-Whitney U is the AUC

# %%
y = rng.binomial(1, 0.4, 500)
score = y + rng.normal(0, 1.2, 500)
u = stats.mannwhitneyu(score[y == 1], score[y == 0]).statistic
auc_from_u = u / (np.sum(y == 1) * np.sum(y == 0))
assert np.isclose(auc_from_u, roc_auc_score(y, score))
print(f"U/(n1 n0) = {auc_from_u:.4f} = AUC {roc_auc_score(y, score):.4f}")

# %%
print("\nAll checks passed.")
