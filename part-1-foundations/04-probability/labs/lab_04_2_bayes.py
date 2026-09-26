# %% [markdown]
# # Lab 04.2: Bayes, base rates and Simpson
#
# 1. The intrusion detector: simulate a million connections and count what the SOC actually sees.
# 2. Odds form: combining independent evidence.
# 3. Simpson's paradox: the kidney stone data, and a checker for any 2x2x2 table.
# 4. Beta-binomial updating, and how much the prior matters as data grows.
# 5. The autoregressive factorization: the probability of a sequence as a product of conditionals.

# %%
import numpy as np
from scipy import stats
from scipy.integrate import trapezoid

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. A 99% accurate detector on rare attacks

# %%
def posterior(prior, tpr, fpr):
    return tpr * prior / (tpr * prior + fpr * (1 - prior))


n = 1_000_000
attack = rng.uniform(size=n) < 1e-4
alert = np.where(attack, rng.uniform(size=n) < 0.99, rng.uniform(size=n) < 0.01)
tp, fp = np.sum(alert & attack), np.sum(alert & ~attack)
print(f"attacks: {attack.sum()}, alerts: {alert.sum()}, true alerts: {tp}, false alerts: {fp}")
print(f"precision (simulated): {tp / (tp + fp):.4f}   (Bayes): {posterior(1e-4, 0.99, 0.01):.4f}")
assert abs(tp / (tp + fp) - posterior(1e-4, 0.99, 0.01)) < 0.005
assert posterior(1e-4, 0.99, 0.01) < 0.01

fpr_needed = 0.99 * 1e-4 / 0.9999          # FPR for which precision = 0.5
assert np.isclose(posterior(1e-4, 0.99, fpr_needed), 0.5)
print(f"FPR needed for 50% precision: {fpr_needed:.2e}")

# %% [markdown]
# ## 2. Odds form and independent evidence

# %%
prior = 0.2
odds = prior / (1 - prior) * 3 * 20
p_spam = odds / (1 + odds)
assert np.isclose(p_spam, 15 / 16)

# Same thing by simulation: generate emails where the two cues are conditionally independent given the class
m = 2_000_000
spam = rng.uniform(size=m) < prior
p_invoice = np.where(spam, 0.3, 0.1)       # likelihood ratio 3
p_newdom = np.where(spam, 0.4, 0.02)       # likelihood ratio 20
invoice = rng.uniform(size=m) < p_invoice
newdom = rng.uniform(size=m) < p_newdom
both = invoice & newdom
print(f"P(spam | both cues): simulated {spam[both].mean():.4f}, odds form {p_spam:.4f}")
assert abs(spam[both].mean() - p_spam) < 0.01

# %% [markdown]
# ## 3. Simpson's paradox

# %%
def simpson(table):
    """table[group][option] = (successes, trials). Returns (per-group winners, overall winner)."""
    rate = lambda s, t: s / t
    per_group = [max(g, key=lambda o: rate(*g[o])) for g in table.values()]
    totals = {o: tuple(map(sum, zip(*(g[o] for g in table.values())))) for o in next(iter(table.values()))}
    overall = max(totals, key=lambda o: rate(*totals[o]))
    return per_group, overall, totals


kidney = {"small": {"A": (81, 87), "B": (234, 270)}, "large": {"A": (192, 263), "B": (55, 80)}}
per_group, overall, totals = simpson(kidney)
print("kidney stones: best per group", per_group, "| best overall", overall, totals)
assert per_group == ["A", "A"] and overall == "B"

mine = {"easy": {"X": (9, 10), "Y": (80, 100)}, "hard": {"X": (30, 100), "Y": (2, 10)}}
per_group, overall, _ = simpson(mine)
assert per_group == ["X", "X"] and overall == "Y"

# %% [markdown]
# ## 4. Beta-binomial updating

# %%
a0, b0 = 1, 1
k, n_obs = 12, 200
post = stats.beta(a0 + k, b0 + n_obs - k)
print(f"posterior Beta({a0 + k}, {b0 + n_obs - k}): mean {post.mean():.4f}, 95% interval "
      f"({post.ppf(0.025):.4f}, {post.ppf(0.975):.4f})")
assert np.isclose(post.mean(), 13 / 202)

# Numerical Bayes on a grid gives the same posterior
theta = np.linspace(1e-6, 1 - 1e-6, 20001)
unnorm = stats.beta(a0, b0).pdf(theta) * theta**k * (1 - theta) ** (n_obs - k)
grid_post = unnorm / trapezoid(unnorm, theta)
assert np.max(np.abs(grid_post - post.pdf(theta))) < 1e-3 * post.pdf(theta).max()

# A strong wrong prior vs growing data: the likelihood wins eventually
true_rate = 0.06
for n_i in (20, 200, 2000, 20000):
    k_i = rng.binomial(n_i, true_rate)
    stubborn = stats.beta(50 + k_i, 50 + n_i - k_i).mean()    # prior centred at 0.5, worth 100 pseudo-observations
    print(f"n={n_i:6d}: posterior mean with a 'it's 50%' prior = {stubborn:.4f}")
assert abs(stubborn - true_rate) < 0.01

# %% [markdown]
# ## 5. Chain rule of probability = autoregressive model
#
# A toy "language model" over the alphabet {a, b} with a first-order dependence. The probability of a sequence is the
# product of conditionals, and those probabilities must sum to 1 over all sequences of a given length.

# %%
from itertools import product

p_first = {"a": 0.7, "b": 0.3}
p_next = {"a": {"a": 0.2, "b": 0.8}, "b": {"a": 0.6, "b": 0.4}}


def seq_prob(s):
    p = p_first[s[0]]
    for prev, cur in zip(s, s[1:]):
        p *= p_next[prev][cur]
    return p


total = sum(seq_prob("".join(t)) for t in product("ab", repeat=6))
assert np.isclose(total, 1.0)
print(f"P('abab') = {seq_prob('abab'):.4f}; all 64 sequences of length 6 sum to {total:.6f}")

# %%
print("\nAll checks passed.")
