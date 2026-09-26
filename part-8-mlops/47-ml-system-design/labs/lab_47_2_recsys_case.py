# %% [markdown]
# # Lab 47.2: A design case study: recommendations, offline evaluation and feedback loops
#
# A simulated catalog: 2,000 users, 300 items, true preferences from hidden user and item factors plus item
# popularity. Clicks depend on preference and on position (people click what's on top).
# 1. Evaluating a new ranker offline from the current system's logs: the naive estimate, the inverse-propensity
#    estimate, and the truth (run online, which the simulation lets us do).
# 2. Feedback loops: a recommender retrained on its own clicks, round after round. What happens to exposure?

# %%
import numpy as np

rs = np.random.default_rng(472)
U, I, D, SLOTS = 2000, 300, 8, 5
user_f = rs.normal(0, 1, (U, D))
item_f = rs.normal(0, 1, (I, D))
popularity = rs.normal(0, 1, I)
true_pref = user_f @ item_f.T / np.sqrt(D) + 0.8 * popularity                    # the hidden truth
POS_BIAS = 1 / np.log2(np.arange(SLOTS) + 2)                                     # click-through by position
sig = lambda z: 1 / (1 + np.exp(-z))


def p_click(users, items, pos):
    return sig(true_pref[users, items] - 2.0) * POS_BIAS[pos]


def softmax_rank(scores, temp, rng):
    """Sample a ranking of SLOTS items without replacement (Plackett-Luce); return items and their probabilities."""
    p = np.exp((scores - scores.max()) / temp); p /= p.sum()
    items = rng.choice(len(p), SLOTS, replace=False, p=p)
    return items, p


def online_ctr(score_fn, temp, n_users=2000, seed=0):
    rng = np.random.default_rng(seed)
    users = rng.integers(0, U, n_users)
    clicks = 0.0
    for u in users:
        items, _ = softmax_rank(score_fn(u), temp, rng)
        clicks += p_click(np.full(SLOTS, u), items, np.arange(SLOTS)).sum()
    return clicks / n_users


# %% [markdown]
# ## 1. Offline evaluation from logs

# %%
personal = true_pref + np.random.default_rng(1).normal(0, 0.8, true_pref.shape)    # a personalized candidate ranker
NEW_TEMP = 0.3
p_new = np.exp((personal - personal.max(1, keepdims=True)) / NEW_TEMP); p_new /= p_new.sum(1, keepdims=True)
new_top5 = np.argsort(-personal, axis=1)[:, :SLOTS]
truth_new = (p_new * sig(true_pref - 2.0) * POS_BIAS[0]).sum(1).mean()             # exact: the online truth


def offline_estimates(log_temp, sessions=20_000, seed=1):
    """Log the top slot of a popularity ranker with the given randomization; estimate the new ranker's top-slot CTR."""
    rng = np.random.default_rng(seed)
    p_log = np.exp((popularity - popularity.max()) / log_temp); p_log /= p_log.sum()
    users = rng.integers(0, U, sessions)
    items = rng.choice(I, sessions, p=p_log)
    clicked = rng.random(sessions) < p_click(users, items, np.zeros(sessions, int))
    match = (new_top5[users] == items[:, None]).any(1)
    w = p_new[users, items] / p_log[items]
    boot = rng.integers(0, sessions, (500, sessions))
    ips_b = (w[boot] * clicked[boot]).mean(1)
    return {"logging policy (truth)": (p_log[None, :] * sig(true_pref - 2.0) * POS_BIAS[0]).sum(1).mean(),
            "naive": clicked[match].mean(), "IPS": np.mean(w * clicked),
            "IPS 95% interval": tuple(np.percentile(ips_b, [2.5, 97.5])),
            "self-normalized IPS": np.sum(w * clicked) / np.sum(w), "max weight": w.max()}


print(f"the metric: click-through rate of the top slot. New personalized ranker, online truth: {truth_new:.3f}")
est = {}
for log_temp in (0.5, 2.0):
    e = est[log_temp] = offline_estimates(log_temp)
    print(f"\nlogger = popularity ranker, randomization temperature {log_temp} (its own CTR {e['logging policy (truth)']:.3f}):")
    print(f"  naive estimate (logged cases where it showed what the new ranker likes)  {e['naive']:.3f}")
    print(f"  IPS estimate                                   {e['IPS']:.3f}, 95% interval [{e['IPS 95% interval'][0]:.3f}, "
          f"{e['IPS 95% interval'][1]:.3f}]; largest weight {e['max weight']:.0f}")
    print(f"  self-normalized IPS                            {e['self-normalized IPS']:.3f}")
print("logs only contain what the old system chose to show. The naive estimate keeps the few logged cases where the old")
print("system happened to show something the new ranker likes (mostly popular items) and is biased by the old system's")
print("choices. Inverse propensity scoring reweights every logged outcome by how much more often the new policy would")
print("show it: unbiased if the logger recorded its probabilities and gave every item a chance, but with a logger that")
print("barely explores, the weights explode and so does the variance. Log propensities, and explore a little, or offline")
print("evaluation of anything new is guesswork. Offline estimates choose candidates; the online test (50.2) decides.")
lo, hi = est[2.0]["IPS 95% interval"]
assert lo <= truth_new <= hi and (hi - lo) < (est[0.5]["IPS 95% interval"][1] - est[0.5]["IPS 95% interval"][0])

# %% [markdown]
# ## 2. Feedback loops

# %%
def feedback_loop(explore, rounds=8, seed=3):
    """Each round: serve with the current click-rate estimates, collect clicks, retrain on everything so far."""
    rng = np.random.default_rng(seed)
    shows, clicks = np.ones(I), np.zeros(I) + 0.05                               # a weak, equal prior
    history = []
    for r in range(rounds):
        est = clicks / shows
        exposure = np.zeros(I)
        for u in rng.integers(0, U, 2000):
            if rng.random() < explore:
                items = rng.choice(I, SLOTS, replace=False)
            else:
                items = np.argsort(-est)[:SLOTS]                                  # same top items for everyone
            c = rng.random(SLOTS) < p_click(np.full(SLOTS, u), items, np.arange(SLOTS))
            shows[items] += 1; clicks[items] += c; exposure[items] += 1
        share = np.sort(exposure)[::-1]
        history.append((share[:10].sum() / share.sum(), (exposure > 0).sum()))
    best_true = np.argsort(-sig(true_pref - 2.0).mean(0))[:SLOTS]
    return history, len(set(np.argsort(-clicks / shows)[:SLOTS]) & set(best_true))


for explore in (0.0, 0.1):
    hist, overlap = feedback_loop(explore)
    print(f"\nexploration {explore:.0%}: share of impressions going to the top 10 items, by round: "
          + " ".join(f"{s:.2f}" for s, _ in hist))
    print(f"  items shown at least once in the last round: {hist[-1][1]} of {I}; the final top 5 includes "
          f"{overlap} of the 5 truly best items")
print("a system trained on its own outputs only learns about what it already shows: early winners get all the")
print("exposure, the rest never get a chance to prove themselves. A small amount of exploration (and logging its")
print("probabilities, for part 1) keeps the data informative. Every recommender, ranker and fraud model with")
print("human feedback has this loop; the design doc should say how it's broken.")

print("\nAll checks passed.")
