# %% [markdown]
# # Lab 05.4: Causal inference on data where we know the truth
#
# We simulate worlds with a known causal effect, then see which estimators recover it.
#
# 1. Confounding: naive comparison vs regression adjustment vs IPW vs matching.
# 2. Collider bias: filtering the dataset creates a correlation from nothing.
# 3. Adjusting for a mediator destroys the effect you want to measure.
# 4. Difference-in-differences.
# 5. Instrumental variables with random encouragement (Wald estimator and 2SLS).

# %%
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(0)
n = 40_000

# %% [markdown]
# ## 1. The app and churn
#
# Loyalty (unobserved in reality, measured here as `tenure` and `orders`) drives both app installation and spend.
# The TRUE effect of the app on monthly spend is +5.

# %%
tenure = rng.gamma(2, 12, n)                      # months as a customer
orders = rng.poisson(1 + tenure / 10)
logit = -3 + 0.06 * tenure + 0.25 * orders
app = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
spend = 20 + 0.8 * tenure + 3 * orders + 5 * app + rng.normal(0, 10, n)
TRUE = 5.0
X = np.column_stack([tenure, orders])

naive = spend[app].mean() - spend[~app].mean()
reg = LinearRegression().fit(np.column_stack([app, X]), spend).coef_[0]

e = LogisticRegression(C=1e6, max_iter=1000).fit(X, app).predict_proba(X)[:, 1]
e = np.clip(e, 0.01, 0.99)
ipw = np.mean(app * spend / e - (~app) * spend / (1 - e))

# 1-nearest-neighbour matching on standardized covariates (effect on the treated)
Xs = (X - X.mean(0)) / X.std(0)
nn = NearestNeighbors(n_neighbors=1).fit(Xs[~app])
_, idx = nn.kneighbors(Xs[app])
att_match = np.mean(spend[app] - spend[~app][idx[:, 0]])

print(f"true effect {TRUE}   naive {naive:.2f}   regression {reg:.2f}   IPW {ipw:.2f}   matching (ATT) {att_match:.2f}")
assert naive > 3 * TRUE, "the naive comparison is badly confounded"
assert abs(reg - TRUE) < 0.4 and abs(ipw - TRUE) < 1.0 and abs(att_match - TRUE) < 1.0

# With the confounders NOT measured, adjustment can't help: that's the untestable assumption.
reg_partial = LinearRegression().fit(np.column_stack([app, orders]), spend).coef_[0]
print(f"adjusting only for orders (tenure unmeasured): {reg_partial:.2f}  <- still biased")
assert reg_partial > TRUE + 2

# %% [markdown]
# ## 2. Collider bias: selecting on a consequence
#
# Bugs and enterprise plan are independent. Both make users file tickets. Among ticket filers, they look negatively related.

# %%
bug = rng.uniform(size=n) < 0.2
enterprise = rng.uniform(size=n) < 0.3
ticket = rng.uniform(size=n) < 0.05 + 0.5 * bug + 0.4 * enterprise
corr_all = np.corrcoef(bug, enterprise)[0, 1]
corr_tickets = np.corrcoef(bug[ticket], enterprise[ticket])[0, 1]
print(f"corr(bug, enterprise): all users {corr_all:+.3f}   ticket filers only {corr_tickets:+.3f}")
assert abs(corr_all) < 0.02 and corr_tickets < -0.15

# %% [markdown]
# ## 3. Don't adjust for a mediator
#
# Discount -> items in cart -> revenue. The total effect of the discount on revenue is 2 (via items) + 1 (direct) = 3.

# %%
disc = rng.integers(0, 2, n)                       # randomized!
items = 3 + 1.0 * disc + rng.normal(0, 1, n)
revenue = 10 + 2.0 * items + 1.0 * disc + rng.normal(0, 2, n)
total = LinearRegression().fit(disc[:, None], revenue).coef_[0]
with_mediator = LinearRegression().fit(np.column_stack([disc, items]), revenue).coef_[0]
print(f"total effect {total:.2f} (truth 3)   'controlling' for items {with_mediator:.2f} (only the direct part)")
assert abs(total - 3) < 0.1 and abs(with_mediator - 1) < 0.1

# %% [markdown]
# ## 4. Difference-in-differences
#
# Weekly accident rates (per 100 vehicles, say) on two streets with different levels but parallel trends; the bike lane lowers accidents by 0.1/week.

# %%
weeks = np.arange(104)
after = weeks >= 52
trend = -0.003 * weeks                               # both streets improving over time
street_a = rng.poisson(np.clip(0.9 + trend - 0.1 * after, 0.01, None) * 100) / 100
street_b = rng.poisson(np.clip(1.1 + trend, 0.01, None) * 100) / 100
did = (street_a[after].mean() - street_a[~after].mean()) - (street_b[after].mean() - street_b[~after].mean())
before_after_only = street_a[after].mean() - street_a[~after].mean()
print(f"DiD {did:+.3f} (truth -0.100)   before/after on the treated street alone {before_after_only:+.3f}")
assert abs(did + 0.1) < 0.05 and before_after_only < -0.2, "the naive comparison mixes in the time trend"

# %% [markdown]
# ## 5. Instrumental variables: random encouragement
#
# Motivation (unobserved) drives both feature use and retention. A randomized nudge shifts feature use. True effect: +0.10.

# %%
motivation = rng.normal(size=n)
nudge = rng.integers(0, 2, n)
use = (0.8 * motivation + 1.0 * nudge + rng.normal(size=n)) > 0.8
retention = 0.3 + 0.10 * use + 0.15 * motivation + rng.normal(0, 0.2, n)

naive_iv = retention[use].mean() - retention[~use].mean()
wald = (retention[nudge == 1].mean() - retention[nudge == 0].mean()) / (use[nudge == 1].mean() - use[nudge == 0].mean())
# 2SLS: regress use on the nudge, then retention on the predicted use
use_hat = LinearRegression().fit(nudge[:, None], use).predict(nudge[:, None])
tsls = LinearRegression().fit(use_hat[:, None], retention).coef_[0]
print(f"naive {naive_iv:.3f}   Wald IV {wald:.3f}   2SLS {tsls:.3f}   (truth 0.100)")
assert naive_iv > 0.2 and abs(wald - 0.1) < 0.03 and np.isclose(wald, tsls)

# %%
print("\nAll checks passed.")
