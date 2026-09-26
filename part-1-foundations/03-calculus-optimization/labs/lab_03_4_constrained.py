# %% [markdown]
# # Lab 03.4: Constraints, multipliers and projections
#
# 1. Lagrange by hand vs scipy's SLSQP on the textbook examples; the multiplier as a shadow price.
# 2. PCA as a constrained problem: projected gradient ascent on the unit sphere finds the top eigenvector.
# 3. Maximum entropy with a mean constraint gives a softmax (the loaded die).
# 4. KKT on a small quadratic program: active constraints, complementary slackness.

# %%
import numpy as np
from scipy.optimize import brentq, minimize

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Textbook examples

# %%
# min x^2 + y^2  s.t.  x + 2y = 5  -> (1, 2), lambda = 2
res = minimize(lambda v: v @ v, x0=[0.0, 0.0], method="SLSQP",
               constraints=[{"type": "eq", "fun": lambda v: v[0] + 2 * v[1] - 5}])
assert np.allclose(res.x, [1, 2], atol=1e-6)

# max xy s.t. x + y = c : optimum c^2/4, multiplier c/2. Shadow price check by finite difference.
def best_xy(c):
    r = minimize(lambda v: -v[0] * v[1], x0=[1.0, 1.0], method="SLSQP",
                 constraints=[{"type": "eq", "fun": lambda v: v[0] + v[1] - c}])
    return -r.fun


lam = 5.0
shadow = (best_xy(10.1) - best_xy(9.9)) / 0.2
print(f"d(optimum)/d(c) at c=10: {shadow:.4f}   multiplier: {lam}")
assert np.isclose(shadow, lam, rtol=1e-4)

# %% [markdown]
# ## 2. PCA by projected gradient ascent on the sphere
#
# max w^T C w  s.t. ||w|| = 1. Gradient step, then project back (normalize). The Lagrange condition says the answer
# is the top eigenvector and the multiplier is its eigenvalue.

# %%
A = rng.normal(size=(5, 5))
C = A @ A.T                                    # a covariance-like PSD matrix
w = rng.normal(size=5)
w /= np.linalg.norm(w)
eta = 0.5 / np.linalg.norm(C, 2)
for _ in range(5000):
    w = w + eta * 2 * C @ w                    # ascent step on w^T C w
    w /= np.linalg.norm(w)                     # projection onto the unit sphere
lam_all, Q = np.linalg.eigh(C)
assert np.isclose(abs(w @ Q[:, -1]), 1.0, atol=1e-6), "converged to the top eigenvector"
multiplier = w @ C @ w                          # from 2Cw = 2 lambda w
assert np.isclose(multiplier, lam_all[-1])
assert np.allclose(C @ w, multiplier * w, atol=1e-6)
print(f"projected ascent found variance {multiplier:.4f}; top eigenvalue {lam_all[-1]:.4f}")

# %% [markdown]
# ## 3. The loaded die: maximum entropy with mean 4.5

# %%
faces = np.arange(1, 7)


def gibbs(beta):
    e = np.exp(beta * faces)
    return e / e.sum()


beta = brentq(lambda b: gibbs(b) @ faces - 4.5, -5, 5)
p = gibbs(beta)
print(f"beta = {beta:.4f}\np = {np.round(p, 3)}  (mean {p @ faces:.3f})")
assert np.isclose(p @ faces, 4.5)

# Compare with a generic constrained solver maximizing entropy directly
neg_entropy = lambda q: np.sum(q * np.log(np.clip(q, 1e-12, None)))
res = minimize(neg_entropy, x0=np.full(6, 1 / 6), method="SLSQP", bounds=[(1e-9, 1)] * 6,
               constraints=[{"type": "eq", "fun": lambda q: q.sum() - 1},
                            {"type": "eq", "fun": lambda q: q @ faces - 4.5}], options={"ftol": 1e-12})
assert np.allclose(res.x, p, atol=1e-4), "the generic solver agrees with the softmax form"

# Any other distribution with mean 4.5 has lower entropy
H = lambda q: -np.sum(q * np.log(q))
for _ in range(200):
    q = rng.dirichlet(np.ones(6))
    # nudge q to have mean 4.5 by mixing with a point mass, if possible, then compare entropies
    target = 4.5
    m = q @ faces
    point = np.zeros(6); point[5 if m < target else 0] = 1
    t = (target - m) / (point @ faces - m)
    q2 = (1 - t) * q + t * point
    if 0 <= t <= 1 and np.all(q2 > 0):
        assert H(q2) <= H(p) + 1e-9
print("max-entropy check against 200 random alternatives: ok")

# %% [markdown]
# ## 4. KKT on a tiny quadratic program
#
# min (x-2)^2 + (y-1)^2  s.t.  x + y <= 2,  x >= 0,  y >= 0
# The unconstrained optimum (2, 1) violates x + y <= 2, so that constraint is active; the others are not.

# %%
f = lambda v: (v[0] - 2) ** 2 + (v[1] - 1) ** 2
cons = [{"type": "ineq", "fun": lambda v: 2 - v[0] - v[1]},     # scipy's convention: fun(v) >= 0
        {"type": "ineq", "fun": lambda v: v[0]},
        {"type": "ineq", "fun": lambda v: v[1]}]
res = minimize(f, x0=[0.0, 0.0], method="SLSQP", constraints=cons)
x_star = res.x
print("x* =", np.round(x_star, 6))
assert np.allclose(x_star, [1.5, 0.5], atol=1e-6)

# Recover multipliers from stationarity: grad f + mu * grad h = 0 with h = x + y - 2 (active), others inactive.
grad_f = 2 * (x_star - np.array([2.0, 1.0]))
grad_h = np.array([1.0, 1.0])
mu = -(grad_f @ grad_h) / (grad_h @ grad_h)
assert mu > 0, "dual feasibility"
assert np.allclose(grad_f + mu * grad_h, 0, atol=1e-6), "stationarity"
assert abs(mu * (x_star.sum() - 2)) < 1e-8, "complementary slackness (active constraint)"
print(f"mu for x + y <= 2: {mu:.4f}; multipliers for x >= 0, y >= 0: 0 (inactive)")

# %%
print("\nAll checks passed.")
