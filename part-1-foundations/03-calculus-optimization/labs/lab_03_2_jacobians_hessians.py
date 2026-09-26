# %% [markdown]
# # Lab 03.2: Jacobians, VJPs, Hessians and Newton
#
# 1. Softmax Jacobian: formula vs numerical, and the O(n) VJP.
# 2. The chain rule as Jacobian products; VJP chains without materializing Jacobians.
# 3. Classify critical points with Hessian eigenvalues.
# 4. Hessian-vector products by finite differences of gradients, and power iteration on them.
# 5. Newton vs gradient descent on an ill-conditioned quadratic and on logistic regression.

# %%
import numpy as np

rng = np.random.default_rng(0)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def numerical_jacobian(f, x, h=1e-6):
    fx = f(x)
    J = np.zeros((fx.size, x.size))
    for j in range(x.size):
        e = np.zeros_like(x); e[j] = h
        J[:, j] = (f(x + e) - f(x - e)) / (2 * h)
    return J

# %% [markdown]
# ## 1. Softmax Jacobian

# %%
z = rng.normal(size=6)
p = softmax(z)
J = np.diag(p) - np.outer(p, p)
assert np.allclose(J, numerical_jacobian(softmax, z), atol=1e-8)
assert np.allclose(J, J.T) and np.allclose(J @ np.ones(6), 0), "symmetric, and blind to adding a constant"
g = rng.normal(size=6)
vjp_fast = p * (g - p @ g)
assert np.allclose(vjp_fast, J.T @ g)
print("softmax Jacobian and O(n) VJP: ok")

# %% [markdown]
# ## 2. Chain rule = Jacobian product; backprop = chained VJPs
#
# f(x) = sum( softmax( W2 @ relu(W1 @ x) ) * c ) : a tiny network with a linear readout.

# %%
d_in, d_h, d_out = 4, 7, 5
W1, W2, c = rng.normal(size=(d_h, d_in)), rng.normal(size=(d_out, d_h)), rng.normal(size=d_out)
x = rng.normal(size=d_in)


def net(x):
    return np.array([softmax(W2 @ np.maximum(W1 @ x, 0)) @ c])


# Full Jacobians, multiplied (fine for tiny sizes, hopeless for big ones)
h1 = W1 @ x
a1 = np.maximum(h1, 0)
p = softmax(W2 @ a1)
J_relu = np.diag((h1 > 0).astype(float))
J_soft = np.diag(p) - np.outer(p, p)
J_full = c[None, :] @ J_soft @ W2 @ J_relu @ W1          # (1, d_in)

# VJPs, right to left, never building a matrix bigger than the weights we already have
g = c.copy()                                  # d f / d p
g = p * (g - p @ g)                           # through softmax
g = W2.T @ g                                  # through W2
g = g * (h1 > 0)                              # through ReLU: a mask
g = W1.T @ g                                  # through W1
assert np.allclose(J_full.ravel(), g)
assert np.allclose(g, numerical_jacobian(net, x).ravel(), atol=1e-7)
print("Jacobian product == chained VJPs == numerical: ok")

# %% [markdown]
# ## 3. Classify critical points of f(x, y) = x^3 - 3x + y^2

# %%
def hess(xy):
    return np.array([[6 * xy[0], 0.0], [0.0, 2.0]])


kinds = {}
for pt in ([1.0, 0.0], [-1.0, 0.0]):
    lam = np.linalg.eigvalsh(hess(pt))
    kinds[tuple(pt)] = "minimum" if np.all(lam > 0) else ("maximum" if np.all(lam < 0) else "saddle")
    print(pt, "eigenvalues", lam, "->", kinds[tuple(pt)])
assert kinds == {(1.0, 0.0): "minimum", (-1.0, 0.0): "saddle"}

# %% [markdown]
# ## 4. Hessian-vector products without the Hessian
#
# For logistic regression the Hessian is X^T D X / n with D = diag(p(1-p)). We compute Hv from gradients only, then
# find the top eigenvalue by power iteration on those products.

# %%
n, d = 400, 10
X = rng.normal(size=(n, d)) * np.linspace(0.2, 3.0, d)       # features with very different scales
w_true = rng.normal(size=d)
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
sig = lambda t: 1 / (1 + np.exp(-t))


def loss(w):
    z = X @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def grad(w):
    return X.T @ (sig(X @ w) - y) / n


def hessian(w):
    s = sig(X @ w)
    return X.T @ (X * (s * (1 - s))[:, None]) / n


def hvp(w, v, eps=1e-5):
    return (grad(w + eps * v) - grad(w - eps * v)) / (2 * eps)


w = rng.normal(size=d) * 0.1
v = rng.normal(size=d)
assert np.allclose(hvp(w, v), hessian(w) @ v, rtol=1e-6)

u = rng.normal(size=d)
for _ in range(300):
    u = hvp(w, u); u /= np.linalg.norm(u)
top = u @ hvp(w, u)
assert np.isclose(top, np.linalg.eigvalsh(hessian(w))[-1], rtol=1e-4)
print(f"top Hessian eigenvalue from HVPs only: {top:.4f}  -> max stable GD step ~ {2 / top:.3f}")

# %% [markdown]
# ## 5. Newton vs gradient descent

# %%
# (a) Ill-conditioned quadratic: Newton in one step, GD crawls.
Hq = np.diag([100.0, 1.0])
fq = lambda w: 0.5 * w @ Hq @ w
w_gd = np.array([1.0, 1.0])
steps_gd = 0
while fq(w_gd) > 1e-8:
    w_gd = w_gd - (1.9 / 100.0) * (Hq @ w_gd)        # near the largest stable step size
    steps_gd += 1
w_nt = np.array([1.0, 1.0]) - np.linalg.solve(Hq, Hq @ np.array([1.0, 1.0]))
assert np.allclose(w_nt, 0)
print(f"quadratic, condition number 100: Newton 1 step, GD {steps_gd} steps")
assert steps_gd > 200

# (b) Logistic regression: Newton converges quadratically.
w = np.zeros(d)
print("Newton on logistic regression, gradient norm per step:")
norms = []
for it in range(8):
    gnorm = np.linalg.norm(grad(w))
    norms.append(gnorm)
    print(f"  step {it}: |grad| = {gnorm:.2e}   loss = {loss(w):.6f}")
    if gnorm < 1e-12:
        break
    w = w - np.linalg.solve(hessian(w), grad(w))
assert norms[-1] < 1e-10, "Newton should nail this in a handful of steps"

from sklearn.linear_model import LogisticRegression
sk = LogisticRegression(C=1e12, fit_intercept=False, tol=1e-12, max_iter=10000).fit(X, y)
assert np.allclose(sk.coef_.ravel(), w, atol=1e-4), "same optimum as scikit-learn's L-BFGS"

# %%
print("\nAll checks passed.")
