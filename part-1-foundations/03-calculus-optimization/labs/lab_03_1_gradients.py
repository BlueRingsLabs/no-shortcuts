# %% [markdown]
# # Lab 03.1: Gradients you can trust
#
# 1. A reusable numerical gradient checker (central differences, relative error).
# 2. Check the identity table from the lesson: linear, quadratic, least squares, log-sum-exp, softmax + CE.
# 3. The worked neuron example: manual forward and backward pass, checked numerically.
# 4. The step-size trade-off: why h = 1e-6 and not 1e-12.

# %%
import numpy as np

rng = np.random.default_rng(0)


def numerical_grad(f, x, h=1e-6):
    x = x.astype(float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e.flat[i] = h
        g.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def rel_err(a, b, eps=1e-12):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), eps)


def check(name, f, grad, x, tol=1e-7):
    err = rel_err(grad(x), numerical_grad(f, x))
    print(f"{name:32s} rel err {err:.1e}")
    assert err < tol, f"{name}: gradient looks wrong"

# %% [markdown]
# ## 1-2. The identity table

# %%
d, n = 5, 20
a = rng.normal(size=d)
A = rng.normal(size=(d, d))
S = A + A.T
X, y = rng.normal(size=(n, d)), rng.normal(size=n)
w0 = rng.normal(size=d)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def logsumexp(z):
    m = z.max()
    return m + np.log(np.sum(np.exp(z - m)))


t = np.zeros(d); t[2] = 1.0                                  # one-hot target

check("a^T w", lambda w: a @ w, lambda w: a, w0)
check("w^T A w (general A)", lambda w: w @ A @ w, lambda w: (A + A.T) @ w, w0)
check("w^T S w (symmetric S)", lambda w: w @ S @ w, lambda w: 2 * S @ w, w0)
check("||w||^2", lambda w: w @ w, lambda w: 2 * w, w0)
check("||Xw - y||^2", lambda w: np.sum((X @ w - y) ** 2), lambda w: 2 * X.T @ (X @ w - y), w0)
check("ridge objective", lambda w: 0.5 * np.sum((X @ w - y) ** 2) + 0.5 * 0.3 * w @ w,
      lambda w: X.T @ (X @ w - y) + 0.3 * w, w0)
check("logsumexp", logsumexp, softmax, w0)
check("cross-entropy(softmax(z), t)", lambda z: -np.sum(t * np.log(softmax(z))), lambda z: softmax(z) - t, w0)

# And a wrong gradient must fail the check (a check that can't fail is worthless)
try:
    check("||Xw - y||^2, forgot the 2", lambda w: np.sum((X @ w - y) ** 2), lambda w: X.T @ (X @ w - y), w0)
    raise RuntimeError("the checker should have caught the missing factor")
except AssertionError:
    print("   -> caught the bug, as it should")

# Ridge closed form: gradient is zero there
lam = 0.3
w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
assert np.allclose(X.T @ (X @ w_ridge - y) + lam * w_ridge, 0, atol=1e-10)

# %% [markdown]
# ## 3. The neuron from the lesson, by hand

# %%
sigma = lambda z: 1 / (1 + np.exp(-z))


def neuron_loss(params, x=2.0, y=1.0):
    w, b = params
    return (sigma(w * x + b) - y) ** 2


def neuron_backward(params, x=2.0, y=1.0):
    w, b = params
    z = w * x + b                    # forward
    a_ = sigma(z)
    dL_da = 2 * (a_ - y)             # backward, right to left
    dL_dz = dL_da * a_ * (1 - a_)
    return np.array([dL_dz * x, dL_dz * 1.0])


g = neuron_backward(np.array([0.5, -1.0]))
assert np.allclose(g, [-0.5, -0.25]), g                       # the numbers worked out in the lesson
check("neuron (w=0.5, b=-1)", neuron_loss, neuron_backward, np.array([0.5, -1.0]))
check("neuron (random params)", neuron_loss, neuron_backward, rng.normal(size=2))

g_sat = neuron_backward(np.array([3.0, 0.0]), y=0.0)          # saturated AND wrong
print(f"saturated sigmoid, target 0: gradient {g_sat}  <- tiny despite a terrible prediction")
assert np.abs(g_sat).max() < 0.02

# %% [markdown]
# ## 4. Choosing h
#
# Truncation error shrinks like h^2, rounding error grows like eps/h. The sweet spot for central differences in
# float64 is around h ~ 1e-5 .. 1e-6.

# %%
f = lambda x: np.exp(np.sin(x[0]))
true = np.array([np.cos(1.0) * np.exp(np.sin(1.0))])
errors = {}
for h in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13):
    errors[h] = rel_err(numerical_grad(f, np.array([1.0]), h=h), true)
    print(f"h={h:.0e}  rel err {errors[h]:.1e}")
best_h = min(errors, key=errors.get)
assert 1e-7 <= best_h <= 1e-3, "optimum should be in the middle, not at the extremes"
assert errors[1e-13] > errors[best_h] * 100 and errors[1e-1] > errors[best_h] * 100

# %%
print("\nAll checks passed.")
