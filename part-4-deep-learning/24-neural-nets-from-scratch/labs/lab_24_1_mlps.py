# %% [markdown]
# # Lab 24.1: What an MLP is, and what it can represent
#
# NumPy only; PyTorch arrives in Module 25.
#
# 1. Five linear layers are one linear layer. XOR stays unsolvable.
# 2. Activations and saturation: how much gradient survives 10 sigmoid layers vs 10 ReLU layers.
# 3. Universal approximation by hand: a ReLU network that interpolates sin(x), no training. Error vs width.
# 4. Depth vs width: a sawtooth with 2^k pieces from 2k units in k layers.
# 5. Counting the linear pieces of random ReLU networks along a line.
# 6. Training MLPs of different shapes on two interleaved spirals (hand-written backprop, previewing 24.2).

# %%
import numpy as np

rng = np.random.default_rng(241)
relu = lambda z: np.maximum(z, 0)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# %% [markdown]
# ## 1. Linear layers collapse

# %%
dims = [2, 16, 16, 16, 16, 1]
Ws = [rng.normal(size=(dims[i + 1], dims[i])) for i in range(5)]
bs = [rng.normal(size=dims[i + 1]) for i in range(5)]
X = rng.normal(size=(1000, 2))
h = X.T
for W, b in zip(Ws, bs):
    h = W @ h + b[:, None]                                      # no activation
W_tot, b_tot = np.eye(2), np.zeros(2)
for W, b in zip(Ws, bs):
    W_tot, b_tot = W @ W_tot, W @ b_tot + b
assert np.allclose(h, W_tot @ X.T + b_tot[:, None])
print(f"a 5-layer linear network with {sum(W.size + b.size for W, b in zip(Ws, bs))} parameters equals one linear map with {W_tot.size + b_tot.size}")

Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); yx = np.array([0, 1, 1, 0])
# any linear function of (x1, x2) thresholded: brute-force check over a grid of weights finds no separator
best = max(np.mean(((Xx @ np.array([a, b]) + c) > 0) == yx) for a in np.linspace(-3, 3, 31) for b in np.linspace(-3, 3, 31) for c in np.linspace(-3, 3, 31))
print(f"best accuracy of any linear threshold on XOR (grid search): {best:.2f}")
assert best == 0.75

# %% [markdown]
# ## 2. Saturation and vanishing gradients

# %%
def grad_norm_at_input(act, dact, depth=10, width=64, init_scale=1.0):
    x = rng.normal(size=(width, 256))
    Ws_, zs = [], []
    h_ = x
    for _ in range(depth):
        W = rng.normal(size=(width, width)) * init_scale / np.sqrt(width)
        z = W @ h_; Ws_.append(W); zs.append(z); h_ = act(z)
    g = np.ones_like(h_)                                         # d(sum of outputs)/d(output)
    for W, z in zip(reversed(Ws_), reversed(zs)):
        g = W.T @ (g * dact(z))
    return np.linalg.norm(g) / np.sqrt(g.size)


for name, act, dact, scale in [("sigmoid", sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z)), 1.0),
                               ("tanh", np.tanh, lambda z: 1 - np.tanh(z) ** 2, 1.0),
                               ("ReLU", relu, lambda z: (z > 0).astype(float), np.sqrt(2))]:
    print(f"{name:8s}: gradient size at the input after 10 layers {grad_norm_at_input(act, dact, init_scale=scale):.2e}")
g_sig = grad_norm_at_input(sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z)))
g_relu = grad_norm_at_input(relu, lambda z: (z > 0).astype(float), init_scale=np.sqrt(2))
assert g_sig < 1e-3 * g_relu
print("(ReLU with the He scale sqrt(2/fan_in) keeps the gradient's size; 26.2 derives why)")

# %% [markdown]
# ## 3. Universal approximation by hand
#
# f(x) = f(x0) + sum_k (slope_k - slope_{k-1}) * ReLU(x - x_k): one hidden unit per knot, weights written down, not learned.

# %%
def interpolating_relu_net(f, a, b, n_knots):
    knots = np.linspace(a, b, n_knots + 1)
    slopes = np.diff(f(knots)) / np.diff(knots)
    w_out = np.r_[slopes[0], np.diff(slopes)]                    # change of slope at each knot
    bias_hidden = -knots[:-1]                                     # unit k: ReLU(x - x_k)
    return lambda x: f(knots[0]) + relu(x[:, None] + bias_hidden[None, :]) @ w_out


xs = np.linspace(0, 2 * np.pi, 5000)
errs = {}
for n in (4, 8, 16, 32, 64):
    net = interpolating_relu_net(np.sin, 0, 2 * np.pi, n)
    errs[n] = np.abs(net(xs) - np.sin(xs)).max()
    print(f"{n:3d} hidden units: max error {errs[n]:.5f}")
print(f"doubling the width divides the error by about {errs[32] / errs[64]:.2f} (theory for smooth functions: 4)")
assert 3.5 < errs[32] / errs[64] < 4.5

# %% [markdown]
# ## 4. Depth vs width: the sawtooth

# %%
tent = lambda x: 2 * relu(x) - 4 * relu(x - 0.5)                # two ReLU units, maps [0,1] onto [0,1]


def count_pieces(y, x):
    slopes = np.round(np.diff(y) / np.diff(x), 6)
    return 1 + int(np.sum(np.abs(np.diff(slopes)) > 1e-3))


x01 = np.linspace(0, 1, 2**14 + 1)
for k in (1, 3, 6, 10):
    y = x01.copy()
    for _ in range(k):
        y = tent(y)
    print(f"{k:2d} layers x 2 ReLUs = {2 * k:2d} units: {count_pieces(y, x01):5d} linear pieces "
          f"(a one-hidden-layer net would need >= {2**k - 1} units)")
    if k == 10:
        assert count_pieces(y, x01) == 2**10

# %% [markdown]
# ## 5. Linear pieces of random ReLU networks along a line

# %%
def pieces_along_line(widths, n_points=200_001):
    t = np.linspace(-3, 3, n_points)
    direction = rng.normal(size=2); direction /= np.linalg.norm(direction)
    h_ = (t[:, None] * direction[None, :]).T
    pattern = []
    for i, w in enumerate(widths):
        W = rng.normal(size=(w, h_.shape[0])) * np.sqrt(2 / h_.shape[0]); b = rng.normal(size=(w, 1)) * 0.5
        z = W @ h_ + b
        pattern.append(z > 0)
        h_ = relu(z)
    act = np.vstack(pattern)
    return 1 + int(np.sum(np.any(act[:, 1:] != act[:, :-1], axis=0)))     # a new piece whenever any unit flips


for widths in ([48], [24, 24], [16, 16, 16], [12, 12, 12, 12]):
    counts = [pieces_along_line(widths) for _ in range(10)]
    print(f"{str(widths):18s} ({sum(widths)} units): {np.mean(counts):5.1f} pieces along a line (mean of 10 random nets)")
print("(for *random* networks depth adds fewer pieces than the worst case suggests; trained networks can exploit it better)")

# %% [markdown]
# ## 6. Training MLPs on two spirals

# %%
def spirals(n):
    t = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    cls = rng.integers(0, 2, n)
    r = t / (3 * np.pi)
    x = np.column_stack([r * np.cos(t + np.pi * cls), r * np.sin(t + np.pi * cls)]) + rng.normal(0, 0.02, (n, 2))
    return x, cls


def train_mlp(X, y, widths, epochs=3000, lr=0.05, seed=0):
    r = np.random.default_rng(seed)
    sizes = [2, *widths, 2]
    params = [(r.normal(size=(a, b)) * np.sqrt(2 / a), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]
    Y = np.eye(2)[y]
    vel = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    for _ in range(epochs):
        acts, pre = [X], []
        for i, (W, b) in enumerate(params):
            z = acts[-1] @ W + b; pre.append(z)
            acts.append(relu(z) if i < len(params) - 1 else z)
        Z = acts[-1] - acts[-1].max(1, keepdims=True)
        P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
        g = (P - Y) / len(X)                                     # softmax + cross-entropy gradient (17.4)
        for i in reversed(range(len(params))):
            W, b = params[i]
            gW, gb = acts[i].T @ g, g.sum(0)
            if i > 0:
                g = (g @ W.T) * (pre[i - 1] > 0)
            vW, vb = vel[i]
            vW[:] = 0.9 * vW - lr * gW; vb[:] = 0.9 * vb - lr * gb
            params[i] = (W + vW, b + vb)

    def predict(Xn):
        h_ = Xn
        for i, (W, b) in enumerate(params):
            h_ = h_ @ W + b
            if i < len(params) - 1:
                h_ = relu(h_)
        return h_.argmax(1)
    return predict, sum(W.size + b.size for W, b in params)


Xs, ys = spirals(1500)
Xs_te, ys_te = spirals(3000)
acc = {}
for widths in ([], [8], [64], [16, 16], [32, 32, 32]):
    predict, n_params = train_mlp(Xs, ys, widths)
    acc[str(widths)] = np.mean(predict(Xs_te) == ys_te)
    print(f"hidden layers {str(widths):13s} ({n_params:5d} params): test accuracy {acc[str(widths)]:.3f}")
assert acc["[]"] < 0.7 and acc["[32, 32, 32]"] > 0.95

print("\nAll checks passed.")
