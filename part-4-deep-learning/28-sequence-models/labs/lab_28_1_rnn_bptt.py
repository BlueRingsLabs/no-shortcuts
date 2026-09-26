# %% [markdown]
# # Lab 28.1: Recurrent networks and backprop through time
#
# 1. An RNN cell by hand, matched against nn.RNN.
# 2. BPTT by hand in NumPy, matched against autograd.
# 3. The Jacobian product: how ||dh_T/dh_t|| scales with the distance T - t.
# 4. The silent failure: a task whose answer depends only on the first input, with a growing delay.
# 5. Truncated BPTT: a window shorter than the dependency gives no training signal for it.
# 6. A character-level language model trained on this course, with clipping, perplexity and temperature.

# %%
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(281)
torch.set_num_threads(4)
rng = np.random.default_rng(281)

# %% [markdown]
# ## 1. The cell

# %%
T, B, D, H = 7, 3, 4, 5
rnn = nn.RNN(D, H, batch_first=True)
x = torch.randn(B, T, D)
out_ref, h_ref = rnn(x)
Wxh, Whh, bxh, bhh = rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0
h, outs = torch.zeros(B, H), []
for t in range(T):
    h = torch.tanh(x[:, t] @ Wxh.T + bxh + h @ Whh.T + bhh)          # PyTorch keeps two biases; their sum is the b of the lesson
    outs.append(h)
assert torch.allclose(torch.stack(outs, 1), out_ref, atol=1e-6) and torch.allclose(h, h_ref[0], atol=1e-6)
print(f"hand-written loop matches nn.RNN; parameters: {sum(p.numel() for p in rnn.parameters())}, for any sequence length")

# %% [markdown]
# ## 2. BPTT by hand
#
# Many-to-many: y_t = W_hy h_t + c, squared error at every step. The backward pass walks the sequence in reverse,
# carrying dL/dh from the future (through W_hh) and adding the local error at each step.

# %%
def bptt_numpy(x, y, Wxh, Whh, b, Why, c):
    T = len(x)
    hs = [np.zeros(Whh.shape[0])]
    for t in range(T):
        hs.append(np.tanh(Wxh @ x[t] + Whh @ hs[-1] + b))
    preds = [Why @ hs[t + 1] + c for t in range(T)]
    loss = 0.5 * sum(((p - yt) ** 2).sum() for p, yt in zip(preds, y))
    g = {k: np.zeros_like(v) for k, v in dict(Wxh=Wxh, Whh=Whh, b=b, Why=Why, c=c).items()}
    dh_next = np.zeros(Whh.shape[0])                              # dL/dh_t arriving from step t+1
    for t in reversed(range(T)):
        dy = preds[t] - y[t]
        g["Why"] += np.outer(dy, hs[t + 1]); g["c"] += dy
        dh = Why.T @ dy + dh_next                                   # local error + error from the future
        dz = (1 - hs[t + 1] ** 2) * dh                             # through tanh
        g["Wxh"] += np.outer(dz, x[t]); g["Whh"] += np.outer(dz, hs[t]); g["b"] += dz
        dh_next = Whh.T @ dz                                       # through the recurrence, to step t-1
    return loss, g


T, D, H, O = 12, 3, 6, 2
P = dict(Wxh=rng.normal(0, 0.5, (H, D)), Whh=rng.normal(0, 0.4, (H, H)), b=rng.normal(0, 0.1, H),
         Why=rng.normal(0, 0.5, (O, H)), c=rng.normal(0, 0.1, O))
xs, ys = rng.normal(size=(T, D)), rng.normal(size=(T, O))
loss_np, g_np = bptt_numpy(xs, ys, **P)

Pt = {k: torch.tensor(v, requires_grad=True) for k, v in P.items()}
h = torch.zeros(H, dtype=torch.float64)
loss_t = 0
for t in range(T):
    h = torch.tanh(Pt["Wxh"] @ torch.tensor(xs[t]) + Pt["Whh"] @ h + Pt["b"])
    loss_t = loss_t + 0.5 * ((Pt["Why"] @ h + Pt["c"] - torch.tensor(ys[t])) ** 2).sum()
loss_t.backward()
err = max(np.abs(g_np[k] - Pt[k].grad.numpy()).max() for k in P)
print(f"\nBPTT by hand vs autograd: loss {loss_np:.6f} vs {loss_t.item():.6f}, max gradient difference {err:.1e}")
assert abs(loss_np - loss_t.item()) < 1e-10 and err < 1e-10

# %% [markdown]
# ## 3. The product of Jacobians
#
# W_hh has i.i.d. entries with standard deviation g / sqrt(H), so its spectral radius is about g. Run a tanh RNN on
# random inputs and measure the norm of dh_T / dh_t (the spectral norm of the product of per-step Jacobians) as the
# distance grows. A linear RNN (no tanh) is shown for comparison: there the product is just W_hh^(T-t).

# %%
H, T = 64, 120
print("\n||dh_T/dh_t|| (largest singular value) at distance T - t:")
print("   g   model     d=1        d=10       d=40       d=100")
slopes = {}
for g in (0.8, 1.0, 1.5, 3.0):
    W = torch.tensor(rng.normal(0, g / np.sqrt(H), (H, H)), dtype=torch.float64)
    U = torch.tensor(rng.normal(0, 1 / np.sqrt(4), (H, 4)), dtype=torch.float64)
    xs = torch.tensor(rng.normal(size=(T, 4)), dtype=torch.float64)
    for model in ("tanh", "linear"):
        h, J, norms = torch.zeros(H, dtype=torch.float64), torch.eye(H, dtype=torch.float64), []
        hs = []
        for t in range(T):
            z = W @ h + U @ xs[t]
            h = torch.tanh(z) if model == "tanh" else z
            hs.append(h)
        # dh_T/dh_{T-d} = J_T J_{T-1} ... J_{T-d+1}, with J_i = diag(tanh'(z_i)) W
        for d in range(1, 101):
            i = T - d
            Ji = ((1 - hs[i] ** 2)[:, None] * W) if model == "tanh" else W
            J = J @ Ji
            norms.append(torch.linalg.matrix_norm(J, ord=2).item())
        norms = np.array(norms)
        slopes[(g, model)] = np.polyfit(np.arange(20, 101), np.log(norms[19:]), 1)[0]
        print(f"{g:4.1f}   {model:7s} " + "  ".join(f"{norms[d - 1]:9.2e}" for d in (1, 10, 40, 100)))
print("linear: the product grows or shrinks like g^d, as the algebra says. tanh: every derivative is at most 1 and the")
print("saturated units contribute nearly 0, so g = 1 and even g = 1.5 still vanish. At g = 3 the dynamics turn chaotic and")
print("the product explodes anyway, just far more slowly than the linear one. A spectral radius above 1 is necessary")
print("for explosion, not sufficient.")
assert slopes[(0.8, "tanh")] < -0.1 and slopes[(1.0, "tanh")] < -0.1 and slopes[(1.5, "tanh")] < -0.05
assert slopes[(1.5, "linear")] > 0.2
assert slopes[(3.0, "tanh")] > 0.1 and slopes[(3.0, "tanh")] < slopes[(3.0, "linear")] - 0.5

# %% [markdown]
# ## 4. The silent failure
#
# Sequence of length L: x_1 is +1 or -1, the label is its sign; every later input is Gaussian noise. The network must
# carry one bit for L steps. Chance is 50%.

# %%
def delay_task(n, L, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, L, 1, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    x[:, 0, 0] = y * 2.0 - 1
    return x, y


class Classifier(nn.Module):
    def __init__(self, cell="rnn", h=32):
        super().__init__()
        self.rnn = (nn.RNN if cell == "rnn" else nn.GRU)(1, h, batch_first=True)
        self.out = nn.Linear(h, 2)

    def forward(self, x):
        return self.out(self.rnn(x)[1][0])                        # the final hidden state


def train_delay(L, cell="rnn", steps=600, seed=0):
    torch.manual_seed(seed)
    m = Classifier(cell)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    X, y = delay_task(4096, L, seed=1)
    losses = []
    for s in range(steps):
        i = torch.randint(0, len(X), (64,))
        loss = F.cross_entropy(m(X[i]), y[i])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); losses.append(loss.item())
    Xt, yt = delay_task(2000, L, seed=2)
    with torch.no_grad():
        return (m(Xt).argmax(1) == yt).float().mean().item()


t0 = time.time()
print("\nvanilla RNN, accuracy on a held-out set by delay:")
acc_delay = {}
for L in (5, 20, 50, 100):
    acc_delay[L] = train_delay(L)
    print(f"  L = {L:3d}: {acc_delay[L]:.3f}")
print(f"({time.time() - t0:.0f}s) the network is the same, the optimizer is the same, only the distance changed.")
assert acc_delay[5] > 0.95
assert acc_delay[100] < 0.7

# %% [markdown]
# ## 5. Truncated BPTT
#
# One long stream, processed in chunks with the hidden state carried across chunk boundaries. Input: random bits;
# target at step t: the bit from step t - 8. With chunks of 4 steps, every target's cause lies in an earlier chunk, so
# its gradient is always cut. With chunks of 32, most causes are inside the chunk.

# %%
DELAY = 8


def train_stream(window, steps=800, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rnn, head = nn.GRU(1, 32, batch_first=True), nn.Linear(32, 1)  # a gated cell (28.2), so the cell isn't the bottleneck
    opt = torch.optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=3e-3)
    bits = torch.randint(0, 2, (32, steps * window + DELAY), generator=g).float()
    h = torch.zeros(1, 32, 32)
    for s in range(steps):
        a = DELAY + s * window
        xw = bits[:, a:a + window, None]
        yw = bits[:, a - DELAY:a - DELAY + window]
        out, h = rnn(xw, h)
        loss = F.binary_cross_entropy_with_logits(head(out)[..., 0], yw)
        opt.zero_grad(); loss.backward(); opt.step()
        h = h.detach()                                             # the truncation: keep the state, cut the gradient
    test = torch.randint(0, 2, (256, 400), generator=g).float()
    with torch.no_grad():
        pred = head(rnn(test[..., None])[0])[..., 0] > 0
    return (pred[:, DELAY:] == test[:, :-DELAY].bool()).float().mean().item()


t0 = time.time()
acc_short, acc_long = train_stream(4, steps=4800), train_stream(32, steps=600)
print(f"\nrecall the bit from {DELAY} steps ago, same number of time steps seen in training:")
print(f"  chunks of  4 steps: {acc_short:.3f}   chunks of 32 steps: {acc_long:.3f}   ({time.time() - t0:.0f}s)")
print("with the short window the state could carry the bit; nothing ever told the model to keep it.")
assert acc_short < 0.6 and acc_long > 0.9

# %% [markdown]
# ## 6. A character-level language model, trained on this course
#
# The lessons of Parts I to III. A vanilla tanh RNN, teacher forcing, truncated BPTT over 100-character windows taken
# at random positions (state reset per window: simpler, and fine for a demo), gradient clipping.

# %%
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "part-5-language-models" / "_shared"))
from corpus import lesson_files, read_lesson  # noqa: E402  (the frozen course text, tools/snapshot_course.py)
files = lesson_files(("part-1-foundations", "part-2-data-engineering", "part-3-classical-ml"))
text = "\n".join(read_lesson(p) for p in files)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
split = int(0.95 * len(data))
train_d, val_d = data[:split], data[split:]
V = len(chars)
print(f"\ncorpus: {len(files)} lessons, {len(text):,} characters, {V} distinct")


class CharRNN(nn.Module):
    def __init__(self, V, h=256):
        super().__init__()
        self.emb = nn.Embedding(V, 64)
        self.rnn = nn.RNN(64, h, batch_first=True)
        self.out = nn.Linear(h, V)

    def forward(self, x, h=None):
        o, h = self.rnn(self.emb(x), h)
        return self.out(o), h


def batch(d, bs=64, L=100, g=None):
    i = torch.randint(0, len(d) - L - 1, (bs,), generator=g)
    return torch.stack([d[j:j + L] for j in i]), torch.stack([d[j + 1:j + L + 1] for j in i])


@torch.no_grad()
def val_loss(m, n=20):
    g = torch.Generator().manual_seed(0)
    m.eval()
    ls = [F.cross_entropy(m(x)[0].reshape(-1, V), y.reshape(-1)).item() for x, y in (batch(val_d, g=g) for _ in range(n))]
    m.train()
    return float(np.mean(ls))


torch.manual_seed(0)
lm = CharRNN(V)
opt = torch.optim.Adam(lm.parameters(), lr=2e-3)
t0, gnorms = time.time(), []
for step in range(1500):
    x, y = batch(train_d)
    loss = F.cross_entropy(lm(x)[0].reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    gnorms.append(torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0).item())   # returns the norm before clipping
    opt.step()
    if step % 500 == 0 or step == 1499:
        print(f"  step {step:4d}  train loss {loss.item():.3f}  val loss {val_loss(lm):.3f}  ({time.time() - t0:.0f}s)")

counts = torch.bincount(train_d, minlength=V).double() + 1
unigram_ce = -(counts / counts.sum()).log()[val_d].mean().item()
ce = val_loss(lm)
print(f"validation perplexity: uniform {V}, unigram frequencies {np.exp(unigram_ce):.1f}, char-RNN {np.exp(ce):.1f}")
print(f"gradient norm before clipping: median {np.median(gnorms):.2f}, max {np.max(gnorms):.1f}, clipped on {np.mean(np.array(gnorms) > 1):.0%} of steps")
print("clipping rarely fires on this run. It's insurance: one bad batch at step 30,000 is enough to need it.")
assert np.exp(ce) < np.exp(unigram_ce) / 3


@torch.no_grad()
def sample(m, prompt, n=300, temperature=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    m.eval()
    x = torch.tensor([[stoi[c] for c in prompt]])
    logits, h = m(x)
    out = list(prompt)
    for _ in range(n):
        last = logits[0, -1]
        nxt = last.argmax() if temperature == 0 else torch.multinomial((last / temperature).softmax(0), 1, generator=g)[0]
        out.append(chars[nxt])
        logits, h = m(nxt.view(1, 1), h)
    return "".join(out)


for temp in (0, 0.5, 1.0, 1.5):
    s = sample(lm, "## Things that ", temperature=temp)
    print(f"\n--- temperature {temp} ---\n{s}")
greedy = sample(lm, "## Things that ", temperature=0)
tail = greedy[-150:]
repeats = max(tail.count(tail[i:i + 20]) for i in range(0, 130))
print(f"\ngreedy decoding: the most repeated 20-character substring appears {repeats} times in the last 150 characters")
print("greedy decoding falls into a loop; sampling at 0.5 to 1.0 wanders more usefully; 1.5 is noise.")
assert repeats >= 3

print("\nAll checks passed.")
