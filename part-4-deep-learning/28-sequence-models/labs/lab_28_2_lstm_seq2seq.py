# %% [markdown]
# # Lab 28.2: LSTM, GRU, and seq2seq with attention
#
# 1. An LSTM cell by hand, matched against nn.LSTM (and its gate order).
# 2. 28.1's delay task again: vanilla RNN vs GRU vs LSTM, and the forget-gate bias.
# 3. Seq2seq on reversing digit strings: without attention, with attention, by input length.
# 4. The attention matrix, read as an algorithm.
# 5. Teacher-forced accuracy vs free-running accuracy: the number that flatters.

# %%
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(282)
torch.set_num_threads(4)

# %% [markdown]
# ## 1. The LSTM cell

# %%
D, H, T, B = 3, 4, 6, 2
lstm = nn.LSTM(D, H, batch_first=True)
x = torch.randn(B, T, D)
out_ref, (h_ref, c_ref) = lstm(x)
W_ih, W_hh = lstm.weight_ih_l0, lstm.weight_hh_l0
b = lstm.bias_ih_l0 + lstm.bias_hh_l0
h, c, outs = torch.zeros(B, H), torch.zeros(B, H), []
for t in range(T):
    gates = x[:, t] @ W_ih.T + h @ W_hh.T + b                     # one matmul, 4H outputs
    i, f, g, o = gates.chunk(4, dim=1)                             # PyTorch's order: input, forget, cell candidate, output
    i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    c = f * c + i * g                                              # the additive path
    h = o * torch.tanh(c)
    outs.append(h)
assert torch.allclose(torch.stack(outs, 1), out_ref, atol=1e-6) and torch.allclose(c, c_ref[0], atol=1e-6)
print(f"hand-written LSTM matches nn.LSTM; weight_ih is {tuple(W_ih.shape)} = (4 x hidden, input)")
print(f"parameters: nn.LSTM(32, 64) {sum(p.numel() for p in nn.LSTM(32, 64).parameters())}, "
      f"nn.GRU(32, 64) {sum(p.numel() for p in nn.GRU(32, 64).parameters())} (exercise 2)")

# %% [markdown]
# ## 2. Remembering one bit for L steps
#
# Same task as lab 28.1: x_1 is +1 or -1, the label is its sign, everything after it is noise. Same budget for every model.

# %%
def delay_task(n, L, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, L, 1, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    x[:, 0, 0] = y * 2.0 - 1
    return x, y


class Classifier(nn.Module):
    def __init__(self, cell, h=32, forget_bias=None):
        super().__init__()
        self.rnn = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell](1, h, batch_first=True)
        if forget_bias is not None:                                # PyTorch has two bias vectors; set the forget slice of both
            with torch.no_grad():
                for bias in (self.rnn.bias_ih_l0, self.rnn.bias_hh_l0):
                    bias[h:2 * h] = forget_bias / 2
        self.out = nn.Linear(h, 2)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.out(o[:, -1])


def train_delay(L, cell, steps=500, seed=0, forget_bias=None):
    torch.manual_seed(seed)
    m = Classifier(cell, forget_bias=forget_bias)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    X, y = delay_task(4096, L, seed=1)
    for s in range(steps):
        i = torch.randint(0, len(X), (64,))
        loss = F.cross_entropy(m(X[i]), y[i])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    Xt, yt = delay_task(2000, L, seed=2)
    with torch.no_grad():
        return (m(Xt).argmax(1) == yt).float().mean().item()


t0 = time.time()
res = {}
configs = [("vanilla RNN", "rnn", None), ("GRU", "gru", None), ("LSTM, forget bias 0", "lstm", None),
           ("LSTM, forget bias 2", "lstm", 2.0), ("LSTM, forget bias 4", "lstm", 4.0)]
print("\naccuracy on held-out sequences (chance 0.5), one seed")
print("                         L = 20, 500 steps    L = 100, 1000 steps")
for name, cell, fb in configs:
    a20 = train_delay(20, cell, steps=500, forget_bias=fb)
    a100 = train_delay(100, cell, steps=1000, forget_bias=fb) if cell != "gru" else float("nan")   # the GRU is slow on CPU; exercise 7
    res[name] = (a20, a100)
    print(f"  {name:22s} {a20:10.3f}          {a100:10.3f}")
print(f"({time.time() - t0:.0f}s)")
print("at L = 20 the plain RNN wins: the gated cells start out forgetting (sigmoid(0) = 0.5 per step) and need longer to learn")
print("to keep. At L = 100 the forget gate's starting value decides everything: sigmoid(2)^100 = 3e-6 of the memory")
print("survives, sigmoid(4)^100 = 0.16. The additive path only helps once the gate is open.")
assert res["vanilla RNN"][0] > 0.95 and res["vanilla RNN"][1] < 0.6
assert res["LSTM, forget bias 0"][0] < 0.7 and res["LSTM, forget bias 2"][0] > 0.95
assert res["LSTM, forget bias 2"][1] < 0.6 and res["LSTM, forget bias 4"][1] > 0.95

# %% [markdown]
# ## 3. Seq2seq: reverse a string of digits
#
# Source: 4 to 20 random digits. Target: the same digits reversed, then an end token. Reversal is easy to state and hard
# for a bottleneck: the first output digit is the last input digit, and the last output digit was read 20 steps ago.

# %%
V, SOS, EOS, PAD = 10, 10, 11, 12
NTOK = 13


def make_batch(n, lo=4, hi=20, g=None):
    lens = torch.randint(lo, hi + 1, (n,), generator=g)
    Lmax = int(lens.max())
    src = torch.full((n, Lmax), PAD); tgt = torch.full((n, Lmax + 1), PAD)
    for k, L in enumerate(lens.tolist()):
        d = torch.randint(0, V, (L,), generator=g)
        src[k, :L] = d
        tgt[k, :L] = d.flip(0); tgt[k, L] = EOS
    return src, lens, tgt


class Seq2Seq(nn.Module):
    def __init__(self, attention, H=128):
        super().__init__()
        self.attention = attention
        self.emb = nn.Embedding(NTOK, 64)
        self.enc = nn.GRU(64, H, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * H, H)
        self.dec = nn.GRUCell(64 + (2 * H if attention else 0), H)
        self.Wa = nn.Linear(H, 2 * H, bias=False)                  # Luong's "general" score: h^T W s
        self.out = nn.Linear(H + (2 * H if attention else 0), NTOK)

    def encode(self, src, lens):
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(src), lens, batch_first=True, enforce_sorted=False)
        S, h = self.enc(packed)
        S, _ = nn.utils.rnn.pad_packed_sequence(S, batch_first=True, total_length=src.shape[1])
        h0 = torch.tanh(self.bridge(torch.cat([h[0], h[1]], 1)))   # final states of both directions: the bottleneck
        mask = torch.arange(src.shape[1])[None] < lens[:, None]
        return S, h0, mask

    def step(self, tok, h, S, mask):
        e = self.emb(tok)
        if not self.attention:
            h = self.dec(e, h)
            return self.out(h), h, None
        scores = torch.einsum("bh,bth->bt", self.Wa(h), S).masked_fill(~mask, -1e9)   # never attend to padding
        alpha = scores.softmax(1)
        ctx = torch.einsum("bt,bth->bh", alpha, S)
        h = self.dec(torch.cat([e, ctx], 1), h)
        return self.out(torch.cat([h, ctx], 1)), h, alpha

    def forward(self, src, lens, tgt):                             # teacher forcing
        S, h, mask = self.encode(src, lens)
        tok, logits = torch.full((len(src),), SOS), []
        for t in range(tgt.shape[1]):
            lg, h, _ = self.step(tok, h, S, mask)
            logits.append(lg)
            tok = tgt[:, t]                                        # the TRUE previous token, not the prediction
        return torch.stack(logits, 1)

    @torch.no_grad()
    def generate(self, src, lens, max_len):
        S, h, mask = self.encode(src, lens)
        tok, outs, alphas = torch.full((len(src),), SOS), [], []
        for _ in range(max_len):
            lg, h, a = self.step(tok, h, S, mask)
            tok = lg.argmax(1)                                     # greedy: the model's own output goes back in
            outs.append(tok); alphas.append(a)
        return torch.stack(outs, 1), alphas


def train_s2s(attention, steps=1200, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = Seq2Seq(attention)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for s in range(steps):
        src, lens, tgt = make_batch(64, g=g)
        loss = F.cross_entropy(m(src, lens, tgt).reshape(-1, NTOK), tgt.reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
    return m


def evaluate(m, L, n=500, seed=123):
    g = torch.Generator().manual_seed(seed + L)
    src, lens, tgt = make_batch(n, L, L, g=g)
    pred, alphas = m.generate(src, lens, L + 1)
    free_tok = (pred == tgt).float().mean().item()
    exact = (pred == tgt).all(1).float().mean().item()
    with torch.no_grad():
        forced_tok = (m(src, lens, tgt).argmax(-1) == tgt).float().mean().item()
    return exact, free_tok, forced_tok, (src, pred, alphas)


models, res_s2s = {}, {}
for attention in (False, True):
    t0 = time.time()
    models[attention] = train_s2s(attention)
    print(f"\nseq2seq {'with' if attention else 'without'} attention trained in {time.time() - t0:.0f}s")
print("\nexact-match accuracy (whole string right, greedy decoding) by input length:")
print("   L    no attention   attention")
for L in (4, 8, 12, 16, 20):
    res_s2s[L] = {a: evaluate(models[a], L) for a in (False, True)}
    print(f"{L:4d}   {res_s2s[L][False][0]:12.3f}   {res_s2s[L][True][0]:9.3f}")
print("one vector for the whole input holds a few digits well and 20 digits badly. The attention model reads the digit it needs.")
assert res_s2s[20][True][0] > res_s2s[20][False][0] + 0.5
assert res_s2s[4][False][0] > 0.9

# %% [markdown]
# ## 4. Reading the attention matrix

# %%
src, pred, alphas = res_s2s[12][True][3]
A = torch.stack([a[0] for a in alphas])[:12, :12]                 # decoder step x source position, first example
print("\nattention weights for one 12-digit input (rows: output step, columns: input position; '#' > 0.5, '+' > 0.1):")
print("        " + " ".join(str(d.item()) for d in src[0, :12]))
for t in range(12):
    row = "".join(" #" if a > 0.5 else " +" if a > 0.1 else " ." for a in A[t])
    print(f"out {pred[0, t].item()}: {row}")
anti = A.flip(1).diagonal().mean().item()
print(f"mean weight on the anti-diagonal (output t reads input L-1-t): {anti:.2f}. Nobody told it to reverse; it found the alignment.")
assert anti > 0.5

# %% [markdown]
# ## 5. Teacher forcing flatters

# %%
print("\ntoken accuracy for the no-attention model: teacher-forced vs free-running (greedy)")
for L in (8, 12, 20):
    _, free_tok, forced_tok, _ = res_s2s[L][False]
    print(f"  L = {L:2d}: teacher-forced {forced_tok:.3f}   free-running {free_tok:.3f}")
print("teacher forcing hands the model the right previous digit at every step; in production nobody does.")
assert res_s2s[20][False][2] > res_s2s[20][False][1] + 0.1

print("\nAll checks passed.")
