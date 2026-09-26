# %% [markdown]
# # Lab 39.2: Multimodal models, in miniature
#
# Part A, vision-language: a small image encoder, a projector that turns its features into a few "soft tokens", and
# the course's language model reading them as a prefix (the LLaVA recipe). Captions of colored shapes (30.2's data).
#   A1. Train only the projector and encoder, with the language model frozen (LLaVA stage 1).
#   A2. Also fine-tune the language model (stage 2). Test on color-shape combinations never seen in training.
# Part B, speech: a log-mel spectrogram from scratch, and a small model trained with CTC to read digit sequences from
#   synthetic phone-keypad (DTMF) tones: alignment-free sequence recognition, the idea under Whisper's predecessors.

# %%
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "part-4-deep-learning" / "_shared"))
from shapes import CLASSES, COLORS, make_colored  # noqa: E402
from tinygpt import load_or_train  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(392)
lm, tok = load_or_train()
V, D = tok.vocab_size, lm.tok.weight.shape[1]
COLOR_NAMES = list(COLORS)
HELD_OUT = {("circle", "red"), ("square", "blue"), ("triangle", "yellow"), ("plus", "cyan")}

# %% [markdown]
# ## A. Vision-language

# %%
X_np, s_np, c_np = make_colored(6000, seed=0)
X = torch.tensor(X_np)
caption = lambda s, c: f" a {COLOR_NAMES[c]} {CLASSES[s]}."
seen = np.array([(CLASSES[s], COLOR_NAMES[c]) not in HELD_OUT for s, c in zip(s_np, c_np)])
tr = np.flatnonzero(seen[:5000]); te_seen = 5000 + np.flatnonzero(seen[5000:]); te_unseen = 5000 + np.flatnonzero(~seen[5000:])
MEAN, STD = X[tr].mean((0, 2, 3), keepdim=True), X[tr].std((0, 2, 3), keepdim=True)
cap_ids = [tok.encode(caption(s, c)) for s, c in zip(s_np, c_np)]
N_SOFT = 4
print(f"caption tokens, e.g. {caption(0, 0)!r} -> {[tok.decode([i]) for i in tok.encode(caption(0, 0))]}")


class VLM(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = copy.deepcopy(lm)
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, N_SOFT * D))   # image -> 4 soft tokens

    def forward(self, img, ids):
        soft = self.proj(self.enc((img - MEAN) / STD)).view(len(img), N_SOFT, D)
        x = torch.cat([soft, self.lm.tok(ids)], 1)
        x = x + self.lm.pos(torch.arange(x.shape[1]))
        for b in self.lm.blocks:
            x, _ = b(x)
        return self.lm.head(self.lm.ln_f(x))[:, N_SOFT - 1:-1]                       # predictions for each caption token

    @torch.no_grad()
    def caption(self, img, max_new=8):
        soft = self.proj(self.enc((img - MEAN) / STD)).view(len(img), N_SOFT, D)
        out = torch.zeros(len(img), 0, dtype=torch.long)
        for _ in range(max_new):
            x = torch.cat([soft, self.lm.tok(out)], 1) + self.lm.pos(torch.arange(N_SOFT + out.shape[1]))
            for b in self.lm.blocks:
                x, _ = b(x)
            nxt = self.lm.head(self.lm.ln_f(x))[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
        return [tok.decode(o.tolist()).split(".")[0] + "." for o in out]


def train_vlm(m, steps, train_lm, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    for p in m.lm.parameters():
        p.requires_grad = train_lm
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    L = max(len(c) for c in cap_ids)
    for s in range(steps):
        i = tr[torch.randint(0, len(tr), (64,), generator=g).numpy()]
        ids = torch.zeros(64, L, dtype=torch.long); tgt = torch.full((64, L), -100)
        for k, j in enumerate(i):
            ids[k, :len(cap_ids[j])] = torch.tensor(cap_ids[j]); tgt[k, :len(cap_ids[j])] = torch.tensor(cap_ids[j])
        loss = F.cross_entropy(m(X[i], ids).reshape(-1, V), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def exact(m, idx):
    caps = m.caption(X[idx])
    return np.mean([c.strip() == caption(s_np[j], c_np[j]).strip() for c, j in zip(caps, idx)])


def attributes(m, idx):
    """Share of captions naming the right color, and the right shape, separately."""
    caps = m.caption(X[idx])
    color = np.mean([f" {COLOR_NAMES[c_np[j]]} " in f" {c} " for c, j in zip(caps, idx)])
    shape = np.mean([f" {CLASSES[s_np[j]]}." in f" {c}" for c, j in zip(caps, idx)])
    return color, shape


t0 = time.time()
stage1 = train_vlm(VLM(lm), 500, train_lm=False)
acc1 = (exact(stage1, te_seen[:300]), exact(stage1, te_unseen))
stage2 = train_vlm(copy.deepcopy(stage1), 300, train_lm=True, lr=5e-4)
acc2 = (exact(stage2, te_seen[:300]), exact(stage2, te_unseen))
n_proj = sum(p.numel() for n, p in stage1.named_parameters() if not n.startswith("lm."))
print(f"\ntrained in {time.time() - t0:.0f}s; encoder + projector {n_proj:,} parameters, language model {sum(p.numel() for p in lm.parameters()):,}")
print("exact caption match               seen combinations   never-seen combinations")
print(f"  stage 1 (language model frozen)   {acc1[0]:17.3f}   {acc1[1]:23.3f}")
print(f"  stage 2 (language model tuned)    {acc2[0]:17.3f}   {acc2[1]:23.3f}")
print("examples (stage 2, never-seen combinations):", stage2.caption(X[te_unseen[:4]]),
      "truth:", [caption(s_np[j], c_np[j]).strip() for j in te_unseen[:4]])
col, shp = attributes(stage2, te_unseen)
print(f"on never-seen combinations, stage 2 names the right color {col:.0%} of the time and the right shape {shp:.0%}")
print("a frozen language model can be taught to 'read' images through four vectors it never saw in training: the")
print("projector learns to write in the language model's embedding space. But a combination never seen in training")
print("is almost never produced: the model recognizes each attribute and then falls back on pairs it has seen.")
print("Compositional generalization doesn't come free; it comes from training data that covers combinations.")
assert acc1[0] > 0.5 and acc2[0] >= acc1[0] - 0.05 and max(col, shp) > acc2[1] + 0.2

# %% [markdown]
# ## B. Speech: a log-mel spectrogram and CTC
#
# Each digit is a DTMF tone: the sum of a low and a high frequency. A clip is 3 to 6 digits with random durations,
# gaps and noise. The model never learns where each digit starts: CTC sums over all alignments.

# %%
SR = 8000
LOW, HIGH = [697, 770, 852, 941], [1209, 1336, 1477]
KEYS = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (2, 0), 8: (2, 1), 9: (2, 2), 0: (3, 1)}


def clip(r):
    digits = list(r.integers(0, 10, r.integers(3, 7)))
    parts = [np.zeros(int(SR * r.uniform(0.05, 0.2)))]
    for d in digits:
        lo, hi = KEYS[d]
        n = int(SR * r.uniform(0.08, 0.2)); t = np.arange(n) / SR
        parts += [np.sin(2 * np.pi * LOW[lo] * t) + np.sin(2 * np.pi * HIGH[hi] * t), np.zeros(int(SR * r.uniform(0.04, 0.15)))]
    x = np.concatenate(parts)
    return x + r.normal(0, 0.5, len(x)), digits


def mel_filterbank(n_fft, n_mels, sr):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(0), mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r_ = bins[i], bins[i + 1], bins[i + 2]
        fb[i, l:c] = (np.arange(l, c) - l) / max(1, c - l)
        fb[i, c:r_] = (r_ - np.arange(c, r_)) / max(1, r_ - c)
    return fb


FB = mel_filterbank(256, 40, SR)


def log_mel(x, hop=80):
    frames = np.lib.stride_tricks.sliding_window_view(x, 256)[::hop] * np.hanning(256)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log(power @ FB.T + 1e-6).astype(np.float32)                             # (frames, 40): 10 ms per frame


class CTCModel(nn.Module):
    def __init__(self):
        super().__init__()
        # two stride-2 convolutions: 4x fewer frames (40 ms each), so each digit spans a few outputs, not 8 to 20
        self.conv = nn.Sequential(nn.Conv1d(40, 128, 5, stride=2, padding=2), nn.ReLU(),
                                  nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.ReLU())
        self.rnn = nn.GRU(128, 96, batch_first=True, bidirectional=True)
        self.out = nn.Linear(192, 11)                                                  # 10 digits + the CTC blank (index 10)

    def forward(self, feats):
        h = self.conv(feats.transpose(1, 2)).transpose(1, 2)
        return self.out(self.rnn(h)[0]).log_softmax(-1)


def batch(r, n):
    xs = [clip(r) for _ in range(n)]
    feats = [torch.tensor(log_mel(x)) for x, _ in xs]
    T = max(f.shape[0] for f in feats)
    F_ = torch.zeros(n, T, 40)
    for i, f in enumerate(feats):
        F_[i, :len(f)] = (f - f.mean()) / (f.std() + 1e-6)
    return F_, torch.tensor([len(f) for f in feats]), [d for _, d in xs]


def greedy_ctc(logp, length):
    path = logp[:length].argmax(-1).tolist()
    out, prev = [], None
    for p in path:
        if p != prev and p != 10:
            out.append(p)
        prev = p
    return out


t0 = time.time()
torch.manual_seed(0)
asr = CTCModel()
opt = torch.optim.Adam(asr.parameters(), lr=3e-3)
r = np.random.default_rng(0)
out_len = lambda lens: (lens + 3) // 4                                                # frames after the two stride-2 convs
feats_te, lens_te, digits_te = batch(np.random.default_rng(99), 200)
curve = []
for step in range(700):
    feats, lens, digits = batch(r, 16)
    logp = asr(feats)
    targets = torch.tensor([int(d) for ds in digits for d in ds])
    loss = F.ctc_loss(logp.transpose(0, 1), targets, out_len(lens), torch.tensor([len(d) for d in digits]), blank=10)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 99:
        with torch.no_grad():
            lp = asr(feats_te)
        hyp = [greedy_ctc(lp[i], out_len(lens_te)[i]) for i in range(200)]
        curve.append((step + 1, loss.item(), np.mean([h == [int(x) for x in d] for h, d in zip(hyp, digits_te)])))
seq_acc = curve[-1][2]
print(f"\nCTC digit recognizer trained in {time.time() - t0:.0f}s on log-mel features (40 mel bands, 10 ms frames, 4x downsampled)")
print("  step   CTC loss   exact sequence accuracy (200 new noisy clips)")
for st, l, a in curve:
    print(f"  {st:4d}   {l:8.3f}   {a:.3f}")
print(f"e.g. heard {hyp[0]} for {[int(x) for x in digits_te[0]]}")
print("for hundreds of steps the model outputs only blanks (the loss is stuck near 2.5 per digit), then it finds the")
print("alignment and accuracy jumps. That plateau is typical of CTC; without the downsampling it lasted past 1,200 steps")
print("here. No frame-level labels anywhere: CTC learned where each digit is. Whisper replaced CTC with an encoder-")
print("decoder transformer trained on 680,000 hours; the front end, a log-mel spectrogram, is the same.")
assert curve[0][2] < 0.5 and seq_acc > 0.8

print("\nAll checks passed.")
