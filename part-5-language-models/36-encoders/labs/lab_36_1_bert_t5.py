# %% [markdown]
# # Lab 36.1: Encoders, masked language modeling and fine-tuning
#
# 1. Masked language modeling: BERT's 15% masking with the 80/10/10 rule.
# 2. Pretrain a tiny bidirectional encoder (MLM) and a tiny causal decoder of the same size on this course.
# 3. Fill in the blank: both sides of the context vs the left side only.
# 4. Fine-tune the encoder to classify paragraphs (34.2's task): pretrained vs random initialization vs TF-IDF.
# 5. Token classification with subwords: aligning word labels to BPE pieces.
# 6. T5's span corruption: what the training examples look like.

# %%
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from bpe import BPE  # noqa: E402
from corpus import PARTS, labeled_paragraphs, lesson_files, make_ner, read_lesson  # noqa: E402

torch.manual_seed(361)
torch.set_num_threads(4)

files = lesson_files()
train_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 != 4)
val_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 == 4)
tok = BPE.train(train_text, n_merges=1000)
MASK, CLS, PAD = tok.vocab_size, tok.vocab_size + 1, tok.vocab_size + 2               # special tokens after the byte/merge ids
V = tok.vocab_size + 3
tr_ids, va_ids = torch.tensor(tok.encode(train_text)), torch.tensor(tok.encode(val_text))
CTX = 64

# %% [markdown]
# ## 1. Masking

# %%
def mlm_mask(x, g, p=0.15):
    """Choose 15% of positions as targets. Of those: 80% -> [MASK], 10% -> a random token, 10% unchanged.
    Loss is computed only at the chosen positions (labels -100 elsewhere)."""
    chosen = torch.rand(x.shape, generator=g) < p
    labels = torch.where(chosen, x, torch.full_like(x, -100))
    r = torch.rand(x.shape, generator=g)
    x = x.clone()
    x[chosen & (r < 0.8)] = MASK
    rand = chosen & (r >= 0.8) & (r < 0.9)
    x[rand] = torch.randint(0, tok.vocab_size, (int(rand.sum()),), generator=g)
    return x, labels


demo = torch.tensor([tok.encode(" the gradient points uphill, so gradient descent walks the other way")])
masked, labels = mlm_mask(demo, torch.Generator().manual_seed(3), p=0.3)
show = lambda ids: " ".join("[MASK]" if i == MASK else repr(tok.decode([i])) for i in ids)
print("original:", show(demo[0].tolist()))
print("input:   ", show(masked[0].tolist()))
print("the random and unchanged 20% stop the model from learning that only [MASK] positions matter: at fine-tuning time")
print("there are no [MASK] tokens at all.")

# %% [markdown]
# ## 2. Two tiny pretrained models
#
# Both use RoPE (29.3, 35.5). My first version used learned position embeddings, and the masked-LM encoder sat at
# unigram-level loss (about 5.6 nats) for the whole budget while the decoder trained fine: a masked position carries no
# information about its own token, so the encoder has to learn to read its neighbors, and learning where the
# neighbors are through 64 position embeddings took longer than this lab has. With RoPE it gets there in a few hundred
# steps. Encoders are known for a long "unigram plateau" early in training; positions are part of why.

# %%
def rope(x, base=10000.0):
    n, hd = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    ang = torch.arange(n, dtype=torch.float32)[:, None] * theta
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, causal):
        super().__init__()
        self.h, self.causal, self.ln1, self.ln2 = h, causal, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pad_mask=None):
        B, n, d = x.shape
        q, k, v = (t.view(B, n, self.h, d // self.h).transpose(1, 2) for t in self.qkv(self.ln1(x)).chunk(3, -1))
        q, k = rope(q), rope(k)
        attn_mask = None if pad_mask is None else (~pad_mask)[:, None, None, :]      # True = may attend
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.causal and attn_mask is None)
        x = x + self.proj(o.transpose(1, 2).reshape(B, n, d))
        return x + self.mlp(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, causal, d=128, L=4, h=4):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(Block(d, h, causal) for _ in range(L))
        self.ln = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)

    def encode(self, idx, pad_mask=None):
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x, pad_mask)
        return self.ln(x)

    def forward(self, idx):
        return self.lm_head(self.encode(idx))


def windows(ids, bs, g):
    i = torch.randint(0, len(ids) - CTX - 1, (bs,), generator=g)
    return torch.stack([ids[j:j + CTX + 1] for j in i])


def pretrain(causal, steps=1000):
    torch.manual_seed(0)
    m = Transformer(causal)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 2e-3 * min(1, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        w = windows(tr_ids, 32, g)
        if causal:
            loss = F.cross_entropy(m(w[:, :-1]).reshape(-1, V), w[:, 1:].reshape(-1))
        else:
            x, labels = mlm_mask(w[:, :-1], g)
            loss = F.cross_entropy(m(x).reshape(-1, V), labels.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m.eval()


t0 = time.time()
bert, gpt = pretrain(causal=False), pretrain(causal=True)
print(f"\npretrained a bidirectional encoder (MLM) and a causal decoder, same size, same data and steps, in {time.time() - t0:.0f}s")

# %% [markdown]
# ## 3. Fill in the blank
#
# Hide one token in the middle of a validation window. The encoder sees both sides; the decoder only sees what came
# before (the next-token prediction at that position).

# %%
g = torch.Generator().manual_seed(7)
w = windows(va_ids, 2000, g)[:, :CTX]
pos = CTX // 2
target = w[:, pos].clone()
with torch.no_grad():
    x = w.clone(); x[:, pos] = MASK
    bert_logits = bert(x)[:, pos]
    gpt_logits = gpt(w[:, :pos])[:, -1]
hit_b = (bert_logits.argmax(-1) == target).float().numpy()
hit_g = (gpt_logits.argmax(-1) == target).float().numpy()
nll = lambda lg: F.cross_entropy(lg, target).item()
d = hit_b - hit_g
bs = np.random.default_rng(0)
boot = [d[bs.integers(0, len(d), len(d))].mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
common = np.bincount(tr_ids.numpy(), minlength=V).argmax()
baseline = (target == common).float().mean().item()
print(f"\nmasked middle token, {len(target):,} validation windows (always guessing the most frequent token: {baseline:.3f}):")
print(f"  bidirectional encoder: accuracy {hit_b.mean():.3f}, loss {nll(bert_logits):.3f}")
print(f"  causal decoder (left context only): accuracy {hit_g.mean():.3f}, loss {nll(gpt_logits):.3f}")
print(f"  encoder - decoder: {d.mean():+.3f}, paired 95% CI [{lo:+.3f}, {hi:+.3f}]")
if lo > 0:
    verdict = "the encoder, which sees both sides of the blank, is ahead, but only by a little"
elif hi < 0:
    verdict = "the decoder is ahead, although it sees only half the context"
else:
    verdict = "neither is clearly ahead, although the encoder sees twice the context"
print(f"{verdict}. It trained on the same windows but")
print("learned from only the 15% of tokens it masked; the decoder learned from every token. That sample inefficiency is")
print("what ELECTRA (replaced-token detection on every position) was designed to fix. And the encoder can't generate: it")
print("was never trained to produce text one token at a time from the left.")
assert hit_b.mean() > baseline + 0.1 and hit_g.mean() > baseline + 0.1 and d.mean() < 0.1

# %% [markdown]
# ## 4. Fine-tuning for classification
#
# 34.2's task: which Part does a paragraph come from? Input: [CLS] + the first 63 tokens. The classifier reads the
# final [CLS] vector. Same fine-tuning recipe for the pretrained encoder and for a randomly initialized one.

# %%
data = labeled_paragraphs()
random.Random(0).shuffle(data)
split = int(0.8 * len(data))


def encode_batch(items):
    X = torch.full((len(items), CTX), PAD)
    for i, (text, _) in enumerate(items):
        ids = [CLS] + tok.encode(text)[:CTX - 1]
        X[i, :len(ids)] = torch.tensor(ids)
    return X, torch.tensor([l for _, l in items])


Xtr, ytr = encode_batch(data[:split])
Xte, yte = encode_batch(data[split:])


def finetune(encoder, epochs=6, lr=5e-4, seed=0):
    torch.manual_seed(seed)
    enc = encoder
    head = nn.Linear(128, len(PARTS))
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": lr}, {"params": head.parameters(), "lr": lr * 5}], weight_decay=0.01)
    enc.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), 16):
            i = perm[s:s + 16]
            h = enc.encode(Xtr[i], pad_mask=Xtr[i] == PAD)[:, 0]
            loss = F.cross_entropy(head(h), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        return (head(enc.encode(Xte, pad_mask=Xte == PAD)[:, 0]).argmax(1) == yte).float().numpy()


t0 = time.time()
import copy  # noqa: E402
hits_pre = finetune(copy.deepcopy(bert))
torch.manual_seed(0)
hits_rand = finetune(Transformer(causal=False))
acc_pre, acc_rand = hits_pre.mean(), hits_rand.mean()
dd = hits_pre - hits_rand
boot = [dd[np.random.default_rng(b).integers(0, len(dd), len(dd))].mean() for b in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\nparagraph classification, {len(Xtr)} training paragraphs, {len(Xte)} test ({time.time() - t0:.0f}s):")
print(f"  encoder pretrained with MLM, fine-tuned   {acc_pre:.3f}")
print(f"  same encoder, random initialization       {acc_rand:.3f}")
print(f"  TF-IDF + logistic regression (lab 34.2)   ~0.81 (5-fold CV on all paragraphs, full length)")
print(f"  pretrained - random: {acc_pre - acc_rand:+.3f}, paired 95% CI [{lo:+.3f}, {hi:+.3f}] over {len(dd)} test paragraphs")
if lo > 0:
    print("pretraining helps even at this toy scale.")
elif hi < 0:
    print("at this toy scale, pretraining hurt: 350,000 tokens of masked LM taught less than it cost.")
else:
    print("at this toy scale, pretraining makes no measurable difference: 350,000 tokens of masked LM teach little.")
print("TF-IDF beats both by a wide margin: an encoder reading only the first 63 tokens is no match for counting every")
print("word of the paragraph. A real")
print("encoder, pretrained on billions of tokens, is what makes fine-tuning with a few hundred labels competitive. Always keep")
print("the linear baseline.")
assert max(acc_pre, acc_rand) < 0.75 and min(acc_pre, acc_rand) > 1 / len(PARTS) + 0.1

# %% [markdown]
# ## 5. Token classification with subwords
#
# NER labels are per word; the encoder sees BPE pieces. Standard practice: label the first piece of each word, and
# ignore the other pieces in the loss (-100). At prediction time, read the first piece's label.

# %%
toks_, tags_ = make_ner(1, seed=4)[0]
ids, labels_ = [], []
for w_, t_ in zip(toks_, tags_):
    pieces = tok.encode(" " + w_)
    ids += pieces
    labels_ += [t_] + ["-100"] * (len(pieces) - 1)
print("\nword-level tags -> BPE pieces:")
print("  " + "  ".join(f"{tok.decode([i])!r}:{l}" for i, l in zip(ids, labels_)))
print("a name split into three pieces must not get three B-PER tags; mislabeled continuation pieces are a classic bug.")

# %% [markdown]
# ## 6. T5's span corruption

# %%
def span_corrupt(tokens, rate=0.15, span=3, seed=0):
    """Replace non-overlapping spans (about rate x len tokens in total) with sentinels; the target lists each sentinel
    followed by the tokens it replaced, and ends with a final sentinel."""
    r = random.Random(seed)
    n_spans = max(1, round(len(tokens) * rate / span))
    starts = []
    for c in r.sample(range(len(tokens) - span + 1), len(tokens) - span + 1):         # candidates in random order
        if all(abs(c - s_) > span for s_ in starts):                                  # keep spans apart
            starts.append(c)
        if len(starts) == n_spans:
            break
    inp, tgt, i = [], [], 0
    for k, s_ in enumerate(sorted(starts)):
        inp += tokens[i:s_] + [f"<extra_id_{k}>"]
        tgt += [f"<extra_id_{k}>"] + tokens[s_:s_ + span]
        i = s_ + span
    inp += tokens[i:]
    tgt += [f"<extra_id_{len(starts)}>"]
    return " ".join(inp), " ".join(tgt)


words_ = "Thank you for inviting me to your party last week , it was a great evening and the food was excellent".split()
inp, tgt = span_corrupt(words_, rate=0.3, seed=2)
print(f"\nT5 span corruption:\n  input:  {inp}\n  target: {tgt}")
print("the encoder reads the corrupted text; the decoder writes only the missing spans, each after its sentinel. Short")
print("targets make pretraining cheaper than reconstructing the whole input.")

print("\nAll checks passed.")
