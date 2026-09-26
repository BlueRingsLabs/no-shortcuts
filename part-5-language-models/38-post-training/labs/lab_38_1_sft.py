# %% [markdown]
# # Lab 38.1: Supervised fine-tuning
#
# The course's small pretrained GPT knows how the lessons sound. Fine-tune it to answer questions about the course
# ("What is lesson 29.1 about?") in a chat format, and measure:
# 1. What the base model does with a question (it continues the text; it doesn't answer).
# 2. SFT with the loss on the answer only (prompt tokens masked) vs on everything.
# 3. Exact-match accuracy on the phrasings seen in training and on a phrasing never seen, and whether it stops.
# 4. Forgetting: what fine-tuning does to the model's language modeling of the course, and mixing pretraining data back in.

# %%
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from instructions import END, dataset, format_chat  # noqa: E402
from tinygpt import corpus_split, load_or_train, val_loss  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(381)
base, tok = load_or_train()
V, CTX = tok.vocab_size, base.ctx
train_items, heldout_items = dataset("train"), dataset("heldout")
end_ids = tok.encode(" " + END)
print(f"{len(train_items)} training examples, {len(heldout_items)} held-out-phrasing examples; e.g.")
q, a, _ = train_items[0]
print("  " + repr(format_chat(q, a)[0] + format_chat(q, a)[1]))


@torch.no_grad()
def answer(model, question, max_new=40):
    """Greedy decoding with the KV cache (37.2); stops at the end marker."""
    ids = tok.encode(format_chat(question))
    start = len(ids)
    logits, caches = model(torch.tensor([ids]), [None] * len(model.blocks), 0)
    for _ in range(max_new):
        ids.append(int(logits[0, -1].argmax()))
        if ids[-len(end_ids):] == end_ids:
            return tok.decode(ids[start:-len(end_ids)]).strip(), True
        if len(ids) >= CTX:
            break
        logits, caches = model(torch.tensor([ids[-1:]]), caches, len(ids) - 1)
    return tok.decode(ids[start:]).strip(), False


# %% [markdown]
# ## 1. The base model

# %%
for q_ in ("What is lesson 29.1 about?", "How many hours does module 35 take?"):
    out, stopped = answer(base, q_)
    print(f"\nbase model, {q_!r}:\n  -> {out[:160]!r} (stopped: {stopped})")
print("\nit continues the document. Nothing in pretraining taught it that a question is followed by an answer and a stop.")

# %% [markdown]
# ## 2. SFT
#
# Each example is prompt + answer + end marker; labels are the next tokens, with -100 (ignored) on the prompt when
# masking. Short examples are padded; pad positions are ignored too.

# %%
def encode(items, mask_prompt):
    X = torch.zeros(len(items), CTX, dtype=torch.long)
    Y = torch.full((len(items), CTX), -100)
    for i, (q_, a_, _) in enumerate(items):
        p, r = format_chat(q_, a_)
        pi, ri = tok.encode(p), tok.encode(r)
        ids = (pi + ri)[:CTX + 1]
        X[i, :len(ids) - 1] = torch.tensor(ids[:-1])
        labels = ids[1:]
        if mask_prompt:
            labels = [-100] * (len(pi) - 1) + labels[len(pi) - 1:]                  # predict only the answer tokens
        Y[i, :len(labels)] = torch.tensor(labels)
    return X, Y


train_text, _ = corpus_split()
pre_ids = torch.tensor(tok.encode(train_text))


def sft(mask_prompt=True, pretrain_mix=0.0, epochs=12, lr=1e-3, bs=32, seed=0):
    torch.manual_seed(seed)
    m = copy.deepcopy(base).train()
    X, Y = encode(train_items, mask_prompt)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    steps = epochs * math.ceil(len(X) / bs)
    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / steps))
        i = torch.randint(0, len(X), (bs,), generator=g)
        loss = F.cross_entropy(m(X[i]).reshape(-1, V), Y[i].reshape(-1), ignore_index=-100)
        if pretrain_mix:                                                             # keep some language modeling in the diet
            j = torch.randint(0, len(pre_ids) - CTX - 1, (8,), generator=g)
            xp = torch.stack([pre_ids[k:k + CTX] for k in j]); yp = torch.stack([pre_ids[k + 1:k + CTX + 1] for k in j])
            loss = (1 - pretrain_mix) * loss + pretrain_mix * F.cross_entropy(m(xp).reshape(-1, V), yp.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m.eval()


def evaluate(m, items, n=100):
    sub = items[:n]
    outs = [answer(m, q_) for q_, _, _ in sub]
    exact = np.mean([o == a_ for (o, _), (_, a_, _) in zip(outs, sub)])
    stops = np.mean([s_ for _, s_ in outs])
    return exact, stops


t0 = time.time()
variants = {"SFT, loss on answers only": dict(mask_prompt=True),
            "SFT, loss on prompt and answer": dict(mask_prompt=False),
            "SFT, answers only + 25% pretraining mix": dict(mask_prompt=True, pretrain_mix=0.25)}
models, results = {}, {}
base_val = val_loss(base, tok)
for name, kw in variants.items():
    models[name] = sft(**kw)
    tr_acc, tr_stop = evaluate(models[name], train_items)
    ho_acc, ho_stop = evaluate(models[name], heldout_items)
    results[name] = (tr_acc, ho_acc, ho_stop, val_loss(models[name], tok))
print(f"\nfine-tuned three variants in {time.time() - t0:.0f}s")
print("                                            exact match       exact match          stops on its   course text")
print("                                            (seen phrasing)   (unseen phrasing)    own (unseen)   val loss")
print(f"  {'base model':42s} {'-':>15s}   {'-':>17s}    {'-':>12s}   {base_val:9.3f}")
for name, (tr, ho, st, vl) in results.items():
    print(f"  {name:42s} {tr:15.3f}   {ho:17.3f}    {st:12.3f}   {vl:9.3f}")

print("\nexamples, unseen phrasing (answers-only model):")
for q_, a_, _ in heldout_items[:4]:
    out, stopped = answer(models["SFT, loss on answers only"], q_)
    print(f"  {q_!r:62s} -> {out!r}{'' if stopped else ' (no stop)'}   [truth: {a_!r}]")

r_mask, r_all, r_mix = results.values()
print("\nthe format transfers to phrasings it never saw: it answers and stops. The facts it memorized in training are recalled")
print("less reliably under a new phrasing. SFT teaches a behavior quickly; it's a poor way to teach knowledge.")
print("and the price is forgetting: the course text itself got harder to predict. A little pretraining data in the mix")
print("keeps most of the language modeling while learning the new behavior.")
assert r_mask[2] > 0.9 and r_mask[0] > r_mask[1]
assert r_mask[3] > base_val and r_mix[3] < r_mask[3]

print("\nAll checks passed.")
