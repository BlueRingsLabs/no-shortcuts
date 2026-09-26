# %% [markdown]
# # Lab 38.2: LoRA and QLoRA
#
# 1. A LoRA layer from scratch: W x + (alpha / r) B A x, with W frozen, A small random, B zero.
# 2. LoRA at ranks 1, 4 and 16 vs full fine-tuning on 38.1's task: trainable parameters, accuracy, forgetting.
# 3. Merging the adapter into the weights: same outputs, zero inference overhead.
# 4. QLoRA in miniature: a frozen int4 base (37.3's quantizer) with LoRA adapters on top.
# 5. Memory arithmetic for a 7B model.

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
from instructions import END, dataset, format_chat  # noqa: E402
from tinygpt import load_or_train, val_loss  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(382)
base, tok = load_or_train()
V, CTX = tok.vocab_size, base.ctx
train_items, heldout_items = dataset("train"), dataset("heldout")
end_ids = tok.encode(" " + END)

# %% [markdown]
# ## 1. The LoRA layer

# %%
class LoRALinear(nn.Module):
    def __init__(self, linear, r=4, alpha=8):
        super().__init__()
        self.base = linear
        for p in self.base.parameters():
            p.requires_grad = False                                                  # the pretrained weights never change
        self.A = nn.Parameter(torch.randn(r, linear.in_features) / math.sqrt(linear.in_features))
        self.B = nn.Parameter(torch.zeros(linear.out_features, r))                  # zero: the adapted model starts identical
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged(self):
        lin = copy.deepcopy(self.base)
        with torch.no_grad():
            lin.weight += self.scale * self.B @ self.A
        return lin


def add_lora(model, r, alpha=None, targets=("qkv", "proj", "mlp.0", "mlp.2")):
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad = False
    for block in m.blocks:
        for name in targets:
            parent, attr = (block.mlp, int(name[-1])) if name.startswith("mlp") else (block, name)
            lin = parent[attr] if isinstance(attr, int) else getattr(parent, attr)
            new = LoRALinear(lin, r, alpha or 2 * r)
            if isinstance(attr, int):
                parent[attr] = new
            else:
                setattr(parent, attr, new)
    return m


probe = add_lora(base, 4)
x = torch.randint(0, V, (2, 20))
with torch.no_grad():
    assert torch.allclose(probe(x), base(x), atol=1e-6)
print("LoRA-wrapped model before training: identical outputs to the base model (B starts at zero)")

# %% [markdown]
# ## 2. LoRA vs full fine-tuning

# %%
def encode(items):
    X = torch.zeros(len(items), CTX, dtype=torch.long)
    Y = torch.full((len(items), CTX), -100)
    for i, (q_, a_, _) in enumerate(items):
        p, r_ = format_chat(q_, a_)
        pi, ri = tok.encode(p), tok.encode(r_)
        ids = pi + ri
        X[i, :len(ids) - 1] = torch.tensor(ids[:-1])
        Y[i, len(pi) - 1:len(ids) - 1] = torch.tensor(ri)                            # loss on the answer only
    return X, Y


X_all, Y_all = encode(train_items)


def train(m, lr, epochs=12, bs=32, seed=0):
    torch.manual_seed(seed)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    steps = epochs * math.ceil(len(X_all) / bs)
    m.train()
    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / steps))
        i = torch.randint(0, len(X_all), (bs,), generator=g)
        loss = F.cross_entropy(m(X_all[i]).reshape(-1, V), Y_all[i].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return m.eval(), sum(p.numel() for p in params)


@torch.no_grad()
def answer(model, question, max_new=40):
    ids = tok.encode(format_chat(question))
    start = len(ids)
    logits, caches = model(torch.tensor([ids]), [None] * len(model.blocks), 0)
    for _ in range(max_new):
        ids.append(int(logits[0, -1].argmax()))
        if ids[-len(end_ids):] == end_ids:
            return tok.decode(ids[start:-len(end_ids)]).strip()
        logits, caches = model(torch.tensor([ids[-1:]]), caches, len(ids) - 1)
    return tok.decode(ids[start:]).strip()


def accuracy(m, items, n=100):
    return np.mean([answer(m, q_) == a_ for q_, a_, _ in items[:n]])


VALID = {a_ for _, a_, _ in train_items}                                             # every answer the course has


def well_formed(m, items, n=100):
    """Share of answers that are *some* valid answer (a real title or module name), right or not: the format."""
    return np.mean([answer(m, q_) in VALID for q_, _, _ in items[:n]])


t0 = time.time()
rows = {}
full, n_full = train(copy.deepcopy(base), lr=1e-3)
rows["full fine-tuning"] = (n_full, accuracy(full, train_items), accuracy(full, heldout_items), well_formed(full, heldout_items),
                            val_loss(full, tok))
lora_models = {}
for r, epochs in ((1, 12), (4, 12), (16, 12), (4, 30)):
    m, n_tr = train(add_lora(base, r), lr=1e-2, epochs=epochs)                   # LoRA wants a ~10x higher learning rate
    lora_models[(r, epochs)] = m
    rows[f"LoRA r={r}" + (f", {epochs} epochs" if epochs != 12 else "")] = (
        n_tr, accuracy(m, train_items), accuracy(m, heldout_items), well_formed(m, heldout_items), val_loss(m, tok))
total = sum(p.numel() for p in base.parameters())
print(f"\ntrained in {time.time() - t0:.0f}s. Base model: {total:,} parameters; course-text val loss {val_loss(base, tok):.3f}")
print("                         trainable   % of model   exact (seen)   exact (unseen)   well-formed (unseen)   course val loss")
for k, (n_tr, a1, a2, wf, vl) in rows.items():
    print(f"  {k:22s} {n_tr:9,}   {100 * n_tr / total:9.2f}%   {a1:12.3f}   {a2:14.3f}   {wf:20.3f}   {vl:15.3f}")
print("capacity and budget matter: rank 1 and 4 learn little in 12 epochs (and forget little); rank 4 for 30 epochs")
share = rows["LoRA r=16"][1] / rows["full fine-tuning"][1]
print(f"learns more. Rank 16 learns {share:.0%} of what full fine-tuning memorizes on the training phrasings, and does")
print("better on phrasings it never saw: full fine-tuning memorized the templates. The low ranks barely disturb the base")
print("model's knowledge of the course text; rank 16, which learns the most, forgets almost as much as full fine-tuning.")
print("LoRA learns less and forgets less, in proportion to how much it's allowed to change (Biderman et al., 2024).")
assert rows["LoRA r=16"][1] > 0.4 and rows["LoRA r=16"][2] > rows["full fine-tuning"][2]
assert rows["LoRA r=4, 30 epochs"][1] > rows["LoRA r=4"][1]
assert all(rows[k][4] < rows["full fine-tuning"][4] - 0.5 for k in ("LoRA r=1", "LoRA r=4"))

# %% [markdown]
# ## 3. Merging

# %%
m = lora_models[(4, 12)]
merged = copy.deepcopy(m)
for block in merged.blocks:
    for name in ("qkv", "proj"):
        setattr(block, name, getattr(block, name).merged())
    for i in (0, 2):
        block.mlp[i] = block.mlp[i].merged()
x = torch.randint(0, V, (3, 40))
with torch.no_grad():
    diff = (m(x) - merged(x)).abs().max().item()
print(f"\nadapter model vs merged weights: max logit difference {diff:.1e}, and the merged model is a plain GPT again")
assert diff < 1e-4

# %% [markdown]
# ## 4. QLoRA in miniature

# %%
def quantize_int4(w, group=32):
    out, inp = w.shape
    wg = w.reshape(out, inp // group, group)
    s = wg.abs().amax(-1, keepdim=True) / 7
    return (torch.clamp(torch.round(wg / s), -8, 7) * s).reshape(out, inp)


q_base = copy.deepcopy(base)
with torch.no_grad():
    for name, lin in q_base.named_modules():
        if isinstance(lin, nn.Linear) and name != "head":
            lin.weight.copy_(quantize_int4(lin.weight))
q_lora, _ = train(add_lora(q_base, 16), lr=1e-2)
qrow = (accuracy(q_lora, train_items), accuracy(q_lora, heldout_items))
print(f"\nint4 base (val loss {val_loss(q_base, tok):.3f}) + LoRA r=16: exact match seen {qrow[0]:.3f}, unseen {qrow[1]:.3f} "
      f"(LoRA on the full-precision base: {rows['LoRA r=16'][1]:.3f}, {rows['LoRA r=16'][2]:.3f})")
print("the adapters train in full precision on top of a 4-bit base: that's how a 65B model is fine-tuned on one 48 GB GPU.")

# %% [markdown]
# ## 5. Memory for a 7B model

# %%
N = 7e9
full_bytes = 16 * N                                                                   # 33.1: mixed precision + Adam
lora_params = 40e6                                                                    # rank 16 on all linear layers, roughly
lora_bytes = 2 * N + 16 * lora_params
qlora_bytes = 0.5 * 1.03 * N + 16 * lora_params
print(f"\n7B, before activations: full fine-tuning {full_bytes / 1e9:.0f} GB; LoRA (bf16 base) {lora_bytes / 1e9:.0f} GB; "
      f"QLoRA (4-bit base) {qlora_bytes / 1e9:.1f} GB")
print("and every adapter is a few tens of MB: one base model in memory can serve many fine-tunes, swapped per request.")

print("\nAll checks passed.")
