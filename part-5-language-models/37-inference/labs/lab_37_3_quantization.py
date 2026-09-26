# %% [markdown]
# # Lab 37.3: Quantization
#
# The shared tinygpt model, its weights quantized several ways, each measured by validation loss (the only number
# that matters is what quantization does to the model, not to the weights):
#
# 1. Round-to-nearest (RTN) int8: per-tensor vs per-output-channel scales.
# 2. Int4 and int3 with per-group scales, and why group size matters.
# 3. Outliers: one large weight ruins a per-tensor scale; outlier activations are why activation quantization is hard.
# 4. GPTQ: quantize column by column and push each column's rounding error onto the columns not yet quantized, using
#    second-order information from calibration data.
# 5. Memory and speed arithmetic for real models.

# %%
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from tinygpt import corpus_split, load_or_train, val_loss  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(373)
model, tok = load_or_train()
base_loss = val_loss(model, tok)
linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and n != "head"]   # head is tied to the embedding
print(f"fp32 validation loss {base_loss:.4f}; quantizing {len(linears)} linear layers "
      f"({sum(m.weight.numel() for _, m in linears):,} weights), embeddings kept in full precision")


def quantize(w, bits, axis=None, group=None):
    """Symmetric round-to-nearest. axis=None: one scale per tensor; axis=1: one per output row; group: per row, per
    block of `group` input columns. Returns the dequantized weights (what the matmul effectively uses)."""
    qmax = 2 ** (bits - 1) - 1
    if group:
        out, inp = w.shape
        wg = w.reshape(out, inp // group, group)
        scale = wg.abs().amax(-1, keepdim=True) / qmax
        return (torch.clamp(torch.round(wg / scale), -qmax - 1, qmax) * scale).reshape(out, inp)
    scale = (w.abs().max() if axis is None else w.abs().amax(axis, keepdim=True)) / qmax
    return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


def quantized_model(fn):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for name, lin in m.named_modules():
            if isinstance(lin, nn.Linear) and name != "head":
                lin.weight.copy_(fn(lin.weight))
    return m


# %% [markdown]
# ## 1 and 2. Round-to-nearest

# %%
configs = {
    "int8, per tensor": lambda w: quantize(w, 8),
    "int8, per output channel": lambda w: quantize(w, 8, axis=1),
    "int4, per output channel": lambda w: quantize(w, 4, axis=1),
    "int4, groups of 32": lambda w: quantize(w, 4, group=32),
    "int3, per output channel": lambda w: quantize(w, 3, axis=1),
    "int3, groups of 32": lambda w: quantize(w, 3, group=32),
}
print("\nweights                        bits/weight   validation loss   increase")
rtn = {}
for name, fn in configs.items():
    bits = int(name[3])
    overhead = 16 / 32 if "groups" in name else 0                                      # an fp16 scale per 32 weights
    rtn[name] = val_loss(quantized_model(fn), tok)
    print(f"  {name:28s} {bits + overhead:11.2f}   {rtn[name]:15.4f}   {rtn[name] - base_loss:+8.4f}")
print("int8 is nearly free. At 4 bits, one scale per row is too coarse: a row's largest weight sets the step size for")
print("all of them. Per-group scales cost half a bit per weight and buy back about a third of the damage here.")
assert rtn["int8, per output channel"] - base_loss < 0.01
assert rtn["int4, groups of 32"] < rtn["int4, per output channel"] and rtn["int3, groups of 32"] < rtn["int3, per output channel"]

# %% [markdown]
# ## 3. Outliers

# %%
w = model.blocks[0].mlp[0].weight.detach().clone()
w_out = w.clone(); w_out[0, 0] = w.abs().max() * 20                                   # one weight 20x the largest
for name, ww in (("normal", w), ("with one outlier", w_out)):
    e_t = (quantize(ww, 8) - ww).pow(2).mean().item()
    e_c = (quantize(ww, 8, axis=1) - ww).pow(2).mean().item()
    print(f"\nint8 reconstruction error, {name:17s}: per tensor {e_t:.2e}, per channel {e_c:.2e}")
print("one outlier stretches a shared scale and rounds everything else toward zero; finer-grained scales contain it.")

# activation outliers: capture the input to one layer, add a few large feature dimensions, quantize per tensor
acts = []
h = model.blocks[2].mlp[0].register_forward_hook(lambda mod, i, o: acts.append(i[0].detach()))
_, val_text = corpus_split()
ids = torch.tensor([tok.encode(val_text[:3000])[:128]])
with torch.no_grad():
    model(ids)
h.remove()
x = acts[0][0]
x_out = x.clone(); x_out[:, :3] *= 30                                                 # a few "massive activation" features
for name, xx in (("typical activations", x), ("3 outlier features", x_out)):
    err = ((quantize(xx, 8) - xx).pow(2).mean() / xx.pow(2).mean()).item()
    print(f"int8 activations, per tensor, {name:20s}: relative error {err:.2e}")
print("large LLMs develop a few feature dimensions with huge activations (Dettmers et al., 2022). Weights can be")
print("quantized offline with any scheme; activations change per token, which is why weight-only int4 is the common case.")
assert (quantize(w_out, 8) - w_out).pow(2).mean() > 10 * (quantize(w_out, 8, axis=1) - w_out).pow(2).mean()

# %% [markdown]
# ## 4. GPTQ
#
# For a linear layer with weight W and calibration inputs X, minimize ||W X - Q X||^2 over quantized Q. GPTQ (Frantar
# et al., 2023) processes the input columns in order: quantize column j, then adjust the not-yet-quantized columns to
# compensate for its error, using the inverse Hessian H^-1, H = X^T X. Here with per-group scales computed up front.

# %%
calib = []
hooks = [m.register_forward_hook(lambda mod, i, o, n=n: calib.append((n, i[0].detach().reshape(-1, i[0].shape[-1]))))
         for n, m in linears]
train_text, _ = corpus_split()
tids = tok.encode(train_text[:200000])
with torch.no_grad():
    for s in range(0, 16 * 128, 128):                                                 # 16 calibration sequences
        model(torch.tensor([tids[s:s + 128]]))
for h_ in hooks:
    h_.remove()
X_by_layer = {}
for n, x_ in calib:
    X_by_layer.setdefault(n, []).append(x_)
X_by_layer = {n: torch.cat(v).double() for n, v in X_by_layer.items()}


def gptq(W, X, bits, group, damp=0.01):
    W = W.clone().double()
    qmax = 2 ** (bits - 1) - 1
    H = X.T @ X
    H += damp * H.diag().mean() * torch.eye(H.shape[0], dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)   # upper Cholesky of H^-1
    Q = torch.zeros_like(W)
    for j in range(W.shape[1]):
        if j % group == 0:                                                           # scales for the next group of columns
            scale = W[:, j:j + group].abs().amax(1) / qmax
        q = torch.clamp(torch.round(W[:, j] / scale), -qmax - 1, qmax) * scale
        Q[:, j] = q
        err = (W[:, j] - q) / Hinv[j, j]
        W[:, j + 1:] -= err[:, None] * Hinv[j, j + 1:][None, :]                      # compensate in the remaining columns
    return Q.float()


def gptq_model(bits, group):
    m = copy.deepcopy(model)
    mods = dict(m.named_modules())
    with torch.no_grad():
        for n, _ in linears:
            mods[n].weight.copy_(gptq(mods[n].weight, X_by_layer[n], bits, group))
    return m


def layer_error(n, Q):
    W, X = dict(model.named_modules())[n].weight.double(), X_by_layer[n]
    return ((X @ (W - Q.double()).T).pow(2).mean() / (X @ W.T).pow(2).mean()).item()


n0 = linears[4][0]
W0 = dict(model.named_modules())[n0].weight.detach()
print(f"\nlayer '{n0}', int3 groups of 32, relative output error on calibration data: "
      f"RTN {layer_error(n0, quantize(W0, 3, group=32)):.4f}, GPTQ {layer_error(n0, gptq(W0, X_by_layer[n0], 3, 32)):.4f}")
res_gptq = {}
for bits in (4, 3):
    res_gptq[bits] = val_loss(gptq_model(bits, 32), tok)
    print(f"int{bits}, groups of 32: RTN {rtn[f'int{bits}, groups of 32']:.4f}   GPTQ {res_gptq[bits]:.4f}   (fp32 {base_loss:.4f})")
print("same bits, same format, better choice of which way to round: the error of each column is steered into directions")
print("the calibration data says the layer doesn't care about. The gain grows as bits shrink.")
assert res_gptq[3] < rtn["int3, groups of 32"]

# %% [markdown]
# ## 5. What it buys

# %%
print("\nweights only; decoding at batch 1 reads every weight once per token, so tokens/s <= bandwidth / model size")
for name, n in (("8B", 8e9), ("70B", 70e9)):
    for fmt, bpw in (("bf16", 16), ("int8", 8), ("int4 (g128)", 4.125)):
        gb = n * bpw / 8 / 1e9
        print(f"  {name:4s} {fmt:12s} {gb:7.1f} GB   H100 (3.35 TB/s) <= {3350 / gb:6.0f} tokens/s   "
              f"laptop (100 GB/s) <= {100 / gb:5.1f} tokens/s")
print("4-bit weights make a 70B model fit on one 80 GB GPU (with room for a small KV cache) and run 3-4x faster than")
print("bf16 at batch 1, because decoding is memory-bound (33.1). The quality cost has to be measured on your task.")

print("\nAll checks passed.")
