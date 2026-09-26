# %% [markdown]
# # Lab 49.3: Making models small and fast
#
# One large model (a 1024x1024 MLP, 1.1M parameters) on a 10-class tabular task, and the three standard ways to make
# it cheaper to serve:
# 1. Distillation: train a small student on the teacher's outputs. When does it beat training the student on labels?
# 2. Quantization: int8 and int4 weights, per-tensor vs per-channel scales, accuracy and size; and the speed of int8
#    kernels where PyTorch provides them.
# 3. Pruning: zeroing small weights (unstructured) vs removing whole neurons (structured). Which one is faster?
# 4. Latency at batch 1 and batch 256 for each variant.
# ONNX Runtime and TensorRT are covered in the lesson; they aren't installed here, and the ideas are the same.

# %%
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from torch import nn

torch.set_num_threads(1)                                                        # timings on one core, comparable
torch.manual_seed(0)
X, y = make_classification(n_samples=70_000, n_features=64, n_informative=32, n_redundant=16, n_classes=10,
                           n_clusters_per_class=3, class_sep=1.2, flip_y=0.02, random_state=0)
X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
Xtr, ytr, Xte, yte = map(torch.tensor, (X[:50_000], y[:50_000], X[60_000:], y[60_000:]))


def mlp(h):
    return nn.Sequential(nn.Linear(64, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 10))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def acc(m):
    return (m(Xte).argmax(1) == yte).float().mean().item()


def train(m, X, y=None, soft=None, epochs=20, lr=1e-3, T=4.0, alpha=0.9, seed=0, masks=None):
    """Cross-entropy on labels (y = -1 means unlabeled), plus distillation on the teacher's logits if given:
    KL between softened distributions, scaled by T^2 (Hinton et al., 2015)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(X) + 255) // 256))
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=g).split(256):
            out = m(X[idx])
            loss = torch.zeros(())
            if soft is not None:
                loss = alpha * T * T * F.kl_div(F.log_softmax(out / T, 1), F.softmax(soft[idx] / T, 1), reduction="batchmean")
            if y is not None and (y[idx] >= 0).any():
                lab = y[idx]; keep = lab >= 0
                loss = loss + (1 - alpha if soft is not None else 1.0) * F.cross_entropy(out[keep], lab[keep])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if masks:                                                            # pruned weights stay pruned
                with torch.no_grad():
                    for p, mk in masks.items():
                        p.mul_(mk)
    return m


# %% [markdown]
# ## 1. Distillation

# %%
t0 = time.time()
teacher = train(mlp(1024), Xtr, ytr, epochs=15)
with torch.no_grad():
    teacher_logits = teacher(Xtr)
few = ytr.clone(); few[2000:] = -1                                               # only the first 2,000 labels kept
rows = {
    "teacher, 50k labels": teacher,
    "student, 50k labels": train(mlp(256), Xtr, ytr, epochs=30),
    "student, 50k labels + teacher": train(mlp(256), Xtr, ytr, soft=teacher_logits, epochs=30),
    "student, 2k labels": train(mlp(256), Xtr[:2000], ytr[:2000], epochs=200),
    "student, 2k labels + teacher on 48k unlabeled": train(mlp(256), Xtr, few, soft=teacher_logits, epochs=30),
}
dist = {k: acc(m) for k, m in rows.items()}
print(f"teacher: {n_params(teacher):,} parameters; student: {n_params(rows['student, 50k labels']):,} "
      f"({n_params(teacher) / n_params(rows['student, 50k labels']):.0f}x fewer)   ({time.time() - t0:.0f}s)")
for k, v in dist.items():
    print(f"  {k:48s} test accuracy {v:.3f}")
print("with the same 50k labels, the teacher's soft targets add nothing here: the student is limited by its size, not")
print("by its supervision, and on this data the teacher's 'dark knowledge' (which wrong classes are nearly right) is")
print("thin. Distillation's big win is the last row: a teacher labeling data nobody labeled. With 2,000 labels the")
print("student reaches 0.53; with the same labels plus the teacher's outputs on 48,000 unlabeled inputs, it almost matches")
print("the fully supervised student. That's how most production distillation works: a large model (often an LLM,")
print("prompted) labels your traffic, and a small model trained on those labels serves it for a fraction of the cost.")
assert dist["student, 2k labels + teacher on 48k unlabeled"] > dist["student, 2k labels"] + 0.2
assert dist["teacher, 50k labels"] > dist["student, 50k labels"]
student = rows["student, 50k labels"]

# %% [markdown]
# ## 2. Quantization
#
# Symmetric integer quantization of the weights: w ~ s * q, q an integer in [-(2^(b-1) - 1), 2^(b-1) - 1], one scale s
# per tensor, per output row (channel), or per group of 32 weights. Weights only; activations stay float here.

# %%
def quantize(w, bits, granularity):
    qmax = 2 ** (bits - 1) - 1
    if granularity == "tensor":
        s = (w.abs().max() / qmax).clamp_min(1e-12)
    elif granularity == "channel":
        s = (w.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-12)
    else:                                                                         # groups of 32 along each row
        g = w.reshape(w.shape[0], -1, 32)
        s = (g.abs().amax(2, keepdim=True) / qmax).clamp_min(1e-12)
        return (torch.clamp(torch.round(g / s), -qmax, qmax) * s).reshape(w.shape), s.numel()
    return torch.clamp(torch.round(w / s), -qmax, qmax) * s, s.numel()


def quantized_copy(m, bits, granularity):
    q = mlp(m[0].out_features)
    q.load_state_dict(m.state_dict())
    size = 0
    with torch.no_grad():
        for layer in q:
            if isinstance(layer, nn.Linear):
                w, n_scales = quantize(layer.weight, bits, granularity)
                layer.weight.copy_(w)
                size += layer.weight.numel() * bits / 8 + n_scales * 2 + layer.bias.numel() * 4   # fp16 scales
    return q, size


print(f"\n{'teacher weights':32s} {'size':>9s} {'test accuracy':>14s}")
fp32_size = n_params(teacher) * 4
print(f"  {'float32':30s} {fp32_size / 1e6:7.2f}MB {acc(teacher):14.3f}")
quant = {}
for bits, gran in ((8, "tensor"), (8, "channel"), (4, "tensor"), (4, "channel"), (4, "group")):
    qm, size = quantized_copy(teacher, bits, gran)
    quant[(bits, gran)] = acc(qm)
    print(f"  {f'int{bits}, one scale per {gran}':30s} {size / 1e6:7.2f}MB {quant[(bits, gran)]:14.3f}")
print("int8 is free: a quarter of the size and the same accuracy, whatever the granularity. At 4 bits there are only")
print("15 levels, and one scale per tensor spends them on the range of the largest weight; finer scales (per channel,")
print("per group, as GPTQ/AWQ and llama.cpp's formats use) recover the loss for a little extra storage. The loss is")
print("small on this model; on LLMs, whose weights and activations have large outliers, granularity decides whether 4-bit")
print("works at all.")
assert abs(quant[(8, "channel")] - acc(teacher)) < 0.01
assert quant[(4, "group")] >= quant[(4, "tensor")]

# the size is only half the point; speed needs integer kernels. PyTorch's dynamic int8 quantization provides them for
# Linear layers on CPU (weights int8, activations quantized on the fly). It's deprecated in favor of the torchao
# package, so this part runs only if it's still there.
int8_teacher = None
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torch.ao.quantization import quantize_dynamic
        int8_teacher = quantize_dynamic(teacher, {nn.Linear}, dtype=torch.qint8)
        print(f"PyTorch dynamic int8 teacher: test accuracy {acc(int8_teacher):.3f}")
except Exception as e:                                                             # noqa: BLE001
    print(f"(PyTorch dynamic quantization not available: {type(e).__name__}; skipping its timings)")

# %% [markdown]
# ## 3. Pruning

# %%
def magnitude_masks(m, sparsity):
    """Global magnitude pruning: zero the smallest weights across all layers (Han et al., 2015)."""
    ws = [l.weight for l in m if isinstance(l, nn.Linear)]
    thresh = torch.quantile(torch.cat([w.detach().abs().flatten() for w in ws]), sparsity)
    return {w: (w.detach().abs() > thresh).float() for w in ws}


def copy_of(m):
    c = mlp(m[0].out_features); c.load_state_dict(m.state_dict()); return c


print(f"\n{'unstructured pruning of the teacher':38s} {'no fine-tuning':>15s} {'2 epochs fine-tuning':>21s}")
prune = {}
for sp in (0.5, 0.8, 0.95):
    m = copy_of(teacher)
    masks = magnitude_masks(m, sp)
    with torch.no_grad():
        for w, mk in masks.items():
            w.mul_(mk)
    before = acc(m)
    train(m, Xtr, ytr, epochs=2, lr=3e-4, masks=masks)
    prune[sp] = (before, acc(m))
    print(f"  {f'{sp:.0%} of weights zeroed':36s} {before:15.3f} {prune[sp][1]:21.3f}")


def structured(m, keep):
    """Remove whole hidden units: keep the `keep` units of each hidden layer with the largest outgoing weight norms."""
    new = mlp(keep)
    l1, l2, l3 = m[0], m[2], m[4]
    k1 = l2.weight.detach().norm(dim=0).argsort(descending=True)[:keep]         # units of layer 1, by their use in layer 2
    k2 = l3.weight.detach().norm(dim=0).argsort(descending=True)[:keep]
    with torch.no_grad():
        new[0].weight.copy_(l1.weight[k1]); new[0].bias.copy_(l1.bias[k1])
        new[2].weight.copy_(l2.weight[k2][:, k1]); new[2].bias.copy_(l2.bias[k2])
        new[4].weight.copy_(l3.weight[:, k2]); new[4].bias.copy_(l3.bias)
    return new


small = structured(teacher, 256)
before = acc(small)
train(small, Xtr, ytr, epochs=5, lr=5e-4)
structured_acc = acc(small)
print(f"  structured: 1024 -> 256 units per layer ({n_params(small):,} parameters): {before:.3f} before fine-tuning, "
      f"{structured_acc:.3f} after 5 epochs")
print(f"  (the 256-unit student trained from scratch: {dist['student, 50k labels']:.3f})")
print("half the weights can be zeroed with no loss, and a short fine-tune recovers most of what's lost at high sparsity")
print("(at 80% it even ends above the teacher: those two epochs are also two more epochs of training).")
print("But a matrix with 90% zeros stored densely is exactly as big and as slow as before: unstructured sparsity needs")
print("sparse formats and kernels to pay off, and on CPUs and GPUs those win only at very high sparsity or with special")
print("patterns (NVIDIA's 2:4). Removing whole units gives a smaller dense model that's faster everywhere; after")
print("fine-tuning it's about as good as training that size from scratch, which is the honest comparison to make.")
assert prune[0.5][0] > acc(teacher) - 0.02 and prune[0.95][1] > prune[0.95][0]

# %% [markdown]
# ## 4. Latency

# %%
def latency(models, batch, rounds=9, calls=200):
    """Best time per call for each model. The models take turns in every round, so a noisy stretch on a shared
    machine hits all of them instead of whichever happened to be measured then; the minimum is the least noisy."""
    x = Xte[:batch]
    n = calls if batch == 1 else max(5, calls // 20)
    best = {k: float("inf") for k in models}
    with torch.no_grad():
        for m in models.values():
            for _ in range(20):
                m(x)
        for _ in range(rounds):
            for k, m in models.items():
                t0 = time.perf_counter()
                for _ in range(n):
                    m(x)
                best[k] = min(best[k], (time.perf_counter() - t0) / n)
    return best

variants = {"teacher, float32": teacher, "teacher, 95% unstructured sparsity": copy_of(teacher),
            "structured-pruned (256 units)": small, "student (256 units)": student}
with torch.no_grad():
    for w, mk in magnitude_masks(variants["teacher, 95% unstructured sparsity"], 0.95).items():
        w.mul_(mk)
if int8_teacher is not None:
    variants["teacher, int8 dynamic quantization"] = int8_teacher
print(f"\n{'one core':38s} {'batch 1, us':>12s} {'batch 256, us per row':>22s}")
b1, b256 = latency(variants, 1), latency(variants, 256)
lat = {k: (b1[k] * 1e6, b256[k] * 1e6 / 256) for k in variants}
for name in variants:
    print(f"  {name:36s} {lat[name][0]:12.1f} {lat[name][1]:22.2f}")
ratio = n_params(teacher) / n_params(student)
t, st = lat["teacher, float32"], lat["student (256 units)"]
print(f"the {ratio:.0f}x smaller model is {t[0] / st[0]:.0f}x faster at batch 1 and {t[1] / st[1]:.0f}x per row at batch 256: at batch 1")
print("a fixed cost per call (Python, dispatch, allocation) keeps small models from getting proportionally faster.")
print("Zeros in a dense matrix cost the same as any other number. int8 kernels speed up the big model at both batch")
print("sizes, with the same accuracy, which makes them the cheapest win on this list when they're available. Measure")
print("on the hardware and batch size you'll serve at: the ranking changes with both.")
assert lat["student (256 units)"][1] < lat["teacher, float32"][1] / 3
assert abs(lat["teacher, 95% unstructured sparsity"][1] / lat["teacher, float32"][1] - 1) < 0.5

print("\nAll checks passed.")
