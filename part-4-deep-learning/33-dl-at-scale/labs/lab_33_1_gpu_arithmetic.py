# %% [markdown]
# # Lab 33.1: The arithmetic of training
#
# No GPU needed: the accounting is the same on any hardware, and this is where budgets are won or lost.
#
# 1. Memory for weights, gradients and optimizer state: measured on a real model, against the formula.
# 2. Activation memory: count the bytes autograd saves for the backward pass, with and without checkpointing.
# 3. FLOPs: count them with PyTorch's FlopCounterMode, against 6 N per token (plus attention).
# 4. Arithmetic intensity: measure this machine's roofline with matmuls and elementwise ops.
# 5. Back-of-the-envelope numbers for real models and real GPUs.

# %%
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.flop_counter import FlopCounterMode

torch.manual_seed(331)
torch.set_num_threads(4)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, n, d = x.shape
        q, k, v = (t.view(B, n, self.h, d // self.h).transpose(1, 2) for t in self.qkv(self.ln1(x)).chunk(3, -1))
        a = (q @ k.transpose(-2, -1) / (d // self.h) ** 0.5).masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
        x = x + self.proj((a.softmax(-1) @ v).transpose(1, 2).reshape(B, n, d))      # plain attention: the n x n matrix exists
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, V=1000, ctx=256, d=256, h=4, L=4, ckpt=False):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
        self.ckpt = ckpt

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if self.ckpt else b(x)
        return self.head(self.ln(x))


def nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


# %% [markdown]
# ## 1. Weights, gradients, optimizer state

# %%
model = GPT()
N = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
idx = torch.randint(0, 1000, (8, 256))
F.cross_entropy(model(idx).reshape(-1, 1000), idx.reshape(-1)).backward()
opt.step()
w = nbytes(model.parameters())
g = nbytes(p.grad for p in model.parameters())
o = nbytes(t for st in opt.state.values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1)
print(f"parameters N = {N:,}")
print(f"weights {w / 1e6:.1f} MB, gradients {g / 1e6:.1f} MB, AdamW state (m and v) {o / 1e6:.1f} MB")
print(f"total {(w + g + o) / N:.1f} bytes per parameter (fp32 everything: 4 + 4 + 8 = 16)")
assert abs((w + g + o) / N - 16) < 0.01

# %% [markdown]
# ## 2. Activations
#
# Everything autograd keeps for the backward pass goes through saved_tensors_hooks. Count it. (Some saved tensors
# share storage, so this is an upper bound; it's how people estimate activation memory anyway.)

# %%
def saved_bytes(model, idx):
    total = [0]
    weights = {p.data_ptr() for p in model.parameters()}          # linear layers also "save" their weights: not activations

    def pack(t):
        if t.data_ptr() not in weights:
            total[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = F.cross_entropy(model(idx).reshape(-1, 1000), idx.reshape(-1))
    loss.backward()
    return total[0]


print("\nactivation bytes saved for backward, batch 8:")
print("  context   plain       per token   checkpointed   ratio")
act = {}
for n in (64, 128, 256):
    x = torch.randint(0, 1000, (8, n))
    plain = saved_bytes(GPT(), x)
    ck = saved_bytes(GPT(ckpt=True), x)
    act[n] = (plain, ck)
    print(f"  {n:7d}   {plain / 1e6:6.1f} MB   {plain / (8 * n) / 1e3:6.1f} kB    {ck / 1e6:6.1f} MB      {plain / ck:.1f}x")
per_tok = [act[n][0] / (8 * n) for n in (64, 128, 256)]
print("per token, activation memory grows with the context: the n x n attention matrices (4 heads x n numbers per token,")
print("several copies per layer) are the part that isn't linear. FlashAttention (29.3) never stores them.")
print("checkpointing keeps only each block's input and recomputes the rest during backward: roughly one extra forward pass.")
assert per_tok[2] > per_tok[0] * 1.2 and act[256][0] / act[256][1] > 5

# %% [markdown]
# ## 3. FLOPs

# %%
def count_flops(model, idx, backward=True):
    with FlopCounterMode(display=False) as fc:
        loss = F.cross_entropy(model(idx).reshape(-1, 1000), idx.reshape(-1))
        if backward:
            loss.backward()
    return fc.get_total_flops()


m = GPT()
L_, d_, n_ = 4, 256, 256
N_nonemb = sum(p.numel() for n, p in m.named_parameters() if "blocks" in n) + m.head.weight.numel()   # the matmul weights
tokens = 8 * n_
fwd = count_flops(m, idx, backward=False)
total = count_flops(m, idx)
attn_fwd = 4 * n_ ** 2 * d_ * L_ * 8                   # QK^T and AV: 2 matmuls x 2 n^2 d FLOPs per layer, 8 sequences
print(f"\nforward FLOPs counted {fwd / 1e9:.2f} G; 2 N per token = {2 * N_nonemb * tokens / 1e9:.2f} G, "
      f"+ attention scores {attn_fwd / 1e9:.2f} G = {(2 * N_nonemb * tokens + attn_fwd) / 1e9:.2f} G")
print(f"forward + backward counted {total / 1e9:.2f} G; 3x forward = {3 * fwd / 1e9:.2f} G  (backward costs about twice the forward)")
assert abs(fwd - (2 * N_nonemb * tokens + attn_fwd)) / fwd < 0.05
assert abs(total / fwd - 3) < 0.1

# %% [markdown]
# ## 4. Arithmetic intensity on this machine
#
# FLOPs per byte moved. A square matmul of size n does 2 n^3 FLOPs on 3 n^2 numbers: intensity grows with n. An
# elementwise add does 1 FLOP per 12 bytes (two reads, one write, fp32), however big it is.

# %%
def bench(fn, reps=5):
    """Best of `reps` timings: the least disturbed run is the closest to what the hardware can do."""
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


print("\n   op                     intensity (FLOP/byte)   achieved GFLOP/s   achieved GB/s")
roof = {}
for n in (32, 128, 512, 2048):
    a, b = torch.randn(n, n), torch.randn(n, n)
    t = bench(lambda: a @ b, reps=max(5, int(2e8 / n ** 3)))
    flops, byts = 2 * n ** 3, 3 * n * n * 4
    roof[f"matmul {n}"] = flops / t / 1e9
    print(f"   matmul {n:5d}x{n:<5d}   {flops / byts:21.1f}   {flops / t / 1e9:16.1f}   {byts / t / 1e9:13.1f}")
for n in (10 ** 6, 2 * 10 ** 7):
    a, b = torch.randn(n), torch.randn(n)
    out = torch.empty(n)
    t = bench(lambda: torch.add(a, b, out=out))
    roof[f"add {n}"] = n / t / 1e9
    print(f"   add, {n:>10,} floats   {1 / 12:21.2f}   {n / t / 1e9:16.1f}   {12 * n / t / 1e9:13.1f}")
print("small matmuls and every elementwise op are limited by memory traffic (and overheads), not arithmetic: the")
print("FLOP/s they reach is a small fraction of what a big matmul gets. That's the roofline, and it's why fusion matters.")
big = max(roof["matmul 512"], roof["matmul 2048"])
assert big > 5 * roof["matmul 32"] and big > 10 * roof[f"add {2 * 10 ** 7}"]

# %% [markdown]
# ## 5. Real models, real hardware
#
# Mixed-precision training with Adam: bf16 weights (2) + bf16 gradients (2) + fp32 master weights (4) + fp32 Adam m
# and v (8) = 16 bytes per parameter, before activations. (Some setups keep fp32 gradients: 18.)

# %%
print("\nmemory before activations, mixed precision + Adam, 16 bytes per parameter:")
for name, n_params in (("GPT-2 small", 124e6), ("7B", 7e9), ("70B", 70e9)):
    print(f"  {name:12s} {16 * n_params / 1e9:8.0f} GB   (inference in bf16: {2 * n_params / 1e9:6.1f} GB)")
print("a 7B model needs about 112 GB to fine-tune fully with Adam: more than one 80 GB GPU, before a single activation.")

# Training time for a 7B model on 2T tokens, 1,024 H100s, at 40% model FLOPs utilization (MFU)
flops = 6 * 7e9 * 2e12
peak = 989e12                                   # H100 SXM, dense bf16
days = flops / (1024 * peak * 0.4) / 86400
print(f"\n6 N D for 7B params, 2T tokens: {flops:.2e} FLOPs. On 1,024 H100s at 40% MFU: {days:.1f} days")
print(f"H100 ridge point: {peak:.3g} FLOP/s / 3.35e12 B/s = {peak / 3.35e12:.0f} FLOP/byte. Below that, you're memory-bound.")
print("decoding one token at a time with batch 1 does about 1 FLOP per byte of weights read: far below the ridge.")
print("that's why LLM inference is memory-bound, and why batching (37.2) and quantization (37.3) help so much.")

print("\nAll checks passed.")
