# %% [markdown]
# # Lab 45.2: Self-hosting: sizing, throughput and the break-even
#
# 1. Memory: weights at each precision and the KV cache per token, from real model configurations. How many
#    concurrent sequences fit on a GPU?
# 2. Throughput: decoding is memory-bound, so batching is nearly free until compute catches up. Measured on the
#    course's tiny GPT on this CPU, then estimated for real GPUs with the roofline (33.1, 37.2).
# 3. Money: cost per million tokens against utilization, and where self-hosting beats an API.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "part-5-language-models" / "_shared"))
from tinygpt import load_or_train  # noqa: E402

torch.set_num_threads(4)

# %% [markdown]
# ## 1. Memory
#
# Configurations as published in the models' config files (layers, KV heads, head dimension). The KV cache stores a
# key and a value vector per layer, per KV head, per token: 2 x layers x kv_heads x head_dim x bytes.

# %%
MODELS = {  # parameters, layers, KV heads, head dim
    "Llama 3.1 8B": (8.0e9, 32, 8, 128),
    "Llama 3.1 70B": (70.6e9, 80, 8, 128),
    "Mistral 7B": (7.2e9, 32, 8, 128),
}
GPUS = {"L4 24 GB": (24e9, 300e9, 121e12), "A100 80 GB": (80e9, 2.0e12, 312e12), "H100 80 GB": (80e9, 3.35e12, 990e12)}
# memory bytes, memory bandwidth bytes/s, dense bf16 FLOP/s (vendor figures)

print(f"{'model':15s} {'weights bf16':>13s} {'int8':>8s} {'int4':>8s} {'KV per token':>13s} {'KV, 8k tokens':>14s}")
for name, (n, layers, kvh, hd) in MODELS.items():
    kv_tok = 2 * layers * kvh * hd * 2                                               # bf16 cache
    print(f"{name:15s} {2 * n / 1e9:12.1f}G {n / 1e9:7.1f}G {0.5 * n / 1e9:7.1f}G {kv_tok / 1e3:11.0f} kB {kv_tok * 8192 / 1e9:12.2f} GB")


def max_sequences(model, gpu, ctx, bytes_per_weight=2, overhead=0.10):
    n, layers, kvh, hd = MODELS[model]
    mem = GPUS[gpu][0] * (1 - overhead) - n * bytes_per_weight                         # ~10% for activations, runtime
    kv_seq = 2 * layers * kvh * hd * 2 * ctx
    return max(0, int(mem // kv_seq))


print("\nconcurrent sequences that fit (4,096-token contexts):")
for model, gpu, bpw in (("Llama 3.1 8B", "L4 24 GB", 2), ("Llama 3.1 8B", "L4 24 GB", 0.5), ("Llama 3.1 8B", "H100 80 GB", 2),
                        ("Llama 3.1 70B", "H100 80 GB", 2), ("Llama 3.1 70B", "H100 80 GB", 0.5)):
    k = max_sequences(model, gpu, 4096, bpw)
    print(f"  {model:14s} {'bf16' if bpw == 2 else 'int4':4s} on one {gpu:11s}: {k:4d}" + ("   (doesn't fit)" if k == 0 else ""))
print("grouped-query attention (8 KV heads instead of 32 or 64) is what keeps these caches affordable. Long contexts")
print("eat concurrency: the cache, not the weights, is usually what limits how many users one GPU can serve.")
assert max_sequences("Llama 3.1 70B", "H100 80 GB", 4096, 2) == 0 and max_sequences("Llama 3.1 70B", "H100 80 GB", 4096, 0.5) > 0

# %% [markdown]
# ## 2. Throughput: batching

# %%
model, tok = load_or_train(verbose=False)
prompt = torch.tensor([tok.encode("The KV cache stores")])


@torch.no_grad()
def decode_throughput(batch, steps=48):
    ids = prompt.repeat(batch, 1)
    logits, caches = model(ids, [None] * len(model.blocks), 0)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    t0 = time.perf_counter()
    for s in range(steps):
        logits, caches = model(nxt, caches, ids.shape[1] + s)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
    return batch * steps / (time.perf_counter() - t0)


decode_throughput(1, 8)                                                                 # warm up
print(f"\ncourse tiny GPT ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters), decoding on this CPU:")
tp = {}
for b in (1, 4, 16, 64):
    tp[b] = max(decode_throughput(b) for _ in range(3))
    print(f"  batch {b:3d}: {tp[b]:8,.0f} tokens/s   ({tp[b] / tp[1]:5.1f}x batch 1)")
print("one sequence at a time leaves the hardware mostly idle; each step reads all the weights to produce one token per")
print("sequence, so more sequences per step are almost free until arithmetic becomes the limit. Continuous batching")
print("(37.2) keeps the batch full as requests come and go; it's the main reason vLLM-style servers exist.")
assert tp[16] > 3 * tp[1]

# roofline estimate for a real GPU: per decode step, read the weights once (memory) and do 2 FLOPs per parameter
# per sequence (compute); the step takes the larger of the two times
print("\nestimated decode throughput, Llama 3.1 8B bf16 on one H100 (roofline, ignoring the KV cache reads):")
n = MODELS["Llama 3.1 8B"][0]
mem_bytes, bw, flops = GPUS["H100 80 GB"]
for b in (1, 8, 64, 256):
    t_step = max(2 * n / bw, 2 * n * b / flops)
    print(f"  batch {b:3d}: {b / t_step:8,.0f} tokens/s")
print("real servers reach a good fraction of these at moderate batch sizes; KV-cache reads (which grow with batch and")
print("context), scheduling and prefill take the rest.")

# %% [markdown]
# ## 3. The break-even

# %%
gpu_hour = 3.00                                                                        # illustrative on-demand H100 price
realistic_peak = 5000                                                                  # output tokens/s sustained under real, mixed traffic: well below the roofline
api_price = {"small API model": 0.50, "mid-size API model": 5.00}                      # $/M output tokens, illustrative
print(f"\none GPU at ${gpu_hour:.2f}/hour, sustaining {realistic_peak:,} tokens/s when busy:")
print("  utilization   $ per million tokens")
for u in (0.05, 0.2, 0.5, 0.9):
    per_m = gpu_hour / (realistic_peak * u * 3600 / 1e6)
    print(f"  {u:11.0%}   {per_m:8.2f}")
print(f"API prices for comparison: " + ", ".join(f"{k} ${v:.2f}/M" for k, v in api_price.items()))
print("self-hosting is cheap per token only when the GPU is busy. At low or bursty utilization you pay for idle")
print("hardware, plus the engineers who run it. It wins on steady high volume, data that can't leave your network,")
print("latency you control, or a fine-tuned small model replacing a large API model; rarely on cost alone at low volume.")

print("\nAll checks passed.")
