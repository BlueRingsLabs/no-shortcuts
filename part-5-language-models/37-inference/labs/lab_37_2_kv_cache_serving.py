# %% [markdown]
# # Lab 37.2: KV cache, batching and serving
#
# 1. The KV cache: identical outputs, and how the cost per generated token changes.
# 2. KV cache memory, for this model and for real ones.
# 3. Batching: throughput vs batch size.
# 4. Static vs continuous batching: a simulation of a server under mixed request lengths.
# 5. Paged KV memory: contiguous preallocation vs 16-token pages.
# 6. Speculative decoding with a cheap draft model: same output as the big model, fewer big-model steps.
# 7. Prefix caching: reusing a shared system prompt.

# %%
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from tinygpt import corpus_split, load_or_train  # noqa: E402

torch.set_num_threads(4)
model, tok = load_or_train()
L, d, H = len(model.blocks), model.tok.weight.shape[1], model.blocks[0].h
prompt = tok.encode("## Things that will bite you\n\n- **")

# %% [markdown]
# ## 1. The KV cache

# %%
@torch.no_grad()
def generate_nocache(ids, n):
    ids = list(ids)
    for _ in range(n):
        ids.append(int(model(torch.tensor([ids]))[0, -1].argmax()))                 # recompute the whole prefix every step
    return ids


@torch.no_grad()
def generate_cache(ids, n):
    logits, caches = model(torch.tensor([ids]), [None] * L, 0)                        # prefill: the prompt in one pass
    out = list(ids)
    for _ in range(n):
        t = int(logits[0, -1].argmax())
        out.append(t)
        logits, caches = model(torch.tensor([[t]]), caches, len(out) - 1)             # decode: one token, attend to the cache
    return out


N_NEW = 100
t0 = time.perf_counter(); a = generate_nocache(prompt, N_NEW); t_nc = time.perf_counter() - t0
t0 = time.perf_counter(); b = generate_cache(prompt, N_NEW); t_c = time.perf_counter() - t0
print(f"{N_NEW} tokens greedy: without cache {t_nc:.2f}s, with cache {t_c:.2f}s ({t_nc / t_c:.1f}x); outputs identical: {a == b}")
print("without a cache, token t costs a forward pass over t tokens (quadratic in total); with one, each new token costs a")
print("pass over one token plus attention reads of the cached keys and values (linear).")
assert a == b and t_c < t_nc

# %% [markdown]
# ## 2. How big is the cache?

# %%
def kv_bytes(layers, kv_heads, head_dim, tokens, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per


print(f"\nthis model: {kv_bytes(L, H, d // H, 1, 4) / 1024:.1f} KiB per token (fp32), {kv_bytes(L, H, d // H, 128, 4) / 1024:.0f} KiB for a full context")
for name, (layers, kvh, hd) in {"Llama 3 8B (32 layers, 8 KV heads x 128)": (32, 8, 128),
                                "Llama 3 70B (80 layers, 8 KV heads x 128)": (80, 8, 128)}.items():
    per_tok = kv_bytes(layers, kvh, hd, 1)
    print(f"{name}: {per_tok / 1024:.0f} KiB per token in bf16; 32 sequences x 8k tokens = {per_tok * 32 * 8192 / 1e9:.0f} GB")
print("the weights of the 8B model are 16 GB in bf16. At scale the cache, not the weights, decides how many requests fit.")

# %% [markdown]
# ## 3. Batching
#
# Decoding one token for one sequence reads every weight once for very little arithmetic (33.1). Decoding a batch reads
# the weights once for many sequences.

# %%
@torch.no_grad()
def batched_decode_throughput(B, steps=40):
    ids = torch.tensor([prompt] * B)
    logits, caches = model(ids, [None] * L, 0)
    pos = ids.shape[1]
    t0 = time.perf_counter()
    for s in range(steps):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        logits, caches = model(nxt, caches, pos + s)
    return B * steps / (time.perf_counter() - t0)


print("\nbatch   tokens/s")
thr = {}
for B in (1, 4, 16, 64):
    thr[B] = batched_decode_throughput(B)
    print(f"{B:5d}   {thr[B]:8.0f}")
print("throughput grows almost linearly with batch at first: the per-step cost is dominated by fixed overheads and weight")
print("reads, shared by every sequence in the batch. Until the cache reads and the arithmetic catch up.")
assert thr[16] > 4 * thr[1]

# %% [markdown]
# ## 4. Static vs continuous batching (simulation)
#
# 200 requests with output lengths from 10 to 500 tokens, a batch of 16 slots, and a step cost of a + b x (active
# sequences) milliseconds. Static batching fills 16 slots and waits for the longest request of the batch to finish.
# Continuous batching refills a slot as soon as its request finishes.

# %%
def simulate(continuous, n_req=200, slots=16, a=20.0, b=0.5, seed=0):
    r = np.random.default_rng(seed)
    lengths = r.integers(10, 500, n_req)
    queue = list(range(n_req))
    active, t, finish, busy_slot_steps, steps = {}, 0.0, {}, 0, 0
    while queue or active:
        if continuous or not active:
            while queue and len(active) < slots:
                i = queue.pop(0); active[i] = lengths[i]
        t += a + b * len(active)
        steps += 1
        busy_slot_steps += sum(1 for v in active.values() if v > 0)
        for i in list(active):
            if active[i] <= 0:
                continue                                                            # finished, waiting for its batch (static)
            active[i] -= 1
            if active[i] == 0:
                finish[i] = t                                                       # its last token was generated now
                if continuous:
                    del active[i]
        if not continuous and all(v <= 0 for v in active.values()):
            active = {}
    return lengths.sum() / (t / 1000), np.mean(list(finish.values())) / 1000, busy_slot_steps / (steps * slots)


print("\n              tokens/s   mean completion time   slot utilization")
sim = {}
for cont in (False, True):
    sim[cont] = simulate(cont)
    print(f"{'continuous' if cont else 'static':12s} {sim[cont][0]:9.0f}   {sim[cont][1]:19.1f}s   {sim[cont][2]:16.0%}")
print("static batches idle behind their longest request; continuous batching (Orca, vLLM) keeps every slot busy.")
assert sim[True][0] > 1.3 * sim[False][0]

# %% [markdown]
# ## 5. Paged KV memory
#
# Reserving max_len tokens of cache per request up front wastes whatever the request doesn't use. PagedAttention
# allocates fixed-size blocks (pages) on demand, like virtual memory.

# %%
r = np.random.default_rng(1)
used = r.integers(10, 2048, 1000)                                                    # final lengths of 1,000 requests
max_len, page = 2048, 16
contiguous = len(used) * max_len
paged = int(np.sum(np.ceil(used / page) * page))
print(f"\ncache slots reserved for 1,000 requests: contiguous (max length {max_len}) {contiguous:,}; paged ({page}-token pages) {paged:,}")
print(f"actually used {used.sum():,}: contiguous wastes {1 - used.sum() / contiguous:.0%}, paged {1 - used.sum() / paged:.1%}")
print("less waste means more concurrent requests in the same memory, which (section 3) means more throughput. Pages also")
print("let requests with a common prefix share its blocks (section 7).")
assert paged < 0.6 * contiguous

# %% [markdown]
# ## 6. Speculative decoding
#
# A cheap draft model proposes k tokens; the big model checks all k in one forward pass and keeps the longest prefix it
# agrees with, plus one token of its own. For greedy decoding the output is exactly the big model's greedy output.
# The draft here is a trigram model counted from the same corpus: nearly free.

# %%
train_text, _ = corpus_split()
ids_all = tok.encode(train_text)
tri = defaultdict(Counter)
for x, y, z in zip(ids_all, ids_all[1:], ids_all[2:]):
    tri[(x, y)][z] += 1
bi = defaultdict(Counter)
for x, y in zip(ids_all, ids_all[1:]):
    bi[x][y] += 1


def draft_next(seq):
    c = tri.get((seq[-2], seq[-1])) or bi.get(seq[-1])
    return c.most_common(1)[0][0] if c else 0


@torch.no_grad()
def speculative(ids, n, k=4):
    out, target_calls, proposed, accepted = list(ids), 0, 0, 0
    while len(out) < len(ids) + n:
        draft = []
        for _ in range(k):
            draft.append(draft_next(out + draft))
        logits = model(torch.tensor([out + draft]))[0]                              # one big-model pass checks all k
        target_calls += 1
        base = len(out) - 1
        n_ok = 0
        for j, t in enumerate(draft):
            if int(logits[base + j].argmax()) == t:
                n_ok += 1
            else:
                break
        out += draft[:n_ok] + [int(logits[base + n_ok].argmax())]                    # plus the big model's own next token
        proposed += k; accepted += n_ok
    return out[:len(ids) + n], target_calls, accepted / proposed


spec, calls, acc_rate = speculative(prompt, N_NEW)
print(f"\nspeculative decoding, {N_NEW} tokens: output identical to plain greedy: {spec == a}; big-model forward passes "
      f"{calls} instead of {N_NEW}; draft tokens accepted {acc_rate:.0%}")
print("a poor draft (a trigram counter, rarely right) still saves a good share of the big-model passes, because every pass")
print("yields at least one token and sometimes more. Drafts from a small model of the same family are accepted 60-80% of")
print("the time. Each verification pass processes k + 1 tokens, which at small batch costs about as much as one: decoding")
print("is memory-bound (33.1). The output is unchanged: exactly, for greedy; by a rejection-sampling rule, for sampling.")
assert spec == a and calls < N_NEW

# %% [markdown]
# ## 7. Prefix caching

# %%
system = tok.encode("You are a careful assistant for a machine learning course. Answer briefly, cite the lesson number, "
                    "and say so when you don't know. ")
questions = [tok.encode(q) for q in ["What is perplexity?", "Why use a KV cache?", "What is RoPE?", "Explain dropout."]]
with torch.no_grad():
    t0 = time.perf_counter()
    for q in questions:
        model(torch.tensor([system + q]), [None] * L, 0)                                # prefill everything, every time
    t_full = time.perf_counter() - t0
    _, sys_cache = model(torch.tensor([system]), [None] * L, 0)
    t0 = time.perf_counter()
    for q in questions:
        model(torch.tensor([q]), [(k.clone(), v.clone()) for k, v in sys_cache], len(system))   # reuse the system prompt's cache
    t_prefix = time.perf_counter() - t0
    full_last = model(torch.tensor([system + questions[0]]))[0, -1]
    reuse_last = model(torch.tensor([questions[0]]), [(k.clone(), v.clone()) for k, v in sys_cache], len(system))[0][0, -1]
print(f"\nsystem prompt {len(system)} tokens, questions ~{np.mean([len(q) for q in questions]):.0f} tokens:")
n_full = sum(len(system) + len(q) for q in questions)
n_reuse = len(system) + sum(len(q) for q in questions)
print(f"prefill all 4 from scratch {t_full * 1000:.0f} ms; reusing the cached system prompt {t_prefix * 1000:.0f} ms; "
      f"same logits: {torch.allclose(full_last, reuse_last, atol=1e-4)}")
print(f"tokens run through the model: {n_full} vs {n_reuse} ({1 - n_reuse / n_full:.0%} fewer). On this tiny model per-call overhead")
print("hides the saving in wall-clock time; on a 70B model with a 2,000-token system prompt, it's most of the prefill cost.")
print("Hosted APIs bill cached prompt tokens at a discount for this reason: keep the stable part of a prompt at the start.")
assert torch.allclose(full_last, reuse_last, atol=1e-4)

print("\nAll checks passed.")
