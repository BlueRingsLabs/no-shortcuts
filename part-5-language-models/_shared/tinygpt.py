"""A small GPT trained on this course, shared by the labs of Modules 37 and 38.

    from tinygpt import load_or_train
    model, tok = load_or_train()          # trains once (~3 minutes on 4 CPU cores), then loads from a cache

The model is 35.2's GPT with one addition: forward() accepts and returns a KV cache, for 37.2. Training is
deterministic, and the cache file name includes a hash of the configuration and of the corpus, so editing the lessons
retrains it instead of loading a stale model. Cache location: $NO_SHORTCUTS_CACHE or ~/.cache/no-shortcuts.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bpe import BPE
from corpus import lesson_files, read_lesson

CONFIG = dict(n_merges=1000, ctx=128, d=128, n_layers=4, n_heads=4, steps=1200, batch=32, lr=3e-3, seed=0, train_threads=4)
# train_threads: floating-point sums are grouped differently with different thread counts, so a model trained with 4
# threads and one trained with 1 differ in the last bits, and after 1,200 steps, in more than that. The shared models
# always train with the same count, whatever the lab that triggers the training has set.


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cache=None):
        B, n, d = x.shape
        q, k, v = (t.view(B, n, self.h, d // self.h).transpose(1, 2) for t in self.qkv(self.ln1(x)).chunk(3, -1))
        if cache is not None and cache[0] is not None:                      # append this step's keys and values
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        new_cache = (k, v)
        causal = n > 1                                                       # one new token may attend to everything cached
        if causal and k.shape[2] > n:                                        # a chunk after a cached prefix: offset causal mask
            mask = torch.ones(n, k.shape[2], dtype=torch.bool).tril(k.shape[2] - n)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        x = x + self.proj(o.transpose(1, 2).reshape(B, n, d))
        return x + self.mlp(self.ln2(x)), new_cache


class GPT(nn.Module):
    def __init__(self, V, ctx, d, n_layers, n_heads):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * n_layers) if name.endswith(("proj.weight", "mlp.2.weight")) else 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx, caches=None, start_pos=0):
        """idx (B, n). With caches (a list with one (k, v) per layer, or None entries), returns (logits, new caches)."""
        x = self.tok(idx) + self.pos(torch.arange(start_pos, start_pos + idx.shape[1]))
        new = []
        for i, b in enumerate(self.blocks):
            x, c = b(x, None if caches is None else caches[i])
            new.append(c)
        logits = self.head(self.ln_f(x))
        return (logits, new) if caches is not None else logits


def corpus_split():
    files = lesson_files()
    train = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 != 4)
    val = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 == 4)
    return train, val


def _cache_path(train_text):
    key = hashlib.sha1((repr(sorted(CONFIG.items())) + train_text).encode()).hexdigest()[:12]
    root = Path(os.environ.get("NO_SHORTCUTS_CACHE", Path.home() / ".cache" / "no-shortcuts"))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"tinygpt-{key}.pt"


def load_or_train(verbose=True):
    train_text, _ = corpus_split()
    c = CONFIG
    tok = BPE.train(train_text, n_merges=c["n_merges"])
    model = GPT(tok.vocab_size, c["ctx"], c["d"], c["n_layers"], c["n_heads"])
    path = _cache_path(train_text)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        if verbose:
            print(f"tinygpt: loaded from {path}")
        return model.eval(), tok
    if verbose:
        print(f"tinygpt: training ({c['steps']} steps; cached afterwards at {path})")
    ids = torch.tensor(tok.encode(train_text))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(c["train_threads"])
    torch.manual_seed(c["seed"])
    model = GPT(tok.vocab_size, c["ctx"], c["d"], c["n_layers"], c["n_heads"])
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=c["lr"], betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(c["seed"])
    for s in range(c["steps"]):
        lr = c["lr"] * min(1, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / c["steps"])))
        for gr in opt.param_groups:
            gr["lr"] = lr
        i = torch.randint(0, len(ids) - c["ctx"] - 1, (c["batch"],), generator=g)
        x = torch.stack([ids[j:j + c["ctx"]] for j in i])
        y = torch.stack([ids[j + 1:j + c["ctx"] + 1] for j in i])
        loss = F.cross_entropy(model(x).reshape(-1, tok.vocab_size), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.set_num_threads(old_threads)
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(path)                                                        # atomic: never a half-written cache file
    return model.eval(), tok


def val_loss(model, tok, n_batches=20, seed=1):
    _, val = corpus_split()
    ids = torch.tensor(tok.encode(val))
    g = torch.Generator().manual_seed(seed)
    ctx = model.ctx
    with torch.no_grad():
        out = []
        for _ in range(n_batches):
            i = torch.randint(0, len(ids) - ctx - 1, (32,), generator=g)
            x = torch.stack([ids[j:j + ctx] for j in i]); y = torch.stack([ids[j + 1:j + ctx + 1] for j in i])
            out.append(F.cross_entropy(model(x).reshape(-1, tok.vocab_size), y.reshape(-1)).item())
    return float(np.mean(out))
