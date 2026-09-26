"""A small instruction-tuning dataset about this course, built from course.yaml, for Modules 38.x.

Questions and answers about lessons and modules ("What is lesson 29.1 about?" -> "Attention from first principles."),
in several phrasings. Training uses some phrasings; evaluation uses a phrasing never seen in training for the same
facts (does the model follow instructions, or only memorize strings?).

The chat template uses plain-text markers, tokenized like any other text by the course's BPE. Production templates use
dedicated special tokens instead (and must never let user text produce them, 35.1).
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

from corpus import snapshot

ROOT = Path(__file__).resolve().parents[2]
USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|end|>"

TRAIN_TEMPLATES = {
    "lesson_title": ["What is lesson {lid} about?", "Lesson {lid}: what's the topic?", "Give me the title of lesson {lid}."],
    "lesson_module": ["Which module is lesson {lid} in?", "Lesson {lid} belongs to which module?"],
    "module_title": ["What is module {mid} called?", "Name module {mid}."],
    "module_hours": ["How many hours does module {mid} take?", "How long is module {mid}?"],
}
HELDOUT_TEMPLATES = {                                                     # never used in training
    "lesson_title": ["Tell me what lesson {lid} covers."],
    "lesson_module": ["In what module do I find lesson {lid}?"],
    "module_title": ["What's the name of module {mid}?"],
    "module_hours": ["Roughly how much time should I plan for module {mid}?"],
}


def facts():
    course = yaml.safe_load(snapshot()["course.yaml"])                     # frozen with the lessons (tools/snapshot_course.py)
    out = []
    for part in course["parts"]:
        for m in part["modules"]:
            mid, mtitle = m["id"], m["title"]
            out.append(("module_title", {"mid": mid}, f"{mtitle}."))
            out.append(("module_hours", {"mid": mid}, f"About {m['hours']} hours."))
            for les in m["lessons"]:
                out.append(("lesson_title", {"lid": les["id"]}, f"{les['title']}."))
                out.append(("lesson_module", {"lid": les["id"]}, f"Module {mid}: {mtitle}."))
    return out


def format_chat(question, answer=None):
    """Prompt part and full text. The model is trained to produce `answer` + END after the assistant marker."""
    prompt = f"{USER} {question}\n{ASSISTANT} "
    return prompt if answer is None else (prompt, f"{answer} {END}")


def dataset(split="train", seed=0):
    """split='train': every fact in the training phrasings. split='heldout': every fact in a held-out phrasing."""
    r = random.Random(seed)
    items = []
    for kind, slots, answer in facts():
        templates = TRAIN_TEMPLATES[kind] if split == "train" else HELDOUT_TEMPLATES[kind]
        for t in templates:
            items.append((t.format(**slots), answer, kind))
    r.shuffle(items)
    return items


SFT_CONFIG = dict(epochs=12, lr=1e-3, bs=32, pretrain_mix=0.25, seed=0, train_threads=4)


def load_or_train_sft(verbose=True):
    """38.1's recipe (loss on answers only, 25% pretraining data mixed in), on the shared tinygpt; cached like it."""
    import copy
    import hashlib
    import math
    import os

    import torch
    import torch.nn.functional as F

    from tinygpt import _cache_path, corpus_split, load_or_train

    base, tok = load_or_train(verbose=verbose)
    train_text, _ = corpus_split()
    key = hashlib.sha1((repr(sorted(SFT_CONFIG.items())) + snapshot()["course.yaml"] + train_text
                        + _cache_path(train_text).name).encode()).hexdigest()[:12]      # retrain if the base model changes
    root = Path(os.environ.get("NO_SHORTCUTS_CACHE", Path.home() / ".cache" / "no-shortcuts"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"tinygpt-sft-{key}.pt"
    model = copy.deepcopy(base)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        if verbose:
            print(f"sft model: loaded from {path}")
        return model.eval(), tok
    if verbose:
        print(f"sft model: training (cached afterwards at {path})")
    c, V, ctx = SFT_CONFIG, tok.vocab_size, base.ctx
    items = dataset("train")
    X = torch.zeros(len(items), ctx, dtype=torch.long)
    Y = torch.full((len(items), ctx), -100)
    for i, (q, a, _) in enumerate(items):
        p, r = format_chat(q, a)
        pi, ri = tok.encode(p), tok.encode(r)
        ids = pi + ri
        X[i, :len(ids) - 1] = torch.tensor(ids[:-1])
        Y[i, len(pi) - 1:len(ids) - 1] = torch.tensor(ri)
    pre = torch.tensor(tok.encode(train_text))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(c["train_threads"])
    torch.manual_seed(c["seed"])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=0.0)
    g = torch.Generator().manual_seed(c["seed"])
    steps = c["epochs"] * math.ceil(len(X) / c["bs"])
    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = c["lr"] * min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / steps))
        i = torch.randint(0, len(X), (c["bs"],), generator=g)
        loss = F.cross_entropy(model(X[i]).reshape(-1, V), Y[i].reshape(-1), ignore_index=-100)
        j = torch.randint(0, len(pre) - ctx - 1, (8,), generator=g)
        xp = torch.stack([pre[k:k + ctx] for k in j]); yp = torch.stack([pre[k + 1:k + ctx + 1] for k in j])
        loss = (1 - c["pretrain_mix"]) * loss + c["pretrain_mix"] * F.cross_entropy(model(xp).reshape(-1, V), yp.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.set_num_threads(old_threads)
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(path)
    return model.eval(), tok
