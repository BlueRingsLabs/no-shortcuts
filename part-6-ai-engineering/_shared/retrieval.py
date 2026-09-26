"""Retrieval over this course, for Module 41's labs: documents, chunks, labeled queries, BM25 and a small trained encoder.

The corpus is Parts 0 to V of the course (the text of each lesson before its exercises): a fixed knowledge base, so
the numbers in the lessons don't move every time a later lesson is written. The queries are real and come with labels
for free: every lesson's exercises, whose relevant document is the lesson they come from, and which aren't in the
searchable text. A second query set is made of code identifiers that appear in exactly one lesson (`ctc_loss`,
`apply_chat_template`): the kind of query that embeddings are bad at.

    lessons(max_part=5) -> list of Lesson(id, title, part, path, body, sections, exercises)
    chunk(lessons, strategy="sections", size=120, overlap=0) -> list of Chunk(lesson_id, heading, text)
    exercise_queries(lessons) / identifier_queries(lessons) -> list of (query, lesson_id)
    reference_queries(lessons) -> list of (query, lesson_id, source_lesson_id): cross-references between lessons
    tokenize(text) -> BM25 terms;  BM25(texts).scores(query) -> np.array
    load_or_train_encoder() -> (Encoder, tokenizer); Encoder.embed(texts) -> unit vectors (numpy)
    evaluate(ranked_lesson_ids, relevant_ids, k) -> recall@k, MRR@10
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "part-5-language-models" / "_shared"))
from corpus import read_lesson, snapshot  # noqa: E402


@dataclass
class Lesson:
    id: str
    title: str
    part: int
    path: Path
    body: str
    sections: list = field(default_factory=list)                          # (heading, text)
    exercises: list = field(default_factory=list)


@dataclass
class Chunk:
    lesson_id: str
    heading: str
    text: str


def _clean(md):
    md = re.sub(r"<details>.*?</details>", "", md, flags=re.S)
    return md


def lessons(max_part=5):
    out = []
    for p in sorted(ROOT / k for k in snapshot() if k.endswith(".md")):    # the frozen text of Parts 0 to V
        if int(p.parts[-3].split("-")[1]) > max_part:
            continue
        text = read_lesson(p)
        lid = p.name.split("-")[0]
        first = text.splitlines()[0]
        title = re.sub(r"^#\s*[0-9.]+\s*", "", first).strip()
        body, _, rest = text.partition("\n## Exercises")
        ex_block = rest.split("<details>")[0]
        exercises = [re.sub(r"\s+", " ", e).strip() for e in re.split(r"\n\d+\.\s", "\n" + ex_block) if e.strip()]
        exercises = [e for e in exercises if len(e.split()) >= 6]
        body = "\n".join(body.splitlines()[1:])                            # drop the title line
        sections, head, buf = [], "Introduction", []
        for line in body.splitlines():
            if line.startswith("## "):
                if "".join(buf).strip():
                    sections.append((head, "\n".join(buf).strip()))
                head, buf = line[3:].strip(), []
            else:
                buf.append(line)
        if "".join(buf).strip():
            sections.append((head, "\n".join(buf).strip()))
        out.append(Lesson(lid, title, int(p.parts[-3].split("-")[1]), p, body, sections, exercises))
    return out


def _paragraphs(text):
    return [re.sub(r"[ \t]+", " ", b).strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def chunk(lessons_, strategy="sections", size=120, overlap=0):
    """strategy: 'sections' packs whole paragraphs, within one section, up to about `size` words;
    'fixed' cuts each lesson into windows of `size` words with `overlap`, ignoring structure;
    'whole' is one chunk per section."""
    out = []
    for les in lessons_:
        if strategy == "fixed":
            w = les.body.split()
            step = max(1, size - overlap)
            for i in range(0, max(1, len(w) - overlap), step):
                out.append(Chunk(les.id, "", " ".join(w[i:i + size])))
            continue
        for head, text in les.sections:
            if strategy == "whole":
                out.append(Chunk(les.id, head, text))
                continue
            buf = []
            for para in _paragraphs(text):
                if buf and len(" ".join(buf).split()) + len(para.split()) > size:
                    out.append(Chunk(les.id, head, "\n\n".join(buf)))
                    buf = []
                buf.append(para)
            if buf:
                out.append(Chunk(les.id, head, "\n\n".join(buf)))
    return out


def exercise_queries(lessons_):
    return [(e, les.id) for les in lessons_ for e in les.exercises]


def identifier_queries(lessons_, n=200, seed=0):
    """Backticked code identifiers (with an underscore, a dot or a parenthesis) found in exactly one lesson body."""
    where = {}
    for les in lessons_:
        for ident in set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*(?:[._][A-Za-z0-9_]+)+)(?:\(\))?`", les.body)):
            if len(ident) >= 6:
                where.setdefault(ident, set()).add(les.id)
    unique = sorted((i, next(iter(s))) for i, s in where.items() if len(s) == 1 and not i.startswith("lab_")
                    and sum(i in les.body for les in lessons_) == 1)
    random.Random(seed).shuffle(unique)
    return unique[:n]


def reference_queries(lessons_):
    """Sentences that point to exactly one other lesson, like "Streaming doesn't make the answer faster (37.2).", with
    the pointer removed. The query is a concept described in another lesson's words; the answer is the lesson that
    explains it. Returns (query, relevant lesson, source lesson): exclude the source when ranking."""
    ids = {les.id for les in lessons_}
    out = []
    for les in lessons_:
        for s in _sentences(les.body):
            refs = set(re.findall(r"\b(\d{2}\.\d)\b", s))
            if len(refs) == 1:
                ref = refs.pop()
                q = re.sub(r"\s*\((?:see )?\d{2}\.\d(?:[^)]*)\)|\b\d{2}\.\d\b", "", s).strip()
                if ref in ids and ref != les.id and len(q.split()) >= 8:
                    out.append((q, ref, les.id))
    return out


STOP = set("""a an the and or of to in on for with is are be was were it its this that these those as at by from
not no but if then than so such can could would should will may might do does did have has had i you we they he she
there here what which who whom how why when where your our their my me us them into about over under between each
all any some more most other also just only very""".split())


def tokenize(text):
    t = text.lower()
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in re.findall(r"[a-z0-9]+", t)]
    idents = re.findall(r"[a-z0-9]+(?:[_.][a-z0-9]+)+", t)                 # keep code identifiers whole, too
    return [w for w in words if w not in STOP] + idents


class BM25:
    """Okapi BM25 (Robertson and Zaragoza, 2009) as a sparse matrix: score(q, d) = sum over query terms of
    idf(t) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))."""

    def __init__(self, texts, k1=1.2, b=0.75):
        docs = [tokenize(t) for t in texts]
        self.vocab = {}
        rows, cols, vals = [], [], []
        for i, d in enumerate(docs):
            counts = {}
            for w in d:
                counts[w] = counts.get(w, 0) + 1
            for w, c in counts.items():
                rows.append(i); cols.append(self.vocab.setdefault(w, len(self.vocab))); vals.append(c)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=np.float64)
        dl = np.array([len(d) for d in docs], dtype=float)
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        self.idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
        tf = tf.tocoo()
        norm = k1 * (1 - b + b * dl[tf.row] / dl.mean())
        w = tf.data * (k1 + 1) / (tf.data + norm) * self.idf[tf.col]
        self.W = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query):
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return np.zeros(self.W.shape[0])
        return np.asarray(self.W[:, cols].sum(axis=1)).ravel()


def lesson_ranking(chunk_scores, chunks_, k=10, exclude=None):
    """Chunk scores -> lesson IDs ranked by their best chunk (document-level results from passage retrieval)."""
    order = np.argsort(-chunk_scores)
    seen, out = {exclude}, []
    for i in order:
        lid = chunks_[i].lesson_id
        if lid not in seen:
            seen.add(lid)
            out.append(lid)
            if len(out) == k:
                break
    return out


def evaluate(rankings, relevant, k=5):
    """rankings: list of ranked lesson-ID lists; relevant: list of the one relevant ID per query."""
    recall = np.mean([r in rk[:k] for rk, r in zip(rankings, relevant)])
    mrr = np.mean([1 / (rk.index(r) + 1) if r in rk[:10] else 0 for rk, r in zip(rankings, relevant)])
    return recall, mrr


def rrf(*score_arrays, k=60):
    """Reciprocal rank fusion (Cormack et al., 2009): sum of 1 / (k + rank) over the systems."""
    total = np.zeros(len(score_arrays[0]))
    for s in score_arrays:
        ranks = np.empty(len(s)); ranks[np.argsort(-s)] = np.arange(1, len(s) + 1)
        total += 1 / (k + ranks)
    return total


# ---------------------------------------------------------------------------------------------------------------------
# A small dense encoder: the course's tiny GPT (35.2), mean-pooled, fine-tuned contrastively on Parts I to III.

ENC_CONFIG = dict(steps=1200, bs=64, lr=3e-4, temp=0.05, seed=0, train_threads=4)


def _sentences(text):
    text = re.sub(r"```.*?```|\$\$.*?\$\$", " ", text, flags=re.S)
    return [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if len(s.split()) >= 5]


def training_passages():
    """Parts I to III only, split into section chunks with their headings. No exercise ever appears here."""
    return [f"{c.heading}. {c.text}" for c in chunk([les for les in lessons() if les.part <= 3], "sections", size=120)]


def _crop(ids, r):
    """A random contiguous span of 10% to 50% of the passage: two independent crops of one passage are a positive pair
    (Contriever, Izacard et al., 2022). Fresh crops every step, so there's no fixed set of pairs to memorize."""
    n = len(ids)
    length = r.randint(max(8, n // 10), max(9, n // 2))
    start = r.randint(0, max(0, n - length))
    return ids[start:start + length][:128]


class Encoder:
    def __init__(self, gpt, tok, proj):
        self.gpt, self.tok, self.proj = gpt, tok, proj

    def _forward(self, texts, max_len):
        return self._forward_ids([self.tok.encode(t)[:max_len] or [0] for t in texts])

    def _forward_ids(self, ids):
        import torch
        L = max(len(i) for i in ids)
        X = torch.zeros(len(ids), L, dtype=torch.long)
        M = torch.zeros(len(ids), L)
        for k, i in enumerate(ids):
            X[k, :len(i)] = torch.tensor(i); M[k, :len(i)] = 1
        g = self.gpt
        x = g.tok(X) + g.pos(torch.arange(L))
        for b in g.blocks:
            x, _ = b(x)
        h = g.ln_f(x)
        pooled = (h * M[..., None]).sum(1) / M.sum(1, keepdim=True)          # mean over real tokens (causal: pads unseen)
        return torch.nn.functional.normalize(self.proj(pooled), dim=-1)

    def embed(self, texts, max_len=128, batch=64):
        import torch
        with torch.no_grad():
            return np.concatenate([self._forward(texts[i:i + batch], max_len).numpy() for i in range(0, len(texts), batch)])


def load_or_train_encoder(verbose=True, trained=True):
    """trained=False returns the pretrained language model, mean-pooled, with an identity projection: the baseline."""
    import copy

    import torch
    import torch.nn.functional as F

    from tinygpt import load_or_train

    base, tok = load_or_train(verbose=verbose)
    gpt = copy.deepcopy(base)
    d = gpt.tok.weight.shape[1]
    proj = torch.nn.Linear(d, d)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(d)); proj.bias.zero_()
    enc = Encoder(gpt, tok, proj)
    if not trained:
        return enc, tok
    passages = training_passages()
    from tinygpt import _cache_path, corpus_split
    key = hashlib.sha1((repr(sorted(ENC_CONFIG.items())) + "\n".join(passages)
                        + _cache_path(corpus_split()[0]).name).encode()).hexdigest()[:12]      # retrain if the base changes
    root = Path(os.environ.get("NO_SHORTCUTS_CACHE", Path.home() / ".cache" / "no-shortcuts"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"encoder-{key}.pt"
    if path.exists():
        state = torch.load(path, weights_only=True)
        gpt.load_state_dict(state["gpt"]); proj.load_state_dict(state["proj"])
        if verbose:
            print(f"encoder: loaded from {path}")
        return enc, tok
    c = ENC_CONFIG
    if verbose:
        print(f"encoder: training on random crops of {len(passages):,} passages ({c['steps']} steps; cached afterwards at {path})")
    ids = [i for i in (tok.encode(p) for p in passages) if len(i) >= 40]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(c["train_threads"])
    torch.manual_seed(c["seed"])
    r = random.Random(c["seed"])
    params = list(gpt.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=0.01)
    gpt.train()
    for s in range(c["steps"]):
        for gr in opt.param_groups:
            gr["lr"] = c["lr"] * min(1, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / c["steps"]))
        batch = [ids[r.randrange(len(ids))] for _ in range(c["bs"])]
        q = enc._forward_ids([_crop(b, r) for b in batch])
        p = enc._forward_ids([_crop(b, r) for b in batch])
        logits = q @ p.T / c["temp"]                                        # in-batch negatives (InfoNCE)
        labels = torch.arange(len(batch))
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    gpt.eval()
    torch.set_num_threads(old_threads)
    tmp = path.with_suffix(".tmp")
    torch.save({"gpt": gpt.state_dict(), "proj": proj.state_dict()}, tmp)
    tmp.replace(path)
    return enc, tok
