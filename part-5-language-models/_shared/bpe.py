"""The byte-level BPE tokenizer from lab 35.1, packaged for the later labs.

    tok = BPE.train(text, n_merges=1000)
    ids = tok.encode("some text"); text = tok.decode(ids)

Encoding caches the result for each distinct pre-tokenized chunk, which makes encoding a whole corpus fast (most
chunks repeat). Lab 35.1 explains every line; this file just removes the need to paste it again.
"""

from __future__ import annotations

import re
from collections import Counter

PAT = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


class BPE:
    def __init__(self, merges: dict, vocab: dict):
        self.merges, self.vocab = merges, vocab
        self._cache: dict = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, text: str, n_merges: int) -> "BPE":
        chunks = Counter(tuple(c.encode("utf-8")) for c in PAT.findall(text))
        words = {w: list(w) for w in chunks}
        pairs, where = Counter(), {}
        for w, cnt in chunks.items():
            for p in zip(w, w[1:]):
                pairs[p] += cnt
                where.setdefault(p, set()).add(w)
        merges, vocab = {}, {i: bytes([i]) for i in range(256)}
        for k in range(n_merges):
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], p))
            new = 256 + k
            merges[best] = new
            vocab[new] = vocab[best[0]] + vocab[best[1]]
            for w in list(where.get(best, ())):
                seq, cnt = words[w], chunks[w]
                for p in zip(seq, seq[1:]):
                    pairs[p] -= cnt
                    if pairs[p] <= 0:
                        del pairs[p]
                    where[p].discard(w)
                out, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                        out.append(new); i += 2
                    else:
                        out.append(seq[i]); i += 1
                words[w] = out
                for p in zip(out, out[1:]):
                    pairs[p] += cnt
                    where.setdefault(p, set()).add(w)
        return cls(merges, vocab)

    def _encode_chunk(self, chunk: str) -> list:
        if chunk in self._cache:
            return self._cache[chunk]
        seq = list(chunk.encode("utf-8"))
        while len(seq) > 1:
            pair = min(zip(seq, seq[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new, out, i = self.merges[pair], [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                    out.append(new); i += 2
                else:
                    out.append(seq[i]); i += 1
            seq = out
        self._cache[chunk] = seq
        return seq

    def encode(self, text: str) -> list:
        ids = []
        for chunk in PAT.findall(text):
            ids += self._encode_chunk(chunk)
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")
