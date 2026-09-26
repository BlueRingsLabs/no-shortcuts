# %% [markdown]
# # Lab 35.1: Byte-level BPE, from scratch
#
# 1. Characters, code points and bytes: what "one character" costs in UTF-8.
# 2. Training byte-level BPE on this course: pre-tokenize, count, merge the most frequent pair, repeat.
# 3. Encoding and decoding: exact round trips on any string, including ones the tokenizer never saw.
# 4. Compression vs vocabulary size, on held-out English, Spanish, Chinese, and Python code: the tokenizer tax.
# 5. The weird parts: numbers, leading spaces, capitalization, and repeated characters.

# %%
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from corpus import lesson_files, read_lesson  # noqa: E402

# %% [markdown]
# ## 1. Bytes

# %%
samples = {"English": "the model", "Spanish": "el niño comió", "Chinese": "机器学习", "emoji": "🙂👍", "math": "∑ x² ≤ ε"}
print("text              characters   UTF-8 bytes")
for k, s in samples.items():
    print(f"{k:8s} {s!r:16s} {len(s):5d}   {len(s.encode('utf-8')):11d}")
print("ASCII is one byte per character; accented Latin letters two; Chinese characters three; emoji four.")
print("A byte-level tokenizer starts from these 256 byte values, so it can encode anything, and never needs <unk>.")

# %% [markdown]
# ## 2. Training
#
# Pre-tokenization splits text into chunks (words with their leading space, runs of digits, punctuation, whitespace)
# so merges never cross chunk boundaries: "dog." and "dog!" share the token for " dog". This pattern is a simplified
# version of GPT-2's (Python's re has no \p{L}; [^\W\d_] means "a letter").

# %%
PAT = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")
# My first version of this pattern had " ?[^\s\w]+" for punctuation, which silently dropped every underscore
# (a "word" character that is neither a letter nor a digit), and the round trip below failed. A pre-tokenizer must be
# lossless; test it on its own:
probe = "snake_case __init__ x_1 naïve café 机器 🙂 \t\n  end"
assert "".join(PAT.findall(probe)) == probe

files = lesson_files()
train_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 != 4)
held_out = "\n".join(read_lesson(f) for i, f in enumerate(files) if i % 10 == 4)


def train_bpe(text, n_merges):
    """Count every adjacent pair across all distinct chunks (weighted by chunk frequency), merge the most frequent pair
    into a new token, and repeat. Pair counts are updated only for the chunks that contain the merged pair."""
    chunks = Counter(tuple(c.encode("utf-8")) for c in PAT.findall(text))   # each distinct chunk once, with its count
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
        best = max(pairs, key=lambda p: (pairs[p], p))                          # ties broken deterministically
        new = 256 + k
        merges[best] = new
        vocab[new] = vocab[best[0]] + vocab[best[1]]
        for w in list(where.get(best, ())):
            seq, cnt = words[w], chunks[w]
            for p in zip(seq, seq[1:]):                                          # remove this chunk's old pairs
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
            for p in zip(out, out[1:]):                                          # and add its new ones
                pairs[p] += cnt
                where.setdefault(p, set()).add(w)
    return merges, vocab


def encode(text, merges):
    ids = []
    for chunk in PAT.findall(text):
        seq = list(chunk.encode("utf-8"))
        while len(seq) > 1:                                                      # apply merges in the order they were learned
            pair = min(zip(seq, seq[1:]), key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            new, out, i = merges[pair], [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                    out.append(new); i += 2
                else:
                    out.append(seq[i]); i += 1
            seq = out
        ids += seq
    return ids


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


t0 = time.time()
merges, vocab = train_bpe(train_text, 2000)
print(f"\ntrained 2,000 merges on {len(train_text.encode('utf-8')):,} bytes in {time.time() - t0:.0f}s")
print("first 12 merges:", [vocab[256 + i].decode("utf-8", "replace") for i in range(12)])
print("merges 1990-2000:", [vocab[256 + i].decode("utf-8", "replace") for i in range(1990, 2000)])
longest = sorted(vocab.values(), key=len)[-8:]
print("longest tokens:", [t.decode("utf-8", "replace") for t in longest])

# %% [markdown]
# ## 3. Round trips

# %%
tests = ["The model's F1 was 0.93.", "Ünïcödé, 机器学习 and 🙂 are all just bytes.", "def f(x):\n    return x ** 2\n",
         "   leading and trailing spaces   ", held_out[:5000]]
for s in tests:
    assert decode(encode(s, merges), vocab) == s
print(f"\nencode -> decode is exact for {len(tests)} test strings, including characters never seen in training")

# %% [markdown]
# ## 4. Compression and the tokenizer tax

# %%
other = {
    "held-out English (this course)": held_out[:20000],
    "Python code": "\n".join((Path(__file__).resolve().parents[3] / "tools" / f).read_text(encoding="utf-8")
                             for f in ("build_docs.py", "run_labs.py"))[:20000],
}
print("\nbytes per token (higher = better compression), by number of merges")
print("                                 256 (bytes)   " + "   ".join(f"{n:>6,}" for n in (500, 1000, 2000)))
for name, text in other.items():
    row = [len(text.encode("utf-8")) / len(encode(text, {p: i for p, i in merges.items() if i < 256 + n})) for n in (0, 500, 1000, 2000)]
    print(f"  {name:30s} {row[0]:11.2f}   " + "   ".join(f"{r:6.2f}" for r in row[1:]))

# the same paragraph in three languages (my own translations)
parallel = {
    "English": ("Language models don't understand text the way we do: they see a sequence of pieces, and every piece costs "
                "the same. A tokenizer trained almost only on English splits words in other languages into more pieces, so "
                "the same content takes more tokens, costs more money and leaves less room in the context. It isn't a "
                "technical detail: it's a fee paid by the users of other languages."),
    "Spanish": ("Los modelos de lenguaje no entienden el texto como nosotros: ven una secuencia de fragmentos, y cada "
                "fragmento cuesta lo mismo. Un tokenizador entrenado casi solo con inglés parte las palabras de otros idiomas "
                "en más pedazos, así que el mismo contenido ocupa más tokens, cuesta más dinero y deja menos espacio en el "
                "contexto. No es un detalle técnico: es una tarifa que pagan los usuarios de otros idiomas."),
    "Chinese": ("语言模型并不像我们这样理解文本：它们看到的是一串片段，每个片段的成本都一样。几乎只用英文训练的分词器会把其他语言的词"
                "切成更多的片段，所以同样的内容需要更多的词元，花更多的钱，在上下文中留下的空间也更少。这不是一个技术细节，而是其他语言的用户要付的一笔费用。"),
}
n_tok = {k: len(encode(v, merges)) for k, v in parallel.items()}
print("\nthe same paragraph, 2,000 merges trained on English:")
for k, v in parallel.items():
    print(f"  {k:8s} {len(v):4d} characters, {len(v.encode('utf-8')):4d} bytes, {n_tok[k]:4d} tokens  ({n_tok[k] / n_tok['English']:.1f}x English)")
print("same content, more tokens: more money per request, less room in the context, and often worse quality for the")
print("languages the tokenizer's training data ignored. Multilingual tokenizers sample languages deliberately (35.3).")
assert n_tok["English"] < n_tok["Spanish"] < n_tok["Chinese"]
assert len(held_out[:20000].encode()) / len(encode(held_out[:20000], merges)) > 2.5

# %% [markdown]
# ## 5. The weird parts

# %%
def show(s):
    return [vocab[i].decode("utf-8", "replace") for i in encode(s, merges)]


print()
for s in ["1234567", "12345678", "2024", " 2025", "the", " the", " The", " THE", "hello", "hellooooooo", " model.", " model!"]:
    print(f"  {s!r:15s} -> {show(s)}")
print("numbers split in chunks of up to three digits by the pre-tokenizer, and then however the merges fall: arithmetic")
print("on these pieces is harder than it looks. ' the', 'the' and ' The' are different tokens with different embeddings,")
print("and a trailing space at the end of a prompt leaves the model to predict a token that rarely starts with one.")
assert show(" the") != show("the") and len(show(" THE")) > len(show(" the"))

print("\nAll checks passed.")
