# %% [markdown]
# # Lab 34.1: Text, tokens and n-gram language models
#
# 1. Normalization: Unicode forms, case, and what a "word" is.
# 2. Zipf's law and the long tail: why unseen words never go away.
# 3. n-gram models: maximum likelihood (infinite perplexity), add-one, add-k, interpolated Kneser-Ney.
# 4. Generating text from a trigram model.
# 5. Converting word perplexity to bits per character, to compare with lab 28.1's character RNN.

# %%
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from corpus import lesson_files, read_lesson  # noqa: E402

rng = np.random.default_rng(341)

# %% [markdown]
# ## 1. Normalization

# %%
a, b = "café", "café"                                                  # one code point vs e + combining accent
print(f"'{a}' == '{b}': {a == b}; lengths {len(a)} and {len(b)}; after NFC: {unicodedata.normalize('NFC', a) == unicodedata.normalize('NFC', b)}")
s = "The U.S. model's F1 was 0.93, e.g. on 2024-05-01. Don't trust it."
print("split on spaces: ", s.split())
print("regex words:     ", re.findall(r"[a-z0-9']+", s.lower()))
print("naive sentences: ", [x for x in re.split(r"(?<=\.)\s+", s)])
print("every choice here (case, punctuation, numbers, abbreviations) changes the vocabulary and every number downstream.")
assert a != b and unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)

# %% [markdown]
# ## 2. Zipf and the long tail
#
# Train on 90% of the lesson files, test on the other 10% (whole files, so test text is genuinely new).

# %%
TOKEN = re.compile(r"[a-z0-9']+|[.,;:!?()]")


def tokenize(text):
    out = []
    for line in text.lower().split("\n"):
        toks = TOKEN.findall(line)
        if toks:
            out += ["<s>"] + toks + ["</s>"]                                        # a line is our "sentence"
    return out


files = lesson_files()
test_idx = set(range(4, len(files), 10))                                            # every 10th file, spread across parts
train_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i not in test_idx)
test_text = "\n".join(read_lesson(f) for i, f in enumerate(files) if i in test_idx)
train, test = tokenize(train_text), tokenize(test_text)
counts = Counter(train)
freqs = np.array(sorted(counts.values(), reverse=True))
slope = np.polyfit(np.log(np.arange(1, len(freqs) + 1))[10:2000], np.log(freqs[10:2000]), 1)[0]
print(f"\ntrain {len(train):,} tokens, test {len(test):,}; vocabulary {len(counts):,}")
print(f"log-log slope of frequency vs rank (Zipf's law says about -1): {slope:.2f}")
print(f"words seen exactly once: {np.mean(freqs == 1):.0%} of the vocabulary")
print("\ntraining tokens   vocabulary   test tokens never seen in training (OOV rate)")
for frac in (0.05, 0.2, 0.5, 1.0):
    sub = Counter(train[:int(frac * len(train))])
    print(f"{int(frac * len(train)):15,}   {len(sub):10,}   {np.mean([t not in sub for t in test]):.1%}")
print("more data shrinks the OOV rate slowly and never to zero: the tail is where language lives.")
assert -1.3 < slope < -0.7

# %% [markdown]
# ## 3. n-gram models and perplexity
#
# Vocabulary: training words seen at least twice; the rest become <unk> (in train and test alike). Perplexity =
# exp(mean negative log-likelihood per token) over the test set.

# %%
vocab = {w for w, c in counts.items() if c >= 2} | {"<s>", "</s>"}
unk = lambda seq: [t if t in vocab else "<unk>" for t in seq]
tr, te = unk(train), unk(test)
V = len(vocab) + 1


def ngram_counts(seq, n):
    c = Counter()
    for i in range(len(seq) - n + 1):
        c[tuple(seq[i:i + n])] += 1
    return c


c1, c2, c3 = ngram_counts(tr, 1), ngram_counts(tr, 2), ngram_counts(tr, 3)
ctx1 = Counter()
for (a_, b_), c in c2.items():
    ctx1[(a_,)] += c
ctx2 = Counter()
for (a_, b_, c_), c in c3.items():
    ctx2[(a_, b_)] += c
N1 = sum(c1.values())


def perplexity(prob_fn, seq, n):
    nll, count = 0.0, 0
    for i in range(n - 1, len(seq)):
        if seq[i] == "<s>":
            continue                                                                  # sentence starts are given, not predicted
        p = prob_fn(tuple(seq[i - n + 1:i]), seq[i])
        if p == 0:
            return float("inf")
        nll -= math.log(p); count += 1
    return math.exp(nll / count)


mle2 = lambda h, w: c2[h + (w,)] / ctx1[h] if ctx1[h] else 0.0
addk = lambda k: (lambda h, w: (c2[h + (w,)] + k) / (ctx1[h] + k * V))
uni = lambda h, w: c1[(w,)] / N1

# interpolated Kneser-Ney (trigram), discount d = 0.75
d = 0.75
cont = Counter(w for (_, w) in c2)                                                   # in how many distinct contexts w appears
n_bigram_types = len(c2)
follow1 = Counter(a_ for (a_, _) in c2)                                              # distinct words after a context
follow2 = Counter((a_, b_) for (a_, b_, _) in c3)
cont2 = Counter((b_, c_) for (_, b_, c_) in c3)                                      # continuation counts for the bigram level
cont2_ctx = Counter(b_ for (_, b_, _) in c3)
cont2_follow = Counter(b_ for (b_, _) in cont2)


def kn_uni(w):
    return max(cont[w], 0.5) / n_bigram_types                                       # continuation probability (floored for <unk>)


def kn_bi(h1, w):
    n_ctx = cont2_ctx[h1]
    if n_ctx == 0:
        return kn_uni(w)
    return max(cont2[(h1, w)] - d, 0) / n_ctx + d * cont2_follow[h1] / n_ctx * kn_uni(w)


def kn_tri(h, w):
    if len(h) < 2 or ctx2[h] == 0:
        return kn_bi(h[-1], w)
    return max(c3[h + (w,)] - d, 0) / ctx2[h] + d * follow2[h] / ctx2[h] * kn_bi(h[-1], w)


results = {
    "unigram (MLE)": perplexity(uni, te, 1),
    "bigram, MLE": perplexity(mle2, te, 2),
    "bigram, add-one": perplexity(addk(1.0), te, 2),
    "bigram, add-0.01": perplexity(addk(0.01), te, 2),
    "trigram, interpolated Kneser-Ney": perplexity(kn_tri, te, 3),
}
print(f"\ntest perplexity (vocabulary {V:,} including <unk>):")
for k_, v in results.items():
    print(f"  {k_:34s} {v:10.1f}")
print("MLE assigns zero probability to any unseen bigram: one is enough for infinite perplexity. Add-one steals so much")
print("mass for unseen events that it's worse than the unigram model. Kneser-Ney discounts and backs off to a smarter")
print("lower-order model: how many different contexts a word appears in, not how often.")
assert math.isinf(results["bigram, MLE"])
assert results["bigram, add-one"] > results["bigram, add-0.01"] > results["trigram, interpolated Kneser-Ney"]
assert results["trigram, interpolated Kneser-Ney"] < 0.6 * results["unigram (MLE)"]

# sanity check: KN probabilities over the vocabulary sum to 1 for a context
h = ("the", "model")
tot = sum(kn_tri(h, w) for w in list(vocab) + ["<unk>"])
print(f"KN probabilities after 'the model' sum to {tot:.4f}")
assert abs(tot - 1) < 0.02

# %% [markdown]
# ## 4. Generating from the trigram model

# %%
vocab_list = sorted(vocab) + ["<unk>"]


def generate(n_words=40, seed=0):
    r = np.random.default_rng(seed)
    h = ("<s>", "the")
    out = ["the"]
    for _ in range(n_words):
        cands = [w for (a_, b_, w) in c3 if (a_, b_) == h] or vocab_list          # sample only from seen continuations + backoff
        p = np.array([kn_tri(h, w) for w in cands]); p /= p.sum()
        w = cands[r.choice(len(cands), p=p)]
        if w == "</s>":
            break
        out.append(w); h = (h[1], w)
    return " ".join(out)


print("\ntrigram samples:")
for sd in range(3):
    print("  " + generate(seed=sd))
print("locally fluent, globally aimless: three words of memory. Every n-gram sample reads like a tired consultant.")

# %% [markdown]
# ## 5. Bits per character
#
# Word-level and character-level perplexities aren't comparable directly. Total bits over the test text are:
# bits per char = (tokens x log2(word perplexity)) / characters.

# %%
kn_ppl = results["trigram, interpolated Kneser-Ney"]
n_pred = sum(1 for t in te if t != "<s>")
bpc = n_pred * math.log2(kn_ppl) / len(test_text)
print(f"\nKneser-Ney trigram: {bpc:.2f} bits per character of the raw test text")
print(f"lab 28.1's character RNN: perplexity about 5.1 per character = {math.log2(5.1):.2f} bits per character")
print("don't read that as counting beating a neural network. The word model predicts lowercased words, spends one symbol")
print("on every rare word (<unk>), and never pays for the math, code and markdown my tokenizer threw away; the RNN")
print("predicted every raw character. Perplexities are only comparable on the same text with the same tokenization.")
print("That caveat is also why LLM perplexities from different tokenizers can't be compared (35.1).")
assert bpc < 3

print("\nAll checks passed.")
