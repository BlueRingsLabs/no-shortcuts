# %% [markdown]
# # Lab 35.3: A pretraining data pipeline, in miniature
#
# A synthetic "crawl" with known ground truth: good documents (paragraphs of this course), boilerplate, keyword spam,
# machine gibberish, non-English pages, exact and near duplicates, personal data, and leaked benchmark questions.
# Then the pipeline stages that real datasets (C4, RefinedWeb, FineWeb, Dolma) run, each measured against the truth:
#
# 1. Heuristic quality filters (Gopher/C4-style rules).
# 2. Language identification with character n-grams.
# 3. Exact deduplication by hashing, near-duplicate detection with MinHash + LSH.
# 4. PII scrubbing with regular expressions, and its false positives.
# 5. Benchmark contamination: 13-gram overlap.
# 6. A classifier-based quality filter, the way FineWeb-Edu and DCLM do it.

# %%
import hashlib
import zlib
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from corpus import labeled_paragraphs  # noqa: E402

r = random.Random(353)
paras = [p for p, _ in labeled_paragraphs(min_words=40)]
r.shuffle(paras)
good, reference = paras[:400], paras[400:]                                           # reference: "trusted" text for the classifier

# %% [markdown]
# ## Building the crawl

# %%
BOILER = ["Accept all cookies", "Home | About | Products | Contact | Login", "Subscribe to our newsletter!",
          "© 2025 All rights reserved.", "Share on Facebook Share on Twitter", "Click here to read more >>",
          "Privacy Policy Terms of Service Sitemap", "Loading...", "Sign up now and get 10% off"]
SPAM_WORDS = ["cheap", "best", "buy", "online", "discount", "free", "shipping", "deal", "sale", "top", "price", "now"]
SPANISH = ["Los modelos de lenguaje no entienden el texto como nosotros, y un tokenizador entrenado casi solo con inglés "
           "parte las palabras de otros idiomas en más pedazos, lo que encarece cada consulta y reduce el contexto útil.",
           "La calidad de los datos importa más que la cantidad: un conjunto pequeño y limpio suele ganarle a uno enorme y "
           "sucio, sobre todo cuando se entrena un modelo durante varias épocas sobre los mismos ejemplos.",
           "Antes de entrenar hay que deduplicar, filtrar y revisar a mano una muestra; si nadie mira los datos, los datos "
           "terminan mirando al modelo desde cada respuesta extraña que produce en producción."]
BENCH = [f"Question {i}: " + p.split(". ")[0] + "?" for i, p in enumerate(reference[:30])]   # a "benchmark" built from text


def kind_doc(kind):
    if kind == "good":
        return r.choice(good)
    if kind == "boilerplate":
        return "\n".join(r.choice(BOILER) for _ in range(r.randint(4, 9)))
    if kind == "spam":
        return " ".join(r.choice(SPAM_WORDS) for _ in range(r.randint(40, 120)))
    if kind == "gibberish":
        letters = "abcdefghijklmnopqrstuvwxyz"
        return " ".join("".join(r.choice(letters) for _ in range(r.randint(2, 11))) for _ in range(r.randint(50, 150)))
    if kind == "spanish":
        return " ".join(r.sample(SPANISH, 2))
    raise ValueError(kind)


docs, truth = [], []
for kind, n in (("good", 400), ("boilerplate", 60), ("spam", 60), ("gibberish", 60), ("spanish", 40)):
    for i in range(n):
        docs.append(good[i] if kind == "good" else kind_doc(kind)); truth.append(kind)
for i in range(60):                                                                     # exact duplicates of good pages
    docs.append(good[i]); truth.append("exact-dup")
near_pairs = set()
for i in range(60, 120):                                                                # near duplicates: a few words changed
    w = good[i].split()
    for _ in range(max(1, len(w) // 30)):
        w[r.randrange(len(w))] = r.choice(["the", "a", "model", "data"])
    docs.append(" ".join(w)); truth.append("near-dup"); near_pairs.add((i, len(docs) - 1))
pii_docs = []
for i in range(120, 150):                                                              # personal data sprinkled into good pages
    user = r.choice(["ana.garcia", "j.okafor", "mei.chen", "l.rossi"])
    extra = (f" Contact {user}@example.com or call +1 415 555 {r.randint(1000, 9999)}. Server at 10.0.{r.randint(0, 255)}.{r.randint(1, 254)}, "
             f"key AKIA{''.join(r.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ234567') for _ in range(16))}.")
    docs.append(good[i] + extra); truth.append("good+pii"); pii_docs.append(len(docs) - 1)
for q in BENCH[:10]:                                                                  # 10 of 30 benchmark questions leak into the crawl
    docs.append(r.choice(good[150:250]) + " " + q + " The answer is in the documentation."); truth.append("contaminated")
truth = np.array(truth)
print(f"crawl: {len(docs)} documents: " + ", ".join(f"{k} {int((truth == k).sum())}" for k in dict.fromkeys(truth)))
bad_kinds = {"boilerplate", "spam", "gibberish"}

# %% [markdown]
# ## 1. Heuristic quality filters

# %%
STOP = {"the", "be", "to", "of", "and", "that", "have", "with", "is", "it", "for", "as", "on", "this", "a", "in"}


def quality_flags(doc):
    ws = doc.split()
    lines = [l for l in doc.split("\n") if l.strip()]
    flags = []
    if len(ws) < 30:
        flags.append("too short")
    if ws and not 3 <= np.mean([len(w) for w in ws]) <= 10:
        flags.append("odd word length")
    if ws and sum(w.lower() in STOP for w in ws) < 2:
        flags.append("no stopwords")
    if ws and len(set(ws)) / len(ws) < 0.3:
        flags.append("repetitive")
    if lines and np.mean([l.rstrip()[-1:] in ".!?\"')" for l in lines]) < 0.5 and len(lines) > 2:
        flags.append("lines without punctuation")
    return flags


flagged = np.array([bool(quality_flags(d)) for d in docs])
is_bad = np.isin(truth, list(bad_kinds))
print("\nheuristic filters:")
print(f"  junk removed: {flagged[is_bad].mean():.0%} of boilerplate, spam and gibberish")
print(f"  good text removed by mistake: {flagged[np.isin(truth, ['good', 'good+pii', 'exact-dup', 'near-dup', 'contaminated'])].mean():.1%}")
print(f"  Spanish removed: {flagged[truth == 'spanish'].mean():.0%}  <- the 'no stopwords' rule is an English rule")
reasons = defaultdict(int)
for d, t in zip(docs, truth):
    for f in quality_flags(d):
        reasons[(t, f)] += 1
print("  why junk was flagged:", {f"{t}: {f}": n for (t, f), n in sorted(reasons.items()) if t in bad_kinds and n >= 20})
assert flagged[is_bad].mean() > 0.95 and flagged[truth == "good"].mean() < 0.05

# %% [markdown]
# ## 2. Language identification
#
# Character trigram profiles from about 500 characters of each language: that's enough, which says something about
# how distinctive languages are at the character level. (In practice: fastText's language ID, 176 languages.)

# %%
def trigrams(s):
    s = f"  {s.lower()}  "
    return [s[i:i + 3] for i in range(len(s) - 2)]


def profile(texts):
    c = defaultdict(int)
    for t in texts:
        for g in trigrams(t):
            c[g] += 1
    return c, sum(c.values())


ES_SAMPLE = ("El conjunto de datos contiene textos de muchas fuentes distintas, y cada una tiene sus propios problemas: "
             "páginas repetidas, menús de navegación, anuncios y a veces información personal que no debería estar ahí. "
             "Por eso el filtrado es una parte central del proceso de entrenamiento. Un buen equipo revisa muestras a mano, "
             "mide qué se elimina en cada paso y se pregunta a quién deja afuera cada regla, porque una regla pensada para "
             "un idioma casi nunca funciona igual de bien para los demás.")
en_sample = " ".join(reference[:3])[:len(ES_SAMPLE)]                                   # the same amount of text per language
en_prof, es_prof = profile([en_sample]), profile([ES_SAMPLE])


def lang(doc):
    """Naive Bayes over character trigrams with add-one smoothing (34.1): an unseen trigram is unlikely, not impossible."""
    tg = trigrams(doc)
    s = {name: sum(np.log((c.get(g, 0) + 1) / (tot + 20000)) for g in tg) for name, (c, tot) in (("en", en_prof), ("es", es_prof))}
    return max(s, key=s.get)


langs = np.array([lang(d) for d in docs])
print(f"\nlanguage ID: Spanish pages detected {np.mean(langs[truth == 'spanish'] == 'es'):.0%}, "
      f"English pages called Spanish {np.mean(langs[truth == 'good'] == 'es'):.1%}")
assert np.mean(langs[truth == "spanish"] == "es") > 0.9 and np.mean(langs[truth == "good"] == "es") < 0.05

# %% [markdown]
# ## 3. Deduplication

# %%
norm = lambda d: re.sub(r"\s+", " ", d.lower()).strip()
seen, exact_dups = {}, []
for i, d in enumerate(docs):
    h = hashlib.sha1(norm(d).encode()).hexdigest()
    if h in seen:
        exact_dups.append(i)
    else:
        seen[h] = i
print(f"\nexact dedup (hash of normalized text): {len(exact_dups)} duplicates found; true exact duplicates: {int((truth == 'exact-dup').sum())}")
print("  (the spam and boilerplate generators also repeat themselves; those count as duplicates too)")

K, BANDS, ROWS = 128, 32, 4                                                            # 128 hashes = 32 bands of 4
rng = np.random.default_rng(0)
A, B = rng.integers(1, 2 ** 31 - 1, K), rng.integers(0, 2 ** 31 - 1, K)
P = 2 ** 31 - 1


def shingles(d, k=5):
    w = norm(d).split()
    return {zlib.crc32(" ".join(w[i:i + k]).encode()) & 0x7FFFFFFF for i in range(max(1, len(w) - k + 1))}   # a stable hash


def minhash(sh):
    x = np.array(list(sh), dtype=np.int64)
    return ((A[:, None] * x[None, :] + B[:, None]) % P).min(1)                        # 128 independent min-hashes


sigs = [minhash(shingles(d)) for d in docs]
buckets = defaultdict(set)
for i, s in enumerate(sigs):
    for b in range(BANDS):
        buckets[(b, tuple(s[b * ROWS:(b + 1) * ROWS]))].add(i)
candidates = {tuple(sorted((i, j))) for ids in buckets.values() if len(ids) < 50 for i in ids for j in ids if i < j}


def jaccard(i, j):
    a, b = shingles(docs[i]), shingles(docs[j])
    return len(a & b) / len(a | b)


found = {p for p in candidates if jaccard(*p) > 0.7}
recall = np.mean([p in found for p in near_pairs])
print(f"MinHash + LSH: {len(candidates):,} candidate pairs from {len(docs) * (len(docs) - 1) // 2:,} possible; "
      f"{len(found)} confirmed at Jaccard > 0.7; recall of the planted near-duplicates {recall:.0%}")
assert recall > 0.9

# %% [markdown]
# ## 4. PII scrubbing

# %%
PII = {"email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
       "phone": re.compile(r"\+?\d{1,3}[ -]?\d{3}[ -]?\d{3}[ -]?\d{4}\b"),
       "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
       "aws_key": re.compile(r"\bAKIA[A-Z2-7]{16}\b")}


def scrub(d):
    hits = {}
    for name, pat in PII.items():
        hits[name] = len(pat.findall(d))
        d = pat.sub(f"<{name.upper()}>", d)
    return d, hits


caught = [scrub(docs[i])[1] for i in pii_docs]
print(f"\nPII in the 30 planted documents: " + ", ".join(f"{k} {sum(c[k] for c in caught)}" for k in PII))
print("example:", scrub(docs[pii_docs[0]])[0][-150:])
fp_text = "Upgrade to version 2.10.3.1 before Friday; the 1.1.1.1 resolver and the ratio 3.14.15.92 are not addresses of people."
print("false positives on innocent text:", scrub(fp_text)[1], "->", scrub(fp_text)[0])
print("regexes catch the formats you thought of and some things that merely look like them. Real pipelines add NER")
print("models (34.2) for names and addresses, and still miss things: treat pretraining data as containing PII.")
assert all(c["email"] == 1 and c["aws_key"] == 1 for c in caught)

# %% [markdown]
# ## 5. Contamination
#
# GPT-3's method: flag a benchmark item if any 13-gram of it appears in the training data. (Our questions are short, so
# we use 8-grams; the idea is identical.)

# %%
def ngrams(text, n=8):
    w = norm(text).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


train_grams = set()
kept = [i for i in range(len(docs)) if not flagged[i]]
for i in kept:
    train_grams |= ngrams(docs[i])
contaminated = [q for q in BENCH if ngrams(q) & train_grams]
planted = BENCH[:10]
caught_planted = [q for q in planted if q in contaminated]
missed = [q for q in planted if q not in contaminated]
print(f"\nbenchmark questions with an 8-gram in the training data: {len(contaminated)} of {len(BENCH)}; "
      f"of the 10 planted, {len(caught_planted)} caught")
for q in missed:
    print(f"  missed: {q!r} ({len(norm(q).split())} words: too short to contain an 8-gram at all)")
print("false positives: none of the 20 clean questions matched. Short items slip through long-n-gram checks, and")
print("paraphrased or translated leaks slip through any exact n-gram check: contamination detection has a recall problem.")
print("Report clean and contaminated subsets separately (39.1), and decontaminate training data against every benchmark.")
assert len(caught_planted) + len(missed) == 10 and all(len(norm(q).split()) < 8 for q in missed)
assert set(contaminated) <= set(planted)

# %% [markdown]
# ## 6. A classifier-based quality filter
#
# Train a classifier to tell "reference-like" text (trusted paragraphs) from random crawl text, then keep the crawl
# documents it scores highly. The reference defines "quality": whatever it over- or under-represents, the filter will too.

# %%
n_ref = len(reference)
crawl_sample = [docs[i] for i in r.sample(range(len(docs)), n_ref)]
X_train = reference + crawl_sample
y_train = [1] * n_ref + [0] * n_ref
print(f"\nquality classifier: {n_ref} trusted paragraphs vs {n_ref} random crawl documents")
qvec = TfidfVectorizer(sublinear_tf=True, min_df=2)
qclf = LogisticRegression(C=3, max_iter=2000).fit(qvec.fit_transform(X_train), y_train)
score = qclf.predict_proba(qvec.transform(docs))[:, 1]
for thr in (0.5, 0.3):
    keep = score > thr
    print(f"threshold {thr}: keeps good {keep[truth == 'good'].mean():.0%}, junk {keep[is_bad].mean():.0%}, "
          f"Spanish {keep[truth == 'spanish'].mean():.0%}")
print("the negatives were a random crawl sample, and half the crawl is good text, so the classifier is unsure about good")
print("pages: the threshold decides how many survive. Choosing negatives and thresholds is most of the work (DCLM, FineWeb-Edu).")
print("And it learned what the reference looks like, which includes 'English, about ML, written by me'. A filter trained on")
print("one voice or one dialect quietly removes others: decide what 'quality' means before a classifier decides for you.")
print("in a real pipeline the heuristics of section 1 run first and remove most of that junk, so the classifier's")
print("threshold can favor recall of good text.")
k3, k5 = score > 0.3, score > 0.5
assert k3[truth == "good"].mean() > k5[truth == "good"].mean() + 0.2 and k5[is_bad].mean() < 0.05 < k3[is_bad].mean()
assert (score > 0.5)[truth == "spanish"].mean() < 0.1

print("\nAll checks passed.")
