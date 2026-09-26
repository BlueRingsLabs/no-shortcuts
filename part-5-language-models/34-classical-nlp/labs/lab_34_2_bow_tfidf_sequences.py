# %% [markdown]
# # Lab 34.2: Bag of words, TF-IDF and sequence labeling
#
# 1. Text classification: which Part of this course did a paragraph come from? Bag of words + naive Bayes, TF-IDF +
#    logistic regression, bigrams, and what the model actually looks at.
# 2. The hashing trick: a fixed-size feature space with no vocabulary to store.
# 3. Named entity recognition on synthetic sentences, tested on people the models have never seen:
#    word identity only, an HMM with Viterbi, a feature-based token classifier, and a linear-chain CRF.
# 4. Entity-level F1, and why per-token accuracy flatters.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from corpus import PARTS, labeled_paragraphs, make_ner  # noqa: E402

torch.manual_seed(342)
torch.set_num_threads(4)

# %% [markdown]
# ## 1. Classifying paragraphs

# %%
data = labeled_paragraphs()
texts, labels = [t for t, _ in data], np.array([l for _, l in data])
print(f"{len(texts)} paragraphs; per Part: {np.bincount(labels).tolist()} ({', '.join(p.split('-', 2)[2] for p in PARTS)})")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
models = {
    "majority class": make_pipeline(CountVectorizer(), DummyClassifier()),
    "bag of words + naive Bayes": make_pipeline(CountVectorizer(), MultinomialNB()),
    "TF-IDF + logistic regression": make_pipeline(TfidfVectorizer(sublinear_tf=True), LogisticRegression(C=10, max_iter=2000)),
    "TF-IDF (words + bigrams) + LR": make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
                                                   LogisticRegression(C=10, max_iter=2000)),
}
res = {}
print("\n5-fold cross-validated accuracy:")
for name, m in models.items():
    res[name] = cross_val_score(m, texts, labels, cv=cv).mean()
    print(f"  {name:34s} {res[name]:.3f}")
assert res["TF-IDF + logistic regression"] > res["majority class"] + 0.2

m = models["TF-IDF + logistic regression"].fit(texts, labels)
vec, clf = m.named_steps["tfidfvectorizer"], m.named_steps["logisticregression"]
names = np.array(vec.get_feature_names_out())
print("\nthe words with the largest weights per class (the model's evidence):")
for c, part in enumerate(PARTS):
    print(f"  {part.split('-', 2)[2]:18s} " + ", ".join(names[np.argsort(-clf.coef_[c])[:10]]))
print("a linear model on word counts is a keyword detector with calibrated weights. Often that's all the task needs.")
print("(with 800 short documents, naive Bayes edges out logistic regression and bigrams add nothing: 19.2's small-data story.)")

# %% [markdown]
# ## 2. The hashing trick

# %%
for n_features in (2 ** 8, 2 ** 12, 2 ** 18):
    hm = make_pipeline(HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2"),
                       LogisticRegression(C=10, max_iter=2000))
    acc = cross_val_score(hm, texts, labels, cv=cv).mean()
    print(f"hashing into {n_features:7,} buckets (vocabulary {len(names):,} words): accuracy {acc:.3f}")
print("no vocabulary to fit, store or keep in sync: new words just land in a bucket. Too few buckets and words collide.")

# %% [markdown]
# ## 3. Named entity recognition
#
# Training sentences use one half of the name lists; test sentences use people, organizations and places never seen in
# training. Sentences start with a capital letter, and 15% of entity tokens are lowercased (tickets, chat, OCR). A model
# that memorized names fails; one that learned what names look like and where they appear has a chance.

# %%
train_ner, test_ner = make_ner(600, seed=0), make_ner(300, seed=1, unseen_names=True)
TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
t2i = {t: i for i, t in enumerate(TAGS)}


def spans(tags):
    out, start = set(), None
    for i, t in enumerate(tags + ["O"]):
        if start is not None and not t.startswith("I-"):
            out.add((start, i, tags[start][2:])); start = None
        if t.startswith("B-") or (t.startswith("I-") and start is None):         # an I- without a B- starts a (broken) span
            start = i
    return out


def evaluate(pred_seqs, data):
    tok_acc = np.mean([p == g for ps, (_, gs) in zip(pred_seqs, data) for p, g in zip(ps, gs)])
    tp = fp = fn = 0
    invalid = 0
    for ps, (_, gs) in zip(pred_seqs, data):
        P, G = spans(ps), spans(gs)
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
        invalid += any(t.startswith("I-") and (i == 0 or ps[i - 1][2:] != t[2:]) for i, t in enumerate(ps))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    per = [(p, g) for ps, (_, gs) in zip(pred_seqs, data) for p, g in zip(ps, gs) if g.endswith("PER")]
    return tok_acc, 2 * prec * rec / max(prec + rec, 1e-9), np.mean([p == g for p, g in per]), invalid / len(data)


# 3a. word identity only: the most frequent tag for each word in training
from collections import Counter, defaultdict  # noqa: E402
word_tags = defaultdict(Counter)
for toks, tags in train_ner:
    for w, t in zip(toks, tags):
        word_tags[w][t] += 1
lookup = [[word_tags[w].most_common(1)[0][0] if w in word_tags else "O" for w in toks] for toks, _ in test_ner]


# 3b. HMM: transitions and emissions counted, add-0.1 smoothing, decoded with Viterbi
def train_hmm(data, alpha=0.1):
    K = len(TAGS)
    trans = np.full((K + 1, K), alpha)                                              # row K = start
    emit = defaultdict(lambda: np.full(K, alpha))
    for toks, tags in data:
        prev = K
        for w, t in zip(toks, tags):
            trans[prev, t2i[t]] += 1; emit[w][t2i[t]] += 1; prev = t2i[t]
    tag_tot = np.sum([v for v in emit.values()], axis=0)
    return np.log(trans / trans.sum(1, keepdims=True)), emit, tag_tot


def viterbi(scores, trans_log, start_log):
    """scores: (T, K) log emission/feature scores. Returns the best tag sequence under the transition model."""
    T, K = scores.shape
    dp, back = start_log + scores[0], np.zeros((T, K), dtype=int)
    for t in range(1, T):
        cand = dp[:, None] + trans_log + scores[t][None, :]
        back[t] = cand.argmax(0); dp = cand.max(0)
    path = [int(dp.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


trans_log, emit, tag_tot = train_hmm(train_ner)
unk_emit = np.log(np.full(len(TAGS), 0.1) / (tag_tot + 0.1 * 50))                   # an unseen word: the same small count everywhere
hmm_pred = []
for toks, _ in test_ner:
    sc = np.stack([np.log(emit[w] / (tag_tot + 0.1 * 50)) if w in emit else unk_emit for w in toks])
    hmm_pred.append([TAGS[i] for i in viterbi(sc, trans_log[:-1], trans_log[-1])])


# 3c. features: the word, its shape, suffixes, and the neighbors
def features(toks, i):
    w = toks[i]
    f = [f"w={w.lower()}", f"shape={'X' if w[0].isupper() else 'x'}", f"suf3={w[-3:].lower()}", f"len={min(len(w), 8)}"]
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(toks):
            f += [f"w{off}={toks[j].lower()}", f"shape{off}={'X' if toks[j][0].isupper() else 'x'}"]
        else:
            f.append(f"w{off}=<pad>")
    return f


feat_vocab = {}
for toks, _ in train_ner:
    for i in range(len(toks)):
        for f in features(toks, i):
            feat_vocab.setdefault(f, len(feat_vocab))


def featurize(toks):
    ids = [[feat_vocab[f] for f in features(toks, i) if f in feat_vocab] or [0] for i in range(len(toks))]
    L = max(len(x) for x in ids)
    return torch.tensor([x + [len(feat_vocab)] * (L - len(x)) for x in ids])           # padded with a dummy feature id


class Tagger(nn.Module):
    """Emission scores = sum of learned feature weights. crf=False: independent softmax per token.
    crf=True: plus a learned transition matrix, trained with the forward algorithm (a linear-chain CRF)."""
    def __init__(self, crf):
        super().__init__()
        self.emb = nn.EmbeddingBag(len(feat_vocab) + 1, len(TAGS), mode="sum", padding_idx=len(feat_vocab))
        self.trans, self.start = nn.Parameter(torch.zeros(len(TAGS), len(TAGS))), nn.Parameter(torch.zeros(len(TAGS)))
        self.crf = crf

    def emissions(self, X):
        return self.emb(X)                                                            # (T, K)

    def nll(self, X, y):
        e = self.emissions(X)
        if not self.crf:
            return nn.functional.cross_entropy(e, y, reduction="sum")
        gold = self.start[y[0]] + e[torch.arange(len(y)), y].sum() + self.trans[y[:-1], y[1:]].sum()
        alpha = self.start + e[0]
        for t in range(1, len(y)):                                                    # forward algorithm: log-sum over all paths
            alpha = torch.logsumexp(alpha[:, None] + self.trans, 0) + e[t]
        return torch.logsumexp(alpha, 0) - gold

    @torch.no_grad()
    def decode(self, X):
        e = self.emissions(X).numpy()
        if not self.crf:
            return e.argmax(1).tolist()
        return viterbi(e, self.trans.numpy(), self.start.numpy())


def train_tagger(crf, epochs=4):
    torch.manual_seed(0)
    m = Tagger(crf)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    Xs = [(featurize(t), torch.tensor([t2i[x] for x in g])) for t, g in train_ner]
    for _ in range(epochs):
        for i in torch.randperm(len(Xs)).tolist():
            loss = m.nll(*Xs[i]) + 1e-4 * m.emb.weight.pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


t0 = time.time()
token_clf, crf = train_tagger(False), train_tagger(True)
print(f"\ntoken classifier and CRF trained in {time.time() - t0:.0f}s")
preds = {
    "word identity (most frequent tag)": lookup,
    "HMM + Viterbi": hmm_pred,
    "features, token by token": [[TAGS[i] for i in token_clf.decode(featurize(t))] for t, _ in test_ner],
    "features + linear-chain CRF": [[TAGS[i] for i in crf.decode(featurize(t))] for t, _ in test_ner],
}
print("\ntest sentences with entities never seen in training:")
print("  model                               token acc   entity F1   PER-token acc   sentences with invalid BIO")
ner_res = {}
for name, p in preds.items():
    ner_res[name] = evaluate(p, test_ner)
    ta, f1, per, inv = ner_res[name]
    print(f"  {name:36s} {ta:8.3f}   {f1:9.3f}   {per:13.3f}   {inv:26.1%}")
print("token accuracy looks respectable for everyone, because most tokens are O. Entity F1 counts an entity only if its")
print("span and type are exactly right. Identity-based models (lookup, HMM) have never seen these entities; features about")
print("shape and context generalize. The token-by-token classifier emits I- tags with no B- before them in about a third")
print("of the sentences: each decision is made alone. The CRF scores whole sequences and never does.")
assert ner_res["features + linear-chain CRF"][1] > ner_res["HMM + Viterbi"][1] + 0.2
assert ner_res["word identity (most frequent tag)"][0] > 0.6 and ner_res["word identity (most frequent tag)"][1] < 0.1
assert ner_res["features + linear-chain CRF"][1] > ner_res["features, token by token"][1]
assert ner_res["features + linear-chain CRF"][3] == 0 and ner_res["features, token by token"][3] > 0.1

print("\nAll checks passed.")
