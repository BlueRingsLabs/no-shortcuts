"""sentence-transformers in miniature: WordPiece tokenizer, a BERT forward pass in NumPy, mean pooling."""
import unicodedata

import numpy as np
from scipy.special import erf


# -- the tokenizer --------------------------------------------------------------------------------
def _is_punctuation(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True                                   # all ASCII symbols count, even $ and ^
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return any(lo <= cp <= hi for lo, hi in [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
                                             (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF),
                                             (0xF900, 0xFAFF), (0x2F800, 0x2FA1F)])


def _clean(text):
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ord(ch) in (0, 0xFFFD) or (cat.startswith("C") and ch not in "\t\n\r"):
            continue                                  # control characters are dropped
        if ch in " \t\n\r" or cat == "Zs":
            out.append(" ")
        elif _is_cjk(ord(ch)):
            out.append(f" {ch} ")                     # each Chinese character is its own word
        else:
            out.append(ch)
    return "".join(out)


def basic_tokenize(text):
    words = []
    for word in _clean(text).split():
        word = unicodedata.normalize("NFD", word.lower())
        word = "".join(ch for ch in word if unicodedata.category(ch) != "Mn")   # café -> cafe
        current = ""
        for ch in word:
            if _is_punctuation(ch):
                if current:
                    words.append(current)
                words.append(ch)
                current = ""
            else:
                current += ch
        if current:
            words.append(current)
    return words


class MiniWordPiece:
    def __init__(self, vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
        self.unk = self.vocab["[UNK]"]

    def wordpiece(self, word):
        if len(word) > 100:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end = len(word)
            while end > start:                        # the longest piece that is in the vocabulary
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                    break
                end -= 1
            else:
                return [self.unk]                     # one piece missing: the whole word is unknown
            start = end
        return ids

    def encode(self, text, max_length=256):
        ids = [i for w in basic_tokenize(text) for i in self.wordpiece(w)]
        return [self.vocab["[CLS]"]] + ids[:max_length - 2] + [self.vocab["[SEP]"]]

    def batch(self, texts, max_length=256):
        rows = [self.encode(t, max_length) for t in texts]
        width = max(len(r) for r in rows)
        ids = np.array([r + [0] * (width - len(r)) for r in rows])        # [PAD] is id 0
        mask = np.array([[1] * len(r) + [0] * (width - len(r)) for r in rows])
        return ids, mask


# -- the transformer --------------------------------------------------------------------------------
def layer_norm(x, w, b, eps=1e-12):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def bert(W, ids, mask, n_layers=6, n_heads=12):
    """W is the model's state_dict as NumPy arrays. Returns one vector per token."""
    b, s = ids.shape
    x = (W["embeddings.word_embeddings.weight"][ids]
         + W["embeddings.position_embeddings.weight"][:s]
         + W["embeddings.token_type_embeddings.weight"][0])
    x = layer_norm(x, W["embeddings.LayerNorm.weight"], W["embeddings.LayerNorm.bias"])
    blocked = np.where(mask[:, None, None, :] == 1, 0.0, -1e9)          # padding gets no attention

    def linear(x, name):
        return x @ W[name + ".weight"].T + W[name + ".bias"]

    def heads(x):                                    # (b, s, 384) -> (b, 12, s, 32)
        return x.reshape(b, s, n_heads, -1).transpose(0, 2, 1, 3)

    for i in range(n_layers):
        p = f"encoder.layer.{i}."
        q, k, v = (heads(linear(x, p + "attention.self." + n)) for n in ("query", "key", "value"))
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1]) + blocked
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        attn = scores / scores.sum(-1, keepdims=True)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, -1)
        x = layer_norm(x + linear(ctx, p + "attention.output.dense"),
                       W[p + "attention.output.LayerNorm.weight"], W[p + "attention.output.LayerNorm.bias"])
        h = gelu(linear(x, p + "intermediate.dense"))
        x = layer_norm(x + linear(h, p + "output.dense"),
                       W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"])
    return x


# -- pooling and the whole encode ------------------------------------------------------------------
def mean_pool(tokens, mask):
    m = mask[:, :, None]
    return (tokens * m).sum(1) / np.clip(m.sum(1), 1e-9, None)


def normalize(v):
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def mini_encode(tok, W, texts):
    ids, mask = tok.batch(texts)
    return normalize(mean_pool(bert(W, ids, mask), mask))
