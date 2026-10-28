# G. sentence-transformers in miniature

This is the chapter where the copy-paste phase is most comfortable, because the library makes it so easy. Two lines and you have embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(["The cat sits on the mat.", "A kitten is resting on a rug."])
```

And then what? What's in `embeddings`, why does the dot product of two of them mean anything, why is a long document's vector no better than its first page's, and what does that model name even refer to? The idea comes from Sentence-BERT ([Reimers and Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", 2019](https://arxiv.org/abs/1908.10084)<sup>\[18\]</sup>): take a BERT-style transformer, average what it outputs for each token, and train the whole thing so that sentences with similar meanings end up with similar averages. The library is a very convenient wrapper around that idea, and under the convenience there are four steps:

1. A **tokenizer** turns text into integer ids.
2. A **transformer** turns the ids into one vector per token.
3. **Pooling** averages those vectors into one vector per text.
4. **Normalization** scales it to length 1.

This chapter rebuilds all four. The tokenizer and the transformer go all the way down to plain Python and NumPy, using the real model's vocabulary and weights, and the result is compared with `model.encode` number by number. The examples use sentence-transformers 6.2.0, transformers 5.21.0 and the model `all-MiniLM-L6-v2`, which is a 90.9 MB download and runs fine on a laptop CPU.

## 1. What encode gives you

Start by asking the result what it is, the same way we asked arrays and tensors:

```python
import numpy as np

sentences = ["The cat sits on the mat.", "A kitten is resting on a rug.",
             "Quarterly revenue grew by 8%.", "The stock market fell sharply today."]
emb = model.encode(sentences)
print(type(emb).__name__, emb.shape, emb.dtype)
print(np.linalg.norm(emb, axis=1).round(6))
print((emb @ emb.T).round(3))
```

```text
ndarray (4, 384) float32
[1. 1. 1. 1.]
[[ 1.     0.641 -0.039  0.085]
 [ 0.641  1.    -0.035  0.093]
 [-0.039 -0.035  1.     0.168]
 [ 0.085  0.093  0.168  1.   ]]
```

A NumPy array, one row of 384 float32 numbers per sentence, and every row has length exactly 1. That last fact is the useful one. For vectors of length 1, the dot product *is* the cosine similarity, so `emb @ emb.T` is the full similarity matrix, and it already says something sensible: the cat and the kitten score 0.641, even though the only word the two sentences share is "on"; the two finance sentences are closer to each other (0.168) than to either animal; and nothing about cats is similar to revenue. `model.similarity(emb, emb)` gives the same matrix (it computes cosine similarity by default); with normalized vectors it's the same arithmetic.

Then ask the model what it's made of:

```python
print(model)
```

```text
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

There are the four steps (the tokenizer lives inside the `Transformer` module). A `SentenceTransformer` is a PyTorch `nn.Sequential` underneath, which means everything in [chapter F](F-pytorch-in-miniature.md) applies: you can index it (`model[0]` is the transformer), count its parameters, and look at its `state_dict`. The downloaded folder says the same thing in files: a `modules.json` listing those three modules, a `1_Pooling/config.json` that says `"pooling_mode_mean_tokens": true`, a `config.json` describing a `BertModel` with 6 layers, 12 attention heads and vectors of size 384, and a `vocab.txt` with the tokenizer's vocabulary. If you never opened `~/.cache/huggingface`, open it now. It's just files, and they're readable.

And then read the model card, because the name `all-MiniLM-L6-v2` hides the most important information. According to its card on the Hugging Face Hub ([sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)<sup>\[19\]</sup>), it started from a small pretrained model, `nreimers/MiniLM-L6-H384-uncased` (that's the L6 and the H384: 6 layers, hidden size 384, lowercased text), and was fine-tuned with a contrastive objective on more than a billion sentence pairs, learning to tell which sentence was paired with which. It's "intended to be used as a sentence and short paragraph encoder", and "by default, input text longer than 256 word pieces is truncated". Keep that last sentence in mind; [section G.6](#6-the-256-token-wall) is about it.

## 2. encode in fifteen lines

The first rebuild uses the lower-level library, `transformers`, for the tokenizer and the transformer, and writes the rest by hand:

```python
import torch
from transformers import AutoModel, AutoTokenizer

name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(name)
bert = AutoModel.from_pretrained(name)


def encode(texts):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        tokens = bert(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    mean = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(mean, dim=1).numpy()
```

Compared with `model.encode` on four sentences of different lengths, the largest difference between the two results is `0.0`. Not "close": the same bits. That's because it's literally the same computation, done by the same PyTorch operations in the same order, and it's worth reading line by line:

- The tokenizer gets a list of texts and pads them to the same length (`padding=True`), cuts them at 256 tokens (`truncation=True, max_length=256`), and returns PyTorch tensors. [Section G.3](#3-the-tokenizer-text-to-integers) opens it.
- `torch.no_grad()`, from [section F.7](F-pytorch-in-miniature.md#7-four-things-that-break-real-loops): we're not training, so don't record the tape. (`from_pretrained` also puts the model in eval mode for you; `bert.training` is `False`. Dropout is off, and the same text always gives the same vector.)
- `last_hidden_state` has shape `(texts, tokens, 384)`: one vector per token, **including the padding**. [Section G.4](#4-bert-in-forty-lines-of-numpy) opens the transformer.
- The `mask` is 1 for real tokens and 0 for padding, and the mean is a **masked** mean: sum only the real tokens, divide by how many there are. [Section G.5](#5-pooling-and-the-padding-bug) shows what happens if you use a plain `.mean()` here.
- `normalize` divides each vector by its length.

Fifteen lines, and the only parts we didn't write are the tokenizer and the transformer. Let's write those too.

## 3. The tokenizer: text to integers

A transformer can't read text. It reads integers, each one the index of a row in a big table of vectors (the embedding table), so the first job is turning text into integers. BERT's tokenizer does it in two stages, and both are simple enough to rebuild.

**Stage 1, the basic tokenizer**, splits text into words. With this model's settings, it lowercases, removes accents, and splits on whitespace and on every punctuation character, so punctuation becomes its own word:

```python
import unicodedata


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
```

```python
print(basic_tokenize("Tokenization isn't hard, it's just fiddly."))
```

```text
['tokenization', 'isn', "'", 't', 'hard', ',', 'it', "'", 's', 'just', 'fiddly', '.']
```

The accent trick is worth a second look. `unicodedata.normalize("NFD", ...)` splits "é" into "e" followed by a separate combining accent (Unicode category `Mn`, "mark, nonspacing"), and the next line throws the marks away. So "café" becomes "cafe" and "São Paulo" becomes "sao paulo", which is why this model can't tell "resume" from "résumé". `_clean` (in the companion code) removes control characters, turns every kind of whitespace into a space, and puts spaces around Chinese characters, so each one is its own word. `_is_punctuation` treats every ASCII symbol as punctuation, including `$`, `^` and `` ` ``, which Unicode itself doesn't call punctuation.

**Stage 2, WordPiece**, splits each word into pieces from a fixed vocabulary of 30,522 entries, `vocab.txt`, where line *n* is the token with id *n*. The algorithm is greedy: take the longest prefix of the word that's in the vocabulary, then the longest piece of what's left, marked with `##` to say "this continues a word", and so on:

```python
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
```

(The `while ... else` is one of Python's odder constructs: the `else` runs only if the loop ended without a `break`, which here means no piece of any length was found.)

"tokenization" becomes `token` + `##ization` (ids 19204 and 3989), and "fiddly" becomes `fi` + `##dd` + `##ly`. Common words are a single token, rare words are a few, and a word with a character the vocabulary has never seen becomes `[UNK]`, all of it. Every text gets `[CLS]` (id 101) at the start and `[SEP]` (id 102) at the end, because BERT was trained that way, and the 256 limit includes those two.

For a batch, the shorter texts are padded with `[PAD]`, id 0, and an **attention mask** records which positions are real:

```python
batch = model.tokenizer(["short", "a slightly longer sentence"], padding=True)
print(batch["input_ids"])
print(batch["attention_mask"])
```

```text
[[101, 2460, 102, 0, 0, 0], [101, 1037, 3621, 2936, 6251, 102]]
[[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
```

The padding exists only so the batch can be one rectangular tensor ([chapter C](C-numpy-in-miniature.md): arrays are rectangles). It means nothing, and the mask is how every later step knows to ignore it. Our `MiniWordPiece.batch` does the same thing, and on 2,000 random texts (with accents, emoji, Chinese characters, odd whitespace and control characters mixed in, some long enough to be truncated) it produces exactly the ids of the real tokenizer. [Section G.7](#7-checking-the-miniature-and-a-disagreement-i-didnt-expect) tells the story of the texts where it didn't, because there were some.

> [!NOTE]
> The real tokenizer is written in Rust (the `tokenizers` library) and is much faster than ours. Its documentation describes the same stages in its own words: normalization, pre-tokenization, the model (WordPiece here; BPE or Unigram for other models) and post-processing, which adds `[CLS]` and `[SEP]`. Once you've written the miniature, the page "The tokenization pipeline" in the `tokenizers` docs reads like a description of your own code.

## 4. BERT in forty lines of NumPy

Now the part that looks scary from the outside. The transformer in this model is a BERT encoder, the architecture of [Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2019](https://arxiv.org/abs/1810.04805)<sup>\[20\]</sup>, at a small size: 6 layers, 12 attention heads, vectors of size 384. Here it is, using the real weights, which we take from the model's `state_dict` ([section F.7](F-pytorch-in-miniature.md#7-four-things-that-break-real-loops)) and convert to NumPy:

```python
W = {k: v.double().numpy() for k, v in model[0].auto_model.state_dict().items()}
```

That's 103 arrays with names like `encoder.layer.0.attention.self.query.weight`, and 22,713,216 numbers in total. At 4 bytes per float32 that's 90.85 MB, which is the size of the file you downloaded (`model.safetensors`, 90.9 MB). Apart from a small header listing the names and shapes, the file is those numbers and nothing else. And here's the whole forward pass:

```python
import numpy as np
from scipy.special import erf


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
```

Read it with [chapters C](C-numpy-in-miniature.md) and [F](F-pytorch-in-miniature.md) in your head, and almost every line is something you've already seen:

- Embeddings are fancy indexing ([section C.5](C-numpy-in-miniature.md#5-fancy-indexing-and-boolean-masks-always-copy)). `word_embeddings[ids]` picks one row of a 30,522 × 384 table per token. A position embedding (one learned vector per position, up to 512) is added so the model knows the order of the words, plus a "token type" vector that BERT uses to tell two sentences apart; for a single text it's always row 0.
- Attention is three linear layers and two matrix products. Each token's vector is projected into a query, a key and a value. The score of token *i* attending to token *j* is the dot product of *i*'s query and *j*'s key, divided by √32. A softmax turns each row of scores into weights that add up to 1, and each token's new vector is the weighted average of all the values. "Multi-head" is the `heads` function: a reshape and a transpose ([section C.4](C-numpy-in-miniature.md#4-transpose-and-reshape-and-when-numpy-copies)) that splits the 384 numbers into 12 groups of 32 and lets each group attend separately. Then the reverse transpose and reshape glue them back together.
- The mask is broadcasting ([section C.6](C-numpy-in-miniature.md#6-broadcasting-is-a-stride-of-zero)). `mask[:, None, None, :]` has shape `(b, 1, 1, s)`, so a single −1,000,000,000 per padded position is broadcast to every head and every query. After the softmax, those positions get a weight of exactly zero. That's how padding stops mattering *inside* the transformer.
- `scores - scores.max(...)` before `np.exp` is the standard trick for a numerically stable softmax: it changes nothing mathematically (the softmax of *x* and of *x − c* are the same) and keeps `exp` from overflowing, which is [section C.9](C-numpy-in-miniature.md#9-dtypes-or-what-happens-when-a-number-doesnt-fit)'s problem of numbers that don't fit.
- Each sublayer is "add, then normalize": `layer_norm(x + f(x))`. That's the *post-norm* arrangement of the original 2017 transformer, which BERT kept. Modern LLMs moved the normalization inside the branch (pre-norm), for reasons the course explains in detail.
- The feed-forward part is two linear layers with a GELU between them, applied to every token separately: 384 → 1,536 → 384.

That's the whole transformer: a loop over six layers, each a fixed sequence of matrix products, softmaxes and normalizations. (One curiosity: the `state_dict` also contains `pooler.dense.weight` and `pooler.dense.bias`, a layer BERT uses for classification. `BertModel` computes it on every call, and sentence-transformers ignores the result. Two of the 103 arrays do work nobody uses.)

Two shapes are worth checking by hand, because getting them wrong is a common bug in hand-written attention. `q @ k.transpose(0, 1, 3, 2)` multiplies `(b, 12, s, 32)` by `(b, 12, 32, s)` and gives `(b, 12, s, s)`: for every head, a score for every pair of tokens. And `attn @ v` multiplies `(b, 12, s, s)` by `(b, 12, s, 32)` and gives `(b, 12, s, 32)` back. If you can predict those two shapes, you understand attention well enough to debug it.

## 5. Pooling, and the padding bug

The transformer returns one vector per token, and we want one per text. sentence-transformers averages them (`pooling_mode_mean_tokens`), with the masked mean from [section G.2](#2-encode-in-fifteen-lines):

```python
def mean_pool(tokens, mask):
    m = mask[:, :, None]
    return (tokens * m).sum(1) / np.clip(m.sum(1), 1e-9, None)


def normalize(v):
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def mini_encode(tok, W, texts):
    ids, mask = tok.batch(texts)
    return normalize(mean_pool(bert(W, ids, mask), mask))
```

(The `clip` calls are there so an empty mask or an all-zero vector gives zeros instead of a division by zero.)

The mask matters because the transformer also produces vectors for the padding positions. The attention mask keeps real tokens from looking at the padding, but nothing stops the padding positions from computing something, and they do. Put "short" in a batch with a long sentence, so it gets 14 padding positions, and compare a plain `.mean(axis=1)` with the real thing:

```python
ids, mask = tok.batch(["short", "a much longer sentence that forces a lot of padding onto the first one"])
tokens = bert(W, ids, mask)
good = normalize(mean_pool(tokens, mask))
naive = normalize(tokens.mean(axis=1))
alone = model.encode(["short"])[0]
print(float(good[0] @ alone), float(naive[0] @ alone))
```

```text
1.0 0.6889
```

With the masked mean, "short" gets the same vector it gets when it's encoded alone: similarity 1.0, as it should be. With the naive mean, its vector has a similarity of **0.69 with itself**, which on the scale of [section G.1](#1-what-encode-gives-you) is about as similar as a cat and a kitten. And it gets worse in the nastiest possible way: the vector of a text now depends on **which other texts happened to be in its batch**. Change the batch size, or sort your documents differently, and your search results change. No error, no warning. (The long sentence, which has no padding, comes out identical either way, so if you only test with texts of similar length you'll never see it.)

This is the kind of bug that can appear when you write your own pooling for speed, or translate code from one library to another and drop the mask because "it's just an average". If you write pooling yourself, test exactly this: the same text, alone and in a batch with a much longer one.

## 6. The 256-token wall

The model card said it: "input text longer than 256 word pieces is truncated". Watch what that does:

```python
text = " ".join(["The experiment was a success."] * 50)        # 302 tokens with [CLS] and [SEP]
a = model.encode([text + " Actually, it failed completely."])
b = model.encode([text])
print(np.array_equal(a, b))
```

```text
True
```

The conclusion changed completely, and the embedding didn't change at all, not even in the last decimal, because everything after token 256 is cut off before the transformer sees it. And `encode` doesn't warn you. (Calling the tokenizer directly does print a warning, "Token indices sequence length is longer than the specified maximum sequence length for this model (302 > 256)", but `encode` truncates silently, because truncating is the documented behavior.)

256 word pieces is a couple of paragraphs of ordinary English. That's why retrieval pipelines usually **split documents into chunks** before embedding them, and why the size of the chunks is one of the first things to tune. If you embed whole documents with this model, you're embedding their first page, and searching only first pages.

> [!WARNING]
> `model.max_seq_length` tells you the limit (256 here; other models have other limits, from 128 tokens to several thousand). Check it for every model you use, and count tokens with the model's own tokenizer, not words: technical text, code and non-English text produce more tokens per word than ordinary English.

## 7. Checking the miniature, and a disagreement I didn't expect

The test for this chapter has two parts. The first tokenizes 2,000 random texts, built from a list of awkward words ("naïve", "Zürich", "東京", "COVID-19", "user@example.com", "¿qué?", emoji, "x²", "½") joined with odd separators (tabs, non-breaking spaces, zero-width spaces, NUL bytes, the Unicode replacement character), some of them long enough to be truncated, and requires exactly the same ids as the real tokenizer. The second runs `mini_encode` on 30 random batches, with padding, and compares the vectors with `model.encode`:

```text
2030 checks against sentence-transformers 6.2.0: all passed
largest difference in the vectors: 2.6e-07
```

The remaining difference is float32 against float64 rounding. Our miniature computes in float64; PyTorch in float32. The last digits depend on the machine: on GitHub's CI runner the same test prints 2.0e-07. Anything around 1e-07 is the same result.

Then I got greedy, and sent random characters from all over Unicode through both tokenizers, and they disagreed a lot. So I compared the first stage on every single code point from U+0020 to U+2FFFF (194,528 of them), each one between an "a" and a "b":

```text
48575 of 194528 code points disagree
 47750  unassigned in Unicode 15.1.0
   419  other (category Mn)
   256  CJK U+2B820 to U+2B91F
   117  other (category Po)
    20  other (category Cf)
     4  other (category Ps)
     4  other (category Pe)
     2  other (category Mc)
     2  other (category Pd)
     1  other (category So)
```

A quarter of the code points xD. My first thought was that my miniature was broken. It isn't, or at least it's not *more* wrong than the original: my rules are the ones in Google's original BERT tokenizer (`tokenization.py` in the google-research/bert repository), which is also what the old pure-Python `BertTokenizer` in `transformers` 4.x did. The library that sentence-transformers actually runs today is the Rust one, and reading its source (`tokenizers/src/normalizers/bert.rs`, version 0.23.3) explains all three groups:

- The 47,750 unassigned code points come from how the Rust code decides what's a control character. It uses a crate called `unicode_categories`, whose last release is from 2016. In the crate's source, the comment on its function `is_other()` says it covers categories Cc, Cf, Cn and Co, but the implementation only checks Cc, Cf and Co. Cn, "unassigned", is missing, so the Rust tokenizer keeps unassigned code points where Python's version drops them.
- The newer characters are a version problem. Python 3.13 knows Unicode 15.1. A 2016 table doesn't know the marks and punctuation added since then, so they're treated as ordinary letters: not stripped, not split off.
- The 256 CJK characters come from one typo. Google's code lists the CJK block that starts at U+2B820 as Chinese characters. The Rust code has `0x2B920..=0x2CEAF`: a 9 where there should be an 8. The 256 characters in between aren't treated as Chinese, so they aren't split into separate words.

For this model it barely matters. Unassigned code points shouldn't appear in real text, the other characters are rare, and when one does appear the difference is only in which neighboring characters end up inside an `[UNK]`. All 2,000 realistic texts matched exactly. It was still worth chasing, for two reasons. First, it shows what "the same tokenizer" means: there's no tokenizer in the abstract, there's an implementation, with its own Unicode version and its own bugs, and two implementations of "BERT's tokenizer" disagree on a quarter of Unicode. If you ever train with one and serve with another (a Python service and a JavaScript client, say), test that they give the same ids on *your* text. Second, it's the FAQ's promise kept: a disagreement between the miniature and the library is a question worth answering, and answering this one taught me more about tokenizers than the 2,000 passing tests did.

The miniature keeps Google's rules, and the test only uses realistic text, where both implementations agree. The full comparison over Unicode is in the companion code, as `unicode_diff.py`, for the curious.

One more number, so you don't get ideas: in one run on 64 sentences, `mini_encode` took 802 ms and `model.encode` 155 ms on the same machine. NumPy in float64 is no match for PyTorch's float32 kernels, so keep the miniature for learning.

## 8. Reading the docs with this in your head

- The model card, first and always. It says what the model was trained on and for, its maximum length, and how to use it. A sentence encoder used on whole documents, or an English model used on Spanish text, is a model used for something its card says it isn't for, and no amount of tuning downstream fixes that.
- The sentence-transformers documentation (sbert.net): the "Semantic Search" and "Computing Embeddings" pages, and the list of pretrained models with their speed and quality. After this chapter, `normalize_embeddings`, `batch_size`, `convert_to_tensor` and the pooling modes are knobs on things you've built yourself.
- The `transformers` docs on tokenizers: padding, truncation, `return_tensors`, and which backend a tokenizer uses, which [section G.7](#7-checking-the-miniature-and-a-disagreement-i-didnt-expect) made concrete.
- The `tokenizers` docs, "The tokenization pipeline": the four stages of [section G.3](#3-the-tokenizer-text-to-integers), in the authors' words.
- The source: `inspect.getsource(type(model[1]).forward)` shows the real pooling, about thirty lines that cover every pooling mode. Compare the mean branch with your `mean_pool`.

## 9. Exercises

1. Change `mean_pool` to take the vector of the `[CLS]` token instead (`tokens[:, 0]`). Compute the similarity matrix of [section G.1](#1-what-encode-gives-you) with it. This model wasn't trained for CLS pooling: what happens to the cat and the kitten?
2. Write `chunk(text, tok, size=256, overlap=32)`, which splits a long text into pieces of at most `size` tokens (counting `[CLS]` and `[SEP]`) that overlap by `overlap` tokens. Embed the pieces of a long article and check, with the test of [section G.6](#6-the-256-token-wall), that the end of the article now affects the result.
3. Extend the padding test of [section G.5](#5-pooling-and-the-padding-bug) into a function `check_pooling(encode_fn)` that fails if a text's vector depends on its batch. Run it on `mini_encode` with the naive mean and with the masked mean.
4. Find a word that this model's tokenizer turns into `[UNK]`, and one that becomes five or more pieces. What does that suggest about using this model for text from your own field?
5. Make `bert` compute in float32 instead of float64 (convert `W` with `.float()` instead of `.double()`). How much closer does it get to `model.encode`? Does it ever match exactly, and why not?
6. Pick another model from the Hub that uses the same architecture (its `config.json` says `"model_type": "bert"`) and a different pooling (read its `1_Pooling/config.json`). Adapt `mini_encode` to it and check it against `encode`.

**Where this shows up in No Shortcuts:** lessons 29.1, "Attention from first principles", and 29.2, "The transformer block, built from scratch", build the transformer itself, including the post-norm versus pre-norm question from [section G.4](#4-bert-in-forty-lines-of-numpy). Lesson 35.1 builds a BPE tokenizer from scratch, the other big family next to WordPiece. And lesson 41.1, "Embeddings and semantic search", uses embeddings for retrieval, measured honestly against a 1990s baseline that is still hard to beat.

---

[← F. PyTorch in miniature](F-pytorch-in-miniature.md) · [Contents](../README.md#contents) · [H. From copying to writing →](H-from-copying-to-writing.md)
