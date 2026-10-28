import glob, os, time
import numpy as np
from sentence_transformers import SentenceTransformer
from mini_st import MiniWordPiece, bert, mean_pool, normalize, mini_encode, basic_tokenize

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
snap = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                              "hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"))[0]
tok = MiniWordPiece(os.path.join(snap, "vocab.txt"))
W = {k: v.double().numpy() for k, v in model[0].auto_model.state_dict().items()}

print("--- encode")
sentences = ["The cat sits on the mat.", "A kitten is resting on a rug.", "Quarterly revenue grew by 8%.", "The stock market fell sharply today."]
emb = model.encode(sentences)
print(type(emb).__name__, emb.shape, emb.dtype)
print(np.linalg.norm(emb, axis=1).round(6))
print((emb @ emb.T).round(3))
print(model.similarity(emb, emb).numpy().round(3))

print("--- tokenizer")
print(basic_tokenize("Tokenization isn't hard, it's just fiddly."))
print(tok.wordpiece("tokenization"), [k for k, v in tok.vocab.items() if v in tok.wordpiece("tokenization")])
print(tok.wordpiece("fiddly"), len(tok.vocab))
batch = model.tokenizer(["short", "a slightly longer sentence"], padding=True)
print(batch["input_ids"]); print(batch["attention_mask"])
print([k for k in W if k.startswith("pooler")])
print(sum(v.size for v in W.values()), sum(p.numel() for p in model.parameters()))

print("--- padding")
ids, mask = tok.batch(["short", "a much longer sentence that forces a lot of padding onto the first one"])
tokens = bert(W, ids, mask)
good = normalize(mean_pool(tokens, mask))
naive = normalize(tokens.mean(axis=1))
alone = model.encode(["short"])[0]
print(ids.shape, mask[0])
print(float(good[0] @ alone).__round__(6), float(naive[0] @ alone).__round__(4))
print(float(good[1] @ model.encode([ "a much longer sentence that forces a lot of padding onto the first one"])[0]).__round__(6), float(naive[1] @ good[1]).__round__(4))
q = model.encode(["a cat on a rug"])[0]
print(float(q @ good[0]).__round__(4), float(q @ naive[0]).__round__(4))

print("--- truncation")
long_text = " ".join(["word"] * 300)
print(len(model.tokenizer(long_text)["input_ids"]), model.max_seq_length)
a = model.encode([long_text]); b = model.encode([long_text + " but the conclusion is the opposite of everything above"])
print(np.array_equal(a, b))
half = " ".join(["The experiment was a success."] * 50)
print(len(model.tokenizer(half)["input_ids"]))
c = model.encode([half + " Actually, it failed completely."]); d = model.encode([half])
print(np.array_equal(c, d))

print("--- speed")
texts = sentences * 16
t0 = time.perf_counter(); mini_encode(tok, W, texts); t1 = time.perf_counter(); model.encode(texts); t2 = time.perf_counter()
print(f"{(t1-t0)*1000:.0f} ms {(t2-t1)*1000:.0f} ms")
