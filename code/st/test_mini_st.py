"""Check the miniature against sentence-transformers: token ids, and the final vectors."""
import glob
import os
import random

import numpy as np
from sentence_transformers import SentenceTransformer, __version__

from mini_st import MiniWordPiece, mini_encode

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
snapshot = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                                  "hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"))[0]
tok = MiniWordPiece(os.path.join(snapshot, "vocab.txt"))
W = {k: v.double().numpy() for k, v in model[0].auto_model.state_dict().items()}

rng = random.Random(0)
WORDS = ("the model learns embeddings from sentences tokenization unbelievable antidisestablishmentarianism "
         "naïve café São Paulo Zürich señor résumé déjà-vu 東京 北京 数据 COVID-19 3.14159 $100 e-mail "
         "user@example.com don't it's #hashtag (parens) [brackets] ¿qué? ¡hola! 🙂 🚀 x² ½ ™ — … "
         "supercalifragilisticexpialidocious").split()
ODD = ["\t", "\n", " ", "​", "\x00", "\x07", "�", "  ", "　"]


def random_text(n_words):
    parts = []
    for _ in range(n_words):
        parts.append(rng.choice(WORDS))
        parts.append(rng.choice(ODD) if rng.random() < 0.1 else " ")
    return "".join(parts)


checks = 0
for _ in range(2000):                                   # token ids, including truncation at 256
    text = random_text(rng.choice([1, 3, 10, 40, 200]))
    expected = model.tokenizer(text, truncation=True, max_length=256)["input_ids"]
    assert tok.encode(text) == expected, text
    checks += 1

worst = 0.0
for _ in range(30):                                     # whole batches, with padding, against encode
    texts = [random_text(rng.choice([1, 5, 20, 80])) for _ in range(rng.randint(1, 8))]
    diff = np.abs(mini_encode(tok, W, texts) - model.encode(texts)).max()
    assert diff < 1e-5, diff
    worst = max(worst, diff)
    checks += 1

print(f"{checks} checks against sentence-transformers {__version__}: all passed")
print(f"largest difference in the vectors: {worst:.1e}")
