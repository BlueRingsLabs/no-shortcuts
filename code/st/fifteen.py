import numpy as np
import torch
from sentence_transformers import SentenceTransformer
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


texts = ["The cat sits on the mat.", "A kitten is resting on a rug.", "short", "Quarterly revenue grew by 8%."]
ours = encode(texts)
real = SentenceTransformer(name).encode(texts)
print(np.abs(ours - real).max())
print(type(bert).__name__, bert.training)
