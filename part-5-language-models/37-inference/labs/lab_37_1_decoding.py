# %% [markdown]
# # Lab 37.1: Decoding strategies
#
# A small GPT trained on this course (the shared tinygpt model: trained once, then cached), and every common way of
# turning its next-token distributions into text:
#
# 1. Greedy decoding, and why it loops.
# 2. Beam search, with and without length normalization.
# 3. Sampling: temperature, top-k, top-p (nucleus), min-p. Diversity vs the model's own confidence.
# 4. Repetition penalties.
# 5. Constrained decoding: masking the vocabulary so the output always parses, and choosing among fixed options by likelihood.

# %%
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from tinygpt import load_or_train  # noqa: E402

torch.set_num_threads(4)
model, tok = load_or_train()
V = tok.vocab_size


@torch.no_grad()
def next_logits(ids):
    return model(torch.tensor([ids[-model.ctx:]]))[0, -1]


def repeated_ngram_rate(ids, n=4):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / max(1, len(grams))


PROMPT = "## Things that will bite you\n\n- **"
prompt_ids = tok.encode(PROMPT)

# %% [markdown]
# ## 1. Greedy

# %%
@torch.no_grad()
def greedy(ids, n=80):
    ids = list(ids)
    for _ in range(n):
        ids.append(int(next_logits(ids).argmax()))
    return ids


g_ids = greedy(prompt_ids)
print("greedy:\n" + tok.decode(g_ids))
print(f"\nfraction of repeated 4-grams in the continuation: {repeated_ngram_rate(g_ids[len(prompt_ids):]):.2f}")
print("the single most likely token at each step leads into loops: once a phrase is likely, repeating it is likelier.")

# %% [markdown]
# ## 2. Beam search
#
# Keep the B best partial sequences by total log-probability. Summing log-probabilities favors short outputs (every
# token lowers the sum), so compare finished sequences by the average per token (length normalization).

# %%
@torch.no_grad()
def beam_search(ids, beam=4, n=30, normalize=True, end_tokens=None):
    beams = [(0.0, list(ids))]
    finished = []
    for _ in range(n):
        cand = []
        for score, seq in beams:
            lp = F.log_softmax(next_logits(seq), -1)
            top = lp.topk(beam)
            for v, i in zip(top.values.tolist(), top.indices.tolist()):
                new = (score + v, seq + [i])
                (finished if end_tokens and i in end_tokens else cand).append(new)
        beams = sorted(cand, key=lambda x: -x[0])[:beam]
    finished += beams
    key = (lambda x: x[0] / (len(x[1]) - len(ids))) if normalize else (lambda x: x[0])
    return max(finished, key=key)


score, seq = beam_search(prompt_ids, beam=4)
cont = seq[len(prompt_ids):]
print(f"\nbeam search, width 4: {len(cont)} tokens, mean log-prob {score / len(cont):.2f}: {tok.decode(cont)!r}")
g_lp = 0.0
with torch.no_grad():
    for t in range(len(prompt_ids), len(prompt_ids) + 30):
        g_lp += F.log_softmax(next_logits(g_ids[:t]), -1)[g_ids[t]].item()
print(f"greedy's first 30 tokens: mean log-prob {g_lp / 30:.2f}")
print("beam search finds higher-probability text than greedy, and it reads like the course's most common shapes glued")
print("together: for translation, where there's one right answer, higher probability is the goal; for open-ended writing,")
print("the most probable text is bland. (Length normalization matters when beams can end at different lengths; see the lesson.)")

# %% [markdown]
# ## 3. Sampling

# %%
def sample_filtered(logits, temperature=1.0, top_k=None, top_p=None, min_p=None, g=None):
    logits = logits / temperature
    if top_k:
        kth = logits.topk(top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(-1)
    if top_p:                                                                        # smallest set with cumulative mass >= p
        sp, si = probs.sort(descending=True)
        keep = sp.cumsum(0) - sp < top_p
        mask = torch.zeros_like(probs, dtype=torch.bool); mask[si[keep]] = True
        probs = probs * mask
    if min_p:                                                                        # keep tokens with prob >= min_p * max prob
        probs = probs * (probs >= min_p * probs.max())
    probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=g))


@torch.no_grad()
def generate(ids, n=60, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    ids, lps = list(ids), []
    for _ in range(n):
        logits = next_logits(ids)
        t = sample_filtered(logits, g=g, **kw)
        lps.append(F.log_softmax(logits, -1)[t].item())                              # the model's own (untempered) log-prob
        ids.append(t)
    return ids, float(np.mean(lps))


def distinct2(seqs):
    bigrams = [tuple(s[i:i + 2]) for s in seqs for i in range(len(s) - 1)]
    return len(set(bigrams)) / len(bigrams)


settings = {"temperature 0.5": dict(temperature=0.5), "temperature 1.0": dict(temperature=1.0),
            "temperature 1.5": dict(temperature=1.5), "top-k 20": dict(top_k=20), "top-p 0.9": dict(top_p=0.9),
            "min-p 0.1": dict(min_p=0.1), "temperature 1.5 + min-p 0.1": dict(temperature=1.5, min_p=0.1)}
print("\n4 samples of 50 tokens each:")
print("  setting                        distinct bigrams   mean log-prob (model's view)   repeated 4-grams")
res = {}
for name, kw in settings.items():
    outs = [generate(prompt_ids, n=50, seed=s, **kw) for s in range(4)]
    conts = [o[0][len(prompt_ids):] for o in outs]
    res[name] = (distinct2(conts), np.mean([o[1] for o in outs]), np.mean([repeated_ngram_rate(c) for c in conts]))
    print(f"  {name:30s} {res[name][0]:16.3f}   {res[name][1]:28.2f}   {res[name][2]:16.2f}")
print("low temperature: confident and repetitive; high temperature: diverse and increasingly nonsense. Truncation (top-k,")
print("top-p, min-p) cuts the long tail of unlikely tokens, where most of the nonsense comes from, so higher temperatures")
print("stay usable: compare 'temperature 1.5' with 'temperature 1.5 + min-p 0.1'.")
print("\nsample at top-p 0.9:\n" + tok.decode(generate(prompt_ids, seed=1, top_p=0.9)[0]))
assert res["temperature 0.5"][1] > res["temperature 1.5"][1] and res["temperature 0.5"][0] < res["temperature 1.5"][0]
assert res["temperature 1.5 + min-p 0.1"][1] > res["temperature 1.5"][1]

# %% [markdown]
# ## 4. Repetition penalty
#
# Divide the logits of tokens that already appeared (if positive; multiply if negative) by a penalty > 1 (the CTRL
# paper's rule, which Hugging Face implements).

# %%
@torch.no_grad()
def greedy_penalized(ids, n=80, penalty=1.3):
    ids = list(ids)
    for _ in range(n):
        logits = next_logits(ids).clone()
        seen = torch.tensor(sorted(set(ids)))
        logits[seen] = torch.where(logits[seen] > 0, logits[seen] / penalty, logits[seen] * penalty)
        ids.append(int(logits.argmax()))
    return ids


p_ids = greedy_penalized(prompt_ids)
print(f"\ngreedy with repetition penalty 1.3: repeated 4-grams {repeated_ngram_rate(p_ids[len(prompt_ids):]):.2f} "
      f"(plain greedy {repeated_ngram_rate(g_ids[len(prompt_ids):]):.2f})")
print(tok.decode(p_ids[len(prompt_ids):]))
print("fewer loops, and a blunt instrument: it also penalizes words that should repeat ('the', names, code identifiers).")
assert repeated_ngram_rate(p_ids[len(prompt_ids):]) <= repeated_ngram_rate(g_ids[len(prompt_ids):])

# %% [markdown]
# ## 5. Constrained decoding
#
# (a) The output must be a number: at every step, mask every token that isn't made only of digits (plus one that ends
# the number). The model can only choose among valid continuations, so the output always parses.
# (b) Classification by likelihood: score each allowed answer by its total log-probability after the prompt.

# %%
digit_tokens = [i for i in range(V) if re.fullmatch(r"\d+", tok.decode([i]))]
end_tok = tok.encode(".")[0]


@torch.no_grad()
def constrained_number(prompt, max_len=6):
    ids = tok.encode(prompt)
    allowed = torch.full((V,), float("-inf")); allowed[digit_tokens] = 0; allowed[end_tok] = 0
    out = []
    for _ in range(max_len):
        t = int((next_logits(ids + out) + allowed).argmax())
        if t == end_tok:
            break
        out.append(t)
    return tok.decode(out)


prompts = ["The lab uses a batch size of ", "Module ", "The answer to exercise ", "About "]
free = [tok.decode(greedy(tok.encode(p), n=4)[len(tok.encode(p)):]) for p in prompts]
cons = [constrained_number(p) for p in prompts]
print("\nprompt -> unconstrained greedy | constrained to digits")
for p, f_, c in zip(prompts, free, cons):
    print(f"  {p!r:32s} -> {f_!r:22s} | {c!r}")
print("constrained outputs always parse as numbers. Whether they're right is the model's problem, not the parser's", end="")
print(f": this model answers {cons[0]} whatever the question." if len(set(cons)) == 1 else ".", "A valid format is not a correct answer.")
assert all(re.fullmatch(r"\d+", c) for c in cons)


@torch.no_grad()
def option_logprob(prompt, option):
    p, o = tok.encode(prompt), tok.encode(option)
    lp = F.log_softmax(model(torch.tensor([(p + o)[-model.ctx:]]))[0], -1)
    start = len(p + o) - len(o) - max(0, len(p + o) - model.ctx) - 1
    return sum(lp[start + j, t].item() for j, t in enumerate(o))


options = [" SQL", " gradient", " Kafka"]
question = "To join two tables you write a query in"
scores = {o: option_logprob(question, o) for o in options}
print(f"\n{question!r}: " + ", ".join(f"{o.strip()} {s:.2f}" for o, s in scores.items()) + f" -> {max(scores, key=scores.get).strip()}")
print("scoring a closed set of answers never produces an invalid one, and gives a probability for each. It's how many")
print("benchmarks score multiple choice (39.1), and why a length or tokenization bias between options matters.")

print("\nAll checks passed.")
