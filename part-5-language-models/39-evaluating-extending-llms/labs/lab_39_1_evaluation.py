# %% [markdown]
# # Lab 39.1: Evaluating language models
#
# The course's fine-tuned model (38.1's recipe, shared and cached) as the system under test.
# 1. One set of answers, four scoring rules, four different numbers.
# 2. Prompt sensitivity: the same facts asked four ways.
# 3. Multiple choice by log-likelihood: summed vs length-normalized, and a bias you can measure.
# 4. How sure are you? Bootstrap intervals and a paired comparison of two systems.
# 5. Contamination: can you tell from the model's loss whether it trained on a text? And a confound that fools people.

# %%
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from corpus import lesson_files, read_lesson  # noqa: E402
from instructions import END, TRAIN_TEMPLATES, HELDOUT_TEMPLATES, dataset, facts, format_chat, load_or_train_sft  # noqa: E402
from tinygpt import load_or_train  # noqa: E402

torch.set_num_threads(4)
model, tok = load_or_train_sft()
base, _ = load_or_train(verbose=False)
CTX = model.ctx
end_ids = tok.encode(" " + END)


@torch.no_grad()
def answer(m, question, temperature=0.0, seed=0, max_new=30):
    g = torch.Generator().manual_seed(seed)
    ids = tok.encode(format_chat(question))
    start = len(ids)
    logits, caches = m(torch.tensor([ids]), [None] * len(m.blocks), 0)
    for _ in range(max_new):
        lg = logits[0, -1]
        t = int(lg.argmax()) if temperature == 0 else int(torch.multinomial((lg / temperature).softmax(-1), 1, generator=g))
        ids.append(t)
        if ids[-len(end_ids):] == end_ids:
            return tok.decode(ids[start:-len(end_ids)]).strip()
        if len(ids) >= CTX:
            break
        logits, caches = m(torch.tensor([[t]]), caches, len(ids) - 1)
    return tok.decode(ids[start:]).strip()


# %% [markdown]
# ## 1. Scoring rules

# %%
def normalize(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in s.split() if w not in {"a", "an", "the"})


def token_f1(pred, gold):
    p, g_ = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g_)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g_)
    return 2 * prec * rec / (prec + rec)


items = dataset("heldout")
random.Random(0).shuffle(items)
items = items[:150]
preds = [answer(model, q) for q, _, _ in items]
golds = [a for _, a, _ in items]
scores = {"exact match": np.mean([p == g for p, g in zip(preds, golds)]),
          "normalized exact match": np.mean([normalize(p) == normalize(g) for p, g in zip(preds, golds)]),
          "token F1 (SQuAD-style)": np.mean([token_f1(p, g) for p, g in zip(preds, golds)]),
          "gold contained in the answer": np.mean([normalize(g) in normalize(p) for p, g in zip(preds, golds)])}
print("the same 150 answers (held-out phrasings), scored four ways:")
for k, v in scores.items():
    print(f"  {k:30s} {v:.3f}")
print("none of these is 'the accuracy'. A benchmark number without its scoring rule is not a number.")
assert scores["token F1 (SQuAD-style)"] > scores["exact match"]

# %% [markdown]
# ## 2. Prompt sensitivity

# %%
lesson_facts = [(slots, ans) for kind, slots, ans in facts() if kind == "lesson_title"]
random.Random(1).shuffle(lesson_facts)
lesson_facts = lesson_facts[:80]
templates = TRAIN_TEMPLATES["lesson_title"] + HELDOUT_TEMPLATES["lesson_title"]
print("\nthe same 80 facts (lesson titles), asked four ways:")
per_t = {}
for t in templates:
    per_t[t] = np.mean([answer(model, t.format(**s)) == a for s, a in lesson_facts])
    print(f"  {t:40s} {per_t[t]:.3f}{'   (never seen in training)' if t in HELDOUT_TEMPLATES['lesson_title'] else ''}")
print(f"spread: {max(per_t.values()) - min(per_t.values()):.3f}. Report results over several prompt variants, or you are")
print("partly measuring your prompt. Large models are less sensitive than this one, and not insensitive.")

# %% [markdown]
# ## 3. Multiple choice by likelihood
#
# "What is lesson X about?" with the true title and three other lesson titles as options. Score each option by its
# total log-probability as the answer, or by its average per token.

# %%
all_titles = [a for _, a in [(s, a) for kind, s, a in facts() if kind == "lesson_title"]]


@torch.no_grad()
def option_logprob(m, question, option):
    p, o = tok.encode(format_chat(question)), tok.encode(option + " " + END)
    lp = F.log_softmax(m(torch.tensor([p + o]))[0], -1)
    tok_lp = lp[torch.arange(len(p) - 1, len(p) + len(o) - 1), torch.tensor(o)]
    return tok_lp.sum().item(), tok_lp.mean().item()


r = random.Random(2)
mc = {"sum": [], "mean": []}
correct_is_shortest = []
for slots, ans in lesson_facts[:60]:
    q = f"What is lesson {slots['lid']} about?"
    opts = [ans] + r.sample([t for t in all_titles if t != ans], 3)
    lps = [option_logprob(model, q, o) for o in opts]
    shortest = int(np.argmin([len(tok.encode(o)) for o in opts]))
    correct_is_shortest.append(shortest == 0)
    for j, key in enumerate(("sum", "mean")):
        choice = int(np.argmax([x[j] for x in lps]))
        mc[key].append(choice == 0)
print(f"\n4-option multiple choice, 60 questions (chance 0.25; the correct option is the shortest in {np.mean(correct_is_shortest):.0%} of them):")
for key in ("sum", "mean"):
    print(f"  total log-prob {'(summed)       ' if key == 'sum' else '(per token mean)'}: accuracy {np.mean(mc[key]):.3f}")

# the same questions with four *wrong* options: nothing to know, so what the scoring rule prefers shows through
bias = {"sum": [], "mean": []}
for slots, ans in lesson_facts[:60]:
    q = f"What is lesson {slots['lid']} about?"
    opts = r.sample([t for t in all_titles if t != ans], 4)
    lps = [option_logprob(model, q, o) for o in opts]
    shortest = int(np.argmin([len(tok.encode(o)) for o in opts]))
    for j, key in enumerate(("sum", "mean")):
        bias[key].append(int(np.argmax([x[j] for x in lps])) == shortest)
print(f"with four wrong options, the shortest is picked: summed {np.mean(bias['sum']):.0%}, per-token mean "
      f"{np.mean(bias['mean']):.0%} (no preference would be 25%)")
print("when the model knows the answer, both rules find it. When it doesn't, the rule decides: here the per-token mean")
print("almost never picks the shortest option. Once a long title has started, its later tokens are easy to predict,")
print("which raises its average; the sum pays for every extra token and happens to land near chance on this model.")
print("Neither is neutral in general. Evaluation tools differ in exactly this kind of detail, which is why the same")
print("model gets different scores on the 'same' benchmark from different tools.")
assert np.mean(bias["sum"]) > np.mean(bias["mean"])

# %% [markdown]
# ## 4. Confidence intervals and paired comparisons
#
# System A: greedy decoding. System B: sampling at temperature 0.7. Same 150 questions.

# %%
a_ok = np.array([p == g for p, g in zip(preds, golds)], dtype=float)
b_ok = np.array([answer(model, q, temperature=0.7, seed=i) == a for i, (q, a, _) in enumerate(items)], dtype=float)
rs = np.random.default_rng(0)
boot = rs.integers(0, len(items), (5000, len(items)))
ci = lambda x: np.percentile(x[boot].mean(1), [2.5, 97.5])
diff = (a_ok - b_ok)[boot].mean(1)
print(f"\nA (greedy): {a_ok.mean():.3f}, 95% CI {ci(a_ok).round(3).tolist()}")
print(f"B (T=0.7):  {b_ok.mean():.3f}, 95% CI {ci(b_ok).round(3).tolist()}")
print(f"paired difference A - B: {(a_ok - b_ok).mean():+.3f}, 95% CI {np.percentile(diff, [2.5, 97.5]).round(3).tolist()}")
lo, hi = np.percentile(diff, [2.5, 97.5])
print(f"questions where they disagree: {int((a_ok != b_ok).sum())} of {len(items)}. With 150 items, each system's interval is")
print("several points wide, and a 2-point 'improvement' on a benchmark this size is noise. The paired interval uses only the")
print("questions where the systems disagree, which is why it's narrower than the two separate ones: " +
      ("here it excludes zero, so greedy really is better on this model." if lo > 0 or hi < 0 else "here it includes zero, so the difference could be nothing."))
print("Pair the comparison (same items), and report intervals.")

# %% [markdown]
# ## 5. Membership inference and a confound
#
# The base model was trained on 90% of the lessons of Parts I to III; every 10th lesson was held out. Score paragraphs
# by the model's loss. Training paragraphs should have lower loss. Can the loss tell them apart?

# %%
def paragraphs(files):
    out = []
    for f in files:
        for p in re.split(r"\n\s*\n", read_lesson(f)):
            if len(p) > 300:
                out.append(p[:400])
    return out


@torch.no_grad()
def para_loss(m, text):
    ids = tok.encode(text)[:CTX + 1]
    return F.cross_entropy(m(torch.tensor([ids[:-1]]))[0], torch.tensor(ids[1:])).item()


files = lesson_files()
member = paragraphs([f for i, f in enumerate(files) if i % 10 != 4])
random.Random(3).shuffle(member)
member = member[:150]
nonmember_same = paragraphs([f for i, f in enumerate(files) if i % 10 == 4])[:150]
root = Path(__file__).resolve().parents[3]
nonmember_shift = paragraphs(lesson_files(("part-4-deep-learning",)))[:150]
L_m = [para_loss(base, p) for p in member]
L_s = [para_loss(base, p) for p in nonmember_same]
L_x = [para_loss(base, p) for p in nonmember_shift]
auc_same = roc_auc_score([1] * len(L_m) + [0] * len(L_s), [-x for x in L_m + L_s])
auc_shift = roc_auc_score([1] * len(L_m) + [0] * len(L_x), [-x for x in L_m + L_x])
print(f"\nmean loss: trained-on paragraphs {np.mean(L_m):.3f}; held-out lessons of the same parts {np.mean(L_s):.3f}; "
      f"Part IV lessons {np.mean(L_x):.3f}")
print(f"AUC for 'was this paragraph in training?': vs held-out lessons from the same parts {auc_same:.3f}; vs Part IV {auc_shift:.3f}")
print("against Part IV the attack looks strong, but most of that is topic shift (different words), not memorization: a")
print("fair membership test compares with non-members from the same distribution (Duan et al., 2024). A model trained for")
print("~12 epochs on this data memorizes some; LLMs trained for about one epoch on web text memorize less, except duplicates.")
assert auc_same > 0.55 and auc_shift > auc_same

print("\nAll checks passed.")
