# %% [markdown]
# # Lab 38.3: Preferences, reward models and DPO
#
# Start from 38.1's fine-tuned model (the shared SFT model: trained once, cached). It answers course questions, often
# wrongly, and at temperature 1 it often produces garbled titles. Then:
# 1. Preference pairs, from two kinds of labeler: one who checks facts (which answer is correct?) and one who can only
#    check form (which answer is a real, well-formed lesson title?). Real raters are mostly the second kind.
# 2. A reward model (Bradley-Terry) for each: the SFT model with a scalar head. Which one can it learn?
# 3. Best-of-n with the better of the two: what optimizing an imperfect reward model does.
# 4. Reward hacking: a reward model trained on a labeler who likes long answers.
# 5. DPO: optimize the policy directly on (correct, wrong) pairs, and watch what happens to the likelihoods.

# %%
import copy
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from instructions import END, dataset, format_chat, load_or_train_sft  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(383)
sft, tok = load_or_train_sft()
V, CTX = tok.vocab_size, sft.ctx
end_ids = tok.encode(" " + END)
items = dataset("train")
random.Random(0).shuffle(items)
pair_items, eval_items = items[:400], items[400:520]
heldout = dataset("heldout")[:120]


@torch.no_grad()
def sample_answers(model, question, n, temperature=1.0, max_new=24, seed=0):
    """n answers to one question, sampled in a batch with the KV cache; returns (answer text, answer token ids) pairs."""
    g = torch.Generator().manual_seed(seed)
    prompt = tok.encode(format_chat(question))
    ids = torch.tensor([prompt] * n)
    logits, caches = model(ids, [None] * len(model.blocks), 0)
    out, done = [[] for _ in range(n)], [False] * n
    for step in range(max_new):
        probs = (logits[:, -1] / temperature).softmax(-1) if temperature > 0 else None
        nxt = torch.multinomial(probs, 1, generator=g)[:, 0] if temperature > 0 else logits[:, -1].argmax(-1)
        for i, t in enumerate(nxt.tolist()):
            if not done[i]:
                out[i].append(t)
                done[i] = out[i][-len(end_ids):] == end_ids
        if all(done) or len(prompt) + step + 1 >= CTX:
            break
        logits, caches = model(nxt[:, None], caches, len(prompt) + step)
    res = []
    for o in out:
        body = o[:-len(end_ids)] if o[-len(end_ids):] == end_ids else o
        res.append((tok.decode(body).strip(), o))
    return res


def greedy_accuracy(model, its):
    return np.mean([sample_answers(model, q, 1, temperature=0)[0][0] == a for q, a, _ in its])


# %% [markdown]
# ## 1. Preference pairs

# %%
TITLES = {a for _, a, _ in dataset("train")}                                         # every well-formed answer the course has
t0 = time.time()
pairs, form_pairs = [], []                                                            # (question, chosen ids, rejected ids)
n_correct_samples = n_wellformed = 0
for k, (q, a, _) in enumerate(pair_items):
    samples = sample_answers(sft, q, 4, seed=k)
    n_correct_samples += sum(s == a for s, _ in samples)
    n_wellformed += sum(s in TITLES for s, _ in samples)
    right = [ids for s, ids in samples if s == a]
    wrong = [ids for s, ids in samples if s != a]
    if right and wrong:
        pairs.append((q, right[0], wrong[0]))                                         # the fact-checker's preference
    good = [ids for s, ids in samples if s in TITLES]
    bad = [ids for s, ids in samples if s not in TITLES]
    if good and bad:
        form_pairs.append((q, good[0], bad[0]))                                       # the form-checker's preference
print(f"sampled 4 answers for each of {len(pair_items)} questions in {time.time() - t0:.0f}s: "
      f"{n_correct_samples / (4 * len(pair_items)):.0%} correct, {n_wellformed / (4 * len(pair_items)):.0%} well-formed")
print(f"{len(pairs)} (correct, wrong) pairs; {len(form_pairs)} (well-formed, garbled) pairs")
print("example form pair:", repr(form_pairs[0][0]), "| chosen:", repr(tok.decode(form_pairs[0][1])),
      "| rejected:", repr(tok.decode(form_pairs[0][2])))

# %% [markdown]
# ## 2. A reward model

# %%
class RewardModel(nn.Module):
    def __init__(self, trunk):
        super().__init__()
        self.trunk = copy.deepcopy(trunk)
        self.head = nn.Linear(self.trunk.tok.weight.shape[1], 1)

    def forward(self, ids):                                                          # score = head(final hidden state of the last token)
        x = self.trunk.tok(ids) + self.trunk.pos(torch.arange(ids.shape[1]))
        for b in self.trunk.blocks:
            x, _ = b(x)
        return self.head(self.trunk.ln_f(x))[:, -1, 0]


def seq(question, resp_ids):
    return tok.encode(format_chat(question)) + list(resp_ids)


def train_rm(train_pairs, epochs=3, lr=1e-4, seed=0):
    torch.manual_seed(seed)
    rm = RewardModel(sft)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr)
    for _ in range(epochs):
        for i in torch.randperm(len(train_pairs)).tolist():
            q, w, l = train_pairs[i]
            rw, rl = rm(torch.tensor([seq(q, w)])), rm(torch.tensor([seq(q, l)]))
            loss = -F.logsigmoid(rw - rl).mean()                                     # Bradley-Terry: P(w > l) = sigmoid(rw - rl)
            opt.zero_grad(); loss.backward(); opt.step()
    return rm.eval()


def rm_accuracy(rm, test_pairs):
    with torch.no_grad():
        return np.mean([rm(torch.tensor([seq(q, w)])).item() > rm(torch.tensor([seq(q, l)])).item() for q, w, l in test_pairs])


def cv_accuracy(all_pairs, k=5, seed=0):
    """k-fold cross-validated pair accuracy: every pair is scored once by a reward model that never saw it. With a
    couple of hundred pairs, a single 80/20 split leaves ~40 test pairs and an uncertainty of about +-0.08."""
    order = np.random.default_rng(seed).permutation(len(all_pairs))
    hits = []
    for fold in np.array_split(order, k):
        held = set(fold.tolist())
        m = train_rm([all_pairs[i] for i in order if i not in held])
        hits += [float(rm_accuracy(m, [all_pairs[i]])) for i in fold]
    hits = np.array(hits)
    boot = [hits[np.random.default_rng(b).integers(0, len(hits), len(hits))].mean() for b in range(1000)]
    return hits.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


t0 = time.time()
fact_cv, form_cv = cv_accuracy(pairs), cv_accuracy(form_pairs)
print(f"\nreward models, 5-fold cross-validated pair accuracy ({time.time() - t0:.0f}s):")
print(f"  trained on the fact-checker's labels (correct > wrong):       {fact_cv[0]:.3f}  95% CI [{fact_cv[1]:.3f}, {fact_cv[2]:.3f}]")
print(f"  trained on the form-checker's labels (well-formed > garbled): {form_cv[0]:.3f}  95% CI [{form_cv[1]:.3f}, {form_cv[2]:.3f}]")
if fact_cv[1] <= 0.5:
    print("the fact reward model can't be told apart from a coin. ", end="")
else:
    print("the fact reward model is better than a coin, and far from reliable. ", end="")
print("It starts from the same small model that gets most facts wrong,")
print("so the knowledge it would need to judge correctness mostly isn't in it; form is a surface property it picks up")
print("easily. Real reward models are initialized from models as large as the policy for this reason, and they still")
print("reward what raters can check. (A single 80/20 split of these pairs gave fact accuracies anywhere from 0.47 to 0.62")
print("depending on the run: that's why this is cross-validated.)")
assert form_cv[0] > fact_cv[0] and form_cv[1] > 0.5
fsplit = int(0.8 * len(form_pairs))
rm = train_rm(form_pairs[:fsplit])                                                   # the form reward model used below

# %% [markdown]
# ## 3 and 4. Best-of-n, with the form reward model and with a hackable one
#
# The "length-loving labeler": for pairs of two sampled answers, it prefers the longer one, whatever it says. A reward
# model trained on those labels learns length. Best-of-n picks the answer with the highest reward out of n samples.

# %%
biased_pairs = []
for k, (q, a, _) in enumerate(pair_items[:300]):
    (s1, i1), (s2, i2) = sample_answers(sft, q, 2, seed=1000 + k)
    if len(s1) != len(s2):
        biased_pairs.append((q, i1, i2) if len(s1) > len(s2) else (q, i2, i1))
rm_len = train_rm(biased_pairs)


def best_of_n(reward_model, its, n):
    correct, formed, lengths = [], [], []
    for k, (q, a, _) in enumerate(its):
        cands = sample_answers(sft, q, n, seed=5000 + k)
        with torch.no_grad():
            scores = [reward_model(torch.tensor([seq(q, ids)])).item() for _, ids in cands]
        pick = cands[int(np.argmax(scores))][0]
        correct.append(pick == a); formed.append(pick in TITLES); lengths.append(len(pick))
    return np.mean(correct), np.mean(formed), np.mean(lengths)


print("\nbest-of-n on 60 questions: correct / well-formed / length in characters of the picked answer")
print("   n   form reward model              length-loving reward model")
bon = {}
for n in (1, 4, 16):
    bon[n] = (best_of_n(rm, eval_items[:60], n), best_of_n(rm_len, eval_items[:60], n))
    (a1, f1, l1), (a2, f2, l2) = bon[n]
    print(f"{n:4d}   {a1:.3f} / {f1:.3f} / {l1:5.1f}          {a2:.3f} / {f2:.3f} / {l2:5.1f}")
(c1f, _, l1f), (c16f, _, l16f) = bon[1][0], bon[16][0]
(c1, f1, l1), (c16, f16, l16) = bon[1][1], bon[16][1]
print(f"neither reward model makes best-of-n reliably more correct. The form model is right on {form_cv[0]:.0%} of held-out pairs,")
print("and picking the maximum of 16 samples selects exactly the samples where its errors are largest: correctness goes")
print(f"from {c1f:.2f} to {c16f:.2f}. That's over-optimization (Gao et al., 2023): the harder you optimize an imperfect proxy,")
print("the further you drift from what it approximates. The length-loving model shows the pure case: length goes up with")
print(f"n ({l1:.0f} to {l16:.0f} characters), and correctness goes down ({c1:.2f} to {c16:.2f}). Optimizing a proxy delivers the proxy.")
assert l16 > l1 and c16 < c1 and c16f <= c1f

# %% [markdown]
# ## 5. DPO
#
# loss = -log sigmoid(beta * [(log pi(w) - log ref(w)) - (log pi(l) - log ref(l))]), with log-probabilities summed over
# the response tokens only. No reward model, no sampling during training.

# %%
def response_logprob(model, question, resp_ids):
    ids = seq(question, resp_ids)
    logits = model(torch.tensor([ids[:-1]]))[0]
    lp = F.log_softmax(logits, -1)
    start = len(ids) - len(resp_ids) - 1
    return lp[torch.arange(start, len(ids) - 1), torch.tensor(resp_ids)].sum()


ref = copy.deepcopy(sft).eval()
with torch.no_grad():
    ref_lp = [(response_logprob(ref, q, w).item(), response_logprob(ref, q, l).item()) for q, w, l in pairs]


def dpo(beta=0.1, epochs=2, lr=1e-4, nll=0.0, seed=0):
    """nll > 0 adds the chosen answer's per-token negative log-likelihood to the loss (as in RPO, Pang et al., 2024)."""
    torch.manual_seed(seed)
    pol = copy.deepcopy(sft).train()
    opt = torch.optim.AdamW(pol.parameters(), lr=lr)
    for _ in range(epochs):
        for i in torch.randperm(len(pairs)).tolist():
            q, w, l = pairs[i]
            lw, ll = response_logprob(pol, q, w), response_logprob(pol, q, l)
            loss = -F.logsigmoid(beta * ((lw - ref_lp[i][0]) - (ll - ref_lp[i][1]))) - nll * lw / len(w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
    return pol.eval()


t0 = time.time()
res = {"SFT model": (0.0, 0.0, greedy_accuracy(sft, eval_items), greedy_accuracy(sft, heldout))}
for name, lr, nll in (("DPO, lr 1e-4", 1e-4, 0.0), ("DPO + NLL on chosen, lr 1e-4", 1e-4, 1.0),
                      ("DPO, lr 1e-5", 1e-5, 0.0), ("DPO, lr 3e-6", 3e-6, 0.0)):
    pol = dpo(lr=lr, nll=nll)
    with torch.no_grad():
        new_lp = [(response_logprob(pol, q, w).item(), response_logprob(pol, q, l).item()) for q, w, l in pairs]
    d_w = np.mean([n[0] - r[0] for n, r in zip(new_lp, ref_lp)])
    d_l = np.mean([n[1] - r[1] for n, r in zip(new_lp, ref_lp)])
    res[name] = (d_w, d_l, greedy_accuracy(pol, eval_items), greedy_accuracy(pol, heldout))
print(f"\nDPO on {len(pairs)} (correct, wrong) pairs, four settings, in {time.time() - t0:.0f}s")
print(f"  {'':30s} {'change in log-prob vs reference':>32s}   {'greedy exact match':>20s}")
print(f"  {'':30s} {'chosen':>15s} {'rejected':>16s}   {'other questions':>16s} {'unseen phrasings':>17s}")
for k, (dw, dl, a1, a2) in res.items():
    print(f"  {k:30s} {dw:15.2f} {dl:16.2f}   {a1:16.3f} {a2:17.3f}")
print("every setting widens the margin between chosen and rejected, which is all the loss asks for. Look at which way")
print("each moved: at lr 1e-4 both collapse, the rejected faster, and the model all but forgets how to answer. The loss")
print("only sees the difference, and pushing probability off the rejected answers moved it to text nobody chose")
print("(likelihood displacement, Razin et al., 2024). An NLL term on the chosen answer holds it in place and limits the")
print("damage; small learning rates widen the margin while barely moving the model. None of them makes it more accurate:")
print("preference pairs can only reweight answers the model already produces, and this one mostly produces wrong ones.")
print("Watch the chosen log-probability, not only the margin or the loss.")
collapse, anchored, gentle = res["DPO, lr 1e-4"], res["DPO + NLL on chosen, lr 1e-4"], res["DPO, lr 3e-6"]
assert collapse[1] < collapse[0]                                                     # the rejected fall faster
sft_acc = res["SFT model"][2]
assert collapse[0] < -5 and collapse[2] < sft_acc - 0.3
assert anchored[0] > collapse[0] + 5 and anchored[2] > collapse[2]
assert gentle[0] - gentle[1] > 0.5 and abs(gentle[2] - sft_acc) < 0.15

print("\nAll checks passed.")
