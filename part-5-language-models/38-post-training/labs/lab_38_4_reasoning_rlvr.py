# %% [markdown]
# # Lab 38.4: Reasoning, verifiable rewards and GRPO
#
# A task where correctness is checkable: adding two 3-digit numbers. A small transformer, trained from scratch on
# characters, in two formats:
#   direct:            "123+489=612"
#   chain of thought:  "123+489:3+9=12,2+8+1=11,1+4+1=6=612"  (digits right to left, with carries)
# 1. Same model, same budget: how much does writing out the steps help?
# 2. GRPO on a direct-answer model that's right most of the time: sample several answers per problem, reward 1 for a
#    correct final answer, push up the better-than-average ones. What changes: greedy, sampled, and pass@k accuracy.
# 3. Test-time compute: majority voting over several samples.

# %%
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from tinygpt import GPT  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(384)
CHARS = "0123456789+=:,#"                                                              # '#' ends an answer
stoi = {c: i for i, c in enumerate(CHARS)}
V, CTX = len(CHARS), 48
enc = lambda s: [stoi[c] for c in s]
dec = lambda ids: "".join(CHARS[i] for i in ids)


def problem(r):
    return r.randint(100, 999), r.randint(100, 999)


def target(a, b, cot):
    if not cot:
        return f"{a + b}#"
    steps, carry = [], 0
    for da, db in zip(reversed(str(a)), reversed(str(b))):
        s = int(da) + int(db) + carry
        steps.append(f"{da}+{db}" + (f"+{carry}" if carry else "") + f"={s}")
        carry = s // 10
    return ",".join(steps) + f"={a + b}#"


def prompt(a, b, cot):
    return f"{a}+{b}" + (":" if cot else "=")


print("examples:", repr(prompt(123, 489, False) + target(123, 489, False)), repr(prompt(123, 489, True) + target(123, 489, True)))


def final_answer(text):
    body = text.split("#")[0]
    tail = body.split("=")[-1]
    return int(tail) if tail.isdigit() else None


# %% [markdown]
# ## 1. Supervised training on each format

# %%
def make_batch(bs, cot, r):
    X = torch.zeros(bs, CTX, dtype=torch.long)
    Y = torch.full((bs, CTX), -100)
    for i in range(bs):
        a, b = problem(r)
        p, t = enc(prompt(a, b, cot)), enc(target(a, b, cot))
        ids = p + t
        X[i, :len(ids) - 1] = torch.tensor(ids[:-1])
        Y[i, len(p) - 1:len(ids) - 1] = torch.tensor(t)                              # loss on the answer only
    return X, Y


def train_sft(cot, steps, seed=0):
    torch.manual_seed(seed)
    r = random.Random(seed)
    m = GPT(V, CTX, d=96, n_layers=3, n_heads=4)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 2e-3 * min(1, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        X, Y = make_batch(64, cot, r)
        loss = F.cross_entropy(m(X).reshape(-1, V), Y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def generate(m, prompts, temperature=0.0, max_new=40, g=None):
    """Batched generation with the KV cache. prompts: list of token lists of equal length (all problems here have the
    same prompt length). Returns (texts, generated ids, per-token log-probs under the sampling distribution)."""
    ids = torch.tensor(prompts)
    logits, caches = m(ids, [None] * len(m.blocks), 0)
    B, n0 = ids.shape
    outs, done = [[] for _ in range(B)], torch.zeros(B, dtype=torch.bool)
    for step in range(max_new):
        lg = logits[:, -1]
        nxt = lg.argmax(-1) if temperature == 0 else torch.multinomial((lg / temperature).softmax(-1), 1, generator=g)[:, 0]
        for i in range(B):
            if not done[i]:
                outs[i].append(int(nxt[i]))
        done |= nxt == stoi["#"]
        if done.all() or n0 + step + 1 >= CTX:
            break
        logits, caches = m(nxt[:, None], caches, n0 + step)
    return [dec(o) for o in outs], outs


test_r = random.Random(12345)
TEST = [problem(test_r) for _ in range(500)]


def accuracy(m, cot, probs=TEST):
    texts, _ = generate(m, [enc(prompt(a, b, cot)) for a, b in probs])
    return np.mean([final_answer(t) == a + b for t, (a, b) in zip(texts, probs)])


t0 = time.time()
STEPS = 800
m_direct, m_cot = train_sft(False, STEPS), train_sft(True, STEPS)
acc_d, acc_c = accuracy(m_direct, False), accuracy(m_cot, True)
print(f"\ntrained both formats for {STEPS} steps of 64 problems in {time.time() - t0:.0f}s")
print(f"exact accuracy on 500 new problems: direct answer {acc_d:.3f}, with written-out steps {acc_c:.3f}")
print("the steps turn one hard computation (all digits and carries at once, in a single forward pass) into several easy")
print("ones, each conditioned on text the model already wrote. More tokens of output = more computation per problem.")
assert acc_c > acc_d + 0.5

# %% [markdown]
# ## 2. GRPO with a verifiable reward
#
# GRPO needs a model that's sometimes right: a group of answers that are all wrong (or all right) has no signal. So
# start from the direct-answer model trained longer (1,400 steps), which is right most of the time. For each problem,
# sample G = 8 answers at temperature 1; reward 1 if the final number is right, else 0; advantage = reward minus the
# group mean (divided by the group std). Loss: minus the advantage times the log-probability of each sampled answer,
# averaged over tokens. No reward model, no critic, no correct answer ever shown.

# %%
def seq_logprob(m, prompt_ids, gen_ids):
    ids = torch.tensor([prompt_ids + gen_ids])
    lp = F.log_softmax(m(ids[:, :-1])[0], -1)
    start = len(prompt_ids) - 1
    return lp[torch.arange(start, start + len(gen_ids)), torch.tensor(gen_ids)]


def grpo(m0, steps=40, problems_per_step=16, G=8, lr=1e-5, seed=0):
    torch.manual_seed(seed)
    r = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    m = GPT(V, CTX, d=96, n_layers=3, n_heads=4)
    m.load_state_dict(m0.state_dict())
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    history = []
    for step in range(steps):
        losses, rewards = [], []
        for _ in range(problems_per_step):
            a, b = problem(r)
            p = enc(prompt(a, b, False))
            m.eval()
            texts, gens = generate(m, [p] * G, temperature=1.0, g=g)
            rew = torch.tensor([float(final_answer(t) == a + b) for t in texts])
            rewards.append(rew.mean().item())
            if rew.std() == 0:                                                       # all right or all wrong: no signal
                continue
            adv = (rew - rew.mean()) / (rew.std() + 1e-6)
            m.train()
            for gi, ai in zip(gens, adv):
                losses.append(-ai * seq_logprob(m, p, gi).mean())
        if losses:
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        history.append(np.mean(rewards))
    return m.eval(), history


def sample_k(m, cot, k, probs, seed=0):
    g = torch.Generator().manual_seed(seed)
    answers = []
    for a, b in probs:
        texts, _ = generate(m, [enc(prompt(a, b, cot))] * k, temperature=1.0, g=g)
        answers.append([final_answer(t) for t in texts])
    return answers


sub = TEST[:200]
truth = [a + b for a, b in sub]


def profile(m, cot, k=15):
    """greedy accuracy; mean accuracy of single samples at T=1; majority of 5 and of 15; pass@15."""
    ans = sample_k(m, cot, k, sub)
    maj = lambda n: np.mean([max(set(x[:n]), key=x[:n].count) == t for x, t in zip(ans, truth)])
    per_problem = np.array([np.mean([y == t for y in x]) for x, t in zip(ans, truth)])
    return (accuracy(m, cot, sub), per_problem.mean(), maj(5), maj(15), np.mean([t in x for x, t in zip(ans, truth)]),
            per_problem)


t0 = time.time()
m_base = train_sft(False, 1400)
before = profile(m_base, False)
m_rl, hist = grpo(m_base)
after = profile(m_rl, False)
print(f"\ndirect-answer model after 1,400 SFT steps, then GRPO for 40 steps x 16 problems x 8 samples ({time.time() - t0:.0f}s)")
print(f"  {'':12s} {'greedy':>7s} {'one sample, T=1':>16s} {'pass@15':>8s}")
print(f"  {'before RL':12s} {before[0]:7.3f} {before[1]:16.3f} {before[4]:8.3f}")
print(f"  {'after RL':12s} {after[0]:7.3f} {after[1]:16.3f} {after[4]:8.3f}")
d = after[5] - before[5]
bs = np.random.default_rng(0)
boot = [d[bs.integers(0, len(d), len(d))].mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"one-sample accuracy, after - before: {d.mean():+.3f}, paired 95% CI [{lo:+.3f}, {hi:+.3f}] over {len(d)} problems")
print("nobody showed the model a correct answer during RL; it was only told which of its own attempts were right. The")
print("effect here is small, and on 200 problems not clearly beyond the noise: 40 steps at a learning rate small enough not")
print("to damage the model (in development runs, 3e-4 took greedy accuracy from 0.82 to 0.57). What didn't happen is as")
print("telling: pass@15 didn't rise. RL moves probability toward answers the model can already produce; it didn't make")
print("unsolvable problems solvable (Yue et al., 2025). Large RLVR runs get large gains in pass@1 the same way, from")
print("thousands of steps over problems chosen to be solvable sometimes.")
assert after[1] > before[1] - 0.02 and after[4] <= before[4] + 0.02 and after[0] >= before[0] - 0.03

# %% [markdown]
# ## 3. Test-time compute

# %%
cot_prof = profile(m_cot, True)
print("\n                             greedy   one sample   majority of 5   majority of 15   pass@15")
for name, pr in (("direct, 1,400 SFT steps", before), ("direct + RL", after), ("written-out steps, 800 steps", cot_prof)):
    print(f"  {name:28s} {pr[0]:6.3f} {pr[1]:12.3f} {pr[2]:15.3f} {pr[3]:16.3f} {pr[4]:9.3f}")
print("majority voting (self-consistency) turns several unreliable samples into one more reliable answer, at k times the")
print("cost. Writing out the steps is also test-time compute: more tokens per problem, each an easy computation. On this")
print("task it beats everything else in the table at a fraction of the training.")
assert before[3] >= before[1] and cot_prof[0] > before[0]
print("\nAll checks passed.")
