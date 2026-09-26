# %% [markdown]
# # Lab 46.4: Deep Q-networks
#
# CartPole (the classic pole-balancing task; +1 per step, up to 500): 4 continuous state variables, so no table.
# A small network approximates Q(s, .). Then the two ideas that made DQN work (Mnih et al., 2015), removed one at a time:
#   experience replay: learn from random minibatches of past transitions, not from the latest one
#   target network:    compute targets with a periodically updated copy, so the target doesn't move with every step
# and Double DQN (van Hasselt et al., 2016), 46.3's fix for maximization bias. Two seeds each: RL results vary a lot
# between seeds, and one seed proves nothing.

# %%
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from envs import CartPole  # noqa: E402

torch.set_num_threads(1)                                                     # tiny networks: threads only add overhead
STEPS, GAMMA = 30_000, 0.99


def qnet():
    return nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))


def evaluate(net, episodes=10, seed=1000):
    env, total = CartPole(), 0.0
    for ep in range(episodes):
        s, done, trunc = env.reset(seed=seed + ep), False, False
        while not (done or trunc):
            with torch.no_grad():
                s, r, done, trunc, _ = env.step(int(net(torch.tensor(s)).argmax()))
            total += r
    return total / episodes


def dqn(replay=True, target=True, double=False, seed=0, log_every=5000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPole(seed=seed)
    net = qnet()
    tgt = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    cap = 50_000 if replay else 1
    buf_s = np.zeros((cap, 4), np.float32); buf_a = np.zeros(cap, np.int64); buf_r = np.zeros(cap, np.float32)
    buf_s2 = np.zeros((cap, 4), np.float32); buf_d = np.zeros(cap, np.float32)
    n, i = 0, 0
    s = env.reset()
    curve = []
    for step in range(STEPS):
        eps = max(0.05, 1 - step / 10_000)                                   # explore a lot early, then less
        if rng.random() < eps:
            a = int(rng.integers(2))
        else:
            with torch.no_grad():
                a = int(net(torch.tensor(s)).argmax())
        s2, r, done, trunc, _ = env.step(a)
        buf_s[i], buf_a[i], buf_r[i], buf_s2[i], buf_d[i] = s, a, r, s2, float(done)   # truncation is not an ending
        i = (i + 1) % cap; n = min(n + 1, cap)
        s = env.reset() if (done or trunc) else s2
        if n >= (1000 if replay else 1):
            idx = rng.integers(0, n, 64) if replay else np.array([(i - 1) % cap])
            bs, ba, br = torch.tensor(buf_s[idx]), torch.tensor(buf_a[idx]), torch.tensor(buf_r[idx])
            bs2, bd = torch.tensor(buf_s2[idx]), torch.tensor(buf_d[idx])
            with torch.no_grad():
                evaluator = tgt if target else net
                if double:
                    a2 = net(bs2).argmax(1, keepdim=True)                       # choose with the online net...
                    q2 = evaluator(bs2).gather(1, a2)[:, 0]                      # ...value with the target net
                else:
                    q2 = evaluator(bs2).max(1).values
                y = br + GAMMA * (1 - bd) * q2
            q = net(bs).gather(1, ba[:, None])[:, 0]
            loss = F.smooth_l1_loss(q, y)                                        # Huber: robust to large TD errors
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10)
            opt.step()
            if target and step % 500 == 0:
                tgt.load_state_dict(net.state_dict())
        if (step + 1) % log_every == 0:
            curve.append(evaluate(net, episodes=5))
    return net, curve


variants = {"DQN (replay + target network)": dict(),
            "Double DQN": dict(double=True),
            "no target network": dict(target=False),
            "no replay (learn from each transition once)": dict(replay=False)}
print(f"{STEPS:,} environment steps per run; greedy return (max 500) every 5,000 steps, 2 seeds")
results = {}
t0 = time.time()
for name, kw in variants.items():
    runs = [dqn(seed=sd, **kw) for sd in (0, 1)]
    final = [evaluate(net, episodes=20) for net, _ in runs]
    best = [max(c) for _, c in runs]
    results[name] = (final, best)
    curves = " | ".join(" ".join(f"{v:3.0f}" for v in c) for _, c in runs)
    print(f"  {name:44s} best {np.mean(best):5.1f}  final {np.mean(final):5.1f}   (seed 0 | seed 1: {curves})")
print(f"({time.time() - t0:.0f}s)")
print("without a target network the Q-values chase their own updates and diverge: the agent is worse than random.")
print("Learning from each transition once, in order, is learning from a stream of correlated, non-stationary samples:")
print("it spikes and falls back. Full DQN learns, and then often gets *worse*: once it balances well, the buffer fills")
print("with good states only, and it forgets what to do when things go wrong. The fixes in practice: keep the best")
print("checkpoint by evaluation (not the last one), decay the learning rate, bigger buffers. And look at every seed:")
print("in RL, the variation between runs is often larger than the difference between methods.")
dqn_best = np.mean(results["DQN (replay + target network)"][1])
assert dqn_best > np.mean(results["no replay (learn from each transition once)"][1]) and np.mean(results["no target network"][1]) < 50

print("\nAll checks passed.")
