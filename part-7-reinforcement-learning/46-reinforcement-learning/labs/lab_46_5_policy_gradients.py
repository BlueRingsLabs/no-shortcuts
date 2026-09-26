# %% [markdown]
# # Lab 46.5: Policy gradients and actor-critic
#
# Learn the policy directly: a network outputs action probabilities, and gradient ascent on expected return pushes up
# the log-probability of actions in proportion to how good they turned out to be.
# 1. The variance problem, measured: the spread of REINFORCE gradient estimates with and without a baseline.
# 2. REINFORCE, REINFORCE with a learned baseline, and actor-critic with GAE, on CartPole, at the same number of
#    environment steps, over 3 seeds.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from envs import CartPole  # noqa: E402

torch.set_num_threads(1)
GAMMA = 0.99


def policy_net():
    return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))


def value_net():
    return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))


def rollout(pi, env, n_steps, rng_seed):
    """Collect n_steps of experience (several episodes). Returns tensors and the finished episodes' returns."""
    g = torch.Generator().manual_seed(rng_seed)
    S, A, R, D, T = [], [], [], [], []
    s = env.reset()
    ep_ret, finished = 0.0, []
    for _ in range(n_steps):
        with torch.no_grad():
            a = int(torch.multinomial(torch.softmax(pi(torch.tensor(s)), -1), 1, generator=g))
        s2, r, done, trunc, _ = env.step(a)
        S.append(s); A.append(a); R.append(r); D.append(done); T.append(done or trunc)
        ep_ret += r
        if done or trunc:
            finished.append(ep_ret); ep_ret = 0.0
            s2 = env.reset()
        s = s2
    return (torch.tensor(np.array(S)), torch.tensor(A), torch.tensor(R, dtype=torch.float32),
            torch.tensor(D, dtype=torch.float32), torch.tensor(T, dtype=torch.float32), finished, torch.tensor(s))


def returns_to_go(R, T):
    """Discounted return from each step to the end of its episode (the unfinished last episode is cut off)."""
    G, out = 0.0, torch.zeros_like(R)
    for t in reversed(range(len(R))):
        G = R[t] + GAMMA * G * (1 - T[t])
        out[t] = G
    return out


def gae(R, D, T, V, v_last, lam=0.95):
    """Generalized advantage estimation (Schulman et al., 2016): exponentially weighted n-step TD errors."""
    adv, last = torch.zeros_like(R), 0.0
    for t in reversed(range(len(R))):
        v_next = v_last if t == len(R) - 1 else V[t + 1]
        v_next = v_next * (1 - D[t])                                           # terminal: nothing after it
        delta = R[t] + GAMMA * v_next - V[t]
        last = delta + GAMMA * lam * (1 - T[t]) * last
        adv[t] = last
    return adv


# %% [markdown]
# ## 1. The variance of the gradient estimate

# %%
torch.manual_seed(0)
pi = policy_net()
env = CartPole(seed=0)
base_grads, plain_grads = [], []
flat = lambda: torch.cat([p.grad.flatten() for p in pi.parameters()]).clone()
S, A, R, D, T, _, _ = rollout(pi, env, 20_000, rng_seed=1)
G = returns_to_go(R, T)
logp = torch.log_softmax(pi(S), -1)[torch.arange(len(A)), A]
for k in range(40):                                                            # 40 independent batches of 500 steps
    sl = slice(k * 500, (k + 1) * 500)
    for weights, store in ((G[sl], plain_grads), (G[sl] - G[sl].mean(), base_grads)):
        pi.zero_grad()
        (-(logp[sl] * weights).mean()).backward(retain_graph=True)
        store.append(flat())
var_plain = torch.stack(plain_grads).var(0).sum().item()
var_base = torch.stack(base_grads).var(0).sum().item()
print(f"variance of the policy gradient estimate over 40 batches of 500 steps (sum over parameters):")
print(f"  REINFORCE, weights = return:               {var_plain:10.2f}")
print(f"  REINFORCE, weights = return - baseline:    {var_base:10.2f}   ({var_plain / var_base:.1f}x smaller)")
print("subtracting a baseline doesn't change the expected gradient (the score function has mean zero) and removes part")
print("of its noise: actions no longer get credit just because the episode as a whole went well. Here, at a random")
print("initial policy with short, similar episodes, a constant baseline gains 1.5x; the gain grows as returns spread out,")
print("and a learned, state-dependent baseline (the critic below) removes much more.")
assert var_base < 0.8 * var_plain

# %% [markdown]
# ## 2. Learning CartPole

# %%
def train(method, total_steps=100_000, batch=2000, seed=0, lr=3e-3):
    torch.manual_seed(seed)
    pi, vf = policy_net(), value_net()
    opt_pi = torch.optim.Adam(pi.parameters(), lr=lr)
    opt_v = torch.optim.Adam(vf.parameters(), lr=1e-3)
    env = CartPole(seed=seed)
    curve, recent = [], []
    for it in range(total_steps // batch):
        S, A, R, D, T, finished, s_last = rollout(pi, env, batch, rng_seed=seed * 1000 + it)
        recent = (recent + finished)[-20:]
        curve.append(np.mean(recent) if recent else 0.0)
        if method == "REINFORCE":
            G = returns_to_go(R, T)
            weights = G / (G.std() + 1e-8)                                       # rescaled only: no baseline
        elif method == "REINFORCE + baseline":
            G = returns_to_go(R, T)
            with torch.no_grad():
                weights = G - vf(S)[:, 0]
            for _ in range(20):                                                   # fit the baseline to the returns
                opt_v.zero_grad(); ((vf(S)[:, 0] - G) ** 2).mean().backward(); opt_v.step()
        else:                                                                     # actor-critic with GAE
            with torch.no_grad():
                V = vf(S)[:, 0]; v_last = vf(s_last)[0]
            weights = gae(R, D, T, V, v_last)
            target = weights + V
            for _ in range(20):
                opt_v.zero_grad(); ((vf(S)[:, 0] - target) ** 2).mean().backward(); opt_v.step()
        if method != "REINFORCE":
            weights = (weights - weights.mean()) / (weights.std() + 1e-8)
        logp = torch.log_softmax(pi(S), -1)[torch.arange(len(A)), A]
        opt_pi.zero_grad(); (-(logp * weights).mean()).backward(); opt_pi.step()
    return curve


t0 = time.time()
print(f"\nCartPole, 100,000 environment steps per run, mean return of the last 20 episodes (max 500), 3 seeds:")
print(f"  {'method':24s} {'after 20k':>10s} {'after 50k':>10s} {'after 100k':>11s}")
pg = {}
for method in ("REINFORCE", "REINFORCE + baseline", "actor-critic (GAE)"):
    curves = np.array([train(method, seed=s) for s in range(3)])
    pg[method] = curves
    m = curves.mean(0)
    print(f"  {method:24s} {m[9]:10.1f} {m[24]:10.1f} {m[-1]:11.1f}   per seed at the end: {np.round(curves[:, -1]).astype(int)}")
print(f"({time.time() - t0:.0f}s)")
print("the same gradient, estimated three ways. Plain REINFORCE learns, slowly and noisily. A learned baseline learns")
print("the task on every seed. Actor-critic with GAE keeps up for the first 20k steps, then falls behind, unevenly: its")
print("advantages lean on the critic's estimates, and a critic that's wrong biases every update in the same direction,")
print("where Monte Carlo returns are only noisy. That's the bias-variance dial of 46.3 again, and at lambda = 0.95 with a")
print("critic fitted for 20 steps per batch, the bias cost more than the variance saved. Actor-critic methods win when the")
print("critic is good and episodes are long; PPO (46.6) adds more critic training and a limit on each policy update.")
print("Three seeds is also a small sample: look at the per-seed column before believing any ranking.")
assert pg["REINFORCE + baseline"].mean(0)[-1] > pg["REINFORCE"].mean(0)[-1] + 50
assert pg["actor-critic (GAE)"].mean(0)[9] >= pg["REINFORCE"].mean(0)[9]

print("\nAll checks passed.")
