# %% [markdown]
# # Lab 46.6: PPO, and search
#
# 1. PPO on CartPole: reuse each batch for several epochs of updates, with the clipped objective keeping each update
#    close to the policy that collected the data. Against a single update per batch (46.5's actor-critic), and against
#    several epochs *without* clipping. We watch how far each update moves the policy (KL divergence).
# 2. Search: Monte Carlo tree search on tic-tac-toe, against a random player and against perfect play (minimax), as
#    the number of simulations grows.

# %%
import math
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from envs import CartPole, TicTacToe  # noqa: E402

torch.set_num_threads(1)
GAMMA, LAM = 0.99, 0.95


def mlp(out):
    return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out))


def collect(pi, env, n, seed):
    g = torch.Generator().manual_seed(seed)
    S, A, R, D, T, LP, done_returns = [], [], [], [], [], [], []
    s, ep = env.reset(), 0.0
    for _ in range(n):
        with torch.no_grad():
            logits = pi(torch.tensor(s))
            a = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g))
            LP.append(torch.log_softmax(logits, -1)[a].item())
        s2, r, d, tr, _ = env.step(a)
        S.append(s); A.append(a); R.append(r); D.append(d); T.append(d or tr)
        ep += r
        if d or tr:
            done_returns.append(ep); ep = 0.0; s2 = env.reset()
        s = s2
    t = lambda x, dt=torch.float32: torch.tensor(np.array(x), dtype=dt)
    return t(S), t(A, torch.long), t(R), t(D), t(T), t(LP), done_returns, torch.tensor(s)


def advantages(R, D, T, V, v_last):
    adv, last = torch.zeros_like(R), 0.0
    for i in reversed(range(len(R))):
        v_next = (v_last if i == len(R) - 1 else V[i + 1]) * (1 - D[i])
        last = R[i] + GAMMA * v_next - V[i] + GAMMA * LAM * (1 - T[i]) * last
        adv[i] = last
    return adv


def train(method, total=100_000, batch=2000, seed=0):
    torch.manual_seed(seed)
    pi, vf = mlp(2), mlp(1)
    opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()), lr=3e-4 if method != "one update per batch" else 3e-3)
    env = CartPole(seed=seed)
    epochs = 1 if method == "one update per batch" else 10
    clip = 0.2 if method == "PPO (10 epochs, clipped)" else None
    curve, kls, recent = [], [], []
    for it in range(total // batch):
        S, A, R, D, T, LP_old, finished, s_last = collect(pi, env, batch, seed * 1000 + it)
        recent = (recent + finished)[-20:]
        curve.append(np.mean(recent) if recent else 0.0)
        with torch.no_grad():
            V = vf(S)[:, 0]; adv = advantages(R, D, T, V, vf(s_last)[0]); ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for ep in range(epochs):
            for mb in torch.randperm(batch).split(256 if epochs > 1 else batch):
                lp = torch.log_softmax(pi(S[mb]), -1)[torch.arange(len(mb)), A[mb]]
                ratio = torch.exp(lp - LP_old[mb])                                   # pi_new / pi_old for the taken action
                if clip:
                    obj = torch.min(ratio * adv[mb], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb])
                else:
                    obj = ratio * adv[mb]
                loss = -obj.mean() + 0.5 * ((vf(S[mb])[:, 0] - ret[mb]) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vf.parameters()), 0.5)
                opt.step()
        with torch.no_grad():
            lp_new = torch.log_softmax(pi(S), -1)[torch.arange(batch), A]
            kls.append((LP_old - lp_new).mean().item())                             # approximate KL(old || new)
    return np.array(curve), np.array(kls)


# %% [markdown]
# ## 1. PPO

# %%
t0 = time.time()
print("CartPole, 100,000 steps, batches of 2,000; mean return of the last 20 episodes (max 500), 3 seeds")
print(f"  {'method':30s} {'after 20k':>10s} {'after 50k':>10s} {'after 100k':>11s} {'median KL per update':>21s}")
ppo = {}
for method in ("one update per batch", "10 epochs, no clipping", "PPO (10 epochs, clipped)"):
    runs = [train(method, seed=s) for s in range(3)]
    curves, kls = np.array([c for c, _ in runs]), np.array([k for _, k in runs])
    ppo[method] = curves
    m = curves.mean(0)
    print(f"  {method:30s} {m[9]:10.1f} {m[24]:10.1f} {m[-1]:11.1f} {np.median(kls):21.4f}")
print(f"({time.time() - t0:.0f}s)")
print("one gradient step per batch throws away most of what the batch could teach, even at a 10x larger learning rate.")
print("Ten epochs over the same batch learn more per sample; without a brake, each update moves the policy far from the")
print("one that collected the data (median KL 40x PPO's), where the advantages no longer apply. Here that run is ahead at")
print("50k steps and then falls back, which is the usual shape: fast, then unstable. PPO's clipped ratio removes the")
print("incentive to move any action's probability more than about 20% per batch; it's slower early and ends highest.")
print("With three seeds the gaps between the last two are suggestive, not proven. Clipping, a value baseline and GAE:")
print("that's PPO, the algorithm of RLHF (38.3).")
assert ppo["PPO (10 epochs, clipped)"].mean(0)[-1] > ppo["one update per batch"].mean(0)[-1]

# %% [markdown]
# ## 2. Search: MCTS on tic-tac-toe

# %%
@lru_cache(maxsize=None)
def minimax(board, player):
    """Value of the position for `player` to move, with perfect play by both: +1 win, 0 draw, -1 loss."""
    w = TicTacToe.winner(board)
    if w is None:
        return 0
    if w != 0:
        return w * player
    return max(-minimax(TicTacToe.play(board, m, player), -player) for m in TicTacToe.moves(board))


def perfect_move(board, player, rng):
    vals = {m: -minimax(TicTacToe.play(board, m, player), -player) for m in TicTacToe.moves(board)}
    best = max(vals.values())
    return rng.choice([m for m, v in vals.items() if v == best])


def mcts_move(board, player, n_sims, rng, c=1.4):
    """UCT (Kocsis and Szepesvari, 2006): select by upper confidence bound, expand, random rollout, back up."""
    N, W, children = {}, {}, {}

    def rollout(b, p):
        while True:
            w = TicTacToe.winner(b)
            if w is None:
                return 0
            if w != 0:
                return w
            b = TicTacToe.play(b, rng.choice(TicTacToe.moves(b)), p); p = -p

    for _ in range(n_sims):
        b, p, path = board, player, []
        while True:                                                                 # selection
            w = TicTacToe.winner(b)
            if w != 0:                                                              # game over: +1/-1 won, None drawn
                result = w or 0
                break
            if b not in children:                                                   # expansion
                children[b] = TicTacToe.moves(b)
                N[b] = 0
                result = rollout(b, p)                                              # simulation
                path.append((b, None, p))
                break
            untried = [m for m in children[b] if (b, m) not in N]
            if untried:
                m = rng.choice(untried)
            else:
                m = max(children[b], key=lambda m: W[(b, m)] / N[(b, m)] + c * math.sqrt(math.log(N[b]) / N[(b, m)]))
            path.append((b, m, p))
            b, p = TicTacToe.play(b, m, p), -p
        for node, m, mover in path:                                                  # backup
            N[node] = N.get(node, 0) + 1
            if m is not None:
                N[(node, m)] = N.get((node, m), 0) + 1
                W[(node, m)] = W.get((node, m), 0) + result * mover                  # from the mover's point of view
    return max(children[board], key=lambda m: N.get((board, m), 0))


def play(x_policy, o_policy, rng):
    b, p = (0,) * 9, 1
    while TicTacToe.winner(b) == 0:
        m = (x_policy if p == 1 else o_policy)(b, p, rng)
        b, p = TicTacToe.play(b, m, p), -p
    w = TicTacToe.winner(b)
    return 0 if w is None else w


rng = random.Random(0)
randomp = lambda b, p, r: r.choice(TicTacToe.moves(b))
print(f"\nperfect play from the empty board is a draw: minimax value {minimax((0,) * 9, 1)}")
print("  MCTS simulations per move   vs random (MCTS as X): win / draw / loss    vs perfect play (MCTS as O): draw rate")
mc = {}
for sims in (10, 100, 1000):
    mp = lambda b, p, r, s=sims: mcts_move(b, p, s, r)
    vs_rand = [play(mp, randomp, rng) for _ in range(60)]
    vs_perf = [play(lambda b, p, r: perfect_move(b, p, r), mp, rng) for _ in range(30)]
    mc[sims] = np.mean([g == 0 for g in vs_perf])
    print(f"  {sims:26d}   {np.mean([g == 1 for g in vs_rand]):.2f} / {np.mean([g == 0 for g in vs_rand]):.2f} / "
          f"{np.mean([g == -1 for g in vs_rand]):.2f}                          {mc[sims]:.2f}")
print("with enough simulations, search alone (random rollouts, no learning, no knowledge of the game beyond its rules)")
print("draws almost every game against perfect play, which is the best anyone can do. The rare losses are positions where")
print("random rollouts misjudge a line that a careful opponent punishes. AlphaZero replaces the random rollouts with a")
print("value network and guides the selection with a policy network, both trained on the search's own results: learning")
print("and search improving each other. For LLMs, the analogous ideas are best-of-n, verifier-guided search and tree")
print("search over reasoning steps.")
assert mc[1000] >= 0.9 and mc[1000] > mc[10]

print("\nAll checks passed.")
