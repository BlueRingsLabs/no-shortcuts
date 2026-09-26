# %% [markdown]
# # Lab 46.3: Monte Carlo and temporal difference: SARSA and Q-learning
#
# No model of the world this time: the agent learns from experience only.
# 1. Q-learning and Monte Carlo control on the slippery lake, checked against the true optimal values from 46.2.
# 2. The cliff (Sutton and Barto, example 6.6): SARSA learns the safe path, Q-learning the optimal one, and the
#    optimal one does worse while exploring.
# 3. Maximization bias, and double Q-learning (Sutton and Barto, example 6.7).

# %%
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from envs import CliffWalking, GridWorld  # noqa: E402


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))                                              # break ties randomly


def value_iteration(env, gamma):
    v = np.zeros(env.n_states)
    for _ in range(10_000):
        Q = np.array([[sum(p * (r + (0 if d else gamma * v[s2])) for p, s2, r, d in env.P[s][a]) for a in range(4)]
                      for s in range(16)])
        if np.abs(Q.max(1) - v).max() < 1e-12:
            return Q
        v = Q.max(1)


# %% [markdown]
# ## 1. Learning the slippery lake from experience

# %%
gamma = 0.95
env = GridWorld(slip=2 / 3, seed=0)
Q_star = value_iteration(env, gamma)


def greedy_success(Q, episodes=2000, seed=5):
    e = GridWorld(slip=2 / 3, seed=seed)
    wins = 0
    for _ in range(episodes):
        s, done, trunc = e.reset(), False, False
        while not (done or trunc):
            s, r, done, trunc, _ = e.step(int(Q[s].argmax()))
        wins += r
    return wins / episodes


def q_learning(env, episodes, eps=0.2, seed=0):
    rng = np.random.default_rng(seed)
    Q, N = np.zeros((16, 4)), np.zeros((16, 4))
    for ep in range(episodes):
        s, done, trunc = env.reset(), False, False
        while not (done or trunc):
            a = eps_greedy(Q, s, eps, rng)
            s2, r, done, trunc, _ = env.step(a)
            target = r + (0 if done else gamma * Q[s2].max())                 # off-policy: the greedy next action
            N[s, a] += 1
            Q[s, a] += (target - Q[s, a]) / N[s, a] ** 0.6                     # step sizes that shrink, but not too fast
            s = s2
    return Q


def mc_control(env, episodes, eps=0.2, seed=0):
    """Every-visit Monte Carlo control: wait for the episode's end, then average the returns that followed."""
    rng = np.random.default_rng(seed)
    Q, N = np.zeros((16, 4)), np.zeros((16, 4))
    for ep in range(episodes):
        s, done, trunc, traj = env.reset(), False, False, []
        while not (done or trunc):
            a = eps_greedy(Q, s, eps, rng)
            s2, r, done, trunc, _ = env.step(a)
            traj.append((s, a, r)); s = s2
        G = 0.0
        for s, a, r in reversed(traj):
            G = r + gamma * G
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]
    return Q


print(f"slippery lake; the optimal policy (46.2) succeeds {greedy_success(Q_star):.3f} of the time, start value {Q_star[0].max():.4f}")
print(f"  {'episodes':>9s}   {'method':12s} {'|Q - Q*| max':>13s} {'greedy policy success':>22s}")
for n_ep in (2_000, 20_000):
    for name, fn in (("Q-learning", q_learning), ("Monte Carlo", mc_control)):
        Q = fn(GridWorld(slip=2 / 3, seed=1), n_ep)
        print(f"  {n_ep:9,d}   {name:12s} {np.abs(Q - Q_star).max():13.3f} {greedy_success(Q):22.3f}")
print("both converge toward the optimal values, from experience alone. Q-learning updates after every step by")
print("bootstrapping from its own estimate of the next state; Monte Carlo waits for the actual return of each episode.")
print("Bootstrapping has lower variance and some bias; Monte Carlo, the reverse.")

# %% [markdown]
# ## 2. The cliff: SARSA vs Q-learning

# %%
def td_control(method, episodes=500, alpha=0.5, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    env, Q = CliffWalking(), np.zeros((48, 4))
    returns = []
    for ep in range(episodes):
        s, done, trunc, G = env.reset(), False, False, 0.0
        a = eps_greedy(Q, s, eps, rng)
        while not (done or trunc):
            s2, r, done, trunc, _ = env.step(a)
            G += r
            a2 = eps_greedy(Q, s2, eps, rng)
            nxt = Q[s2, a2] if method == "SARSA" else Q[s2].max()             # on-policy vs off-policy target
            Q[s, a] += alpha * (r + (0 if done else nxt) - Q[s, a])
            s, a = s2, a2
        returns.append(G)
    return Q, np.array(returns)


def greedy_path(Q):
    env, path = CliffWalking(), []
    s = env.reset()
    for _ in range(60):
        s, r, done, trunc, _ = env.step(int(Q[s].argmax())); path.append(s)
        if done:
            break
    return path


print()
cliff = {}
for method in ("SARSA", "Q-learning"):
    runs = [td_control(method, seed=k) for k in range(20)]
    online = np.mean([r[-100:].mean() for _, r in runs])
    path = greedy_path(runs[0][0])
    row_near_cliff = sum(1 for s in path if s // 12 == 2)
    cliff[method] = online
    print(f"{method:10s}: return per episode while learning (last 100, eps=0.1) {online:7.1f}; greedy path length "
          f"{len(path)}, steps along the row next to the cliff {row_near_cliff}")
print("Q-learning learns the values of the greedy policy, which walks right along the edge (13 steps, the optimum).")
print("But it acts epsilon-greedily, and one random step there is a 100-point fall. SARSA learns the values of the policy")
print("it actually follows, exploration included, and keeps its distance. On-policy methods are safer while learning;")
print("off-policy ones learn the best policy from any behavior, which is what makes replay and DQN possible (46.4).")
assert cliff["SARSA"] > cliff["Q-learning"]

# %% [markdown]
# ## 3. Maximization bias
#
# From state A, "right" ends the episode with reward 0; "left" goes to B, where each of 10 actions ends it with a
# reward drawn from N(-0.1, 1). Going left is worse on average, but the max over noisy estimates looks positive.

# %%
def bias_experiment(double, episodes=300, runs=1000, alpha=0.1, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    left_share = np.zeros(episodes)
    for run in range(runs):
        QA = np.zeros((2, 2)); QB = np.zeros((2, 10))                        # two tables for double Q-learning
        for ep in range(episodes):
            qa = QA.sum(0)
            a = rng.integers(2) if rng.random() < eps else int(rng.choice(np.flatnonzero(qa == qa.max())))
            left_share[ep] += a == 0
            if a == 1:                                                          # right: terminal, reward 0
                i = rng.integers(2) if double else 0
                QA[i, 1] += alpha * (0 - QA[i, 1])
                continue
            qb = QB.sum(0) if double else QB[0]
            b = rng.integers(10) if rng.random() < eps else int(rng.choice(np.flatnonzero(qb == qb.max())))
            r = rng.normal(-0.1, 1)
            if double:
                i = rng.integers(2)                                             # update one table, evaluate with the other
                best_b = int(QB[i].argmax())
                QA[i, 0] += alpha * (QB[1 - i, best_b] - QA[i, 0])
                QB[i, b] += alpha * (r - QB[i, b])
            else:
                QA[0, 0] += alpha * (QB[0].max() - QA[0, 0])
                QB[0, b] += alpha * (r - QB[0, b])
    return left_share / runs


shares = {}
for double in (False, True):
    ls = shares[double] = bias_experiment(double)
    print(f"\n{'double Q-learning' if double else 'Q-learning':18s}: share of 'left' (the worse action), episodes 1-50 "
          f"{ls[:50].mean():.2f}, 250-300 {ls[250:].mean():.2f} (optimal: {0.05:.2f}, i.e. exploration only)")
print("the maximum of noisy estimates is biased upward, so Q-learning overvalues B and keeps going left for a long time.")
print("Double Q-learning picks the best action with one estimate and values it with another: no systematic bias.")
print("DQN inherits the problem and the fix (Double DQN, 46.4).")
assert shares[True][:50].mean() < shares[False][:50].mean() - 0.3

print("\nAll checks passed.")
