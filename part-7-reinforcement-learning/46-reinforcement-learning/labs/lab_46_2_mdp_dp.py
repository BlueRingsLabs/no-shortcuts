# %% [markdown]
# # Lab 46.2: MDPs, Bellman equations and dynamic programming
#
# A 4x4 lake (start top-left, goal bottom-right, four holes), deterministic or slippery. The model of the world is
# known: P[s][a] lists (probability, next state, reward, done). With a known model, no learning is needed, only
# planning.
# 1. Policy evaluation: solve the Bellman expectation equation, by iteration and as a linear system.
# 2. Value iteration and policy iteration: the optimal policy, and how fast each gets there.
# 3. What slipperiness and the discount factor do to the optimal policy.

# %%
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from envs import GridWorld  # noqa: E402

ARROWS = "<v>^"


def show(policy, env):
    rows = []
    for r in range(4):
        rows.append(" ".join(env.MAP[r][c] if env.MAP[r][c] in "HG" else ARROWS[policy[r * 4 + c]] for c in range(4)))
    return "\n".join("    " + row for row in rows)


def matrices(env, policy):
    """P_pi (16x16) and r_pi (16) for a deterministic policy: the MDP collapsed to a Markov reward process."""
    P = np.zeros((16, 16)); r = np.zeros(16)
    for s in range(16):
        for p, s2, rew, done in env.P[s][policy[s]]:
            r[s] += p * rew
            if not done:
                P[s, s2] += p
    return P, r


# %% [markdown]
# ## 1. Policy evaluation

# %%
gamma = 0.95
policy = np.full(16, 2)                                                      # the shortest path: down, down, right,
policy[[0, 4, 10]] = 1                                                        # right, down, right
for slip in (0.0, 2 / 3):
    env = GridWorld(slip=slip)
    P, r = matrices(env, policy)
    v_exact = np.linalg.solve(np.eye(16) - gamma * P, r)                     # v = r + gamma P v, solved directly
    v = np.zeros(16)
    for sweep in range(1, 10_000):
        v_new = r + gamma * P @ v                                            # the same equation, as an iteration
        if np.abs(v_new - v).max() < 1e-10:
            break
        v = v_new
    if slip == 0:
        print(f"the shortest-path policy, gamma {gamma}:")
        print(show(policy, env))
    print(f"{'deterministic' if slip == 0 else 'slippery':13s} lake: value of the start {v_exact[0]:.4f} (linear solve), "
          f"{v[0]:.4f} after {sweep} sweeps of iterative evaluation")
    assert np.allclose(v, v_exact, atol=1e-8)
print(f"deterministic: six steps to the goal, worth gamma^5 = {gamma ** 5:.4f}. On the slippery lake the same plan is worth")
print("almost nothing: it walks along holes. Evaluating a policy needs no rollouts when the model is known.")

# %% [markdown]
# ## 2. Value iteration and policy iteration

# %%
def q_from_v(env, v, gamma):
    Q = np.zeros((16, 4))
    for s in range(16):
        for a in range(4):
            Q[s, a] = sum(p * (rew + (0 if done else gamma * v[s2])) for p, s2, rew, done in env.P[s][a])
    return Q


def value_iteration(env, gamma, tol=1e-10):
    v, sweeps = np.zeros(16), 0
    while True:
        v_new = q_from_v(env, v, gamma).max(1)                               # Bellman optimality backup
        sweeps += 1
        if np.abs(v_new - v).max() < tol:
            return v_new, q_from_v(env, v_new, gamma).argmax(1), sweeps
        v = v_new


def policy_iteration(env, gamma):
    policy, rounds = np.zeros(16, dtype=int), 0
    while True:
        P, r = matrices(env, policy)
        v = np.linalg.solve(np.eye(16) - gamma * P, r)                       # evaluate exactly
        new = q_from_v(env, v, gamma).argmax(1)                              # improve greedily
        rounds += 1
        if np.array_equal(new, policy):
            return v, policy, rounds
        policy = new


for slip in (0.0, 2 / 3):
    env = GridWorld(slip=slip)
    v_vi, pi_vi, sweeps = value_iteration(env, gamma)
    v_pi, pi_pi, rounds = policy_iteration(env, gamma)
    print(f"\n{'deterministic' if slip == 0 else 'slippery (the intended move 1/3 of the time)'} lake:")
    print(show(pi_vi, env))
    print(f"value iteration: {sweeps} sweeps; policy iteration: {rounds} evaluate-and-improve rounds; "
          f"same values: {np.allclose(v_vi, v_pi, atol=1e-6)}; start value {v_vi[0]:.4f}")
    env_run = GridWorld(slip=slip, seed=1)
    wins = 0
    for ep in range(2000):
        s, done, trunc = env_run.reset(), False, False
        while not (done or trunc):
            s, rew, done, trunc, _ = env_run.step(pi_vi[s])
        wins += rew
    print(f"success rate over 2,000 episodes following it: {wins / 2000:.3f}")
    assert np.allclose(v_vi, v_pi, atol=1e-6)
print("policy iteration takes few, expensive rounds (a linear solve each); value iteration many cheap sweeps. On the")
print("slippery lake the optimal policy often points *away* from the goal: moving toward a wall is how you avoid")
print("sliding into a hole. It's still the best possible, and it still fails a lot: optimal is not the same as good.")

# %% [markdown]
# ## 3. The discount factor

# %%
env = GridWorld(slip=2 / 3)
for g in (0.5, 0.9, 0.99, 0.999):
    v_g, pi_g, _ = value_iteration(env, g)
    env_run = GridWorld(slip=2 / 3, seed=2)
    wins, steps = 0, []
    for ep in range(2000):
        s, done, trunc, t = env_run.reset(), False, False, 0
        while not (done or trunc):
            s, rew, done, trunc, _ = env_run.step(pi_g[s]); t += 1
        wins += rew; steps.append(t)
    print(f"gamma {g:5}: success {wins / 2000:.3f}, mean episode length {np.mean(steps):5.1f}")
print("a low gamma makes the agent impatient: a reward 20 steps away is worth gamma^20 of one now, so it takes risks to")
print("get there sooner. gamma is part of the problem definition, not a detail: it says how much the future matters.")

print("\nAll checks passed.")
