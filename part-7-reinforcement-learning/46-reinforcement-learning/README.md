# Module 46: Reinforcement Learning

*Part VII: Reinforcement Learning · about 50 hours*

Sutton and Barto, condensed, implemented, and connected to the LLM world.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What is the difference between on-policy and off-policy learning?

## When you finish it, you can

- Solve multi-armed bandits with epsilon-greedy, UCB and Thompson sampling.
- Formulate MDPs, derive the Bellman equations and solve them with value and policy iteration.
- Implement Monte Carlo, SARSA and Q-learning.
- Implement DQN with replay buffers and target networks.
- Derive the policy gradient theorem and implement REINFORCE, actor-critic and PPO.
- Explain MCTS and AlphaZero, and connect PPO/GRPO to LLM post-training.

## Lessons

46.1. [Bandits and exploration](46.1-bandits.md)

46.2. [MDPs, Bellman equations and dynamic programming](46.2-mdps-bellman-dp.md)

46.3. [Monte Carlo and temporal difference: SARSA and Q-learning](46.3-mc-td.md)

46.4. [Deep Q-networks](46.4-dqn.md)

46.5. [Policy gradients and actor-critic](46.5-policy-gradients.md)

46.6. [PPO, search, and RL in the wild](46.6-ppo-search.md)

## Labs

Run them from the repository root, for example:

```bash
python part-7-reinforcement-learning/46-reinforcement-learning/labs/lab_46_1_bandits.py
```

- [`lab_46_1_bandits.py`](labs/lab_46_1_bandits.py)
- [`lab_46_2_mdp_dp.py`](labs/lab_46_2_mdp_dp.py)
- [`lab_46_3_mc_td.py`](labs/lab_46_3_mc_td.py)
- [`lab_46_4_dqn.py`](labs/lab_46_4_dqn.py)
- [`lab_46_5_policy_gradients.py`](labs/lab_46_5_policy_gradients.py)
- [`lab_46_6_ppo_search.py`](labs/lab_46_6_ppo_search.py)

Back to [Part VII](../) · [Syllabus](../../SYLLABUS.md)
