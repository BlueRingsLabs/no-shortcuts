"""Small environments for Module 46, with the Gymnasium-style interface, so the labs need no extra packages.

    env.reset(seed=None) -> observation
    env.step(action) -> (observation, reward, terminated, truncated, info)

Bandit(probs)             Bernoulli arms; step(arm) returns reward 0 or 1 (observation is None).
GridWorld(slip=0.0)       4x4 lake: start top-left, goal bottom-right, holes end the episode. Exposes P for planning:
                          P[s][a] = list of (prob, next_state, reward, done), like FrozenLake.
CliffWalking()            4x12 grid, -1 per step, -100 and back to start for stepping into the cliff (Sutton and
                          Barto, example 6.6).
CartPole()                the classic pole-balancing task (Barto, Sutton and Anderson, 1983), same physics constants and
                          termination as Gymnasium's CartPole-v1: +1 per step, up to 500 steps.
TicTacToe                 board as a tuple of 9 ints (0 empty, 1 X, -1 O); helpers for search (46.6).
"""

from __future__ import annotations

import math

import numpy as np


class Bandit:
    def __init__(self, probs, seed=0):
        self.probs = np.asarray(probs, float)
        self.rng = np.random.default_rng(seed)
        self.n_actions = len(self.probs)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return None

    def step(self, a):
        return None, float(self.rng.random() < self.probs[a]), False, False, {}


class GridWorld:
    MAP = ["SFFF",
           "FHFH",
           "FFFH",
           "HFFG"]
    MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]                                     # (row, col) for left, down, right, up

    def __init__(self, slip=0.0, seed=0):
        self.n, self.slip = 4, slip
        self.n_states, self.n_actions = 16, 4
        self.rng = np.random.default_rng(seed)
        self.P = {s: {a: self._transitions(s, a) for a in range(4)} for s in range(16)}

    def _cell(self, s):
        return self.MAP[s // 4][s % 4]

    def _move(self, s, a):
        r, c = divmod(s, 4)
        dr, dc = self.MOVES[a]
        r, c = min(max(r + dr, 0), 3), min(max(c + dc, 0), 3)
        return r * 4 + c

    def _transitions(self, s, a):
        if self._cell(s) in "HG":
            return [(1.0, s, 0.0, True)]
        outcomes = [(1 - self.slip, a)] + ([(self.slip / 2, (a - 1) % 4), (self.slip / 2, (a + 1) % 4)] if self.slip else [])
        out = []
        for p, aa in outcomes:
            s2 = self._move(s, aa)
            out.append((p, s2, 1.0 if self._cell(s2) == "G" else 0.0, self._cell(s2) in "HG"))
        return out

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s, self.t = 0, 0
        return self.s

    def step(self, a):
        trans = self.P[self.s][a]
        i = self.rng.choice(len(trans), p=[t[0] for t in trans])
        _, self.s, r, done = trans[i]
        self.t += 1
        return self.s, r, done, self.t >= 100, {}


class CliffWalking:
    def __init__(self):
        self.rows, self.cols = 4, 12
        self.n_states, self.n_actions = 48, 4

    def reset(self, seed=None):
        self.s, self.t = 36, 0                                                      # bottom-left
        return self.s

    def step(self, a):
        r, c = divmod(self.s, 12)
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]                               # up, right, down, left
        r, c = min(max(r + dr, 0), 3), min(max(c + dc, 0), 11)
        s2 = r * 12 + c
        self.t += 1
        if r == 3 and 1 <= c <= 10:                                                  # the cliff
            self.s = 36
            return self.s, -100.0, False, self.t >= 500, {"fell": True}
        self.s = s2
        return s2, -1.0, s2 == 47, self.t >= 500, {}


class CartPole:
    gravity, masscart, masspole, length, force_mag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    theta_limit, x_limit = 12 * 2 * math.pi / 360, 2.4

    def __init__(self, max_steps=500, seed=0):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.n_actions, self.obs_dim = 2, 4

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, 4)
        self.t = 0
        return self.state.astype(np.float32)

    def step(self, a):
        x, x_dot, th, th_dot = self.state
        force = self.force_mag if a == 1 else -self.force_mag
        total, pml = self.masspole + self.masscart, self.masspole * self.length
        cos, sin = math.cos(th), math.sin(th)
        temp = (force + pml * th_dot ** 2 * sin) / total
        th_acc = (self.gravity * sin - cos * temp) / (self.length * (4 / 3 - self.masspole * cos ** 2 / total))
        x_acc = temp - pml * th_acc * cos / total
        x, x_dot = x + self.tau * x_dot, x_dot + self.tau * x_acc                   # Euler, as in the original
        th, th_dot = th + self.tau * th_dot, th_dot + self.tau * th_acc
        self.state = np.array([x, x_dot, th, th_dot])
        self.t += 1
        terminated = abs(x) > self.x_limit or abs(th) > self.theta_limit
        return self.state.astype(np.float32), 1.0, terminated, self.t >= self.max_steps, {}


class TicTacToe:
    LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    @staticmethod
    def winner(board):
        for a, b, c in TicTacToe.LINES:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]
        return 0 if 0 in board else None                                            # None: draw; 0: game on

    @staticmethod
    def moves(board):
        return [i for i, v in enumerate(board) if v == 0]

    @staticmethod
    def play(board, i, player):
        b = list(board)
        b[i] = player
        return tuple(b)
