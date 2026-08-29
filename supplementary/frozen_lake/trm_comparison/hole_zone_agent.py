"""Zone Q-learning agent for the Hole-Disable TRM.

Three independent binary clock nodes — 8 nodes total, constant for any
values of c_min, D, x_h:

  x_node : 0 if x < c_min  else 1
  y_node : 0 if y <= D      else 1
  h_node : 0 if h < x_h    else 1

State key : (grid_pos, rm_state, x_node, y_node, h_node, disabled_action)
Action    : grid_action ∈ {UP, DOWN, LEFT, RIGHT}  (d_max=0)

When h_node=0 the disabled_action direction is excluded from the action set.
When h_node=1 disabled_action is -1 (no restriction) in the state key.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hole_grid import HoleGridEnv, ACTIONS, UP, DOWN, LEFT, RIGHT
from .hole_trm import HoleTRMInstance, HoleTRMState, U_TERMINAL


class HoleZoneEnv:
    """Zone-state simulation for the hole-disable TRM."""

    def __init__(
        self,
        trm: HoleTRMInstance,
        slip_prob: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.trm   = trm
        self._grid = HoleGridEnv(slip_prob=slip_prob, seed=seed)
        self._trm_s = HoleTRMState()

    def reset(self) -> Tuple:
        pos = self._grid.reset()
        self._trm_s.reset(self.trm.x_h)
        return pos, self._trm_s.u

    def step(self, action: int) -> Tuple:
        pos, event, done, hole_dir = self._grid.step(action)
        reward, done = self._trm_s.step(action, event, hole_dir, self.trm)
        return pos, self._trm_s.u, reward, done

    def state_key(self) -> tuple:
        t = self._trm_s
        x_node = 0 if t.x < self.trm.c_min else 1
        y_node = 0 if t.y <= self.trm.D    else 1
        h_node = 0 if t.h < self.trm.x_h   else 1
        da     = t.disabled_action if h_node == 0 else -1
        return (self._grid.pos(), t.u, x_node, y_node, h_node, da)

    def valid_actions(self) -> List[int]:
        t = self._trm_s
        if t.h < self.trm.x_h and t.disabled_action >= 0:
            return [a for a in ACTIONS if a != t.disabled_action]
        return list(ACTIONS)


class HoleZoneAgent:
    """Tabular Q-learning over zone-graph states for the hole-disable TRM."""

    def __init__(
        self,
        trm: HoleTRMInstance,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.05,
        slip_prob: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.trm           = trm
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon       = epsilon_start
        self._rng          = random.Random(seed)
        self._seed         = seed
        self.slip_prob     = slip_prob
        self.Q: Dict[tuple, Dict[int, float]] = {}

    def _get_q(self, s: tuple, valid: List[int]) -> Dict[int, float]:
        if s not in self.Q:
            self.Q[s] = {a: 0.0 for a in ACTIONS}
        # Ensure all valid actions have an entry (they should already).
        return self.Q[s]

    def _best_action(self, s: tuple, valid: List[int]) -> int:
        q = self._get_q(s, valid)
        return max(valid, key=lambda a: q[a])

    def _eps_greedy(self, s: tuple, valid: List[int]) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(valid)
        return self._best_action(s, valid)

    def train(
        self,
        episodes: int,
        eval_interval: int = 0,
        eval_episodes: int = 20,
        eval_seed: Optional[int] = None,
    ) -> dict:
        decay = (self.epsilon_start - self.epsilon_min) / max(1, episodes - 1)
        self.epsilon = self.epsilon_start
        checkpoint_evals: List[Tuple[int, float]] = []

        for ep in range(episodes):
            self._run_episode(exploring=True)
            self.epsilon = max(self.epsilon_min, self.epsilon - decay)

            if eval_interval > 0 and (ep + 1) % eval_interval == 0:
                saved = self.epsilon
                self.epsilon = 0.0
                res = self.evaluate(n_episodes=eval_episodes, seed=eval_seed)
                self.epsilon = saved
                checkpoint_evals.append((ep + 1, res["mean"]))

        return {"checkpoint_evals": checkpoint_evals}

    def evaluate(self, n_episodes: int = 100, seed: Optional[int] = None) -> dict:
        saved = self.epsilon
        self.epsilon = 0.0
        base_seed = seed if seed is not None else self._seed
        eval_rng = random.Random(base_seed)
        rewards = [
            self._run_episode(exploring=False, seed=eval_rng.randint(0, 2**31))
            for _ in range(n_episodes)
        ]
        self.epsilon = saved
        arr = np.array(rewards)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "min": float(arr.min()), "max": float(arr.max())}

    def n_states(self) -> int:
        return len(self.Q)

    def _run_episode(self, exploring: bool, seed: Optional[int] = None) -> float:
        ep_seed = seed if seed is not None else self._rng.randint(0, 2**31)
        env = HoleZoneEnv(self.trm, slip_prob=self.slip_prob, seed=ep_seed)
        _, rm = env.reset()
        total_r = 0.0

        for _ in range(500):
            if rm == U_TERMINAL:
                break
            s = env.state_key()
            valid = env.valid_actions()
            action = self._eps_greedy(s, valid) if exploring else self._best_action(s, valid)
            _, rm, reward, done = env.step(action)
            total_r += reward

            if exploring:
                s_new  = env.state_key()
                q      = self._get_q(s, valid)
                q_new  = self._get_q(s_new, env.valid_actions())
                valid_new = env.valid_actions()
                best_next = max(q_new[a] for a in valid_new) if not done else 0.0
                q[action] += self.alpha * (reward + self.gamma * best_next - q[action])

            if done:
                break

        return total_r
