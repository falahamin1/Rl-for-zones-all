"""Region-based Q-learning agent for the single-goal TRM paper example.

State key : (grid_pos, rm_state, y_capped)
            y_capped = min(y, D)  — exact integer clock capped at deadline

For guard  c_min <= y <= D:
  - y is tracked exactly up to D; all y > D share one class.
  - Gives D+1 distinct y states: 0, 1, ..., D.

Zone, by contrast, always has exactly 3 clock nodes regardless of D.

Action : grid_action ∈ {0,1,2,3}  (d_max=0, dt=1 per step)

Q-update:
  Q[(s, a)] += α * (r + γ * max_a' Q[(s', a')] − Q[(s, a)])
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .sg_grid import SGGridEnv, ACTIONS
from .sg_trm import SGTRMInstance, SGTRMState, SG_U_TERMINAL


class SGRegionEnv:
    """Single-goal TRM Frozen Lake with integer-clock region states."""

    def __init__(
        self,
        trm: SGTRMInstance,
        slip_prob: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.trm   = trm
        self._grid  = SGGridEnv(slip_prob=slip_prob, seed=seed)
        self._trm_s = SGTRMState()
        self._y: int = 0

    def reset(self) -> Tuple:
        pos = self._grid.reset()
        self._trm_s.reset()
        self._y = 0
        return pos, self._trm_s.u

    def step(self, grid_action: int):
        pos, event, env_done = self._grid.step(grid_action)
        self._y += 1
        self._trm_s.y = self._y

        rm_reward, done = self._trm_s.step(event, self.trm)
        done = done or env_done
        return pos, self._trm_s.u, rm_reward, done

    def state_key(self) -> tuple:
        y_cap = min(self._y, self.trm.D)
        return (self._grid.pos(), self._trm_s.u, y_cap)


class SGRegionAgent:
    """Tabular Q-learning over (pos, rm_state, y_capped) × grid_actions.

    Parameters
    ----------
    trm           SGTRMInstance
    alpha         Learning rate
    gamma         Discount factor
    epsilon_start Initial ε for ε-greedy
    epsilon_min   Floor for ε decay
    slip_prob     Grid slip probability
    seed          RNG seed
    """

    def __init__(
        self,
        trm: SGTRMInstance,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.05,
        slip_prob: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.trm    = trm
        self.alpha  = alpha
        self.gamma  = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon = epsilon_start
        self._rng    = random.Random(seed)
        self._seed   = seed
        self.slip_prob = slip_prob

        self._actions: List[int] = list(ACTIONS)
        self.Q: Dict[tuple, Dict[int, float]] = {}

    def _get_q(self, s: tuple) -> Dict[int, float]:
        if s not in self.Q:
            self.Q[s] = {a: 0.0 for a in self._actions}
        return self.Q[s]

    def _best_action(self, s: tuple) -> int:
        q = self._get_q(s)
        return max(q, key=q.get)

    def _eps_greedy(self, s: tuple) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self._actions)
        return self._best_action(s)

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
        env = SGRegionEnv(self.trm, slip_prob=self.slip_prob, seed=ep_seed)
        pos, rm = env.reset()
        total_r = 0.0
        max_steps = 600

        for _ in range(max_steps):
            if rm == SG_U_TERMINAL:
                break
            s = env.state_key()
            action = self._eps_greedy(s) if exploring else self._best_action(s)
            _, rm, reward, done = env.step(action)
            total_r += reward

            if exploring:
                s_new = env.state_key()
                q = self._get_q(s)
                q_new = self._get_q(s_new)
                best_next = max(q_new.values()) if not done else 0.0
                q[action] += self.alpha * (reward + self.gamma * best_next - q[action])
            if done:
                break

        return total_r
