"""Full-training gymnasium.Env wrapper for gridtimed (gym_readiness.py's
GridTimedGymEnv was a minimal readiness check -- notably ``point`` there was
switch-clocks-and-y ONLY, missing position. Without position the point
observation is not Markov-sufficient (the same clock values at two
different grid cells are different decisions), which would make it a
HANDICAPPED baseline -- this module fixes that, since the point baseline
must be fair (task instructions, explicit): ``point`` here is
``[position one-hot] ++ [switch clocks, y] ++ [4 move-blocks]``.

Zone tracking is online (propagated one real step at a time), reusing
gym_readiness.py's exact update logic (same edge_delay/guard/reset chain).

mlp_point's per-element grouping is by MOVE (4 blocks, static: which of
UP/DOWN/LEFT/RIGHT an action is doesn't depend on state). The OTHER three
encoders (cnn_dbm/cnn_proj/deepset_h) do NOT get a per-element head here:
their natural grouping is by clock, but which clock a move is "about"
depends on the agent's CURRENT position, not fixed for the whole episode --
there is no static action->clock mapping to broadcast from, so
``task_job_idx`` is left as ``None`` and build_head falls back to a pooled
head for those three (see encoders/base.py's "Pooled fallback" note, and
/tmp/ENCODERS_SETUP.md item 1 for why this is the domain's genuine
architecture, not a shortcut).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from zoneviews.zone import Zone

from .env import GridState, all_actions, initial_state, is_terminal, target_of, step as env_step, _parse
from .instance import DEADLINE, K_BOUND, MOVES, GridInstance, SCALE
from .zone_ops import EDGE_CLOCK, GLOBAL_CLOCK, GridZoneContext, apply_lower_guard, apply_upper_guard, edge_delay, reset_clocks

N_MOVE_STATUS_CATEGORIES = 3  # wall(out-of-bounds) / plain-or-switch / gate


def _move_features(instance: GridInstance, state: GridState, move: str):
    target = target_of(state.pos, move)
    if not instance.in_bounds(target):
        return (1.0, 0.0, 0.0), 0.0
    if instance.is_gate(target):
        a, b = instance.gate_window(target)
        cname = instance.gate_clock(target)
        idx = instance.clock_names.index(cname)
        cur_val = state.clocks[idx]
        span = max(1, b - a + 1)
        if cur_val < a:
            dist = (cur_val - a) / span
        elif cur_val > b:
            dist = (cur_val - b) / span
        else:
            dist = 0.0
        return (0.0, 0.0, 1.0), float(dist)
    return (0.0, 1.0, 0.0), 0.0


class GridTimedTrainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, instance: GridInstance, seed: Optional[int] = None):
        super().__init__()
        self.instance = instance
        self._seed = seed

        self.action_ids: List[str] = all_actions(instance)
        self.n_tasks = len(self.action_ids)
        self.n_clocks = instance.n_switches + 2  # switches + y + e (edge clock), matches zone.d exactly
        self.n_blocks = len(MOVES)  # 4
        self.n_status_categories = N_MOVE_STATUS_CATEGORIES
        self.mlp_group_idx: np.ndarray = np.array(
            [MOVES.index(_parse(aid)[1]) for aid in self.action_ids], dtype=np.int64
        )
        self.task_job_idx = None  # see module docstring: no static action->clock mapping here

        self._grid_cells = instance.grid_n * instance.grid_n
        self._point_dim = self._grid_cells + self.n_clocks + self.n_blocks * (self.n_status_categories + 1)

        self.action_space = spaces.Discrete(self.n_tasks)
        self.observation_space = spaces.Dict(
            {
                "point": spaces.Box(-np.inf, np.inf, shape=(self._point_dim,), dtype=np.float32),
                "mask": spaces.MultiBinary(self.n_tasks),
            }
        )

        self._k_bound_real = max(K_BOUND, instance.deadline + 4)
        self._zctx = GridZoneContext(instance)
        self._state: Optional[GridState] = None
        self._Z = None
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = initial_state(self.instance)
        self._Z = self._zctx.ctx.get_zero_federation()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: int):
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        aid = self.action_ids[int(action)]
        delay, move = _parse(aid)
        target = target_of(self._state.pos, move)

        res = env_step(self.instance, self._state, aid)

        # Same online zone-update logic as gym_readiness.py's
        # GridTimedGymEnv.step -- see that module for the reasoning.
        if self.instance.in_bounds(target) and res.legal and not is_terminal(res.next_state):
            Z2 = edge_delay(self._Z, self._zctx, delay, self.instance.w)
            if self.instance.is_gate(target):
                a, b = self.instance.gate_window(target)
                cname = self.instance.gate_clock(target)
                Z2 = apply_lower_guard(Z2, self._zctx, [(cname, a)])
                Z2 = apply_upper_guard(Z2, self._zctx, [(cname, b)])
            Z2 = apply_upper_guard(Z2, self._zctx, [(GLOBAL_CLOCK, self.instance.deadline)])
            if self.instance.is_switch(target):
                Z2 = reset_clocks(Z2, self._zctx, [self.instance.switch_clock(target)])
            Z2 = reset_clocks(Z2, self._zctx, [EDGE_CLOCK])
            self._Z = Z2
        elif is_terminal(res.next_state):
            self._Z = self._zctx.ctx.get_zero_federation()

        self._step_count += 1
        self._state = res.next_state
        done = is_terminal(self._state)
        reward = -res.cost
        info: Dict[str, Any] = {"legal": res.legal, "elapsed": res.elapsed}
        if done:
            success = self._state.pos == "DONE"
            info["success"] = success
            info["domain_metric"] = float(self._step_count) if success else None

        return self._get_obs(), reward, done, False, info

    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, Any]:
        return {"point": self._get_point(), "zone": self._current_zone(), "mask": self._action_mask()}

    def _get_point(self) -> np.ndarray:
        pos_onehot = np.zeros(self._grid_cells, dtype=np.float32)
        clocks = np.zeros(self.n_clocks, dtype=np.float32)
        if not is_terminal(self._state):
            r, c = self._state.pos
            pos_onehot[r * self.instance.grid_n + c] = 1.0
            n_sw = self.instance.n_switches
            clocks[:n_sw] = self._state.clocks
            clocks[n_sw] = self._state.y
            # clocks[n_sw + 1] (the edge clock e's slot) has no concrete
            # equivalent -- e is zone-only bookkeeping (see zone_ops.py),
            # always reset to 0 by the time any concrete state is read out,
            # so leaving it at its zero-initialized value is exact, not an
            # approximation.

        status = np.zeros((self.n_blocks, self.n_status_categories), dtype=np.float32)
        elapsed = np.zeros(self.n_blocks, dtype=np.float32)
        if not is_terminal(self._state):
            for i, move in enumerate(MOVES):
                cat, dist = _move_features(self.instance, self._state, move)
                status[i] = cat
                elapsed[i] = dist

        return np.concatenate([pos_onehot, clocks, status.reshape(-1), elapsed]).astype(np.float32)

    def _current_zone(self) -> Zone:
        k_vec = tuple([int(round(self._k_bound_real * SCALE))] * len(self._zctx.clock_names))
        return Zone(self._Z, self._zctx.clock_names, k_vec).clip_to_k_box()

    def _action_mask(self) -> np.ndarray:
        if is_terminal(self._state):
            return np.zeros(self.n_tasks, dtype=bool)
        from .env import peek

        return np.array([peek(self.instance, self._state, aid).legal for aid in self.action_ids], dtype=bool)


def make_env(instance: GridInstance, seed: Optional[int] = None) -> GridTimedTrainEnv:
    return GridTimedTrainEnv(instance, seed=seed)
