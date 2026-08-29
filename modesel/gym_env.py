"""gymnasium.Env wrapper for modesel, following train/env.py's established
pattern: fixed Discrete action space, explicit boolean mask, observation
dict with 'point' / 'zone' / 'mask'. Same-process vector env only
(train/vec_env.py) -- Federation objects are not picklable.

Zone tracking is ONLINE here (propagated forward one real step at a time,
via the exact same zone-update logic zones.py uses for DP replay --
zones.advance_zone_for_transition, shared, not duplicated), using REAL
sampled durations for the concrete dynamics and the mode's TRUE interval
for the zone -- the same asymmetry used throughout this project.

Point layout: ``[g] ++ [status one-hot (4: waiting/active/done/failed) per
raw task] ++ [elapsed fraction per raw task]`` -- matches
encoders.mlp_point.MLPPointEncoder's ``n_status_categories=4``,
``n_blocks=n_tasks_total`` contract exactly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from zoneviews.zone import Zone

from .env import ModeSelState, action_mask, initial_state, is_legal, is_terminal, step
from .instance import MODES, ModeSelInstance, SCALE, action_id, parse_action_id
from .zones import ModeSelZoneContext, advance_zone_for_transition

K_MARGIN = 4  # see geometry.py's identical note: scale per instance, not globally


class ModeSelGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, instance: ModeSelInstance, seed: Optional[int] = None):
        super().__init__()
        self.instance = instance
        self._seed = seed

        self.tasks = instance.all_tasks  # order defines both point-vector blocks and action order
        self.n_blocks = len(self.tasks)
        self.n_status_categories = 4  # waiting / active / done / failed
        self.n_tasks = 2 * self.n_blocks  # action space: (task, mode) pairs
        # zone.d = n_jobs + 1 (the global clock g is part of the zone too,
        # unlike job-shop's delta-free zones) -- n_clocks must match that
        # exactly, since it sizes the CNN/DeepSet encoders' input dim.
        # task_job_idx below still only ever indexes 0..n_jobs-1 (no action
        # "owns" g), so g's per-element row is computed but never broadcast
        # to any action -- harmless, not a bug.
        self.n_clocks = instance.n_jobs + 1

        self.action_ids: List[str] = [action_id(t.task_id, m) for t in self.tasks for m in MODES]
        self.task_job_idx: np.ndarray = np.array(
            [instance.job_index(t.job_id) for t in self.tasks for _ in MODES], dtype=np.int64
        )
        # mlp_point's per-element grouping is by raw TASK (n_blocks), not
        # job -- every action's block index, in the SAME action order as
        # action_ids/task_job_idx above (2 consecutive actions per task).
        self.mlp_group_idx: np.ndarray = np.repeat(np.arange(self.n_blocks, dtype=np.int64), len(MODES))
        self._point_dim = 1 + (self.n_status_categories + 1) * self.n_blocks

        self.action_space = spaces.Discrete(self.n_tasks)
        self.observation_space = spaces.Dict(
            {
                "point": spaces.Box(-np.inf, np.inf, shape=(self._point_dim,), dtype=np.float32),
                "mask": spaces.MultiBinary(self.n_tasks),
            }
        )

        self._k_bound_real = max(j.deadline for j in instance.jobs) + K_MARGIN
        self._zctx = ModeSelZoneContext(instance)
        self._state: Optional[ModeSelState] = None
        self._Z = None
        self._rng: Optional[np.random.Generator] = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        eff_seed = seed if seed is not None else self._seed
        self._rng = np.random.default_rng(eff_seed)
        self._state = initial_state(self.instance)
        self._Z = self._zctx.ctx.get_zero_federation()
        return self._get_obs(), {}

    def step(self, action: int):
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        aid = self.action_ids[int(action)]
        task_id, mode = parse_action_id(aid)
        if not is_legal(self.instance, self._state, task_id, mode):
            enabled = [self.action_ids[i] for i, ok in enumerate(action_mask(self.instance, self._state).values()) if ok]
            raise ValueError(f"Action {action} ({aid!r}) is not currently enabled. Enabled: {enabled}")

        task = self.instance.task_by_id(task_id)
        job_idx = self.instance.job_index(task.job_id)

        s_before = self._state
        res = step(self.instance, s_before, task_id, mode, self._rng)
        self._Z = advance_zone_for_transition(self._Z, self._zctx, self.instance, s_before, job_idx, mode, res.trace)
        self._state = res.next_state

        done = res.done
        reward = -res.cost
        info: Dict[str, Any] = {"newly_failed": res.newly_failed}
        if done:
            n_failed_jobs = sum(1 for js in self._state.jobs if js.failed)
            info["domain_metric"] = self._state.g  # makespan
            info["success"] = n_failed_jobs == 0

        return self._get_obs(), reward, done, False, info

    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, Any]:
        return {"point": self._get_point(), "zone": self._current_zone(), "mask": self._action_mask_array()}

    def _get_point(self) -> np.ndarray:
        s = self._state
        shared = np.array([s.g], dtype=np.float32)
        status = np.zeros((self.n_blocks, self.n_status_categories), dtype=np.float32)
        elapsed = np.zeros(self.n_blocks, dtype=np.float32)
        for i, t in enumerate(self.tasks):
            job_idx = self.instance.job_index(t.job_id)
            js = s.jobs[job_idx]
            if js.failed:
                status[i, 3] = 1.0
            elif t.index < js.cursor:
                status[i, 2] = 1.0  # done
            elif t.index == js.cursor and js.active_mode is not None:
                status[i, 1] = 1.0  # active
                if js.active_duration:
                    elapsed[i] = min(1.0, js.c / js.active_duration)
            else:
                status[i, 0] = 1.0  # waiting (includes not-yet-reached tasks)
        return np.concatenate([shared, status.reshape(-1), elapsed]).astype(np.float32)

    def _current_zone(self) -> Zone:
        k_vec = tuple([int(round(self._k_bound_real * SCALE))] * len(self._zctx.clock_names))
        return Zone(self._Z, self._zctx.clock_names, k_vec).clip_to_k_box()

    def _action_mask_array(self) -> np.ndarray:
        mask = action_mask(self.instance, self._state)
        return np.array([mask[aid] for aid in self.action_ids], dtype=bool)


def make_env(instance: ModeSelInstance, seed: Optional[int] = None) -> ModeSelGymEnv:
    return ModeSelGymEnv(instance, seed=seed)
