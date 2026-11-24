"""Full-training gymnasium.Env wrapper for trmgrid, following train/env.py's
design contract exactly (point/zone/mask observation dict, fixed
Discrete action space with an explicit boolean mask, reward = -elapsed,
no shaping) and gridtimed/gym_env.py's own concrete structure (read as
the proven precedent for wiring a NEW domain into run_one.py's actual
integration path -- train.vec_env.SyncVectorEnv, not train/env.py's own
job-shop-specific SyncVectorEnv, which nothing outside train/env.py uses).

DETERMINISM (checked directly, not assumed -- see /tmp/TRMGRID_SWEEP.md
Step 2a): trmgrid.env.peek() uses elapsed=delay+1 throughout, exactly
matching gridtimed's own concrete-DP convention -- w (the real,
physically-unobserved [1,1+w] duration jitter) is used ONLY by the zone
construction (zone_ops.edge_delay's symbolic uncertain elapse), never by
this concrete environment. This environment's dynamics are therefore
FULLY DETERMINISTIC given the policy: two rollouts of the same action
sequence from the same reset produce bit-identical trajectories (verified
directly). This is a deliberate design choice carried over unchanged from
Measurement 1/2/3's own DP (see trmgrid/instance.py's module docstring),
not an oversight -- and it is what makes a single (n=1) greedy episode a
valid, complete evaluation of a deterministic policy's behavior, exactly
as it was for gridtimed_eval.py.

``point`` is the FAIR baseline observation: exact real-valued clock
valuation (t_0..t_{k-1}, g -- currently integer-valued since the concrete
env is deterministic on delay+1, cast to float32; a zeroed slot for the
zone-only edge clock e is still allocated so point_dim tracks the zone's
own dimension d exactly, matching gridtimed's identical convention) PLUS
the full discrete state: grid position (one-hot), and RM phase encoded as
TWO k-dim one-hot slots (first sub-goal primed, second sub-goal primed --
both zero at phase=(), only the first set at phase=(i,), both set at
phase=(i,j)) -- this disambiguates order (phase=(0,1) vs (1,0) are
different states, same primed SET, different active guard), which a
plain "visited" bitmask alone could not.

``has_per_element_head = True`` (mlp_point only): moves are grouped by
direction (4 blocks, static -- UP/DOWN/LEFT/RIGHT doesn't depend on
state), same as every other domain in this repo. ``task_job_idx = None``:
which sub-goal a move is "about" depends on the agent's CURRENT position
AND phase (not a fixed action->clock mapping) -- the identical reasoning
gridtimed's own module docstring gives for setting task_job_idx=None
there. build_head therefore falls back to a pooled head for
deepset_h/deepset_v/cnn_dbm/cnn_proj/gnn here too -- mlp_point is the ONE
encoder with a genuine per-element head, matching gridtimed's own
encoder-by-encoder treatment exactly (see /tmp/TRMGRID_SWEEP.md Step 2).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from zoneviews.zone import Zone

from .env import TRMState, all_actions, initial_state, is_terminal, peek, target_of, _parse
from .instance import DELTA, MOVES, TRMGridInstance
from .zone_ops import EDGE_CLOCK, GLOBAL_CLOCK, TRMZoneContext, apply_lower_guard, apply_upper_guard, edge_delay, reset_clocks

N_MOVE_STATUS_CATEGORIES = 3  # wall(out-of-bounds) / plain / sub-goal-or-goal


def _move_features(instance: TRMGridInstance, state: TRMState, move: str):
    target = target_of(state.pos, move)
    if not instance.in_bounds(target):
        return (1.0, 0.0, 0.0), 0.0

    if target == instance.goal_pos and len(state.phase) == 2:
        j = state.phase[1]
        cap_j = instance.caps[j]
        cur = state.clocks[j]
        span = max(1, cap_j)
        dist = (cur - cap_j) / span if cur > cap_j else 0.0
        return (0.0, 0.0, 1.0), float(dist)

    if instance.is_sub_goal(target):
        tgt_idx = instance.sub_goal_index(target)
        if len(state.phase) == 1 and tgt_idx != state.phase[0]:
            i = state.phase[0]
            lo, hi = instance.windows[(i, tgt_idx)]
            cur = state.clocks[i]
            span = max(1, hi - lo + 1)
            if cur < lo:
                dist = (cur - lo) / span
            elif cur > hi:
                dist = (cur - hi) / span
            else:
                dist = 0.0
            return (0.0, 0.0, 1.0), float(dist)
        return (0.0, 0.0, 1.0), 0.0  # first visit (no guard) or incidental touch

    return (0.0, 1.0, 0.0), 0.0


class TRMGridTrainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, instance: TRMGridInstance, seed: Optional[int] = None):
        super().__init__()
        self.instance = instance
        self._seed = seed

        self.action_ids: List[str] = all_actions(instance)
        self.n_tasks = len(self.action_ids)
        self.k = instance.k
        #: Matches zone.d exactly: k sub-goal clocks + g + e (edge clock,
        #: zone-only bookkeeping -- see module docstring).
        self.n_clocks = self.k + 2
        self.n_blocks = len(MOVES)
        self.n_status_categories = N_MOVE_STATUS_CATEGORIES
        self.mlp_group_idx: np.ndarray = np.array(
            [MOVES.index(_parse(aid)[1]) for aid in self.action_ids], dtype=np.int64
        )
        self.task_job_idx = None  # see module docstring

        self._grid_cells = instance.grid_n * instance.grid_n
        #: position one-hot + (k clocks + g + e-placeholder) + 2*k phase one-hots.
        self._point_dim = self._grid_cells + self.n_clocks + 2 * self.k + self.n_blocks * (self.n_status_categories + 1)

        self.action_space = spaces.Discrete(self.n_tasks)
        self.observation_space = spaces.Dict(
            {
                "point": spaces.Box(-np.inf, np.inf, shape=(self._point_dim,), dtype=np.float32),
                "mask": spaces.MultiBinary(self.n_tasks),
            }
        )

        self._k_bound_real = max(20, instance.deadline + 4)
        self._zctx = TRMZoneContext(instance)
        self._state: Optional[TRMState] = None
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
        phase = self._state.phase

        res = peek(self.instance, self._state, aid)

        # Zone update -- mirrors zones.py's build_state_zones edge-by-edge,
        # applied online here exactly as gridtimed's own gym_env.py does.
        if self.instance.in_bounds(target) and res.legal and not is_terminal(res.next_state):
            Z2 = edge_delay(self._Z, self._zctx, delay, self.instance.w)
            Z2 = apply_upper_guard(Z2, self._zctx, [(GLOBAL_CLOCK, self.instance.deadline)])
            is_sg = self.instance.is_sub_goal(target)
            tgt_idx = self.instance.sub_goal_index(target) if is_sg else None
            reset_name = None
            if phase == () and is_sg:
                reset_name = f"t_{tgt_idx}"
            elif len(phase) == 1 and is_sg and tgt_idx != phase[0]:
                i = phase[0]
                lo, hi = self.instance.windows[(i, tgt_idx)]
                Z2 = apply_lower_guard(Z2, self._zctx, [(f"t_{i}", lo)])
                Z2 = apply_upper_guard(Z2, self._zctx, [(f"t_{i}", hi)])
                reset_name = f"t_{tgt_idx}"
            elif is_sg:
                reset_name = f"t_{tgt_idx}"
            if reset_name is not None:
                Z2 = reset_clocks(Z2, self._zctx, [reset_name])
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
            info["fail_reason"] = res.fail_reason

        return self._get_obs(), reward, done, False, info

    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, Any]:
        return {"point": self._get_point(), "zone": self._current_zone(), "mask": self._action_mask()}

    def _get_point(self) -> np.ndarray:
        pos_onehot = np.zeros(self._grid_cells, dtype=np.float32)
        clocks = np.zeros(self.n_clocks, dtype=np.float32)  # t_0..t_{k-1}, g, e-placeholder(always 0)
        first_onehot = np.zeros(self.k, dtype=np.float32)
        second_onehot = np.zeros(self.k, dtype=np.float32)

        if not is_terminal(self._state):
            r, c = self._state.pos
            pos_onehot[r * self.instance.grid_n + c] = 1.0
            clocks[: self.k] = self._state.clocks
            clocks[self.k] = self._state.g
            # clocks[self.k + 1] (e's slot) has no concrete equivalent --
            # e is zone-only bookkeeping, always left at its zero-init
            # value, exact not approximate (matches gridtimed's identical
            # convention for its own edge clock).
            phase = self._state.phase
            if len(phase) >= 1:
                first_onehot[phase[0]] = 1.0
            if len(phase) >= 2:
                second_onehot[phase[1]] = 1.0

        status = np.zeros((self.n_blocks, self.n_status_categories), dtype=np.float32)
        elapsed = np.zeros(self.n_blocks, dtype=np.float32)
        if not is_terminal(self._state):
            for i, move in enumerate(MOVES):
                cat, dist = _move_features(self.instance, self._state, move)
                status[i] = cat
                elapsed[i] = dist

        return np.concatenate(
            [pos_onehot, clocks, first_onehot, second_onehot, status.reshape(-1), elapsed]
        ).astype(np.float32)

    def _current_zone(self) -> Zone:
        k_vec = tuple([int(round(self._k_bound_real * 20))] * len(self._zctx.clock_names))
        return Zone(self._Z, self._zctx.clock_names, k_vec).clip_to_k_box()

    def _action_mask(self) -> np.ndarray:
        if is_terminal(self._state):
            return np.zeros(self.n_tasks, dtype=bool)
        return np.array([peek(self.instance, self._state, aid).legal for aid in self.action_ids], dtype=bool)


def make_env(instance: TRMGridInstance, seed: Optional[int] = None) -> TRMGridTrainEnv:
    return TRMGridTrainEnv(instance, seed=seed)
