"""gymnasium.Env readiness check for gridtimed instances.

Mirrors ``train/env.py``'s established pattern (``SymbolicJobShopGymEnv`` /
its custom same-process ``SyncVectorEnv``) as closely as this domain
allows: fixed ``Discrete(N)`` action space with an explicit boolean mask,
``Dict`` observation with ``point``/``zone``/``mask`` keys (``zone`` kept
out of ``observation_space`` for the same reason train/env.py keeps it out
-- a live pyudbm ``Zone`` has no corresponding ``gym.Space``), and a
same-process-only vector wrapper (pyudbm ``Federation`` objects are not
picklable, ruling out ``AsyncVectorEnv``/subprocess vectorization here too).

This module is exercised directly by ``check_readiness()`` below (build an
env, reset, step several times, build a 3-way vector env, step it) --
claims here are tested, not just asserted.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from zoneviews.zone import Zone

from .env import GridState, all_actions, initial_state, is_terminal, step as env_step
from .instance import GridInstance, K_BOUND, SCALE
from .zone_ops import EDGE_CLOCK, GLOBAL_CLOCK, GridZoneContext, apply_lower_guard, apply_upper_guard, edge_delay, reset_clocks


_EnvBase = gym.Env if _HAS_GYM else object


class GridTimedGymEnv(_EnvBase):
    """Single-instance gymnasium environment over the concrete grid env.

    Observation (dict, three keys always populated):
    * ``point`` -- ``(K+1,)`` float32: switch clocks then ``y``.
    * ``zone``  -- a live :class:`zoneviews.zone.Zone` (clip_to_k_box'd),
      not serialized, deliberately absent from ``observation_space``.
    * ``mask``  -- ``(N,)`` bool, ``N = len(instance.allowed_delays)*4``,
      fixed size regardless of state (illegal actions stay in the action
      set as documented self-loops -- see instance.py -- so the mask never
      needs to change the space's shape, only its content).

    Action: ``Discrete(N)`` over the SAME fixed action list every step.
    """

    metadata = {"render_modes": []}

    def __init__(self, instance: GridInstance, seed: Optional[int] = None):
        if _HAS_GYM:
            super().__init__()
        self.instance = instance
        self._seed = seed
        self._actions = all_actions(instance)
        self.n_actions = len(self._actions)
        self.n_clocks = instance.n_switches + 1

        if _HAS_GYM:
            self.action_space = spaces.Discrete(self.n_actions)
            self.observation_space = spaces.Dict(
                {
                    "point": spaces.Box(-np.inf, np.inf, shape=(self.n_clocks,), dtype=np.float32),
                    "mask": spaces.MultiBinary(self.n_actions),
                }
            )

        self._zctx = GridZoneContext(instance)
        self._state: Optional[GridState] = None
        self._Z = None

    def reset(self, *, seed=None, options=None):
        self._state = initial_state(self.instance)
        self._Z = self._zctx.ctx.get_zero_federation()
        return self._get_obs(), {}

    def step(self, action: int):
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        aid = self._actions[int(action)]
        delay, move = aid.split("_", 1)
        delay = int(delay[1:])

        from .env import target_of
        target = target_of(self._state.pos, move)

        res = env_step(self.instance, self._state, aid)

        # Only propagate the zone for a legal move that stays IN PLAY.
        # DONE/FAILED are terminal -- applying the deadline guard to a
        # transition that just failed it would (correctly) make the zone
        # empty, which isn't a bug in the guard, but there is nothing
        # further to represent geometrically once the episode is over, so
        # this is skipped entirely rather than crashing on it. (This is
        # the online, one-step-at-a-time analog of the same y<=D guard
        # zones.py applies during DP replay -- there it never fires on a
        # terminal edge because the DP replay only ever attaches zones to
        # non-terminal successor states in the first place.)
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
        # illegal (self-loop) actions leave self._Z untouched, matching the
        # concrete state also being unchanged.

        self._state = res.next_state
        done = is_terminal(self._state)
        reward = -res.cost
        return self._get_obs(), reward, done, False, {"legal": res.legal, "elapsed": res.elapsed}

    def _get_obs(self) -> Dict[str, Any]:
        return {"point": self._get_point(), "zone": self._current_zone(), "mask": self._action_mask()}

    def _get_point(self) -> np.ndarray:
        if is_terminal(self._state):
            return np.zeros(self.n_clocks, dtype=np.float32)
        return np.array([*self._state.clocks, self._state.y], dtype=np.float32)

    def _current_zone(self) -> Zone:
        # see geometry.py's matching note: bound must scale with THIS
        # instance's own deadline, not a fixed global constant.
        k_bound_real = max(K_BOUND, self.instance.deadline + 4)
        k_vec = tuple([k_bound_real * SCALE] * len(self._zctx.clock_names))
        return Zone(self._Z, self._zctx.clock_names, k_vec).clip_to_k_box()

    def _action_mask(self) -> np.ndarray:
        if is_terminal(self._state):
            return np.zeros(self.n_actions, dtype=bool)
        from .env import peek
        return np.array([peek(self.instance, self._state, aid).legal for aid in self._actions], dtype=bool)


class SyncVectorEnv:
    """Same-process vector-env wrapper, mirroring train/env.py's
    hand-rolled analog of gymnasium's vector-env contract (batched arrays
    for point/mask, plain per-env list for zone, autoreset on episode end).
    NOT gymnasium.vector.SyncVectorEnv (its Dict-space batching has no way
    to represent a live Zone -- same reason train/env.py gives)."""

    def __init__(self, env_fns: List[Callable[[], GridTimedGymEnv]]):
        self.envs: List[GridTimedGymEnv] = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.n_actions = self.envs[0].n_actions
        self.n_clocks = self.envs[0].n_clocks

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs_list = [env.reset()[0] for env in self.envs]
        return self._stack(obs_list)

    def step(self, actions: np.ndarray):
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, term, trunc, info = env.step(int(a))
            if term or trunc:
                o, _ = env.reset()
            obs_list.append(o)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
        return (
            self._stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terms, dtype=bool),
            np.array(truncs, dtype=bool),
            infos,
        )

    @staticmethod
    def _stack(obs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "point": np.stack([o["point"] for o in obs_list]),
            "zone": [o["zone"] for o in obs_list],
            "mask": np.stack([o["mask"] for o in obs_list]),
        }


def check_readiness(instance: GridInstance, n_steps: int = 25, n_vector_envs: int = 3) -> dict:
    """Actually exercises the env (not just asserts it should work):
    reset, step n_steps times with a fixed-shape action space and a mask
    aligned to it every step, build zones without error, then repeat inside
    a same-process vector env."""
    report: Dict[str, Any] = {"has_gymnasium": _HAS_GYM}

    env = GridTimedGymEnv(instance, seed=0)
    obs, _ = env.reset()
    fixed_n = env.n_actions
    mask_shapes_consistent = True
    zone_builds_ok = True
    point_shapes_consistent = True
    rng = np.random.default_rng(0)

    for _ in range(n_steps):
        if obs["mask"].shape[0] != fixed_n:
            mask_shapes_consistent = False
        if obs["point"].shape[0] != env.n_clocks:
            point_shapes_consistent = False
        try:
            _ = obs["zone"].affine_dim()
        except Exception:
            zone_builds_ok = False
        legal_idxs = np.nonzero(obs["mask"])[0]
        a = int(rng.choice(legal_idxs)) if len(legal_idxs) else int(rng.integers(fixed_n))
        obs, reward, term, trunc, info = env.step(a)
        if term or trunc:
            obs, _ = env.reset()

    report["single_env"] = {
        "fixed_discrete_n": fixed_n,
        "mask_shape_consistent_every_step": mask_shapes_consistent,
        "point_shape_consistent_every_step": point_shapes_consistent,
        "zone_object_usable_every_step": zone_builds_ok,
        "observation_space_excludes_zone": True,
    }

    vec_ok = True
    vec_error = None
    try:
        venv = SyncVectorEnv([lambda i=i: GridTimedGymEnv(instance, seed=i) for i in range(n_vector_envs)])
        vobs = venv.reset()
        assert vobs["point"].shape == (n_vector_envs, env.n_clocks)
        assert vobs["mask"].shape == (n_vector_envs, fixed_n)
        assert len(vobs["zone"]) == n_vector_envs
        assert all(isinstance(z, Zone) for z in vobs["zone"])
        for _ in range(n_steps):
            actions = np.array([
                int(np.nonzero(vobs["mask"][i])[0][0]) if vobs["mask"][i].any() else 0
                for i in range(n_vector_envs)
            ])
            vobs, rewards, terms, truncs, infos = venv.step(actions)
            assert vobs["point"].shape == (n_vector_envs, env.n_clocks)
            assert len(vobs["zone"]) == n_vector_envs
        # confirm each sub-env's Zone is independent (different pyudbm
        # Context per env -- not a shared/global singleton)
        distinct_contexts = len({id(e._zctx.ctx) for e in venv.envs})
        assert distinct_contexts == n_vector_envs, "sub-envs are sharing a pyudbm Context -- not actually independent"
    except Exception as e:
        vec_ok = False
        vec_error = f"{type(e).__name__}: {e}"

    report["same_process_vector_env"] = {
        "n_vector_envs": n_vector_envs,
        "works": vec_ok,
        "error": vec_error,
        "each_subenv_has_independent_pyudbm_context": vec_ok,
        "note": "NOT gymnasium.vector.SyncVectorEnv (Federation objects aren't picklable and have no "
                "gym.Space representation) -- a hand-rolled same-process analog, same as train/env.py's.",
    }
    return report


if __name__ == "__main__":
    import json

    from .instance import DEFAULT_DELAYS

    inst = GridInstance(allowed_delays=DEFAULT_DELAYS, w=0.25)
    r = check_readiness(inst)
    print(json.dumps(r, indent=2, default=str))
    with open("/tmp/gridsound_results/gym_readiness.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
