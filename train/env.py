"""gymnasium.Env wrapper around ``SymbolicJobShopEnv``, delta-free.

Design carried forward from ``zoneviews.collect``:

* ``delta`` is not part of the zone here either — this module reuses
  ``zoneviews.collect``'s delta-free, job-clock-only zone context and
  propagation helpers directly (``_JobClockContext``, ``_job_clock_k``,
  ``_propagate_branches``, ``_load_instance``) rather than duplicating that
  logic. Nothing under ``prob_jobshop_ext`` is imported for its ``delta``
  machinery (``zone_ops.normalize``/``min_cost``/``ZoneContext`` are never
  used).
* Elapsed time (and hence reward) is read off the *concrete* side, not the
  zone, matching ``region_env.py:86-111``'s convention exactly: reward is
  the negative of the real elapsed time this step (cost minimization framed
  as reward maximization via negation). No reward shaping is added.
  Concretely: ``SymbolicJobShopEnv`` resets the dispatched job's clock to 0
  immediately before advancing to the next decision point, and every job's
  virtual clock (active or waiting) advances by the *same* delta at every
  micro-step in between (standard timed-automata delay semantics — all
  clocks tick together). So the dispatched job's virtual clock, read right
  after ``env.step()`` returns, equals exactly the total elapsed time for
  this whole decision-to-decision transition, with no extra bookkeeping
  needed.

Action space and observation shape are modeled on ``prob_jobshop_ext/env.py``'s
dead ``JobShopEnv``: fixed ``Discrete(N)`` over the flat task list, explicit
boolean action mask, invalid actions raise ``ValueError`` (mirroring that
class's philosophy of a hard invariant rather than silent clipping).

Vectorization
-------------
pyudbm ``Federation`` objects are not picklable, so nothing here may cross a
process boundary — this rules out ``gymnasium.vector.AsyncVectorEnv``/
``SubprocVecEnv`` entirely. It also rules out *using* the generic
``gymnasium.vector.SyncVectorEnv`` in the usual way: that class batches
``Dict`` observations by iterating ``observation_space.spaces``, and there is
no ``gym.Space`` for "a live Python object" — a ``Zone`` cannot be
represented as a ``Box``/``Discrete``/etc. This module therefore provides
its own minimal, same-process :class:`SyncVectorEnv` that mirrors
gymnasium's vector-env step/reset/autoreset *contract* (batched arrays for
the numeric fields, autoreset on episode end) while keeping ``zone`` as a
plain per-env Python list rather than forcing it through space machinery it
was never designed for.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from prob_jobshop_ext.instance import ProbJobShopInstance
from prob_jobshop_ext.symbolic_env import SymbolicJobShopEnv
from prob_jobshop_ext.zone_ops import apply_delay, apply_invariant

from zoneviews.collect import _JobClockContext, _job_clock_k, _load_instance, _propagate_branches
from zoneviews.zone import Zone

__all__ = ["SymbolicJobShopGymEnv", "SyncVectorEnv", "make_env"]


class SymbolicJobShopGymEnv(gym.Env):
    """Single-instance gymnasium environment over ``SymbolicJobShopEnv``.

    Observation (dict, three keys always populated):

    * ``point``  ``(d + 4N,)`` float32 — concrete clock valuation and task
      status: ``d`` job-clock values, then ``N`` one-hot task-status triples
      (waiting/active/done), then ``N`` per-task elapsed-duration fractions
      (0 unless the task is active).
    * ``zone``   a :class:`zoneviews.zone.Zone` (already ``clip_to_k_box``'d)
      — live, not serialized. Deliberately absent from ``observation_space``
      (see module docstring); still always present in the dict returned by
      :meth:`reset`/:meth:`step`.
    * ``mask``   ``(N,)`` bool — ``True`` for currently-dispatchable tasks.

    Action: ``Discrete(N)``, ``N`` = total tasks (fixed job/task order:
    job order, then task order within job). An action for a task that is
    not currently enabled raises ``ValueError`` rather than being silently
    ignored or clipped, mirroring ``prob_jobshop_ext/env.py``'s
    ``JobShopEnv``.

    Reward: ``-elapsed`` (elapsed real time this step, from the concrete
    side — see module docstring). No shaping.
    """

    metadata = {"render_modes": []}

    def __init__(self, instance: "str | ProbJobShopInstance", seed: Optional[int] = None):
        super().__init__()
        self.instance: ProbJobShopInstance = _load_instance(instance) if isinstance(instance, str) else instance
        self._seed = seed

        self._task_ids: List[str] = [t.task_id for j in self.instance.jobs for t in j.tasks]
        self._task_to_idx: Dict[str, int] = {tid: i for i, tid in enumerate(self._task_ids)}
        self._job_ids: List[str] = [j.job_id for j in self.instance.jobs]
        self._job_to_idx: Dict[str, int] = {jid: i for i, jid in enumerate(self._job_ids)}
        self._task_job_id: List[str] = [
            self.instance.job_of_task(tid).job_id for tid in self._task_ids
        ]
        #: Index of each task's owning job/clock, length N, values in [0, d).
        #: Used by per-element-head encoders (e.g. CNN-DBM) to broadcast a
        #: per-job embedding out to that job's task slot(s).
        self.task_job_idx: np.ndarray = np.array(
            [self._job_to_idx[jid] for jid in self._task_job_id], dtype=np.int64
        )

        self.n_tasks = len(self._task_ids)
        self.n_clocks = len(self._job_ids)
        self._point_dim = self.n_clocks + 4 * self.n_tasks

        self._k = _job_clock_k(self.instance)
        self._zctx = _JobClockContext(self.instance)

        self.action_space = spaces.Discrete(self.n_tasks)
        self.observation_space = spaces.Dict(
            {
                "point": spaces.Box(-np.inf, np.inf, shape=(self._point_dim,), dtype=np.float32),
                "mask": spaces.MultiBinary(self.n_tasks),
            }
        )

        self._env: Optional[SymbolicJobShopEnv] = None
        self._loc: Optional[Tuple[str, ...]] = None
        self._actions: List[dict] = []
        self._Z = None

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed
        self._env = SymbolicJobShopEnv(self.instance, seed=effective_seed)
        self._loc, self._actions = self._env.reset()

        Z = self._zctx.ctx.get_zero_federation()
        Z = apply_invariant(Z, self._zctx, self._env.invariant(self._loc))
        Z = apply_delay(Z, self._zctx, self._env.invariant(self._loc), self._env.rate(self._loc))
        self._Z = Z

        return self._get_obs(), {}

    def step(self, action: int):
        if self._env is None:
            raise RuntimeError("step() called before reset()")

        task_id = self._task_ids[int(action)]
        action_dict = next((a for a in self._actions if a["action_id"] == task_id), None)
        if action_dict is None:
            enabled = [a["action_id"] for a in self._actions]
            raise ValueError(f"Action {action} ({task_id!r}) is not currently enabled. Enabled: {enabled}")

        branches = _propagate_branches(self._Z, self._zctx, action_dict, self._env)

        loc_new, actions_new, done, info = self._env.step(task_id)
        sampled_d = info["sampled_d"]

        # Elapsed time from the concrete side (region_env.py:86-111
        # convention) -- see module docstring for why this single lookup is
        # exact.
        job_id = self._task_job_id[self._task_to_idx[task_id]]
        elapsed = float(self._env._virtual_clocks[job_id])

        matching = [Zf for d, Zf in branches if d == sampled_d]
        self._Z = matching[0] if matching else (branches[0][1] if branches else self._Z)

        self._loc, self._actions = loc_new, actions_new
        reward = -elapsed

        return self._get_obs(), reward, bool(done), False, {"elapsed": elapsed}

    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, Any]:
        return {
            "point": self._get_point(),
            "zone": self._current_zone(),
            "mask": self._action_mask(),
        }

    def _get_point(self) -> np.ndarray:
        vc = self._env._virtual_clocks
        job_clock_vec = np.array([vc[jid] for jid in self._job_ids], dtype=np.float32)

        status_onehot = np.zeros((self.n_tasks, 3), dtype=np.float32)
        elapsed_frac = np.zeros(self.n_tasks, dtype=np.float32)
        for i, (tid, jid) in enumerate(zip(self._task_ids, self._task_job_id)):
            s = self._loc[i]
            if s == "waiting":
                status_onehot[i, 0] = 1.0
            elif s == "done":
                status_onehot[i, 2] = 1.0
            else:  # "active_d{X}"
                status_onehot[i, 1] = 1.0
                exact_d = int(s[len("active_d"):])
                if exact_d > 0:
                    elapsed_frac[i] = min(1.0, vc[jid] / exact_d)

        return np.concatenate([job_clock_vec, status_onehot.reshape(-1), elapsed_frac]).astype(np.float32)

    def _current_zone(self) -> Zone:
        return Zone(self._Z, self._zctx.clock_names, self._k).clip_to_k_box()

    def _action_mask(self) -> np.ndarray:
        enabled = {a["action_id"] for a in self._actions}
        return np.array([tid in enabled for tid in self._task_ids], dtype=bool)


def make_env(instance_name: str, seed: Optional[int] = None) -> SymbolicJobShopGymEnv:
    return SymbolicJobShopGymEnv(instance_name, seed=seed)


class SyncVectorEnv:
    """Same-process vector-env wrapper (not ``gymnasium.vector.SyncVectorEnv``
    — see module docstring for why). Mirrors its step/reset/autoreset
    contract for the numeric fields.

    ``step()`` auto-resets any sub-env that just terminated, matching
    gymnasium's vector-env convention (the returned ``obs`` is already the
    new episode's initial observation; ``info[i]['final_observation']``
    carries the terminal one, and ``info[i]['episode_done']`` is set).
    """

    def __init__(self, env_fns: List[Callable[[], SymbolicJobShopGymEnv]]):
        self.envs: List[SymbolicJobShopGymEnv] = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space
        self.n_tasks = self.envs[0].n_tasks
        self.n_clocks = self.envs[0].n_clocks
        self.task_job_idx = self.envs[0].task_job_idx

        self._ep_return = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_length = np.zeros(self.num_envs, dtype=np.int64)
        #: Drained by the trainer each update: list of
        #: {"return": float, "length": int, "makespan": float}.
        self.completed_episodes: List[Dict[str, float]] = []

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs_list = []
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + i
            o, _ = env.reset(seed=env_seed)
            obs_list.append(o)
        self._ep_return[:] = 0.0
        self._ep_length[:] = 0
        return self._stack(obs_list)

    def step(self, actions: np.ndarray):
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, info = env.step(int(a))
            self._ep_return[i] += r
            self._ep_length[i] += 1

            if term or trunc:
                self.completed_episodes.append(
                    {
                        "return": float(self._ep_return[i]),
                        "length": int(self._ep_length[i]),
                        "makespan": float(-self._ep_return[i]),
                    }
                )
                final_obs = o
                o, _ = env.reset()
                info = dict(info)
                info["final_observation"] = final_obs
                info["episode_done"] = True
                self._ep_return[i] = 0.0
                self._ep_length[i] = 0

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

    def drain_episode_stats(self) -> List[Dict[str, float]]:
        out = self.completed_episodes
        self.completed_episodes = []
        return out

    @staticmethod
    def _stack(obs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "point": np.stack([o["point"] for o in obs_list]),
            "zone": [o["zone"] for o in obs_list],
            "mask": np.stack([o["mask"] for o in obs_list]),
        }
