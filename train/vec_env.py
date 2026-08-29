"""Generic same-process vector-env wrapper, shared by modesel and gridtimed's
gym wrappers (train/env.py's SymbolicJobShopGymEnv keeps its own, unchanged,
job-shop-specific copy -- this is not a modification of that file, it's new
machinery for the two new domains).

Same contract as train/env.py's SyncVectorEnv (batched arrays for
point/mask, plain per-env list for zone, autoreset on episode end) --
NOT gymnasium.vector.SyncVectorEnv (Federation objects aren't picklable and
have no gym.Space representation, so its Dict-space batching can't work
here either; see train/env.py's own module docstring for the identical
reasoning). Domain-agnostic: each wrapped env's step() must return an
``info`` dict that includes ``"domain_metric"`` (float or None) and
``"success"`` (bool) on the step where an episode ends.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np


class SyncVectorEnv:
    def __init__(self, env_fns: List[Callable[[], Any]]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space
        self.n_tasks = self.envs[0].n_tasks
        self.n_clocks = self.envs[0].n_clocks
        # None (not an empty array) means "this domain has no static,
        # state-independent action->clock grouping" -- gridtimed, where
        # which clock a move is "about" depends on the agent's current
        # position. build_head falls back to a pooled head for the
        # affected encoders in that case; see encoders/base.py.
        self.task_job_idx = getattr(self.envs[0], "task_job_idx", None)
        # mlp_point's own grouping (may differ from task_job_idx -- e.g.
        # modesel groups by raw task, gridtimed groups by move); default to
        # the identity (one block per action) for envs that don't define
        # one, matching job-shop's original one-block-per-task convention.
        self.n_blocks = getattr(self.envs[0], "n_blocks", self.n_tasks)
        self.n_status_categories = getattr(self.envs[0], "n_status_categories", 3)
        self.mlp_group_idx = getattr(self.envs[0], "mlp_group_idx", np.arange(self.n_tasks))

        self._ep_return = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_length = np.zeros(self.num_envs, dtype=np.int64)
        #: Drained by the trainer each update: list of
        #: {"return": float, "length": int, "domain_metric": float or None, "success": bool}.
        self.completed_episodes: List[Dict[str, Any]] = []

    def reset(self, seed=None) -> Dict[str, Any]:
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
                        "domain_metric": info.get("domain_metric"),
                        "success": bool(info.get("success", False)),
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

    def drain_episode_stats(self) -> List[Dict[str, Any]]:
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
