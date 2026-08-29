"""Replay concrete trajectories into live symbolic Federations, gate_track
version: completion guards/invariants use the true [lo, hi] interval for
each completed task's *nominal* duration, not an exact pinned value --
this is what makes the resulting zones non-degenerate (Measurement 1)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .instance import SCALE, GateJobShopInstance, interval
from .symbolic_env import GateSymbolicJobShopEnv
from .zone_ops import GateZoneContext, apply_delay, apply_lower_guard, apply_invariant, normalize, reset_clocks


def _invariant_of(env: GateSymbolicJobShopEnv, loc: tuple) -> List[Tuple[str, int]]:
    """c_J <= hi for every active task, hi from its NOMINAL duration's
    interval (scaled)."""
    result = []
    for tid, status in zip(env._task_ids, loc):
        if status.startswith("active_d"):
            job_id = env._task_job[tid]
            nominal_d = int(status[len("active_d"):])
            _lo, hi = interval(nominal_d, env.w)
            result.append((f"c_{job_id}", int(round(hi * SCALE))))
    return result


def collect_zones(
    instance: GateJobShopInstance,
    n_episodes: int,
    seed: int,
    m_bound: Optional[int] = None,
    do_normalize: bool = True,
    policy: str = "uniform",
) -> List[Tuple[tuple, "object", GateZoneContext]]:
    zctx = GateZoneContext(instance)
    if m_bound is None:
        m_bound = int(instance.worst_case_makespan_upper_bound() * SCALE) + SCALE

    out = []
    for ep in range(n_episodes):
        env = GateSymbolicJobShopEnv(instance, seed=seed * 1_000_003 + ep)
        loc, actions, info = env.reset()
        Z = zctx.ctx.get_zero_federation()
        rng = np.random.default_rng(seed * 1_000_003 + ep + 777_777)
        done = all(s in ("done", "failed") for s in loc)

        while not done:
            if not actions:
                break
            if do_normalize:
                Z = normalize(Z, zctx, m_bound)
            out.append((loc, Z, zctx))

            a = actions[rng.integers(len(actions))]
            tid = a["action_id"]
            job_id = env._task_job[tid]

            loc_before_dispatch = list(env._loc)
            loc, actions, done, info = env.step(tid)
            d = info["sampled_d"]  # NOMINAL duration -- zones use nominal-derived [lo,hi], not the realized real value

            loc_after_dispatch = list(loc_before_dispatch)
            loc_after_dispatch[env._task_to_idx[tid]] = f"active_d{d}"
            Z = reset_clocks(Z, zctx, [zctx.c(job_id)])
            Z = apply_invariant(Z, zctx, _invariant_of(env, tuple(loc_after_dispatch)))

            for step in env._last_advance_path:
                inv_before = _invariant_of(env, step["loc_before"])
                Z = apply_delay(Z, zctx, inv_before)
                for ctid in step["completing_tasks"]:
                    status = step["loc_before"][env._task_to_idx[ctid]]
                    nominal_d = int(status[len("active_d"):])
                    lo, _hi = interval(nominal_d, env.w)
                    cjid = env._task_job[ctid]
                    Z = apply_lower_guard(Z, zctx, [(zctx.c(cjid), int(round(lo * SCALE)))])
                if Z is None or Z.is_empty():
                    raise RuntimeError("collect_zones: zone unexpectedly empty during replay")

    return out
