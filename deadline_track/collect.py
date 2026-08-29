"""Replay concrete random-policy rollouts into live symbolic Federations.

Mirrors the general approach of ``zoneviews.collect`` (Track 1) but adapted
for this fork's environment: no ``delta``, two clock families (``c_J``,
``g_J``), no blanket deadline invariant, and jobs that can fail. Since we
only need to replay *realized* trajectories (not enumerate every possible
duration branch, as the RL-facing Track 1 collector does), this drives the
zone forward using ``DeadlineSymbolicJobShopEnv._last_advance_path`` (the
exact micro-step completion sequence the concrete simulation just took) --
replaying that sequence into the Federation via the same
reset/invariant/delay/guard primitives ``zone_ops.py`` exposes, so the
resulting Federation is provably kept in lockstep with the concrete
simulation rather than independently re-derived (see ``validate_replay`` in
``geometry.py`` for where the two are cross-checked, matching how
``feasible_best_case`` vs. the arithmetic shortcut was validated rather
than assumed).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .instance import SCALE, DeadlineJobShopInstance
from .symbolic_env import DeadlineSymbolicJobShopEnv
from .zone_ops import DeadlineZoneContext, apply_delay, apply_guard, apply_invariant, apply_lower_guard, normalize, reset_clocks


def _invariant_of(env: DeadlineSymbolicJobShopEnv, loc: tuple) -> List[Tuple[str, int]]:
    result = []
    for tid, status in zip(env._task_ids, loc):
        if status.startswith("active_d"):
            job_id = env._task_job[tid]
            exact_d = int(status[len("active_d"):])
            result.append((f"c_{job_id}", exact_d * SCALE))
    return result


def collect_zones(
    instance: DeadlineJobShopInstance,
    n_episodes: int,
    seed: int,
    m_c: Optional[int] = None,
    m_g: Optional[int] = None,
    do_normalize: bool = True,
) -> List[Tuple[tuple, "object", DeadlineZoneContext]]:
    """Returns a list of (loc, Federation, zctx) at every decision point
    across ``n_episodes`` uniform-random-policy rollouts. Federations are
    NOT clipped here (callers decide clip/normalize policy); ``zctx`` is
    shared across all returned entries (one per instance)."""
    zctx = DeadlineZoneContext(instance)
    # A "waiting" job's own c_J is not bounded by a single task's max
    # duration -- like g_J, it tracks elapsed time until its first dispatch
    # -- so both default to the instance's worst-case-makespan bound unless
    # a caller overrides them. See geometry.py's analyze_zones docstring
    # for the crash this fixed when m_c was task-duration-scale instead.
    if m_c is None:
        m_c = int(instance.worst_case_makespan_upper_bound() * SCALE) + SCALE
    if m_g is None:
        m_g = int(instance.worst_case_makespan_upper_bound() * SCALE) + SCALE

    out = []
    for ep in range(n_episodes):
        env = DeadlineSymbolicJobShopEnv(instance, seed=seed * 1_000_003 + ep)
        loc, actions, info = env.reset()
        Z = zctx.ctx.get_zero_federation()
        rng = np.random.default_rng(seed * 1_000_003 + ep + 777_777)
        done = all(s in ("done", "failed") for s in loc)

        while not done:
            if not actions:
                break
            if do_normalize:
                Z = normalize(Z, zctx, m_c, m_g)
            out.append((loc, Z, zctx))

            a = actions[rng.integers(len(actions))]
            tid = a["action_id"]
            job_id = env._task_job[tid]

            loc_before_dispatch = list(env._loc)
            loc, actions, done, info = env.step(tid)
            d = info["sampled_d"]

            # 1) dispatch: reset c_J, invariant for the newly-active set.
            loc_after_dispatch = list(loc_before_dispatch)
            loc_after_dispatch[env._task_to_idx[tid]] = f"active_d{d}"
            Z = reset_clocks(Z, zctx, [zctx.c(job_id)])
            Z = apply_invariant(Z, zctx, _invariant_of(env, tuple(loc_after_dispatch)))

            # 2) replay every micro-step this dispatch triggered.
            for step in env._last_advance_path:
                inv_before = _invariant_of(env, step["loc_before"])
                Z = apply_delay(Z, zctx, inv_before)
                for ctid in step["completing_tasks"]:
                    status = step["loc_before"][env._task_to_idx[ctid]]
                    exact_d = int(status[len("active_d"):])
                    cjid = env._task_job[ctid]
                    Z = apply_lower_guard(Z, zctx, [(zctx.c(cjid), exact_d * SCALE)])
                if Z is None or Z.is_empty():
                    raise RuntimeError("collect_zones: zone unexpectedly empty during replay")

    return out
