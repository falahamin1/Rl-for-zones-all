"""Cross-check: does the live arithmetic feasibility shortcut
(``DeadlineSymbolicJobShopEnv._check_feasible``) agree with the real
zone-based check (``zone_ops.feasible_best_case``) on sampled concrete
decision points? The two are claimed equivalent in several docstrings
because g_J is never reset and always advances at rate 1 regardless of
which job's task is active -- this script tests that claim rather than
leaving it asserted."""
from __future__ import annotations

import random
from typing import List

import numpy as np

from .instance import SCALE, DeadlineJobShopInstance
from .symbolic_env import DeadlineSymbolicJobShopEnv
from .zone_ops import DeadlineZoneContext, feasible_best_case, zone_at_point


def _point_zone_at(env: DeadlineSymbolicJobShopEnv, zctx: DeadlineZoneContext):
    values = {}
    for j in env.instance.jobs:
        jid = j.job_id
        # c_J: only meaningful while active; 0 is fine otherwise (module
        # docstring of zone_at_point -- every clock must be given a value).
        values[zctx.c(jid)] = int(round(env._virtual_clocks.get(jid, 0.0))) * SCALE
        values[zctx.g(jid)] = int(round(env._g_values.get(jid, 0.0))) * SCALE
    return zone_at_point(zctx, values)


def validate(instance: DeadlineJobShopInstance, n_episodes: int = 50, seed: int = 0) -> dict:
    zctx = DeadlineZoneContext(instance)
    rng = np.random.default_rng(seed)
    n_checks = 0
    n_agree = 0
    disagreements: List[dict] = []

    for ep in range(n_episodes):
        env = DeadlineSymbolicJobShopEnv(instance, seed=seed * 10_000 + ep)
        loc, actions, info = env.reset()
        done = False
        while not done:
            # For every "ready" waiting task at this decision point (not
            # just the ones the env already deemed feasible -- re-derive
            # candidates independently here so infeasible-but-ready ones
            # are also checked), re-run both feasibility checks.
            active_machines = set()
            for tid, status in zip(env._task_ids, loc):
                if status.startswith("active"):
                    active_machines.add(env.instance.task_by_id(tid).machine)
            for tid, status in zip(env._task_ids, loc):
                if status != "waiting":
                    continue
                pred_id = env.instance.predecessor_of(tid)
                if pred_id is not None and loc[env._task_to_idx[pred_id]] != "done":
                    continue
                task = env.instance.task_by_id(tid)
                if task.machine in active_machines:
                    continue
                job_id = env._task_job[tid]
                job_tasks = env._job_task_ids[job_id]
                start = job_tasks.index(tid)
                remaining = [(t, env._min_dur[t] * SCALE) for t in job_tasks[start:]]

                arith = env._check_feasible(job_id, tid)
                Zp = _point_zone_at(env, zctx)
                zoneb = feasible_best_case(Zp, zctx, job_id, remaining, instance.deadline_scaled(job_id))

                n_checks += 1
                if arith == zoneb:
                    n_agree += 1
                else:
                    disagreements.append(
                        {
                            "episode": ep,
                            "loc": loc,
                            "job_id": job_id,
                            "task": tid,
                            "arithmetic": arith,
                            "zone_based": zoneb,
                        }
                    )

            if not actions:
                break
            a = actions[rng.integers(len(actions))]
            loc, actions, done, info = env.step(a["action_id"])

    return {
        "n_checks": n_checks,
        "n_agree": n_agree,
        "agreement_rate": n_agree / n_checks if n_checks else float("nan"),
        "n_disagreements": len(disagreements),
        "disagreements_sample": disagreements[:5],
    }


if __name__ == "__main__":
    import zoneviews  # noqa: F401
    from .instance import load_base_instance, make_deadline_variant

    for name, t in [("PJS_04", 0.1), ("PJS_04", 0.5), ("LJS_01", 0.25)]:
        base = load_base_instance(name)
        inst = make_deadline_variant(base, t=t)
        result = validate(inst, n_episodes=40, seed=1)
        print(f"{name} t={t}: checks={result['n_checks']} agree={result['n_agree']} "
              f"rate={result['agreement_rate']:.4f} disagreements={result['n_disagreements']}")
        if result["disagreements_sample"]:
            print("  sample disagreement:", result["disagreements_sample"][0])
