"""Cross-check: does the live arithmetic feasibility shortcut agree with
the real zone-based check under the interval + single-global-clock model?
Re-derived, not assumed: g is now literally one shared scalar rather than
n provably-equal copies, so the "shortcut" is arguably even more directly
justified than in deadline_track -- but the *best-case remaining work*
computation changed (now interval-aware, using each task's smallest
nominal duration scaled by (1-w)), so this still needs checking, not just
re-asserting the old conclusion."""
from __future__ import annotations

from typing import List

import numpy as np

from .instance import SCALE, GateJobShopInstance, interval
from .symbolic_env import GateSymbolicJobShopEnv
from .zone_ops import GateZoneContext, feasible_best_case, zone_at_point


def _point_zone_at(env: GateSymbolicJobShopEnv, zctx: GateZoneContext):
    # Scale the exact real value FIRST, then round to the nearest tick --
    # rounding to the nearest integer real-time unit before scaling (an
    # earlier version of this line did exactly that) throws away up to 0.5
    # real units = 10 SCALE-ticks of precision, which is exactly what
    # produced 2/3682 spurious disagreements against the environment's own
    # full-precision arithmetic check. This is a bug in this validation
    # script only -- the environment and DP always scale-then-round on the
    # full-precision value (see symbolic_env.py's _check_feasible).
    values = {}
    for j in env.instance.jobs:
        jid = j.job_id
        values[zctx.c(jid)] = int(round(env._virtual_clocks.get(jid, 0.0) * SCALE))
    values["g"] = int(round(env._g * SCALE))
    return zone_at_point(zctx, values)


def validate(instance: GateJobShopInstance, n_episodes: int = 50, seed: int = 0) -> dict:
    zctx = GateZoneContext(instance)
    rng = np.random.default_rng(seed)
    n_checks = 0
    n_agree = 0
    disagreements: List[dict] = []

    for ep in range(n_episodes):
        env = GateSymbolicJobShopEnv(instance, seed=seed * 10_000 + ep)
        loc, actions, info = env.reset()
        done = False
        while not done:
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
                remaining = []
                for t2 in job_tasks[start:]:
                    min_nom = env._min_nominal[t2]
                    lo, hi = interval(min_nom, env.w)
                    remaining.append((t2, int(round(lo * SCALE)), int(round(hi * SCALE))))

                arith = env._check_feasible(job_id, tid)
                Zp = _point_zone_at(env, zctx)
                zoneb = feasible_best_case(Zp, zctx, job_id, remaining, instance.deadline_scaled(job_id))

                n_checks += 1
                if arith == zoneb:
                    n_agree += 1
                else:
                    disagreements.append(
                        {"episode": ep, "loc": loc, "job_id": job_id, "task": tid, "arithmetic": arith, "zone_based": zoneb}
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
    from deadline_track.instance import load_base_instance
    from .instance import make_gate_variant

    total_checks = 0
    total_agree = 0
    for name, t, w in [("PJS_04", 0.25, 0.1), ("PJS_04", 0.25, 0.25), ("LJS_01", 1.0, 0.25), ("PJS_05", 1.0, 0.25)]:
        base = load_base_instance(name)
        inst = make_gate_variant(base, t=t, w=w)
        result = validate(inst, n_episodes=40, seed=1)
        total_checks += result["n_checks"]
        total_agree += result["n_agree"]
        print(f"{name} t={t} w={w}: checks={result['n_checks']} agree={result['n_agree']} "
              f"rate={result['agreement_rate']:.4f} disagreements={result['n_disagreements']}")
        if result["disagreements_sample"]:
            print("  sample disagreement:", result["disagreements_sample"][0])
    print(f"\nTOTAL: {total_agree}/{total_checks} = {total_agree/total_checks:.4%}")
