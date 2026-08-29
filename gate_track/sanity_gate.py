"""Sanity gate -- BLOCKING, run before anything else in this track.

Same hand-built 2-job instance as deadline_track's sanity gate (job A
duration 3, job B duration 5, deterministic, shared machine m1), run at
w=0.0 as a REGRESSION check (an interval of zero width must reproduce
deadline_track's exact result -- V(root)=8.00, correct minimum-slack order
chosen in both configurations) and then re-checked at w=0.25 to confirm the
interval-aware feasibility mechanism doesn't silently break the same
hand-verifiable case.
"""
from __future__ import annotations

from prob_jobshop_ext.instance import Job, ProbJobShopInstance, Task, TaskDistribution

from .dp import explore_and_solve
from .instance import GateJobShopInstance


def _build_instance(name: str, d_a: float, d_b: float, penalty: float, w: float) -> GateJobShopInstance:
    jobs = [
        Job(job_id="A", tasks=[Task("A_T1", "m1", TaskDistribution([3], [1.0]))]),
        Job(job_id="B", tasks=[Task("B_T1", "m1", TaskDistribution([5], [1.0]))]),
    ]
    base = ProbJobShopInstance(name=name, description="gate sanity 2-job instance", machines=["m1"], jobs=jobs)
    return GateJobShopInstance(
        name=name,
        description=base.description,
        machines=base.machines,
        jobs=base.jobs,
        deadlines={"A": d_a, "B": d_b},
        release={"A": 0.0, "B": 0.0},
        tightness=float("nan"),
        width=w,
        penalty=penalty,
        base_name=name,
    )


def run() -> dict:
    penalty = 3.0 * 8.0
    report = {}

    for w in [0.0, 0.25]:
        inst_a_tight = _build_instance(f"SANITY_A_TIGHT_w{w}", d_a=3.0, d_b=8.0, penalty=penalty, w=w)
        inst_b_tight = _build_instance(f"SANITY_B_TIGHT_w{w}", d_a=8.0, d_b=5.0, penalty=penalty, w=w)

        for label, inst, expected_first in [
            (f"A_tight_w{w}", inst_a_tight, "A_T1"),
            (f"B_tight_w{w}", inst_b_tight, "B_T1"),
        ]:
            res = explore_and_solve(inst, state_limit=2000)
            root_actions = res.optimal_actions[res.root_key]
            root_v = res.V[res.root_key] + res.root_initial_penalty
            n_states = len(res.snapshots)

            ok = root_actions == [expected_first]
            report[label] = {
                "w": w,
                "expected_first_dispatch": expected_first,
                "dp_optimal_first_dispatch": root_actions,
                "dp_root_value": root_v,
                "n_states_explored": n_states,
                "passed": ok,
            }
            if not ok:
                raise AssertionError(
                    f"SANITY GATE FAILED ({label}): expected root-optimal action "
                    f"{[expected_first]}, DP gave {root_actions}. STOPPING."
                )

    print("SANITY GATE PASSED for both A_tight and B_tight instances at w=0.0 (regression) and w=0.25.")
    for label, r in report.items():
        print(f"  {label}: DP picks {r['dp_optimal_first_dispatch']} "
              f"(expected {r['expected_first_dispatch']}), "
              f"V(root)={r['dp_root_value']:.2f}, states={r['n_states_explored']}")
    return report


if __name__ == "__main__":
    run()
