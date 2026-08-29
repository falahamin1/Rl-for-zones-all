"""Section 6 sanity gate -- BLOCKING. Run this before anything else.

Hand-builds a 2-job, 1-task-each instance sharing one machine (so dispatch
order is a genuine choice), with duration/deadline numbers chosen so the
optimal dispatch order is computable on paper via "dispatch whichever job
has less slack first" (minimum-slack/least-laxity scheduling), then asserts
the exact DP reproduces it in both directions (an "A tight" instance and a
"B tight" instance with deadlines swapped).

Construction (both instances): job A duration=3 (deterministic), job B
duration=5 (deterministic), same machine m1, so total makespan is 3+5=8
regardless of order -- the *only* thing an order choice affects is which
job's own deadline is hit. slack_J = (D_J - g_J) - W_J, g_J=0 for both at
the root, W_A=3, W_B=5.

Instance 1 ("A tight"): D_A=3 (slack_A=0), D_B=8 (slack_B=3).
  A-first: A done at 3<=3 OK, B done at 8<=8 OK. No penalty. Cost=8 (makespan).
  B-first: B done at 5<=8 OK, A done at 8>3 -> A FAILS. Cost=8+P.
  slack_A(0) < slack_B(3) -> minimum-slack rule says dispatch A first.
  P is set far larger than the 8-unit makespan itself, so there is no
  numeric ambiguity: dispatching A first strictly dominates.

Instance 2 ("B tight"): deadlines swapped, D_A=8 (slack_A=5), D_B=5 (slack_B=0).
  By the same logic, minimum-slack says dispatch B first, and B-first is
  again the strictly dominant, unambiguous choice.

If the DP's argmin action at the root doesn't match this for *both*
instances, this script raises and nothing downstream should be trusted.
"""
from __future__ import annotations

from prob_jobshop_ext.instance import Job, ProbJobShopInstance, Task, TaskDistribution

from .dp import explore_and_solve
from .instance import DeadlineJobShopInstance


def _build_instance(name: str, d_a: float, d_b: float, penalty: float) -> DeadlineJobShopInstance:
    jobs = [
        Job(job_id="A", tasks=[Task("A_T1", "m1", TaskDistribution([3], [1.0]))]),
        Job(job_id="B", tasks=[Task("B_T1", "m1", TaskDistribution([5], [1.0]))]),
    ]
    base = ProbJobShopInstance(name=name, description="sanity-gate 2-job instance", machines=["m1"], jobs=jobs)
    return DeadlineJobShopInstance(
        name=name,
        description=base.description,
        machines=base.machines,
        jobs=base.jobs,
        deadlines={"A": d_a, "B": d_b},
        release={"A": 0.0, "B": 0.0},
        tightness=float("nan"),
        penalty=penalty,
        base_name=name,
    )


def run() -> dict:
    penalty = 3.0 * 8.0  # >> the 8-unit makespan difference; see module docstring

    inst_a_tight = _build_instance("SANITY_A_TIGHT", d_a=3.0, d_b=8.0, penalty=penalty)
    inst_b_tight = _build_instance("SANITY_B_TIGHT", d_a=8.0, d_b=5.0, penalty=penalty)

    report = {}
    for label, inst, expected_first in [
        ("A_tight", inst_a_tight, "A_T1"),
        ("B_tight", inst_b_tight, "B_T1"),
    ]:
        res = explore_and_solve(inst, state_limit=1000)
        root_actions = res.optimal_actions[res.root_key]
        root_v = res.V[res.root_key] + res.root_initial_penalty
        n_states = len(res.snapshots)

        ok = root_actions == [expected_first]
        report[label] = {
            "expected_first_dispatch": expected_first,
            "dp_optimal_first_dispatch": root_actions,
            "dp_root_value": root_v,
            "n_states_explored": n_states,
            "passed": ok,
        }
        if not ok:
            raise AssertionError(
                f"SANITY GATE FAILED ({label}): expected root-optimal action "
                f"{[expected_first]}, DP gave {root_actions}. STOPPING -- "
                f"do not trust any downstream numbers until this is fixed."
            )

    print("SANITY GATE PASSED for both A_tight and B_tight instances.")
    for label, r in report.items():
        print(f"  {label}: DP picks {r['dp_optimal_first_dispatch']} "
              f"(expected {r['expected_first_dispatch']}), "
              f"V(root)={r['dp_root_value']:.2f}, states={r['n_states_explored']}")
    return report


if __name__ == "__main__":
    run()
