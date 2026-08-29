"""Step 1 -- THE PAPER GATE (blocking).

M0: 2 jobs, 1 task each, sharing one machine. d_fast=3, d_slow=5,
cost_penalty (c_fast) = 3, deadline D=9 for both jobs (symmetric).

Hand computation
-----------------
Only one machine exists, so at most one of the two jobs' tasks can be
active at a time; the other must wait. At g=0 both dispatch choices (job,
mode) are legal for either job (each job's own best case, 0+5=5 or 0+3=3,
is well within D=9). Whichever job is dispatched first, the SECOND job's
dispatch decision is the interesting one -- it happens at a g that depends
on the first job's chosen mode, giving the same location two different
concrete g values to compare, hand-computably:

**Zone I -- first job dispatched FAST (arrives at g=3):** second job's
options:
  - SLOW: elapsed 5, cost 5*1 (time) + 0 (no fast penalty) = 5. Feasible
    (3+5=8 <= 9).
  - FAST: elapsed 3, cost 3 + 3 (c_fast) = 6. Feasible (3+3=6 <= 9).
  => optimal: SLOW, V = 5 (5 < 6).

**Zone J -- first job dispatched SLOW (arrives at g=5):** second job's
options:
  - SLOW: would complete at 5+5=10 > 9 -- INFEASIBLE, not even offered as
    a legal action (feasibility is checked on the dispatch edge, not
    after the fact).
  - FAST: elapsed 3, cost 3 + 3 = 6. Feasible (5+3=8 <= 9).
  => optimal: FAST (forced -- the only legal action), V = 6.

The optimal MODE flips from SLOW to FAST purely because of how much slack
remains at the second dispatch -- an inversion driven by the same
decision (which mode for the second job), different g.
"""
from __future__ import annotations

from .dp import explore_and_solve
from .env import initial_state, step_deterministic
from .instance import FAST, SLOW, Job, ModeSelInstance, Task

D_FAST = 3
D_SLOW = 5
COST_PENALTY = 3.0
DEADLINE = 9


def make_m0() -> ModeSelInstance:
    t_a = Task(task_id="J0_T0", job_id="J0", index=0, machine="M0", d_fast=D_FAST, d_slow=D_SLOW)
    t_b = Task(task_id="J1_T0", job_id="J1", index=0, machine="M0", d_fast=D_FAST, d_slow=D_SLOW)
    job_a = Job(job_id="J0", tasks=(t_a,), deadline=DEADLINE)
    job_b = Job(job_id="J1", tasks=(t_b,), deadline=DEADLINE)
    return ModeSelInstance(jobs=(job_a, job_b), cost_penalty=COST_PENALTY, w=0.0, label="M0")


EXPECTED_I = ("J1_T0_SLOW", 5.0)
EXPECTED_J = ("J1_T0_FAST", 6.0)


def run() -> dict:
    instance = make_m0()
    result = explore_and_solve(instance)

    s0 = initial_state(instance)
    zone_i_parent = step_deterministic(instance, s0, "J0_T0", FAST).next_state  # g=3
    zone_j_parent = step_deterministic(instance, s0, "J0_T0", SLOW).next_state  # g=5

    checks = []
    for label, state, (expected_action, expected_v) in [
        ("zone_I (g=3)", zone_i_parent, EXPECTED_I),
        ("zone_J (g=5)", zone_j_parent, EXPECTED_J),
    ]:
        assert state in result.transitions, f"{label} {state} is not even reachable -- design error"
        actual_actions = result.optimal_actions[state]
        actual_v = result.V[state]
        ok = expected_action in actual_actions and abs(actual_v - expected_v) < 1e-6
        checks.append(
            {
                "label": label, "state": str(state),
                "expected_action": expected_action, "actual_optimal_actions": actual_actions,
                "expected_V": expected_v, "actual_V": actual_v, "passed": ok,
            }
        )
        print(f"{label}: expected {expected_action} (V={expected_v}), got {actual_actions} (V={actual_v}) "
              f"-- {'PASS' if ok else 'FAIL'}")

    all_passed = all(c["passed"] for c in checks)
    if not all_passed:
        raise AssertionError(
            "sanity_gate FAILED -- hand computation and exact DP disagree; "
            "STOP, everything downstream would be meaningless. See printed checks."
        )
    print("\nStep 1 (paper gate): PASSED.")
    return {"checks": checks, "n_states": len(result.states), "truncated": result.truncated}


if __name__ == "__main__":
    run()
