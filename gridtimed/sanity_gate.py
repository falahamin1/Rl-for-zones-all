"""Measurement 1 -- THE PAPER GATE (blocking).

Two hand-picked, hand-verified concrete states at the SAME discrete
location (agent at the center cell (1,1), both switches already touched)
that differ only in their clock values, for which the optimal action
provably differs. If the exact DP disagrees with either hand computation,
this raises -- nothing downstream should be trusted.

Hand computation
-----------------
Both states are reached via the fixed 4-move path
``(0,0) --RIGHT--> (0,1)[s1:=0] --DOWN--> (1,1) --LEFT--> (1,0)[s2:=0]
--RIGHT--> (1,1)``, using the *same* delay at every one of the 4 moves
(all-0 for zone I, all-2 for zone J):

Zone I (all delays 0): arriving at (1,1), s1 = 3, s2 = 1, y = 4.
  - RIGHT (gate1 attempt), any delay in {0,1,2}: arrival s1 = 3 + delay + 1
    in {4,5,6}, all inside GATE1_WINDOW=[4,6] -- legal for every delay.
    Cheapest is delay=0: 1 more step to GATE1, then 1 more (DOWN, delay 0)
    to GOAL. Total y = 4 + 1 + 1 = 6 <= D=16. Cost = 2*TIME_COST - 100 = -98.
  - DOWN (gate2 attempt), any delay in {0,1,2}: arrival s2 = 1 + delay + 1
    in {2,3,4}, all OUTSIDE GATE2_WINDOW=[5,7] -- illegal for every delay
    (self-loop). Gate2 is unreachable directly from this state; reaching it
    would require revisiting switch2 (resetting s2, adding >= 2 extra
    moves) and is strictly more expensive than the direct gate1 path no
    matter the exact revisit route (more elapsed time, same +100 bonus).
  => optimal action: RIGHT (delay 0), i.e. "d0_RIGHT".

Zone J (all delays 2): arriving at (1,1), s1 = 9, s2 = 3, y = 12.
  - RIGHT (gate1 attempt): arrival s1 = 9 + delay + 1 in {10,11,12}, all
    OUTSIDE GATE1_WINDOW=[4,6] (already past it -- s1 only increases, so no
    amount of *this* action's delay can bring it back down) -- illegal for
    every delay. Gate1 is unreachable directly from this state.
  - DOWN (gate2 attempt): arrival s2 = 3 + delay + 1 in {4,5,6}. delay=0
    gives 4 (outside [5,7], illegal); delay=1 gives 5 (inside, legal);
    delay=2 gives 6 (inside, legal). Cheapest legal is delay=1: 2 steps to
    GATE2, then 1 more (RIGHT, delay 0) to GOAL. Total y = 12 + 2 + 1 = 15
    <= D=16. Cost = 3*TIME_COST - 100 = -97.
  => optimal action: DOWN (delay 1), i.e. "d1_DOWN".

The optimal action flips from RIGHT to DOWN purely because of which gate's
window the (s1, s2) pair currently satisfies -- an inversion driven by the
same location, different clocks.
"""
from __future__ import annotations

from .dp import explore_and_solve
from .env import GridState
from .instance import DEFAULT_DELAYS, GridInstance

ZONE_I = GridState(pos=(1, 1), clocks=(3, 1), y=4)
ZONE_J = GridState(pos=(1, 1), clocks=(9, 3), y=12)

EXPECTED_I = "d0_RIGHT"
EXPECTED_J = "d1_DOWN"

EXPECTED_V_I = -98.0
EXPECTED_V_J = -97.0


def run() -> dict:
    instance = GridInstance(allowed_delays=DEFAULT_DELAYS)
    result = explore_and_solve(instance)

    checks = []
    for label, state, expected_action, expected_v in [
        ("zone_I", ZONE_I, EXPECTED_I, EXPECTED_V_I),
        ("zone_J", ZONE_J, EXPECTED_J, EXPECTED_V_J),
    ]:
        assert state in result.transitions, f"{label} {state} is not even reachable -- design error"
        actual_actions = result.optimal_actions[state]
        actual_v = result.V[state]
        ok = expected_action in actual_actions and abs(actual_v - expected_v) < 1e-6
        checks.append(
            {
                "label": label,
                "state": str(state),
                "expected_action": expected_action,
                "actual_optimal_actions": actual_actions,
                "expected_V": expected_v,
                "actual_V": actual_v,
                "passed": ok,
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
    print("\nMeasurement 1 (paper gate): PASSED.")
    return {"checks": checks, "n_states": len(result.states), "truncated": result.truncated}


if __name__ == "__main__":
    run()
