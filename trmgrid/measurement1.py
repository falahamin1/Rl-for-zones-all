"""Measurement 1 -- THE PAPER GATE (blocking), v2.

A delay inversion alone (v1's gate) is not enough this time -- v1 had
those and they were too rare to matter (Measurement 3 showed location-
only soundness of 94.29%). This gate requires a genuine PAIR-CHOICE
inversion: a state where the optimal NEXT SUB-GOAL differs depending on
clock values, driven by the branching design's key new mechanism (a
two-sided window, where arriving too soon is exactly as bad as too late).

Demonstration instance: reduced scale from the swept "best setting"
(alpha=0.25, width=8, 5x5, k=3) for hand-tractability -- same alpha and
the same distance-relative calibration FORMULA
(lo_ij = ceil(d_ij*(1+alpha)), hi_ij = lo_ij+width), just a smaller
absolute width=3 so the reachable state space stays small enough to
solve EXACTLY (verified: not truncated) and hand-verify. The mechanism
being demonstrated (a two-sided window inverting the optimal choice) is
identical regardless of width; width=8 vs width=3 changes how MUCH slack
there is, not whether the inversion exists.

Instance: 5x5 grid. START=(0,0), S0=(1,1), S1=(1,3), S2=(3,1),
GOAL=(3,3). d(S0,S1)=d(S0,S2)=2 -> window(0,1)=window(0,2)=[3,6].
d(S1,GOAL)=d(S2,GOAL)=2 -> cap_1=cap_2=6. D=18.

Hand computation
-----------------
Both states are at the SAME location (1,2) -- one step (RIGHT) from
S1=(1,3) -- in RM phase (0,) ["have S0"], differing only in t_0 (time
since S0 was touched), reached via different delay profiles on paths that
never touch S1/S2 (so no incidental reset muddies the comparison).

**State I** (t_0=5, g=7): reached via
``START --RIGHT,DOWN (both d0)--> S0=(1,1)[t_0:=0] --UP,RIGHT (both d0)-->
(0,2) --DOWN (d2)--> (1,2)``. t_0 since reset = 1+1+3 = 5; g = 2+1+1+3 = 7.
  - Enter S1 now (RIGHT, d0): elapsed=1, arrival t_0 = 5+1 = 6. Check
    against window(0,1)=[3,6]: 6 <= 6, SATISFIED (exactly at the upper
    boundary). Transition succeeds: phase -> (0,1), t_1 := 0. Cost = 1.
  - S1 -> GOAL (DOWN,DOWN, d0 each, distance 2): elapsed=2, arrival
    t_1 = 2 <= cap_1=6, fine. SUCCESS. Cost = 2 - 100 = -98.
  - V(state I) = 1 + (-98) = **-97**. Waiting instead of entering now
    would only add elapsed for no benefit (S1 is already safely
    reachable) -- entering immediately is optimal.
  => **optimal action: d0_RIGHT (enter S1 now); pairs with S1.**

**State J** (t_0=6, g=8): reached via the SAME 3-move shape, one tick
slower throughout: ``START --RIGHT,DOWN (d0)--> S0[t_0:=0]
--UP,RIGHT,DOWN (all d1)--> (1,2)``. t_0 since reset = 2+2+2 = 6;
g = 2+2+2+2 = 8.
  - Enter S1 now (RIGHT, ANY delay): MINIMUM elapsed is 1 (d0), giving
    arrival t_0 = 6+1 = 7 > 6 = hi(0,1) already -- exceeded even at the
    fastest possible entry, and more delay only makes arrival LATER.
    **Every legal delay choice on this move is a guard violation
    (terminal failure, cost >= 1+50=51).**
  - Direct approach to S2 instead (distance 3 from (1,2)) fares no
    better: minimum elapsed=3, arrival t_0 = 6+3 = 9 > 6 = hi(0,2) --
    ALSO a guaranteed violation. Neither sub-goal is directly reachable
    from here at t_0=6.
  - The only way out is to retreat and reset: LEFT back to S0=(1,1)
    (d0, elapsed=1) -- since phase is already (0,) (S0 was visited
    first), re-touching S0 is an "incidental" touch (see env.py): no
    guard, phase unchanged, but t_0 resets to 0 (unconditional
    physical-arrival reset). New t_0=0, g=9.
  - From the reset S0, go to S2 (distance 2): DOWN (d0, elapsed=1) ->
    (2,1) [t_0=1, g=10], then DOWN (d1, elapsed=2) -> S2=(3,1): arrival
    t_0 = 1+2 = 3, check against window(0,2)=[3,6]: 3 >= 3, SATISFIED
    (exactly at the LOWER boundary this time). Phase -> (0,2), t_2:=0.
    Cost of this move = 2.
  - S2 -> GOAL (RIGHT,RIGHT, d0 each, distance 2): elapsed=2, arrival
    t_2=2 <= cap_2=6, fine. SUCCESS. Cost = 2-100 = -98.
  - V(state J) = 1 (retreat) + 1 (S0->(2,1)) + 2 (->S2) + (-98) (S2->GOAL)
    = **-94**.
  => **optimal action: d0_LEFT (retreat to S0, resetting t_0); pairs
     with S2, via a genuine backtrack the direct-approach comparison
     above rules out any cheaper alternative for.**

The optimal SUB-GOAL PAIRING flips from S1 to S2 -- and the optimal
IMMEDIATE ACTION flips from "enter now" to "retreat" -- purely because
t_0 crossed hi(0,1) by exactly one unit. This is the two-sided window's
upper bound inverting behavior, the mechanism v1 (a lower-bound-only
chain) could not produce.
"""
from __future__ import annotations

import math

from .dp import explore_and_solve
from .env import TRMState
from .instance import TRMGridInstance, manhattan

START = (0, 0)
S0 = (1, 1)
S1 = (1, 3)
S2 = (3, 1)
GOAL = (3, 3)
GRID_N = 5
ALPHA = 0.25
WIDTH = 3
DEADLINE = 18


def _window(a, b):
    d = manhattan(a, b)
    lo = math.ceil(d * (1 + ALPHA))
    return lo, lo + WIDTH


def _cap(a, goal):
    d = manhattan(a, goal)
    return math.ceil(d * (1 + ALPHA)) + WIDTH


def build_instance() -> TRMGridInstance:
    windows = {
        (0, 1): _window(S0, S1), (0, 2): _window(S0, S2),
        (1, 0): _window(S1, S0), (1, 2): _window(S1, S2),
        (2, 0): _window(S2, S0), (2, 1): _window(S2, S1),
    }
    caps = (_cap(S0, GOAL), _cap(S1, GOAL), _cap(S2, GOAL))
    return TRMGridInstance(
        grid_n=GRID_N, start=START, sub_goals=(S0, S1, S2), goal_pos=GOAL,
        windows=windows, caps=caps, deadline=DEADLINE, w=0.25, label="m1_pair_choice",
    )


STATE_I = TRMState(phase=(0,), pos=(1, 2), clocks=(5, 7, 7), g=7)
STATE_J = TRMState(phase=(0,), pos=(1, 2), clocks=(6, 8, 8), g=8)

EXPECTED_I = "d0_RIGHT"
EXPECTED_J = "d0_LEFT"
EXPECTED_V_I = -97.0
EXPECTED_V_J = -94.0


def run() -> dict:
    instance = build_instance()
    result = explore_and_solve(instance, state_limit=1_000_000)
    assert not result.truncated, "demonstration instance truncated -- not exact, gate would be meaningless"

    checks = []
    for label, state, expected_action, expected_v in [
        ("state_I", STATE_I, EXPECTED_I, EXPECTED_V_I),
        ("state_J", STATE_J, EXPECTED_J, EXPECTED_V_J),
    ]:
        assert state in result.transitions, f"{label} {state} is not even reachable -- design error"
        actual_actions = result.optimal_actions[state]
        actual_v = result.V[state]
        ok = expected_action in actual_actions and abs(actual_v - expected_v) < 1e-6
        checks.append({
            "label": label, "state": str(state),
            "expected_action": expected_action, "actual_optimal_actions": actual_actions,
            "expected_V": expected_v, "actual_V": actual_v, "passed": ok,
        })
        print(f"{label}: expected {expected_action} (V={expected_v}), got {actual_actions} (V={actual_v}) "
              f"-- {'PASS' if ok else 'FAIL'}")

    # Also confirm entering S1 from state_J is indeed a forced guard violation, not just suboptimal.
    right_from_j = result.transitions[STATE_J]["d0_RIGHT"]
    guard_confirmed = right_from_j.newly_failed and right_from_j.fail_reason == "guard_pair"
    print(f"state_J's d0_RIGHT (enter S1) forced failure confirmed: {guard_confirmed} "
          f"(newly_failed={right_from_j.newly_failed}, fail_reason={right_from_j.fail_reason!r})")

    all_passed = all(c["passed"] for c in checks) and guard_confirmed
    if not all_passed:
        raise AssertionError(
            "Measurement 1 (paper gate) FAILED -- hand computation and exact DP disagree; "
            "STOP, everything downstream would be meaningless. See printed checks."
        )
    print("\nMeasurement 1 (paper gate, PAIR-CHOICE inversion): PASSED.")
    return {"checks": checks, "n_states": len(result.states), "truncated": result.truncated,
            "root_V": result.V[result.root], "guard_confirmed": guard_confirmed}


if __name__ == "__main__":
    run()
