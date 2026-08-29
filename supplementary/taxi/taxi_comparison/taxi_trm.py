"""Timed Reward Machine for the Taxi task.

RM structure
------------
States: u0 (terminal), u1 (seek pickup), u2 (seek Green), u3 (seek dropoff)

Guard: x >= c_min  AND  y <= D
  x >= c_min : each phase must accumulate at least c_min time before the next
               event can fire  (x resets to 0 at each successful event)
  y <= D     : total elapsed time must not exceed the global deadline D

Transitions:
  u1 --[a, guard ok ]--> u2   +100  (x resets)
  u1 --[a, guard bad]--> u0   -200  (terminal)
  u1 --[else        ]--> u1   -1

  u2 --[b, guard ok ]--> u3   +100  (x resets)
  u2 --[b, guard bad]--> u0   -200
  u2 --[else        ]--> u2   -1

  u3 --[c, guard ok ]--> u0   +500  (terminal success)
  u3 --[c, guard bad]--> u0   -200
  u3 --[else        ]--> u3   -1

Clocks
------
  x   resets at each successful event — per-phase timer
  y   never resets                    — global timer

Zone graph vs Region graph
--------------------------
Same guard structure x >= c_min AND y <= D as the 3-goal frozen lake TRM.
Zone has exactly 4 clock-state nodes (constant for all instances):
  x-node: 0 if x < c_min  else 1
  y-node: 0 if y <= D     else 1

Region has (c_min+1)×(D+1) clock-state pairs, growing linearly with c_min.

Benchmark instances — fix D=200, d_max=5; vary c_min
----------------------------------------------------
  TAXI_01  c_min= 2  region clock pairs:  3×201=   603  zone: 4
  TAXI_02  c_min= 5  region clock pairs:  6×201= 1,206  zone: 4
  TAXI_03  c_min=10  region clock pairs: 11×201= 2,211  zone: 4
  TAXI_04  c_min=15  region clock pairs: 16×201= 3,216  zone: 4
  TAXI_05  c_min=20  region clock pairs: 21×201= 4,221  zone: 4
  TAXI_06  c_min=25  region clock pairs: 26×201= 5,226  zone: 4
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

# ─── RM state constants ──────────────────────────────────────────────────────
U_TERMINAL = 0
U1 = 1   # seeking "a" (pickup at Red)
U2 = 2   # seeking "b" (arrive at Green with passenger)
U3 = 3   # seeking "c" (dropoff at Blue)

NONTERMINAL_STATES = (U1, U2, U3)
RM_GOAL_EVENTS = {U1: "a", U2: "b", U3: "c"}

# Rewards
R_STEP  = -1
R_GOAL  = 100    # intermediate event with guard satisfied
R_FINAL = 500    # final event (dropoff) with guard satisfied
R_LATE  = -200   # guard failed at event


@dataclass
class TaxiTRMInstance:
    """A single benchmark instance of the Taxi TRM."""
    name:  str
    c_min: int   # lower-bound guard on x  (x >= c_min to advance RM)
    D:     int   # upper-bound guard on y  (y <= D to advance RM)
    d_max: int   # maximum action delay    (action space: {0,...,d_max} × 6)

    @property
    def n_rm_states(self) -> int:
        return 4

    def guard_satisfied(self, clocks: Dict[str, int]) -> bool:
        return clocks["x"] >= self.c_min and clocks["y"] <= self.D


# Standard benchmark suite — D=200, d_max=5 fixed; c_min varies
INSTANCES: Dict[str, TaxiTRMInstance] = {
    "TAXI_01": TaxiTRMInstance(name="TAXI_01", c_min=2,  D=200, d_max=5),
    "TAXI_02": TaxiTRMInstance(name="TAXI_02", c_min=5,  D=200, d_max=5),
    "TAXI_03": TaxiTRMInstance(name="TAXI_03", c_min=10, D=200, d_max=5),
    "TAXI_04": TaxiTRMInstance(name="TAXI_04", c_min=15, D=200, d_max=5),
    "TAXI_05": TaxiTRMInstance(name="TAXI_05", c_min=20, D=200, d_max=5),
    "TAXI_06": TaxiTRMInstance(name="TAXI_06", c_min=25, D=200, d_max=5),
}


class TaxiTRMState:
    """Mutable Taxi TRM execution state (rm_state + clock values)."""

    def __init__(self):
        self.u: int = U1
        self.clocks: Dict[str, int] = {"x": 0, "y": 0}

    def reset(self):
        self.u = U1
        self.clocks = {"x": 0, "y": 0}

    def advance_clocks(self, dt: int = 1):
        self.clocks["x"] += dt
        self.clocks["y"] += dt

    def step(self, event: str, trm: TaxiTRMInstance) -> Tuple[int, bool]:
        """Process one TRM step; return (reward, done).

        event : "" | "a" | "b" | "c"
        """
        u = self.u
        assert u != U_TERMINAL

        goal_event = RM_GOAL_EVENTS.get(u)

        if event == goal_event:
            if trm.guard_satisfied(self.clocks):
                self.clocks["x"] = 0
                if u == U3:
                    self.u = U_TERMINAL
                    return R_FINAL, True
                else:
                    self.u = u + 1
                    return R_GOAL, False
            else:
                self.u = U_TERMINAL
                return R_LATE, True

        return R_STEP, False
