"""Single-Goal TRM for the paper example.

Environment
-----------
6×6 Frozen Lake with one goal C at (5,5). The agent must reach C during a
time window: clock y (elapsed steps from start) must satisfy c_min ≤ y ≤ D.

Clock
-----
  y   global elapsed time, never resets.

Guard
-----
  c_min ≤ y ≤ D

  c_min : goal only becomes collectible after c_min steps (lower bound).
  D     : global deadline — goal expires after D total steps (upper bound).

Zone graph (precomputed from TRM structure)
-------------------------------------------
For guard c_min ≤ y ≤ D, the zone graph has exactly 3 clock nodes:

  y_node = 0  if y < c_min   (pre-window: too early to collect goal)
           1  if c_min ≤ y ≤ D  (in window: guard satisfied)
           2  if y > D           (post-window: deadline exceeded)

These 3 nodes are FIXED regardless of the values of c_min and D — the structure
depends only on the number of thresholds (2 here). Region tracks the exact
integer y value capped at D, giving D+1 distinct states.

State-space comparison
----------------------
  Zone:   3 clock nodes          — constant for all instances
  Region: D+1 clock nodes        — grows linearly with D

Instances (fix c_min=10, vary D)
---------------------------------
  SG_01  D= 30   region nodes:  31  (ratio: 10.3)
  SG_02  D= 60   region nodes:  61  (ratio: 20.3)
  SG_03  D=100   region nodes: 101  (ratio: 33.7)
  SG_04  D=200   region nodes: 201  (ratio: 67.0)
  SG_05  D=400   region nodes: 401  (ratio: 133.7)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

# ─── RM state constants ──────────────────────────────────────────────────────
SG_U_TERMINAL = 0
SG_U1         = 1   # seeking goal C

# Rewards
R_STEP  = -1
R_GOAL  = 500    # guard satisfied → success
R_LATE  = -200   # guard failed   → episode ends
R_HOLE  = -500


@dataclass
class SGTRMInstance:
    """A single benchmark instance of the single-goal TRM."""
    name:  str
    c_min: int   # lower-bound guard on y (goal opens after c_min steps)
    D:     int   # upper-bound guard on y (deadline)

    def guard_satisfied(self, y: int) -> bool:
        return self.c_min <= y <= self.D


# Standard instances — c_min=10 fixed, D varies
INSTANCES = {
    "SG_01": SGTRMInstance(name="SG_01", c_min=10, D=30),
    "SG_02": SGTRMInstance(name="SG_02", c_min=10, D=60),
    "SG_03": SGTRMInstance(name="SG_03", c_min=10, D=100),
    "SG_04": SGTRMInstance(name="SG_04", c_min=10, D=200),
    "SG_05": SGTRMInstance(name="SG_05", c_min=10, D=400),
}


class SGTRMState:
    """Mutable single-goal TRM execution state."""

    def __init__(self):
        self.u: int = SG_U1
        self.y: int = 0

    def reset(self):
        self.u = SG_U1
        self.y = 0

    def step(self, event: str, trm: SGTRMInstance) -> Tuple[int, bool]:
        """Process one TRM step; return (reward, done)."""
        assert self.u != SG_U_TERMINAL

        if event == "h":
            self.u = SG_U_TERMINAL
            return R_HOLE, True

        if event == "c":
            if trm.guard_satisfied(self.y):
                self.u = SG_U_TERMINAL
                return R_GOAL, True
            else:
                self.u = SG_U_TERMINAL
                return R_LATE, True

        return R_STEP, False
