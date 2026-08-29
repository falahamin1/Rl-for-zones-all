"""Hole-Disable TRM for the 3-goal Frozen Lake task.

Extension of the LB-TRM where holes are non-terminal.  Instead of ending
the episode, hitting a hole disables the intended direction for x_h steps,
tracked by a third clock h.

RM structure
------------
States: u0 (terminal), u1 (seek A), u2 (seek B), u3 (seek C)

Guard (goals): x >= c_min  AND  y <= D   (same as LB-TRM)
  x  — per-phase timer, resets to 0 at each successful goal event.
  y  — global elapsed time, never resets.
  h  — hole cooldown timer, resets to 0 when a hole is hit.
       disabled_action is blocked while h < x_h.

Zone graph
----------
Three independent binary clock nodes — 8 nodes total, constant for any
values of c_min, D, x_h:

  x_node : 0 if x < c_min  else 1
  y_node : 0 if y <= D      else 1
  h_node : 0 if h < x_h    else 1   (0 = action disabled, 1 = no restriction)

Region
------
(c_min+1) × (D+1) × (x_h+1) clock triples — grows linearly with each parameter.

Instances — fix c_min=10, D=200, vary x_h
------------------------------------------
  HOLE_01  x_h= 5  region triples: 11×201× 6 =  13,266  zone: 8
  HOLE_02  x_h=10  region triples: 11×201×11 =  24,321  zone: 8
  HOLE_03  x_h=20  region triples: 11×201×21 =  46,431  zone: 8
  HOLE_04  x_h=30  region triples: 11×201×31 =  68,541  zone: 8
  HOLE_05  x_h=50  region triples: 11×201×51 = 112,761  zone: 8
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

# ─── RM state constants ──────────────────────────────────────────────────────
U_TERMINAL = 0
U1         = 1   # seeking A
U2         = 2   # seeking B
U3         = 3   # seeking C

NONTERMINAL_STATES = (U1, U2, U3)
RM_GOAL_EVENTS = {U1: "a", U2: "b", U3: "c"}

# Rewards
R_STEP      = -1
R_HOLE_HIT  = -50    # non-terminal: bounce back + disable action
R_GOAL      = 100    # intermediate goal (A, B) with guard satisfied
R_FINAL     = 500    # final goal (C) with guard satisfied
R_LATE      = -200   # guard failed at a goal event (terminal)


@dataclass
class HoleTRMInstance:
    """A single benchmark instance of the hole-disable TRM."""
    name:  str
    c_min: int   # lower-bound guard on x   (x >= c_min to advance RM)
    D:     int   # upper-bound guard on y   (y <= D to advance RM)
    x_h:   int   # hole cooldown steps      (disabled action while h < x_h)

    @property
    def region_bound(self) -> int:
        return (self.c_min + 1) * (self.D + 1) * (self.x_h + 1)

    def guard_satisfied(self, x: int, y: int) -> bool:
        return x >= self.c_min and y <= self.D


INSTANCES: Dict[str, HoleTRMInstance] = {
    "HOLE_01": HoleTRMInstance(name="HOLE_01", c_min=10, D=200, x_h=5),
    "HOLE_02": HoleTRMInstance(name="HOLE_02", c_min=10, D=200, x_h=10),
    "HOLE_03": HoleTRMInstance(name="HOLE_03", c_min=10, D=200, x_h=20),
    "HOLE_04": HoleTRMInstance(name="HOLE_04", c_min=10, D=200, x_h=30),
    "HOLE_05": HoleTRMInstance(name="HOLE_05", c_min=10, D=200, x_h=50),
}


class HoleTRMState:
    """Mutable execution state for the hole-disable TRM."""

    def __init__(self):
        self.u:               int = U1
        self.x:               int = 0    # per-phase clock
        self.y:               int = 0    # global clock
        self.h:               int = 0    # hole cooldown clock (init below in reset)
        self.disabled_action: int = -1   # -1 = no restriction

    def reset(self, x_h: int):
        self.u = U1
        self.x = 0
        self.y = 0
        self.h = x_h   # start with cooldown already expired (no restriction)
        self.disabled_action = -1

    def step(self, action: int, event: str, hole_dir: int,
             trm: HoleTRMInstance) -> Tuple[int, bool]:
        """Process one environment step; return (reward, done).

        Parameters
        ----------
        action   : intended grid action (used only to advance clocks here).
        event    : checkpoint event "" | "a" | "b" | "c".
        hole_dir : intended direction if a hole was hit (= action), else -1.
        trm      : instance parameters.
        """
        assert self.u != U_TERMINAL

        # Advance all three clocks by 1 step.
        self.x += 1
        self.y += 1
        self.h += 1

        if hole_dir >= 0:
            # Hole hit: bounce back, reset cooldown, disable intended direction.
            self.h = 0
            self.disabled_action = hole_dir
            return R_HOLE_HIT, False

        # Update disabled_action status after clock advance.
        if self.h >= trm.x_h:
            self.disabled_action = -1   # cooldown expired

        goal_event = RM_GOAL_EVENTS.get(self.u)
        if event == goal_event:
            if trm.guard_satisfied(self.x, self.y):
                self.x = 0   # reset per-phase clock
                if self.u == U3:
                    self.u = U_TERMINAL
                    return R_FINAL, True
                else:
                    self.u += 1
                    return R_GOAL, False
            else:
                self.u = U_TERMINAL
                return R_LATE, True

        return R_STEP, False
