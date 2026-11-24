"""The TRM grid instance -- v2: a BRANCHING reward machine over k sub-goals
plus a GOAL cell (v1's fixed A->B->C chain is retired; see
/tmp/TRMGRID_V2.md's root-cause analysis -- a fixed chain makes RM phase a
function of "how many sub-goals visited", which combined with location
nearly determines the optimal action on its own).

Task: visit any TWO distinct sub-goals (agent's choice of WHICH two and in
WHAT order), then reach GOAL. Each ORDERED pair (i, j) carries its own
two-sided window: visiting j while "have i" is accepted only if t_i in
[lo_ij, hi_ij] -- arriving too soon is now exactly as bad as too late,
unlike v1's chain which only ever had a lower OR an upper bound active at
once. From "have i,j", reaching GOAL is accepted only if t_j <= cap_j
(cap indexed by the SECOND sub-goal visited, not by the pair).

RM phase representation: a tuple of visited sub-goal indices, in visit
order -- () = start (nothing visited), (i,) = "have_i", (i, j) = "have_i,j"
(i visited first, then j). Terminal states use pos in ("DONE","FAILED")
exactly as v1 did; phase is irrelevant once terminal.

Clocks: ``t_0 .. t_{k-1}`` (one per sub-goal, resets to 0 on PHYSICAL
arrival at that sub-goal's cell, unconditionally -- same convention as
v1), plus ``g`` (global, never resets, ``g <= D`` throughout), plus one
internal zone-only bookkeeping clock ``e`` (see zone_ops.py, unchanged
from v1).

Guard violations (arriving at a sub-goal outside its window, or at GOAL
outside its cap) are TERMINAL FAILURES with a penalty, checked ON THE
EDGE -- unchanged from v1, not a blocked/illegal move.

Parameter calibration is distance-relative (see generator.py), not a
fixed instance.py default sized for one grid -- v1's own watch-list
lesson (a clip/window bound derived from one instance's parameters,
reused globally, silently emptying zones on a differently-scaled
instance) applies just as much to t_min/t_max/D as to any pyudbm clip
bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

Position = Tuple[int, int]
Phase = Tuple[int, ...]  # () | (i,) | (i, j)

MOVES = ("UP", "DOWN", "LEFT", "RIGHT")
DELTA: Dict[str, Tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

REWARD_SUCCESS = 100.0
TIME_COST = 1.0           # per unit of elapsed g on any legal (moving) action
ILLEGAL_PENALTY = 2.0     # cost of an out-of-bounds move; no time elapses
FAIL_PENALTY = 50.0       # terminal cost: guard violation OR deadline expiry

DEFAULT_DELAYS: Tuple[int, ...] = (0, 1, 2)

#: pyudbm requires exact-integer bounds; w and delay+1+w are real. Every
#: pyudbm-facing bound is scaled by SCALE and rounded; the concrete
#: simulator (env.py/dp.py) stays in integer units throughout.
SCALE = 20

#: w values worth sweeping -- w=0 is Measurement 2's control.
W_VALUES: Tuple[float, ...] = (0.0, 0.25, 0.5)

#: Floor for k-normalization / clip_to_k_box -- a DEFAULT floor only; the
#: measurement code must still take max(K_BOUND, instance.deadline +
#: margin), per instance, never this constant alone (watch-list item).
K_BOUND = 20


@dataclass(frozen=True)
class TRMGridInstance:
    grid_n: int
    start: Position
    sub_goals: Tuple[Position, ...]          # length k, distinct, != start, != goal
    goal_pos: Position
    windows: Dict[Tuple[int, int], Tuple[int, int]]  # ordered (i,j) -> (lo_ij, hi_ij), i != j
    caps: Tuple[int, ...]                    # length k; caps[j] used when j is the SECOND sub-goal visited
    deadline: int
    allowed_delays: Tuple[int, ...] = DEFAULT_DELAYS
    w: float = 0.0
    seed: int = 0
    label: str = "v2"

    def __post_init__(self):
        k = len(self.sub_goals)
        cells = {self.start, self.goal_pos, *self.sub_goals}
        assert len(cells) == 2 + k, "start, goal, and every sub-goal must be distinct cells"
        assert len(self.caps) == k, f"caps must have one entry per sub-goal (k={k}), got {len(self.caps)}"
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                assert (i, j) in self.windows, f"missing window for ordered pair ({i},{j})"
                lo, hi = self.windows[(i, j)]
                assert 0 <= lo <= hi, f"window[{i},{j}] = ({lo},{hi}) must have 0 <= lo <= hi"
        assert self.deadline >= 0

    @property
    def k(self) -> int:
        return len(self.sub_goals)

    @property
    def clock_names(self) -> Tuple[str, ...]:
        return tuple(f"t_{i}" for i in range(self.k))

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.grid_n and 0 <= c < self.grid_n

    def sub_goal_index(self, pos: Position) -> int:
        return self.sub_goals.index(pos)

    def is_sub_goal(self, pos: Position) -> bool:
        return pos in self.sub_goals

    def action_id(self, delay: int, move: str) -> str:
        return f"d{delay}_{move}"


def action_id(delay: int, move: str) -> str:
    return f"d{delay}_{move}"


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
