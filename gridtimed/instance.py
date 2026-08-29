"""The grid instance: parameterized (grid size, switch/gate count and
placement, gate windows, deadline, delay set, duration-jitter width), with
the original 3x3/2-switch/2-gate hand-built instance as the default.

Layout (default, T0)::

    (0,0)START (0,1)SWITCH1 (0,2)
    (1,0)SWITCH2 (1,1)      (1,2)GATE1
    (2,0)        (2,1)GATE2 (2,2)GOAL

Clocks: ``y`` (global, never resets), ``s1..sK`` (one per switch, reset to 0
on entering that switch; before its first activation a switch clock
advances from 0 like any other clock -- i.e. all clocks start identical at
episode time 0 and only diverge once a switch is first touched), plus one
internal bookkeeping clock ``e`` (see zone_ops.py).

Gate ``i`` is associated with switch ``i`` (same index): passable only
while ``s{i+1}`` lies in ``gate_windows[i]``.

SOUNDNESS NOTE (see /tmp/GRIDSOUND.md Task A): earlier versions of this
package built zones by widening the *agent's own delay choice* into a
range -- since delay is a decision variable the agent knows it picked, not
an unobserved random quantity, this made a single zone contain concrete
points reachable only via genuinely different decisions, with different
optimal actions (measured: 40.65% of multi-point zones were internally
non-uniform). Fixed by making the delay component of elapsed time exact
(``delay + 1``, not widened) and introducing ``w`` -- a real, physically
unobserved duration-jitter width -- as the ONLY source of zone geometry: a
move's true duration is drawn uniform on ``[1, 1+w]`` and added on top of
the exact chosen delay. See zone_ops.py / zones.py.

The action set is fixed-size: ``len(allowed_delays) x 4 moves``. Movement is
deterministic in outcome (no slip); illegal moves (walls, or a gate entered
outside its window) are still valid to select -- they self-loop, cost a
small penalty, and advance no clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

Position = Tuple[int, int]

GRID_SIZE = 3
START: Position = (0, 0)
GOAL: Position = (2, 2)
SWITCH1: Position = (0, 1)
SWITCH2: Position = (1, 0)
GATE1: Position = (1, 2)
GATE2: Position = (2, 1)

# Gate windows: gate i passable iff s_i in [a_i, b_i] (both inclusive) at
# the instant of entering it. Chosen so that a genuine, hand-verifiable
# inversion exists at the (1,1)-both-switches-touched location -- see
# sanity_gate.py and /tmp/GRIDGATE.md for the full derivation.
GATE1_WINDOW = (4, 6)  # on s1
GATE2_WINDOW = (5, 7)  # on s2

DEADLINE = 16  # D: y <= D required at the instant the agent reaches GOAL

REWARD_SUCCESS = 100.0
TIME_COST = 1.0          # per unit of elapsed y on any legal (moving) action
ILLEGAL_PENALTY = 2.0    # cost of taking an illegal action (wall / closed gate); no time elapses
DEADLINE_FAIL_PENALTY = 50.0  # terminal cost when y > DEADLINE before reaching GOAL

MOVES = ("UP", "DOWN", "LEFT", "RIGHT")
DELTA: Dict[str, Tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

DEFAULT_DELAYS: Tuple[int, ...] = (0, 1, 2)
RESTRICTED_DELAYS: Tuple[int, ...] = (0,)

#: pyudbm requires exact-integer bounds; w and delay+1+w are real. Every
#: pyudbm-facing bound (gate windows, deadline, edge invariant/guard) is
#: scaled by SCALE and rounded; the concrete simulator (env.py/dp.py) stays
#: in real/exact units throughout, matching gate_track's convention.
SCALE = 20

#: w values swept in Task A's soundness re-measurement. Uniform(1, 1+w) is
#: the true per-move duration; the concrete DP still transitions
#: deterministically on delay+1 (the interval's minimum, not its mean --
#: flagged explicitly in /tmp/GRIDSOUND.md since it does NOT satisfy the
#: "nominal = expectation of a symmetric interval" justification used
#: elsewhere in this repo; Uniform(1,1+w) has mean 1+w/2, not 1).
W_VALUES: Tuple[float, ...] = (0.0, 0.25, 0.5)

# Uniform clock bound (in REAL, unscaled units) used for both k-normalization
# and clip_to_k_box. Multiplied by SCALE wherever it meets pyudbm.
K_BOUND = 20


@dataclass(frozen=True)
class GridInstance:
    grid_n: int = GRID_SIZE
    start: Position = START
    goal: Position = GOAL
    switches: Tuple[Position, ...] = (SWITCH1, SWITCH2)
    gates: Tuple[Position, ...] = (GATE1, GATE2)
    gate_windows: Tuple[Tuple[int, int], ...] = (GATE1_WINDOW, GATE2_WINDOW)
    deadline: int = DEADLINE
    allowed_delays: Tuple[int, ...] = DEFAULT_DELAYS
    w: float = 0.0
    seed: int = 0  # provenance only (which generator seed produced this instance, if any)
    label: str = "T0"

    def __post_init__(self):
        assert len(self.gates) == len(self.switches) == len(self.gate_windows), \
            "gates, switches, and gate_windows must be the same length -- gate i is paired with switch i"

    @property
    def n_switches(self) -> int:
        return len(self.switches)

    @property
    def clock_names(self) -> Tuple[str, ...]:
        return tuple(f"s{i + 1}" for i in range(self.n_switches))

    @property
    def min_delay(self) -> int:
        return min(self.allowed_delays)

    @property
    def max_delay(self) -> int:
        return max(self.allowed_delays)

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.grid_n and 0 <= c < self.grid_n

    def is_gate(self, pos: Position) -> bool:
        return pos in self.gates

    def is_switch(self, pos: Position) -> bool:
        return pos in self.switches

    def switch_index(self, pos: Position) -> int:
        return self.switches.index(pos)

    def gate_index(self, pos: Position) -> int:
        return self.gates.index(pos)

    def gate_window(self, pos: Position) -> Tuple[int, int]:
        return self.gate_windows[self.gate_index(pos)]

    def switch_clock(self, pos: Position) -> str:
        return f"s{self.switch_index(pos) + 1}"

    def gate_clock(self, pos: Position) -> str:
        return f"s{self.gate_index(pos) + 1}"

    def action_id(self, delay: int, move: str) -> str:
        return f"d{delay}_{move}"


def action_id(delay: int, move: str) -> str:
    return f"d{delay}_{move}"
