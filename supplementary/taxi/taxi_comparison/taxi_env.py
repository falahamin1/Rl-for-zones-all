"""5x5 Taxi grid environment for zone vs region comparison.

Layout (row 0 = top, col 0 = left):
    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+

R = Red    (0,0) — passenger pickup location
G = Green  (0,4) — required waypoint (agent must visit while carrying passenger)
Y = Yellow (4,0) — unused
B = Blue   (4,3) — passenger dropoff destination

Starting state: taxi at (2,2), passenger at Red (not yet picked up).

Events
------
  "a" : pickup action at Red (0,0) when passenger is there
  "b" : taxi arrives at Green (0,4) while carrying passenger (position-triggered)
  "c" : dropoff action at Blue (4,3) while carrying passenger

Actions
-------
  S=0, N=1, E=2, W=3  — movement (stochastic with wall constraints)
  PICKUP=4, DROPOFF=5 — deterministic, taxi stays in place

Stochastic slip (is_rainy): 80% intended direction, 10% each perpendicular.
Walls parsed from the original MAP; prevent movement between specific cells.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

ROWS: int = 5
COLS: int = 5

RED    = (0, 0)
GREEN  = (0, 4)
YELLOW = (4, 0)
BLUE   = (4, 3)

TAXI_START = (2, 2)

# Actions
S, N, E, W, PICKUP, DROPOFF = 0, 1, 2, 3, 4, 5
ACTIONS = [S, N, E, W, PICKUP, DROPOFF]
MOVE_ACTIONS = [S, N, E, W]

# East-blocked positions: can't move east from (row, col) → (row, col+1)
# Parsed from MAP: "|" character at desc[1+row, 2*col+2] blocks east move
BLOCKED_EAST = frozenset({
    (0, 1), (1, 1),   # wall between col1–col2 in rows 0–1
    (3, 0), (4, 0),   # wall between col0–col1 in rows 3–4
    (3, 2), (4, 2),   # wall between col2–col3 in rows 3–4
})

# West-blocked positions: mirror of east blocks
BLOCKED_WEST = frozenset({
    (0, 2), (1, 2),
    (3, 1), (4, 1),
    (3, 3), (4, 3),
})


def _move(pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    """Apply action to position, respecting walls and boundaries."""
    r, c = pos
    if action == S:
        return (min(ROWS - 1, r + 1), c)
    elif action == N:
        return (max(0, r - 1), c)
    elif action == E:
        if pos in BLOCKED_EAST:
            return pos
        return (r, min(COLS - 1, c + 1))
    elif action == W:
        if pos in BLOCKED_WEST:
            return pos
        return (r, max(0, c - 1))
    return pos  # PICKUP / DROPOFF: taxi does not move


class TaxiGridEnv:
    """5x5 Taxi grid with stochastic movement and wall constraints.

    Parameters
    ----------
    slip_prob   Probability of sliding to a perpendicular direction.
    seed        RNG seed.
    """

    def __init__(self, slip_prob: float = 0.2, seed: Optional[int] = None):
        self.slip_prob = slip_prob
        self._rng = np.random.default_rng(seed)
        self._pos: Tuple[int, int] = TAXI_START
        self._pass_in_taxi: bool = False

    def reset(self) -> Tuple[Tuple[int, int], bool]:
        self._pos = TAXI_START
        self._pass_in_taxi = False
        return self._pos, self._pass_in_taxi

    def step(self, action: int) -> Tuple[Tuple[int, int], bool, str, bool]:
        """Execute action; return (pos, pass_in_taxi, event, done).

        event : "a" | "b" | "c" | ""
        done  : True only when event == "c" (successful dropoff).
        """
        if action in MOVE_ACTIONS:
            actual = self._sample_action(action)
            self._pos = _move(self._pos, actual)

        event = ""
        if action == PICKUP:
            if self._pos == RED and not self._pass_in_taxi:
                event = "a"
                self._pass_in_taxi = True
        elif action == DROPOFF:
            if self._pos == BLUE and self._pass_in_taxi:
                event = "c"
                self._pass_in_taxi = False
        elif action in MOVE_ACTIONS and self._pass_in_taxi and self._pos == GREEN:
            event = "b"

        done = (event == "c")
        return self._pos, self._pass_in_taxi, event, done

    def pos(self) -> Tuple[int, int]:
        return self._pos

    def pass_in_taxi(self) -> bool:
        return self._pass_in_taxi

    def _sample_action(self, intended: int) -> int:
        if self.slip_prob <= 0.0 or self._rng.random() >= self.slip_prob:
            return intended
        perp = [E, W] if intended in (N, S) else [N, S]
        return int(self._rng.choice(perp))
