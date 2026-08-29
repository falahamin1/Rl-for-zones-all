"""6×6 Frozen Lake grid with a single goal C — paper example variant.

Layout (row 0 = top, col 0 = left):

    S  .  .  .  .  .
    .  .  .  .  .  .
    .  H  .  .  H  .
    .  .  .  .  .  .
    .  .  H  .  .  .
    .  H  .  .  .  C

S = start  (0,0)
C = goal   (5,5)
H = hole   (2,1),(2,4),(4,2),(5,1)
. = frozen (safe)

The agent starts at S and must reach C while satisfying the time-window guard
c_min ≤ y ≤ D on the elapsed clock y.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

# Grid dimensions
ROWS: int = 6
COLS: int = 6

# Special positions
START: Tuple[int, int] = (0, 0)
GOAL_C: Tuple[int, int] = (5, 5)

HOLES = frozenset({(2, 1), (2, 4), (4, 2), (5, 1)})
CHECKPOINTS = {GOAL_C: "c"}

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class SGGridEnv:
    """6×6 stochastic frozen lake with a single goal C.

    Parameters
    ----------
    slip_prob   Probability of sliding 90°.
    seed        RNG seed.
    """

    def __init__(self, slip_prob: float = 0.2, seed: Optional[int] = None):
        self.slip_prob = slip_prob
        self._rng = np.random.default_rng(seed)
        self._pos: Tuple[int, int] = START

    def reset(self) -> Tuple[int, int]:
        self._pos = START
        return self._pos

    def step(self, action: int) -> Tuple[Tuple[int, int], str, bool]:
        """Execute action; return (new_pos, event, done).

        event : "c" | "h" | ""
        done  : True if hole or goal C reached.
        """
        actual = self._sample_action(action)
        dr, dc = _DELTAS[actual]
        r, c = self._pos
        nr, nc = max(0, min(ROWS - 1, r + dr)), max(0, min(COLS - 1, c + dc))
        self._pos = (nr, nc)

        if (nr, nc) in HOLES:
            return self._pos, "h", True
        event = CHECKPOINTS.get((nr, nc), "")
        done = event == "c"
        return self._pos, event, done

    def pos(self) -> Tuple[int, int]:
        return self._pos

    def action_space(self) -> List[int]:
        return ACTIONS

    def _sample_action(self, intended: int) -> int:
        if self.slip_prob <= 0.0 or self._rng.random() >= self.slip_prob:
            return intended
        perp = [LEFT, RIGHT] if intended in (UP, DOWN) else [UP, DOWN]
        return int(self._rng.choice(perp))
