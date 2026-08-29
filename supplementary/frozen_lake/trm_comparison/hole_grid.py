"""6×6 Frozen Lake with non-terminal holes for the hole-disable TRM example.

Same layout as grid.py but holes do NOT end the episode.  When the agent
tries to step into a hole cell it bounces back (stays at its current position)
and the intended direction is returned so the caller can record it as the
disabled action.

Layout (row 0 = top, col 0 = left):

    S  .  .  .  .  .
    .  .  .  .  .  .
    .  H  .  .  H  .
    .  .  .  A  .  .
    .  .  H  .  .  .
    B  H  .  .  .  C

S = start (0,0)   A = checkpoint A (3,3)
B = checkpoint B (5,0)   C = final goal (5,5)
H = hole (bounce back — non-terminal)
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

ROWS: int = 6
COLS: int = 6

START: Tuple[int, int] = (0, 0)
GOAL_A: Tuple[int, int] = (3, 3)
GOAL_B: Tuple[int, int] = (5, 0)
GOAL_C: Tuple[int, int] = (5, 5)

HOLES = frozenset({(2, 1), (2, 4), (4, 2), (5, 1)})
CHECKPOINTS = {GOAL_A: "a", GOAL_B: "b", GOAL_C: "c"}

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class HoleGridEnv:
    """6×6 stochastic Frozen Lake where holes are non-terminal.

    When the agent would enter a hole it stays in place.  The intended
    direction that caused the bounce is returned so the TRM layer can
    disable it for x_h steps.

    Parameters
    ----------
    slip_prob   Probability of sliding 90° to one perpendicular direction.
    seed        RNG seed.
    """

    def __init__(self, slip_prob: float = 0.2, seed: Optional[int] = None):
        self.slip_prob = slip_prob
        self._rng = np.random.default_rng(seed)
        self._pos: Tuple[int, int] = START

    def reset(self) -> Tuple[int, int]:
        self._pos = START
        return self._pos

    def step(self, action: int) -> Tuple[Tuple[int, int], str, bool, int]:
        """Execute action; return (new_pos, event, done, hole_dir).

        hole_dir : the intended action if a hole was hit, else -1.
        event    : "a" | "b" | "c" | ""
        done     : True only when goal C is collected.
        """
        actual = self._sample_action(action)
        dr, dc = _DELTAS[actual]
        r, c = self._pos
        nr, nc = max(0, min(ROWS - 1, r + dr)), max(0, min(COLS - 1, c + dc))

        if (nr, nc) in HOLES:
            return self._pos, "", False, action  # bounce — return intended dir

        self._pos = (nr, nc)
        event = CHECKPOINTS.get((nr, nc), "")
        done = (event == "c")
        return self._pos, event, done, -1

    def pos(self) -> Tuple[int, int]:
        return self._pos

    def action_space(self) -> List[int]:
        return ACTIONS

    def _sample_action(self, intended: int) -> int:
        if self.slip_prob <= 0.0 or self._rng.random() >= self.slip_prob:
            return intended
        perp = [LEFT, RIGHT] if intended in (UP, DOWN) else [UP, DOWN]
        return int(self._rng.choice(perp))
