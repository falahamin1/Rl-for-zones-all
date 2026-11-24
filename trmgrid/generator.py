"""Parameterized instance generator for the v2 branching design.

Windows and caps are calibrated RELATIVE TO ACTUAL MANHATTAN DISTANCE
between the sub-goals involved, not fixed absolute numbers reused across
differently-shaped layouts (v1's own watch-list lesson, applied to timing
parameters this time instead of a pyudbm clip bound):

    lo_ij = ceil(d_ij * (1 + alpha)),  hi_ij = lo_ij + width
    cap_j = ceil(d_j_goal * (1 + alpha)) + width

where d_ij is the Manhattan distance between sub-goal i and sub-goal j,
and d_j_goal is the Manhattan distance from sub-goal j to GOAL. alpha
controls how much slower-than-direct-walking the agent must arrive
(alpha=0 means "no genuine forced wait, direct arrival already legal";
alpha>0 forces some waiting since direct walking arrives before lo_ij);
width controls how much slack the window/cap allows once the lower bound
is met.

The deadline D is ALSO computed relative to this specific layout's actual
distances and windows (never a fixed constant reused across parameter
settings -- same principle), generously, then verified (not assumed) via
a cheap concrete-BFS solvability check with reject+resample, same idiom
as v1's generator.py (read as this file's own prior version, in-place
retune, not a separate copy).
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Tuple

from .env import all_actions, initial_state, is_terminal, peek
from .instance import DEFAULT_DELAYS, Position, TRMGridInstance, manhattan


@dataclass
class GenerationInfo:
    seed: int
    label: str
    attempts: int
    accepted: bool
    rejected_unsolvable: int
    rejected_inconclusive: int
    deadline_used: int = 0


def is_solvable(instance: TRMGridInstance, state_limit: int = 200_000) -> Tuple[bool, bool, int]:
    root = initial_state(instance)
    seen = {root}
    queue = deque([root])
    n = 0
    while queue:
        n += 1
        if n > state_limit:
            return False, True, n
        s = queue.popleft()
        if is_terminal(s):
            if s.pos == "DONE":
                return True, False, n
            continue
        for aid in all_actions(instance):
            res = peek(instance, s, aid)
            if res.next_state not in seen:
                seen.add(res.next_state)
                queue.append(res.next_state)
    return False, False, n


def _windows_and_caps(sub_goals: Tuple[Position, ...], goal_pos: Position, alpha: float, width: int):
    k = len(sub_goals)
    windows: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            d_ij = manhattan(sub_goals[i], sub_goals[j])
            lo = math.ceil(d_ij * (1 + alpha))
            hi = lo + width
            windows[(i, j)] = (lo, hi)
    caps = tuple(math.ceil(manhattan(sub_goals[j], goal_pos) * (1 + alpha)) + width for j in range(k))
    return windows, caps


def _estimate_deadline(start: Position, sub_goals: Tuple[Position, ...], goal_pos: Position,
                        windows: Dict[Tuple[int, int], Tuple[int, int]], caps: Tuple[int, ...]) -> int:
    """Generous (not tight) upper bound: the best-case ordered pair's
    (start->i distance) + hi_ij (worst-case legal wait+arrival at j) +
    cap_j (worst-case legal arrival at GOAL), plus a margin -- verified
    (not assumed) by the solvability check below, so a generous
    over-estimate here costs nothing but a slightly larger state space."""
    k = len(sub_goals)
    best = None
    for i, j in permutations(range(k), 2):
        d_start_i = manhattan(start, sub_goals[i])
        _lo, hi_ij = windows[(i, j)]
        cost = d_start_i + hi_ij + caps[j]
        if best is None or cost < best:
            best = cost
    margin = 4
    return int(best) + margin if best is not None else 4 * len(sub_goals)


def make_instance(
    grid_n: int,
    k: int,
    alpha: float,
    width: int,
    w: float,
    seed: int,
    allowed_delays: Tuple[int, ...] = DEFAULT_DELAYS,
    max_resample: int = 300,
    label: str = "",
) -> Tuple[TRMGridInstance, GenerationInfo]:
    start: Position = (0, 0)
    all_cells = [(r, c) for r in range(grid_n) for c in range(grid_n) if (r, c) != start]
    needed = k + 1  # k sub-goals + goal

    rejected_unsolvable = 0
    rejected_inconclusive = 0

    for attempt in range(1, max_resample + 1):
        sub_seed = seed * 1_000_003 + attempt
        rng = random.Random(sub_seed)
        cells = list(all_cells)
        rng.shuffle(cells)
        sub_goals = tuple(cells[:k])
        goal_pos = cells[k]

        windows, caps = _windows_and_caps(sub_goals, goal_pos, alpha, width)
        D = _estimate_deadline(start, sub_goals, goal_pos, windows, caps)

        inst = TRMGridInstance(
            grid_n=grid_n, start=start, sub_goals=sub_goals, goal_pos=goal_pos,
            windows=windows, caps=caps, deadline=D, allowed_delays=allowed_delays, w=w,
            seed=sub_seed, label=label or f"gen_n{grid_n}_k{k}_a{alpha}_w{width}",
        )

        solvable, inconclusive, _n = is_solvable(inst)
        if solvable:
            info = GenerationInfo(
                seed=sub_seed, label=inst.label, attempts=attempt, accepted=True,
                rejected_unsolvable=rejected_unsolvable, rejected_inconclusive=rejected_inconclusive,
                deadline_used=D,
            )
            return inst, info
        if inconclusive:
            rejected_inconclusive += 1
        else:
            rejected_unsolvable += 1

    raise RuntimeError(
        f"make_instance: no solvable instance found in {max_resample} attempts "
        f"(seed={seed}, grid_n={grid_n}, k={k}, alpha={alpha}, width={width}); "
        f"{rejected_unsolvable} confirmed unsolvable, {rejected_inconclusive} inconclusive"
    )
