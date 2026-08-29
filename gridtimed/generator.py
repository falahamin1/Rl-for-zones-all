"""Task B: a parameterized instance generator, not hand-written tiers.

``make_instance`` places switches, their paired gates, and each gate's
window pseudo-randomly (seeded), then runs a cheap concrete-BFS
solvability check (GOAL reachable within D, ignoring cost/optimality) and
rejects+resamples on failure. The T0 tier (grid_n=3, 2 switches, 2 gates)
is STRUCTURALLY analogous to, but not bit-identical to, the hand-built
canonical instance in instance.py's module defaults -- that hand-built
instance remains the one used for the Measurement 1 sanity gate and is not
touched by this generator.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .env import all_actions, initial_state, is_terminal, peek
from .instance import DEFAULT_DELAYS, GridInstance, Position


@dataclass
class GenerationInfo:
    seed: int
    label: str
    attempts: int
    accepted: bool
    rejected_unsolvable: int
    rejected_inconclusive: int


def is_solvable(instance: GridInstance, state_limit: int = 60_000) -> Tuple[bool, bool, int]:
    """Cheap concrete-state BFS reachability check (ignores cost/optimality
    entirely -- just: is GOAL reachable within D via *some* sequence of
    legal moves?). Returns (solvable, inconclusive, n_states_explored).
    ``inconclusive`` means the state budget was exhausted before either
    finding GOAL or exhausting the reachable set -- distinct from a
    confirmed-unsolvable rejection, and reported separately."""
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


def make_instance(
    grid_n: int,
    n_switches: int,
    n_gates: int,
    window_width: int,
    D: int,
    w: float,
    seed: int,
    allowed_delays: Tuple[int, ...] = DEFAULT_DELAYS,
    max_resample: int = 200,
    label: str = "",
) -> Tuple[GridInstance, GenerationInfo]:
    assert n_switches == n_gates, "gates are paired 1:1 with switches by index"

    start: Position = (0, 0)
    goal: Position = (grid_n - 1, grid_n - 1)
    all_cells = [(r, c) for r in range(grid_n) for c in range(grid_n) if (r, c) not in (start, goal)]
    needed = n_switches + n_gates
    if len(all_cells) < needed:
        raise ValueError(f"grid_n={grid_n} has only {len(all_cells)} free cells, need {needed}")

    rejected_unsolvable = 0
    rejected_inconclusive = 0

    for attempt in range(1, max_resample + 1):
        sub_seed = seed * 1_000_003 + attempt
        rng = random.Random(sub_seed)
        cells = list(all_cells)
        rng.shuffle(cells)
        chosen = cells[:needed]
        switches = tuple(chosen[:n_switches])
        gates = tuple(chosen[n_switches:])

        gate_windows: List[Tuple[int, int]] = []
        for _ in range(n_gates):
            hi_start = max(grid_n, D - window_width - 1)
            a = rng.randint(grid_n, hi_start) if hi_start >= grid_n else grid_n
            gate_windows.append((a, a + window_width))

        inst = GridInstance(
            grid_n=grid_n, start=start, goal=goal,
            switches=switches, gates=gates, gate_windows=tuple(gate_windows),
            deadline=D, allowed_delays=allowed_delays, w=w,
            seed=sub_seed, label=label or f"gen_n{grid_n}_sw{n_switches}_ga{n_gates}",
        )

        solvable, inconclusive, _n = is_solvable(inst)
        if solvable:
            info = GenerationInfo(
                seed=sub_seed, label=inst.label, attempts=attempt, accepted=True,
                rejected_unsolvable=rejected_unsolvable, rejected_inconclusive=rejected_inconclusive,
            )
            return inst, info
        if inconclusive:
            rejected_inconclusive += 1
        else:
            rejected_unsolvable += 1

    raise RuntimeError(
        f"make_instance: no solvable instance found in {max_resample} attempts "
        f"(seed={seed}, grid_n={grid_n}, n_switches={n_switches}); "
        f"{rejected_unsolvable} confirmed unsolvable, {rejected_inconclusive} inconclusive"
    )
