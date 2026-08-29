"""Zone-graph BFS over LOCATIONS (grid cells), for Measurement 2's geometry.

This is deliberately independent of the concrete, bounded-action-set DP in
dp.py: it answers "what (s1, s2, y) valuations are reachable at this
location via *some* sequence of moves respecting each move's
delay-bounded elapsed-time window" -- a genuine timed-automaton
reachability question over the location graph (9 cells, edges = the 4
directions, guarded by a gate's window when entering a gate, reset when
entering a switch), parameterized by ``instance.allowed_delays`` via
``[min_delay+1, max_delay+1]`` on the per-edge bookkeeping clock (see
zone_ops.bounded_delay). Restricting delay to ``{0}`` (Measurement 4)
collapses that per-edge window to a single point and is expected to
degenerate the resulting geometry -- this is not assumed, it is measured.

Unlike gate_track/deadline_track (acyclic once tasks complete), this
location graph has real cycles (a switch can be revisited indefinitely),
so a plain symbolic BFS would not terminate on its own. Two things bound
it: (1) every edge is intersected with the SAME deadline guard ``y <= D``
used by the concrete env's auto-fail (found necessary the hard way: an
earlier version left the zone graph deadline-unconstrained, and some
zones' *minimum* reachable y ended up above K_BOUND purely from loop
revisits, which then silently emptied under clip_to_k_box -- the same
class of bug flagged in deadline_track's clip-bound note. Tying the zone
graph to D directly means "reachable" means the same thing here as it does
for the concrete DP); and (2) every zone is k-normalized
(extrapolate_max_bounds) before being checked against the passed list, and
only added if NOT already subsumed (Federation.__le__, subset) by some
previously-discovered zone at that same location. Measured directly
(see /tmp/GRIDGATE.md): without extrapolation, the same BFS did not
terminate within a 200k-edge budget -- k-normalization is load-bearing
here, not defensive boilerplate.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

from .instance import (
    DEADLINE,
    GATE1,
    GATE2,
    K_BOUND,
    MOVES,
    START,
    DELTA,
    GridInstance,
    Position,
    gate_clock,
    gate_window,
    in_bounds,
    is_gate,
    is_switch,
    switch_clock,
)
from .zone_ops import GridZoneContext, apply_lower_guard, apply_upper_guard, bounded_delay, normalize, reset_clocks


def _target(pos: Position, move: str) -> Position:
    dr, dc = DELTA[move]
    return (pos[0] + dr, pos[1] + dc)


def build_zone_graph(instance: GridInstance, k_bound: int = K_BOUND):
    """Returns ``(zctx, zones_by_loc, n_edges_explored, n_pruned_subsumed,
    n_pruned_empty)``, where ``zones_by_loc: Dict[Position, List[Federation]]``
    is the final passed list -- the set of maximal (non-subsumed), extrapolated
    zones reachable at each location."""
    zctx = GridZoneContext()
    lo = instance.min_delay + 1
    hi = instance.max_delay + 1

    root_zone = zctx.ctx.get_zero_federation()
    zones_by_loc: Dict[Position, List] = {START: [root_zone]}

    queue = deque([(START, root_zone)])
    n_edges = 0
    n_pruned_subsumed = 0
    n_pruned_empty = 0

    while queue:
        pos, Z = queue.popleft()
        for move in MOVES:
            target = _target(pos, move)
            if not in_bounds(target):
                continue
            n_edges += 1

            Z2 = bounded_delay(Z, zctx, lo, hi)
            if is_gate(target):
                a, b = gate_window(target)
                cname = gate_clock(target)
                Z2 = apply_lower_guard(Z2, zctx, [(cname, a)])
                Z2 = apply_upper_guard(Z2, zctx, [(cname, b)])
            Z2 = apply_upper_guard(Z2, zctx, [("y", DEADLINE)])
            if Z2.is_empty():
                n_pruned_empty += 1
                continue
            if is_switch(target):
                Z2 = reset_clocks(Z2, zctx, [switch_clock(target)])
            # e has done its job bounding this edge's elapsed time; reset it
            # back to 0 immediately (not deferred to the next edge's
            # bounded_delay call) so every REPORTED zone has e pinned to a
            # constant, matching a concrete state read out right after
            # arrival (see zone_ops.py's module docstring).
            Z2 = reset_clocks(Z2, zctx, ["e"])

            Z2 = normalize(Z2, zctx, k_bound)
            if Z2.is_empty():
                n_pruned_empty += 1
                continue

            existing = zones_by_loc.setdefault(target, [])
            if any(Z2 <= Zold for Zold in existing):
                n_pruned_subsumed += 1
                continue

            existing.append(Z2)
            queue.append((target, Z2))

    return zctx, zones_by_loc, n_edges, n_pruned_subsumed, n_pruned_empty
