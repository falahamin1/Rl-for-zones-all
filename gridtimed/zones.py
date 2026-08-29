"""The SOUND, canonical zone construction (see /tmp/GRIDSOUND.md Task A).

Exactly one symbolic zone per concrete, exact-DP-reachable state, built by
replaying that state's own discovery path -- same sequence of locations
AND the same EXACT delay chosen at every step (not widened across
``instance.allowed_delays``, which was the bug: the agent's own decision
variable was being treated as unobserved uncertainty, merging concrete
points reachable only via genuinely different decisions into one zone).
The only real width comes from ``instance.w``, a genuinely unobserved
duration jitter (see zone_ops.edge_delay).

Superseded (no longer used): the original ``zone_graph.py``, which built
zones by walking a LOCATION-only graph (delay widened to the full allowed
range, needing k-normalization + subsumption for termination against
revisit cycles). That machinery is retired here: because delay is now
exact per edge, zones are attached 1:1 to states in the concrete DP's own
(already-finite, since the deadline bounds it) transition graph, so no
separate fixed-point BFS over locations is needed at all. Kept in the repo
unmodified as the artifact behind /tmp/GRIDGATE.md's numbers, but
geometry.py and decision_relevance.py now both use this module instead.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .dp import DPResult, explore_and_solve
from .env import GridState, _parse, is_terminal, target_of
from .instance import GridInstance
from .zone_ops import EDGE_CLOCK, GLOBAL_CLOCK, GridZoneContext, apply_lower_guard, apply_upper_guard, edge_delay, reset_clocks


def build_state_zones(instance: GridInstance, dp_result: Optional[DPResult] = None):
    """Returns ``(dp_result, zctx, zones)``, ``zones: Dict[GridState, Federation]``
    covering every non-terminal state in ``dp_result.states``."""
    if dp_result is None:
        dp_result = explore_and_solve(instance)

    zctx = GridZoneContext(instance)
    root = dp_result.root
    zones: Dict[GridState, object] = {root: zctx.ctx.get_zero_federation()}

    for s in dp_result.states:
        if is_terminal(s) or s not in zones:
            continue
        Z_here = zones[s]
        for aid, res in dp_result.transitions.get(s, {}).items():
            nxt = res.next_state
            if nxt == s or nxt in zones or is_terminal(nxt):
                continue
            delay, move = _parse(aid)
            target = target_of(s.pos, move)

            Z2 = edge_delay(Z_here, zctx, delay, instance.w)
            if instance.is_gate(target):
                a, b = instance.gate_window(target)
                cname = instance.gate_clock(target)
                Z2 = apply_lower_guard(Z2, zctx, [(cname, a)])
                Z2 = apply_upper_guard(Z2, zctx, [(cname, b)])
            Z2 = apply_upper_guard(Z2, zctx, [(GLOBAL_CLOCK, instance.deadline)])
            assert not Z2.is_empty(), f"zone became empty replaying a concrete-reachable edge {s}->{aid}->{nxt}"
            if instance.is_switch(target):
                Z2 = reset_clocks(Z2, zctx, [instance.switch_clock(target)])
            Z2 = reset_clocks(Z2, zctx, [EDGE_CLOCK])  # close the book on e, see zone_ops.py docstring
            zones[nxt] = Z2

    return dp_result, zctx, zones


def distinct_zones(dp_result: DPResult, zones: Dict[GridState, object]):
    """One representative federation per (location, distinct zone) --
    geometry stats are computed over these, not over one-per-state (which
    would over-weight zones with many concrete member points)."""
    seen: Dict[Tuple, object] = {}
    for s in dp_result.states:
        if is_terminal(s) or s not in zones:
            continue
        key = (s.pos, str(zones[s]))
        if key not in seen:
            seen[key] = (s.pos, zones[s])
    return list(seen.values())


def validate_containment(instance: GridInstance, zctx: GridZoneContext, zones: Dict[GridState, object]) -> Tuple[int, int]:
    from pyudbm import IntValuation

    from .instance import SCALE

    n_ok, n_bad = 0, 0
    for s, fed in zones.items():
        if is_terminal(s):
            continue
        val = IntValuation(zctx.ctx)
        for cname, cval in zip(instance.clock_names, s.clocks):
            val[cname] = cval * SCALE
        val[GLOBAL_CLOCK] = s.y * SCALE
        val[EDGE_CLOCK] = 0
        if fed.contains(val):
            n_ok += 1
        else:
            n_bad += 1
    return n_ok, n_bad
