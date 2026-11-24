"""The SOUND, canonical zone construction -- v2 (branching RM), in-place
retune of the v1 module (same file, same technique: replay each state's
own exact discovery path, same delay at every step, never widened). Guard/
reset logic on each replayed edge mirrors env.peek's branching conditional
structure exactly, by construction (only ever invoked on edges the
concrete DP already confirmed are non-terminal, so any guard on that edge
already passed concretely).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .dp import DPResult, explore_and_solve
from .env import TRMState, _parse, is_terminal, target_of
from .instance import TRMGridInstance
from .zone_ops import EDGE_CLOCK, GLOBAL_CLOCK, TRMZoneContext, apply_lower_guard, apply_upper_guard, edge_delay, reset_clocks


def build_state_zones(instance: TRMGridInstance, dp_result: Optional[DPResult] = None):
    """Returns ``(dp_result, zctx, zones)``, ``zones: Dict[TRMState, Federation]``
    covering every non-terminal state in ``dp_result.states``."""
    if dp_result is None:
        dp_result = explore_and_solve(instance)

    zctx = TRMZoneContext(instance)
    root = dp_result.root
    zones: Dict[TRMState, object] = {root: zctx.ctx.get_zero_federation()}
    clock_names = instance.clock_names  # ("t_0", ..., "t_{k-1}")

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
            phase = s.phase

            Z2 = edge_delay(Z_here, zctx, delay, instance.w)
            Z2 = apply_upper_guard(Z2, zctx, [(GLOBAL_CLOCK, instance.deadline)])

            is_sg = instance.is_sub_goal(target)
            tgt_idx = instance.sub_goal_index(target) if is_sg else None
            reset_name = None

            if phase == () and is_sg:
                reset_name = clock_names[tgt_idx]
            elif len(phase) == 1 and is_sg and tgt_idx != phase[0]:
                i = phase[0]
                lo, hi = instance.windows[(i, tgt_idx)]
                Z2 = apply_lower_guard(Z2, zctx, [(clock_names[i], lo)])
                Z2 = apply_upper_guard(Z2, zctx, [(clock_names[i], hi)])
                reset_name = clock_names[tgt_idx]
            elif len(phase) == 2 and target == instance.goal_pos:
                # Success transition -> nxt is terminal (DONE), already
                # excluded above; kept for readers matching env.peek 1:1.
                j = phase[1]
                Z2 = apply_upper_guard(Z2, zctx, [(clock_names[j], instance.caps[j])])
            elif is_sg:
                reset_name = clock_names[tgt_idx]

            assert not Z2.is_empty(), f"zone became empty replaying a concrete-reachable edge {s}->{aid}->{nxt}"

            if reset_name is not None:
                Z2 = reset_clocks(Z2, zctx, [reset_name])

            Z2 = reset_clocks(Z2, zctx, [EDGE_CLOCK])  # close the book on e, see zone_ops.py docstring
            zones[nxt] = Z2

    return dp_result, zctx, zones


def distinct_zones(dp_result: DPResult, zones: Dict[TRMState, object]):
    """One representative federation per (location, phase, distinct zone)
    -- geometry stats are computed over these, not over one-per-state."""
    seen: Dict[Tuple, object] = {}
    for s in dp_result.states:
        if is_terminal(s) or s not in zones:
            continue
        key = (s.pos, s.phase, str(zones[s]))
        if key not in seen:
            seen[key] = (s.pos, s.phase, zones[s])
    return list(seen.values())


def validate_containment(instance: TRMGridInstance, zctx: TRMZoneContext, zones: Dict[TRMState, object]) -> Tuple[int, int]:
    from pyudbm import IntValuation

    from .instance import SCALE

    n_ok, n_bad = 0, 0
    for s, fed in zones.items():
        if is_terminal(s):
            continue
        val = IntValuation(zctx.ctx)
        for cname, cval in zip(instance.clock_names, s.clocks):
            val[cname] = cval * SCALE
        val[GLOBAL_CLOCK] = s.g * SCALE
        val[EDGE_CLOCK] = 0
        if fed.contains(val):
            n_ok += 1
        else:
            n_bad += 1
    return n_ok, n_bad
