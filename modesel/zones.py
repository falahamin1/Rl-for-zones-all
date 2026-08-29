"""SOUND zone construction: exactly one zone per concrete DP-reachable
state, built by replaying that state's own discovery path with the EXACT
mode dispatched at each step (never widened across both modes -- that was
gridtimed's soundness bug) and the mode's TRUE real-valued duration
interval ``[d*(1-w), d*(1+w)]`` as the only source of width.

Replays at the MICRO-STEP granularity of ``env.py``'s advance-and-resolve
loop (``StepResult.trace``), since a job's active-task clock must stay
correctly invariant-bounded across however many OTHER jobs' tasks complete
before it does -- not just within the single macro-transition it was
dispatched in.

Design decision flagged explicitly: no g<=D_j deadline guard is baked into
the zone here (unlike gridtimed's y<=D). Reachability under each job's
deadline is already fully determined by WHICH concrete states the DP's own
transition graph contains (infeasible dispatches are never offered;
auto-fail is its own transition) -- adding a redundant guard would not
change which states are reachable, only decorate the zones with a
constraint they already implicitly satisfy by construction.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .dp import DPResult, explore_and_solve
from .env import ModeSelState, is_terminal
from .instance import ModeSelInstance
from .zone_ops import GLOBAL_CLOCK, ModeSelZoneContext, apply_lower_guard, apply_upper_guard, reset_clocks


def interval(d: float, w: float) -> Tuple[float, float]:
    return d * (1 - w), d * (1 + w)


def advance_zone_for_transition(Z_here, zctx: ModeSelZoneContext, instance: ModeSelInstance,
                                 s: ModeSelState, job_idx: int, mode: str, trace) -> object:
    """The one piece of zone-update logic shared by DP replay
    (:func:`build_state_zones`) and live/online tracking during actual gym
    stepping (modesel/gym_env.py) -- identical either way, because the zone
    only ever depends on the mode's TRUE interval and the ORDER micro-step
    completions happen in (``trace``), never on which concrete duration was
    actually realized. ``s`` is the state the transition started FROM (used
    to seed which jobs were already active going in)."""
    task = instance.jobs[job_idx].tasks[s.jobs[job_idx].cursor]
    nominal_d = task.duration(mode)

    # jobs already active in s (dispatched in an earlier transition, not
    # yet completed) keep accumulating on their own clock, which was
    # reset/bounded when THEY were dispatched -- only start tracking a NEW
    # active set here, seeded from s.
    active = {
        i: js.active_duration
        for i, js in enumerate(s.jobs)
        if js.active_mode is not None and not js.failed
    }

    Z2 = reset_clocks(Z_here, zctx, [zctx.c(job_idx)])
    _lo0, hi0 = interval(nominal_d, instance.w)
    Z2 = apply_upper_guard(Z2, zctx, [(zctx.c(job_idx), hi0)])
    assert not Z2.is_empty(), f"zone empty right after dispatch guard, {s}, job={job_idx}, mode={mode}"
    active[job_idx] = nominal_d

    for micro in trace:
        Z2 = Z2.up()
        inv_pairs = [(zctx.c(i), interval(d, instance.w)[1]) for i, d in active.items()]
        Z2 = apply_upper_guard(Z2, zctx, inv_pairs)
        assert not Z2.is_empty(), f"zone empty after invariant, {s}, job={job_idx}, mode={mode}"

        for i in micro.completing_jobs:
            lo, _hi = interval(active[i], instance.w)
            Z2 = apply_lower_guard(Z2, zctx, [(zctx.c(i), lo)])
            assert not Z2.is_empty(), f"zone empty after completion guard, {s}, job={job_idx}, mode={mode}"
            Z2 = reset_clocks(Z2, zctx, [zctx.c(i)])
            del active[i]
        # newly_failed_jobs need no zone treatment: only WAITING (never
        # active) jobs can fail (see env.py's module docstring), so there
        # is no clock to bound for them.

    # A WAITING job's clock is not a real ticking clock in the concrete
    # simulation (env.py: c stays 0 until dispatched -- unlike gridtimed's
    # switch clocks, which genuinely track y until first touched). But
    # pyudbm's up() advances every declared clock uniformly, with no
    # "selective freeze" -- so any job not currently in `active` has been
    # silently drifting with g through every up() call above. Reset erases
    # that drift completely (reset_value replaces ALL of a clock's
    # constraints, discarding whatever relationship it accumulated with
    # other clocks), and nothing during the loop above ever reads or
    # constrains a non-active clock's value, so doing this once here --
    # rather than after every micro-step -- is exactly equivalent and
    # cheaper. Caught by a real containment failure (4/5 states failed
    # validate_containment) before this fix, not by inspection.
    for i in range(instance.n_jobs):
        if i not in active:
            Z2 = reset_clocks(Z2, zctx, [zctx.c(i)])

    return Z2


def build_state_zones(instance: ModeSelInstance, dp_result: Optional[DPResult] = None):
    if dp_result is None:
        dp_result = explore_and_solve(instance)

    zctx = ModeSelZoneContext(instance)
    root = dp_result.root
    zones: Dict[ModeSelState, object] = {root: zctx.ctx.get_zero_federation()}

    for s in dp_result.states:
        if is_terminal(instance, s) or s not in zones:
            continue
        Z_here = zones[s]
        for aid, res in dp_result.transitions.get(s, {}).items():
            nxt = res.next_state
            if nxt == s or nxt in zones or is_terminal(instance, nxt):
                continue
            zones[nxt] = advance_zone_for_transition(
                Z_here, zctx, instance, s, res.dispatched_job_idx, res.dispatched_mode, res.trace
            )

    return dp_result, zctx, zones


def distinct_zones(dp_result: DPResult, zones: Dict[ModeSelState, object], instance: ModeSelInstance):
    """One representative federation per (location, distinct zone) --
    dedupes states sharing an identical zone, same convention as
    gridtimed/zones.py."""
    from .diagnostics import _location_key

    seen: Dict[Tuple, object] = {}
    for s in dp_result.states:
        if is_terminal(instance, s) or s not in zones:
            continue
        key = (_location_key(instance, s), str(zones[s]))
        if key not in seen:
            seen[key] = (_location_key(instance, s), zones[s])
    return list(seen.values())


def validate_containment(instance: ModeSelInstance, zctx: ModeSelZoneContext, zones: Dict[ModeSelState, object]) -> Tuple[int, int]:
    from pyudbm import IntValuation

    from .instance import SCALE

    n_ok, n_bad = 0, 0
    for s, fed in zones.items():
        if is_terminal(instance, s):
            continue
        val = IntValuation(zctx.ctx)
        for i in range(instance.n_jobs):
            val[zctx.c(i)] = int(round(s.jobs[i].c * SCALE))
        val[GLOBAL_CLOCK] = int(round(s.g * SCALE))
        if fed.contains(val):
            n_ok += 1
        else:
            n_bad += 1
    return n_ok, n_bad
