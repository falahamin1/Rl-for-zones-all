"""Zone operations for a trmgrid instance.

Clocks: ``t_A``, ``t_B``, ``t_C`` (one per sub-goal, ``instance.clock_names``),
``g`` (the never-resetting global clock), plus one internal bookkeeping
clock ``e`` ("edge-elapsed"), reset on *every* edge and used to bound a
single edge's elapsed time -- the same technique gridtimed's zone_ops.py
uses (read as a reference, not imported): a per-edge helper clock is the
standard way to give a SINGLE move a bounded, uncertain duration without
that uncertainty leaking into every other clock's own accumulated bound.

SOUND edge construction (see instance.py's module docstring): elapsed time
for one edge is ``delay + duration``, where ``delay`` is the EXACT delay
the agent chose for that specific action (never widened across the
allowed set) and ``duration`` is a genuinely unobserved real value on
``[1, 1+w]``. Built as reset(e) -> up() -> invariant(e <= delay+1+w) ->
guard(e >= delay+1), then ``e`` is reset again immediately after being
used, so every REPORTED zone has ``e`` pinned to a constant.

``g`` never resets -- enforced the same way ``y``/``delta`` are elsewhere
in this repo's other domains: an assertion in ``reset_clocks``, not just a
docstring.
"""
from __future__ import annotations

from typing import List, Tuple

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from pyudbm import Context

from .instance import SCALE, TRMGridInstance

EDGE_CLOCK = "e"
GLOBAL_CLOCK = "g"


def _scale(x) -> int:
    return int(round(x * SCALE))


class TRMZoneContext:
    def __init__(self, instance: TRMGridInstance):
        self._clock_names = list(instance.clock_names) + [GLOBAL_CLOCK, EDGE_CLOCK]
        self.ctx = Context(self._clock_names)
        self.clocks = {name: getattr(self.ctx, name) for name in self._clock_names}

    @property
    def clock_names(self) -> List[str]:
        return list(self._clock_names)


def reset_clocks(Z, zctx: TRMZoneContext, names: List[str]):
    for name in names:
        assert name != GLOBAL_CLOCK, f"{name} is the global clock and must never be reset"
    Z2 = Z
    for name in names:
        Z2 = Z2.reset_value(zctx.clocks[name])
    return Z2


def apply_upper_guard(Z, zctx: TRMZoneContext, pairs: List[Tuple[str, int]]):
    """``pairs`` are (clock_name, bound) in REAL (unscaled) units."""
    Z2 = Z
    for name, bound in pairs:
        Z2 = Z2 & (zctx.clocks[name] <= _scale(bound))
    return Z2


def apply_lower_guard(Z, zctx: TRMZoneContext, pairs: List[Tuple[str, int]]):
    Z2 = Z
    for name, bound in pairs:
        Z2 = Z2 & (zctx.clocks[name] >= _scale(bound))
    return Z2


def edge_delay(Z, zctx: TRMZoneContext, delay: int, w: float):
    """Reset the edge clock, let time pass, and constrain the elapsed
    amount to ``[delay+1, delay+1+w]`` (real units, scaled internally) --
    reset -> up() -> invariant(hi) -> guard(lo), driven by the EXACT delay
    of the action being replayed (see module docstring)."""
    lo = delay + 1
    hi = delay + 1 + w
    Z2 = reset_clocks(Z, zctx, [EDGE_CLOCK])
    Z2 = Z2.up()
    Z2 = apply_upper_guard(Z2, zctx, [(EDGE_CLOCK, hi)])
    Z2 = apply_lower_guard(Z2, zctx, [(EDGE_CLOCK, lo)])
    return Z2


def normalize(Z, zctx: TRMZoneContext, k_bound_real: int):
    bounds = {zctx.clocks[name]: _scale(k_bound_real) for name in zctx.clock_names}
    return Z.extrapolate_max_bounds(bounds)
