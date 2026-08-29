"""Zone operations for a grid instance.

Clocks: ``s1..sK`` (one per switch, ``instance.clock_names``), ``y`` (the
never-resetting global clock), plus one internal bookkeeping clock ``e``
("edge-elapsed"), reset on *every* edge and used to bound a single edge's
elapsed time -- the same role ``c_J`` plays in ``gate_track``'s per-task
invariant/guard chain, just re-armed every move instead of every task
dispatch.

SOUND edge construction (see instance.py's module docstring): elapsed time
for one edge is ``delay + duration``, where ``delay`` is the EXACT delay the
agent chose for that specific action (not widened across the allowed set)
and ``duration`` is a genuinely unobserved real value on ``[1, 1+w]``. This
is built as reset(e) -> up() -> invariant(e <= delay+1+w) -> guard(e >=
delay+1), then ``e`` is reset again immediately after being used (so every
REPORTED zone has ``e`` pinned to a constant -- verified empirically in
geometry.py against a hand check of facet counts before trusting it).

``y`` never resets -- enforced the same way ``delta``/``g`` are elsewhere in
this repo: an assertion in ``reset_clocks``, not just a docstring.
"""
from __future__ import annotations

from typing import List, Tuple

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from pyudbm import Context

from .instance import GridInstance, SCALE

EDGE_CLOCK = "e"
GLOBAL_CLOCK = "y"


def _scale(x) -> int:
    return int(round(x * SCALE))


class GridZoneContext:
    def __init__(self, instance: GridInstance):
        self._clock_names = list(instance.clock_names) + [GLOBAL_CLOCK, EDGE_CLOCK]
        self.ctx = Context(self._clock_names)
        self.clocks = {name: getattr(self.ctx, name) for name in self._clock_names}

    @property
    def clock_names(self) -> List[str]:
        return list(self._clock_names)


def reset_clocks(Z, zctx: GridZoneContext, names: List[str]):
    for name in names:
        assert name != GLOBAL_CLOCK, f"{name} is the global clock and must never be reset"
    Z2 = Z
    for name in names:
        Z2 = Z2.reset_value(zctx.clocks[name])
    return Z2


def apply_upper_guard(Z, zctx: GridZoneContext, pairs: List[Tuple[str, int]]):
    """``pairs`` are (clock_name, bound) in REAL (unscaled) units."""
    Z2 = Z
    for name, bound in pairs:
        Z2 = Z2 & (zctx.clocks[name] <= _scale(bound))
    return Z2


def apply_lower_guard(Z, zctx: GridZoneContext, pairs: List[Tuple[str, int]]):
    Z2 = Z
    for name, bound in pairs:
        Z2 = Z2 & (zctx.clocks[name] >= _scale(bound))
    return Z2


def edge_delay(Z, zctx: GridZoneContext, delay: int, w: float):
    """Reset the edge clock, let time pass, and constrain the elapsed
    amount to ``[delay+1, delay+1+w]`` (real units, scaled internally) --
    reset -> up() -> invariant(hi) -> guard(lo), mirroring gate_track's
    apply_delay + apply_lower_guard chain for a per-task clock, here
    re-armed every move instead of every dispatch, and driven by the
    EXACT delay of the action being replayed (see module docstring)."""
    lo = delay + 1
    hi = delay + 1 + w
    Z2 = reset_clocks(Z, zctx, [EDGE_CLOCK])
    Z2 = Z2.up()
    Z2 = apply_upper_guard(Z2, zctx, [(EDGE_CLOCK, hi)])
    Z2 = apply_lower_guard(Z2, zctx, [(EDGE_CLOCK, lo)])
    return Z2


def normalize(Z, zctx: GridZoneContext, k_bound_real: int):
    bounds = {zctx.clocks[name]: _scale(k_bound_real) for name in zctx.clock_names}
    return Z.extrapolate_max_bounds(bounds)
