"""Zone operations: c_J's (unchanged, per-task reset) plus ONE global
deadline clock ``g`` (never reset) -- see instance.py's module docstring
for why this replaced deadline_track's per-job g_J's.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from pyudbm import Context

from .instance import SCALE

if TYPE_CHECKING:
    from pyudbm.binding.udbm import Federation
    from prob_jobshop_ext.instance import ProbJobShopInstance

GLOBAL_CLOCK = "g"


class GateZoneContext:
    def __init__(self, instance: "ProbJobShopInstance"):
        job_ids = [j.job_id for j in instance.jobs]
        self.c_names: List[str] = [f"c_{jid}" for jid in job_ids]
        self._clock_names = self.c_names + [GLOBAL_CLOCK]
        self.ctx = Context(self._clock_names)
        self.clocks = {name: getattr(self.ctx, name) for name in self._clock_names}
        self.job_ids = job_ids

    @property
    def clock_names(self) -> List[str]:
        return self._clock_names

    def c(self, job_id: str) -> str:
        return f"c_{job_id}"


def apply_guard(Z: "Federation", zctx: GateZoneContext, guards: List[Tuple[str, int]]) -> Optional["Federation"]:
    Z_new = Z
    for cname, bound in guards:
        Z_new = Z_new & (zctx.clocks[cname] <= bound)
    if Z_new.is_empty():
        return None
    return Z_new


def apply_lower_guard(Z: "Federation", zctx: GateZoneContext, guards: List[Tuple[str, int]]) -> Optional["Federation"]:
    Z_new = Z
    for cname, bound in guards:
        Z_new = Z_new & (zctx.clocks[cname] >= bound)
    if Z_new.is_empty():
        return None
    return Z_new


def reset_clocks(Z: "Federation", zctx: GateZoneContext, clock_names: List[str]) -> "Federation":
    """Reset named clocks to 0. The global clock is forbidden here -- the
    single enforcement point for "g never resets", checked with an
    assertion (mirroring ``deadline_track/zone_ops.py`` and, before it,
    ``prob_jobshop_ext/zone_ops.py:76``'s treatment of ``delta``), not
    merely documented."""
    for cname in clock_names:
        assert cname != GLOBAL_CLOCK, f"{cname} is the global deadline clock and must never be reset"
    Z_new = Z
    for cname in clock_names:
        Z_new = Z_new.reset_value(zctx.clocks[cname])
    return Z_new


def apply_invariant(Z: "Federation", zctx: GateZoneContext, inv_pairs: List[Tuple[str, int]]) -> "Federation":
    Z_new = Z
    for cname, bound in inv_pairs:
        Z_new = Z_new & (zctx.clocks[cname] <= bound)
    return Z_new


def apply_delay(Z: "Federation", zctx: GateZoneContext, inv_pairs: List[Tuple[str, int]]) -> "Federation":
    """up() then re-intersect with the task-completion invariant. No
    blanket g <= D_J invariant here either -- same reasoning as
    deadline_track: checked on the dispatch edge only."""
    return apply_invariant(Z.up(), zctx, inv_pairs)


def normalize(Z: "Federation", zctx: GateZoneContext, m_bound: int) -> "Federation":
    """k-normalization, single bound for every clock (c_J's and g alike --
    see deadline_track's own bugfix note: a waiting job's c_J is not
    bounded by a single task's duration, it tracks elapsed time exactly
    like g until its first dispatch)."""
    bounds = {zctx.clocks[cn]: m_bound for cn in zctx.clock_names}
    return Z.extrapolate_max_bounds(bounds)


def zone_at_point(zctx: GateZoneContext, values_scaled: Dict[str, int]) -> "Federation":
    Z = zctx.ctx.get_zero_federation()
    for cname in zctx.clock_names:
        Z = Z.free_clock(zctx.clocks[cname])
    for cname, val in values_scaled.items():
        Z = Z & (zctx.clocks[cname] == val)
        if Z.is_empty():
            raise ValueError(f"zone_at_point: {cname}=={val} produced an empty zone (inconsistent inputs)")
    return Z


def feasible_best_case(
    Z: "Federation",
    zctx: GateZoneContext,
    job_id: str,
    remaining_tasks: List[Tuple[str, int, int]],
    deadline_scaled: int,
) -> bool:
    """Real zone-based feasibility check under interval durations: is
    there any valuation of Z, extended through dispatching
    ``remaining_tasks`` (task_id, lo_scaled, hi_scaled triples, in order --
    lo/hi from instance.interval() at each task's *smallest* nominal
    duration, already SCALE-scaled) back-to-back at their best-case (lo)
    realization, that still satisfies ``g <= deadline_scaled``?

    Chains reset -> invariant(hi) -> delay -> completion-guard(lo) for each
    remaining task, then intersects with the deadline guard on the single
    global clock and checks non-emptiness. See ``symbolic_env.py``'s
    ``_check_feasible`` for the arithmetic shortcut this validates against.
    """
    Zc = Z
    cname = zctx.c(job_id)
    for _tid, lo_s, hi_s in remaining_tasks:
        Zc = reset_clocks(Zc, zctx, [cname])
        Zc = apply_invariant(Zc, zctx, [(cname, hi_s)])
        Zc = Zc.up()
        Zc = apply_invariant(Zc, zctx, [(cname, hi_s)])
        Zc = apply_lower_guard(Zc, zctx, [(cname, lo_s)])
        if Zc is None or Zc.is_empty():
            return False
    Zc = apply_guard(Zc, zctx, [(GLOBAL_CLOCK, deadline_scaled)])
    return Zc is not None and not Zc.is_empty()
