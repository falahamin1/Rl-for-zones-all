"""Zone operations for the deadline-augmented job-shop model.

Fork of ``prob_jobshop_ext.zone_ops``. Two clock families now, both stored
in one pyudbm ``Context`` (no ``delta`` clock at all -- carried forward from
the encoder-slice track, elapsed time is read from the concrete side, same
sign convention as ``region_env.py:86-111``):

* ``c_{job}``  -- unchanged from the base package: per-task, reset on every
  dispatch.
* ``g_{job}``  -- new: starts at 0, advances with every ``up()``, **never**
  reset. Enforced by an assertion in :func:`reset_clocks`, the same pattern
  ``prob_jobshop_ext/zone_ops.py`` already uses for ``delta`` (an assertion,
  not a comment, per the task).

All bounds passed into this module's functions must already be in
*operational SCALE units* (``deadline_track.instance.SCALE = 20``) --
callers (``symbolic_env.py``) are responsible for scaling task durations
and deadlines consistently before calling in here; this module does no
implicit scaling of its own.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap; see zoneviews/__init__.py)
from pyudbm import Context

from .instance import SCALE

if TYPE_CHECKING:
    from pyudbm.binding.udbm import Federation
    from prob_jobshop_ext.instance import ProbJobShopInstance


class DeadlineZoneContext:
    """Wraps a pyudbm Context with 2 * n_jobs clocks: c_J (resetting) and
    g_J (never-resetting deadline clocks)."""

    def __init__(self, instance: "ProbJobShopInstance"):
        job_ids = [j.job_id for j in instance.jobs]
        self.c_names: List[str] = [f"c_{jid}" for jid in job_ids]
        self.g_names: List[str] = [f"g_{jid}" for jid in job_ids]
        self._clock_names = self.c_names + self.g_names
        self.ctx = Context(self._clock_names)
        self.clocks = {name: getattr(self.ctx, name) for name in self._clock_names}
        self.job_ids = job_ids

    @property
    def clock_names(self) -> List[str]:
        return self._clock_names

    def c(self, job_id: str) -> str:
        return f"c_{job_id}"

    def g(self, job_id: str) -> str:
        return f"g_{job_id}"


def apply_guard(Z: "Federation", zctx: DeadlineZoneContext, guards: List[Tuple[str, int]]) -> Optional["Federation"]:
    """Intersect with upper-bound guards ``clock <= bound`` (bound in scaled
    integer units). Returns None if the result is empty."""
    Z_new = Z
    for cname, bound in guards:
        Z_new = Z_new & (zctx.clocks[cname] <= bound)
    if Z_new.is_empty():
        return None
    return Z_new


def apply_lower_guard(Z: "Federation", zctx: DeadlineZoneContext, guards: List[Tuple[str, int]]) -> Optional["Federation"]:
    """Intersect with lower-bound guards ``clock >= bound``."""
    Z_new = Z
    for cname, bound in guards:
        Z_new = Z_new & (zctx.clocks[cname] >= bound)
    if Z_new.is_empty():
        return None
    return Z_new


def reset_clocks(Z: "Federation", zctx: DeadlineZoneContext, clock_names: List[str]) -> "Federation":
    """Reset named clocks to 0. Any ``g_*`` clock is forbidden here -- this
    is the enforcement point for "g_J never resets", checked with an
    assertion (mirroring ``prob_jobshop_ext/zone_ops.py:76``'s treatment of
    ``delta``), not merely documented."""
    for cname in clock_names:
        assert cname not in zctx.g_names, f"{cname} is a deadline clock and must never be reset"
    Z_new = Z
    for cname in clock_names:
        Z_new = Z_new.reset_value(zctx.clocks[cname])
    return Z_new


def apply_invariant(Z: "Federation", zctx: DeadlineZoneContext, inv_pairs: List[Tuple[str, int]]) -> "Federation":
    Z_new = Z
    for cname, bound in inv_pairs:
        Z_new = Z_new & (zctx.clocks[cname] <= bound)
    return Z_new


def apply_delay(Z: "Federation", zctx: DeadlineZoneContext, inv_pairs: List[Tuple[str, int]]) -> "Federation":
    """up() then re-intersect with the (task-completion) invariant. Note
    this does NOT apply any g_J <= D_J invariant unconditionally -- per the
    task, deadlines are checked only on dispatch edges, never as a blanket
    location invariant (an unconditional g_J <= D_J invariant would block
    time elapse entirely past the deadline and deadlock the automaton)."""
    return apply_invariant(Z.up(), zctx, inv_pairs)


def normalize(Z: "Federation", zctx: DeadlineZoneContext, m_c: int, m_g: int) -> "Federation":
    """k-normalization: c_J clocks bounded by m_c (already-scaled max task
    duration), g_J clocks bounded by m_g (already-scaled max deadline across
    jobs, plus headroom) -- unlike the base package's `delta`, g_J clocks
    are NOT excluded from normalization: they are compared against finite
    deadlines and must stay bounded for the zone graph to be finite."""
    bounds = {zctx.clocks[cn]: m_c for cn in zctx.c_names}
    bounds.update({zctx.clocks[gn]: m_g for gn in zctx.g_names})
    return Z.extrapolate_max_bounds(bounds)


def dominates(Z_old: "Federation", Z_new: "Federation") -> bool:
    return Z_new <= Z_old


def zone_at_point(zctx: DeadlineZoneContext, values_scaled: Dict[str, int]) -> "Federation":
    """Build the exact-point Federation pinning every clock in
    ``values_scaled`` (clock_name -> scaled integer value) to an equality
    constant. Starts from the zero federation with every clock immediately
    freed (``free_clock``), not left zero-pinned -- the zero federation
    ties every clock to 0 *and to each other*, so intersecting it directly
    with an equality to a nonzero value is simply empty (0 != 60). Callers
    must supply a value for every clock (c_J and g_J alike; c_J for a
    currently-irrelevant job can be passed as 0). Used to construct "the
    zone at this concrete simulation point" on demand for feasibility-check
    validation, without needing the live env to thread a Federation through
    every step."""
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
    zctx: DeadlineZoneContext,
    job_id: str,
    remaining_tasks: List[Tuple[str, int]],
    deadline_scaled: int,
) -> bool:
    """Real zone-based feasibility check: is there *any* valuation of Z,
    extended through dispatching ``remaining_tasks`` (task_id, min_duration
    pairs, in order, durations already SCALE-scaled) back-to-back at their
    best-case (minimum) durations, that still satisfies
    ``g_{job_id} <= deadline_scaled``?

    This chains reset -> invariant -> delay -> completion-guard for each
    remaining task of the job (the same primitive sequence
    ``symbolic_env.py`` uses for real dispatch), then intersects with the
    deadline guard and checks non-emptiness. Genuinely constructs the DBM
    chain rather than only computing the equivalent scalar arithmetic
    shortcut (``g_J_now + sum(min_durations) <= D_J``) -- see
    ``symbolic_env.py``'s ``_check_feasible`` docstring for why the two are
    provably equivalent here, and ``sanity_gate.py`` / ``calibration.py``
    for where that equivalence is cross-checked rather than assumed.
    """
    Zc = Z
    cname = zctx.c(job_id)
    for _tid, min_d in remaining_tasks:
        Zc = reset_clocks(Zc, zctx, [cname])
        Zc = apply_invariant(Zc, zctx, [(cname, min_d)])
        Zc = Zc.up()
        Zc = apply_invariant(Zc, zctx, [(cname, min_d)])
        Zc = apply_lower_guard(Zc, zctx, [(cname, min_d)])
        if Zc is None or Zc.is_empty():
            return False
    Zc = apply_guard(Zc, zctx, [(zctx.g(job_id), deadline_scaled)])
    return Zc is not None and not Zc.is_empty()
