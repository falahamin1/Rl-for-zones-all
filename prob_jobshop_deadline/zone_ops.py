"""Zone operations for Track 2 (per-job deadlines), pyudbm-backed.

Clocks: TWO per job -- c_J (per-task, resets on every task start, unchanged
behavior from the no-deadline model) and g_J (per-job, NEVER resets, starts
at 0, advances with time -- introduced for deadline/release checks).
Reference clock (index 0) is implicit in pyudbm. Clock count per instance is
therefore 2 * n_jobs.

``delta`` (the no-deadline model's never-resetting global cost clock) does
NOT exist in this module at all -- elapsed time and reward are read off the
concrete side (mirroring ``zoneviews.collect`` / ``region_env.py:86-111``'s
convention), and ``normalize()``/``min_cost()`` (which hard-coded ``delta``
in the no-deadline ``zone_ops.py``) are dropped rather than adapted. Bounded
views for geometry work use ``zoneviews.zone.Zone.clip_to_k_box()`` instead
of k-normalization/extrapolation.

Deadlines are checked against g_J, not c_J: g_J is wall-clock time since
episode start, exactly what "job J must finish by absolute time D_J" means.
c_J resets per task and would make a "deadline" comparison meaningless
after the first reset -- this was a real bug in an earlier attempt.
"""
from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING

from pyudbm import Context

if TYPE_CHECKING:
    from pyudbm.binding.udbm import Federation
    from .instance import ProbJobShopInstance


class ZoneContext:
    """Wraps a pyudbm Context with c_J + g_J clocks for a job-shop instance."""

    def __init__(self, instance: "ProbJobShopInstance"):
        self._c_names = [f"c_{j.job_id}" for j in instance.jobs]
        self._g_names = [f"g_{j.job_id}" for j in instance.jobs]
        self._clock_names = self._c_names + self._g_names
        self.ctx = Context(self._clock_names)
        self.clocks = {name: getattr(self.ctx, name) for name in self._clock_names}

    @property
    def clock_names(self) -> List[str]:
        return self._clock_names

    @property
    def c_names(self) -> List[str]:
        return self._c_names

    @property
    def g_names(self) -> List[str]:
        return self._g_names


# ---------------------------------------------------------------------------
# Zone operations — all return a new Federation, never mutate in place
# ---------------------------------------------------------------------------

def init_zone(zctx: ZoneContext, inv_pairs: List[Tuple[str, int]]) -> "Federation":
    """Z†_0: zero zone intersected with initial invariant."""
    Z = zctx.ctx.get_zero_federation()
    return apply_invariant(Z, zctx, inv_pairs)


def apply_guard(
    Z: "Federation",
    zctx: ZoneContext,
    guards: List[Tuple[str, int]],
) -> Optional["Federation"]:
    """Intersect zone with guard constraints.

    guards: list of (clock_name, upper_bound) representing c <= bound.
    Returns None if the intersection is empty (guard unsatisfiable).
    """
    Z_new = Z
    for cname, bound in guards:
        Z_new = Z_new & (zctx.clocks[cname] <= bound)
    if Z_new.is_empty():
        return None
    return Z_new


def reset_clocks(
    Z: "Federation",
    zctx: ZoneContext,
    clock_names: List[str],
) -> "Federation":
    """Reset named clocks to 0.  No g_J clock may ever be reset.

    Same assertion pattern the no-deadline zone_ops.py uses for delta
    (``assert "delta" not in clock_names``) -- here generalized to the
    whole g_J family, since ANY g_J being reset would silently make its
    deadline check meaningless (this was a real bug in an earlier attempt).
    """
    reset_g = set(clock_names) & set(zctx.g_names)
    assert not reset_g, f"g_J clocks must never be reset, got: {sorted(reset_g)}"
    Z_new = Z
    for cname in clock_names:
        Z_new = Z_new.reset_value(zctx.clocks[cname])
    return Z_new


def apply_delay(
    Z: "Federation",
    zctx: ZoneContext,
    inv_pairs: List[Tuple[str, int]],
    rate: int,
) -> "Federation":
    """Delay operator: up() followed by invariant intersection.

    ``rate`` is unused (kept for call-signature parity with the no-deadline
    zone_ops.py). Invariants only ever constrain c_J (task-duration upper
    bounds), never g_J directly as a blanket invariant -- see
    ``deadline_env.py`` for why an unconditional g_J <= D_J invariant would
    deadlock the automaton by blocking time elapse. g_J bounds are applied
    explicitly, only on dispatch edges, via ``apply_guard``-style
    intersection in ``deadline_env.py``.
    """
    Z_new = Z.up()
    return apply_invariant(Z_new, zctx, inv_pairs)


def apply_invariant(
    Z: "Federation",
    zctx: ZoneContext,
    inv_pairs: List[Tuple[str, int]],
) -> "Federation":
    """Intersect zone with invariant upper bounds."""
    Z_new = Z
    for cname, bound in inv_pairs:
        Z_new = Z_new & (zctx.clocks[cname] <= bound)
    return Z_new


def dominates(Z_old: "Federation", Z_new: "Federation") -> bool:
    """Z_old dominates Z_new iff Z_new ⊆ Z_old (Lemma 4 of the UPTA theory)."""
    return Z_new <= Z_old
