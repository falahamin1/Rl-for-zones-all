"""Zone operations for symbolic RL using pyudbm.

SPIKE MODIFICATION (see /tmp/SPIKE2.md, Task A): the cost clock `delta` has
been REMOVED from the zone entirely. Clocks are now one per job
(c_J1, c_J2, ...) only. Reference clock (index 0) is implicit in pyudbm.

Consequence: `min_cost()` (which read delta's infimum out of the zone) no
longer has anything to read — it has been deleted. Reward/elapsed-time
accounting now comes directly from the concrete side (symbolic_env.py's
`step()` info dict `"elapsed"` key), mirroring what region_env.py's step()
already returns, not from the zone. See symbolic_rl.py's one changed line.

Consequence (flagged, not fixed — out of scope per spike instructions):
`symbolic_optimal.py` imports and uses `min_cost` extensively; it is no
longer importable/functional after this change. Not touched, per
instructions to change nothing beyond zone_ops.py + one symbolic_rl.py line
+ (necessarily) symbolic_env.py's elapsed-time plumbing.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from pyudbm import Context

if TYPE_CHECKING:
    from pyudbm.binding.udbm import Federation
    from .instance import ProbJobShopInstance

# UDBM's internal infinity sentinel (2^30 - 1)
_UDBM_INF = 1073741823


class ZoneContext:
    """Wraps pyudbm Context for a given job-shop instance.

    SPIKE MODIFICATION: no `delta` clock. `_clock_names` is job clocks only.
    """

    def __init__(self, instance: "ProbJobShopInstance"):
        self._clock_names = [f"c_{j.job_id}" for j in instance.jobs]
        self.ctx = Context(self._clock_names)
        # Map name → Clock object
        self.clocks = {name: getattr(self.ctx, name) for name in self._clock_names}

    @property
    def clock_names(self) -> List[str]:
        return self._clock_names


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
    """Reset named clocks to 0."""
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

    rate is unused for job-shop (always 1, cost tracked by delta growing
    with up()), but included for generality.
    Invariant never mentions delta, so delta remains unbounded above (†
    property holds automatically).
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


def normalize(Z: "Federation", zctx: ZoneContext, M: int) -> "Federation":
    """k-normalization by max constant M.

    SPIKE MODIFICATION: with `delta` removed, EVERY clock (all job clocks)
    is now bounded by M — there is no more clock excluded from
    extrapolation. This ensures the zone graph is finite, same as before,
    but no longer guarantees zones are bounded: k-extrapolation relaxes
    bounds that exceed M to infinity, it does not impose an upper bound
    where none existed — a clock that has never been reset (i.e. a job that
    hasn't dispatched its first task yet) can still be unbounded above at a
    given decision point. See /tmp/SPIKE2.md Task A for the measured
    fraction of zones that are actually bounded after this change.
    """
    bounds = {zctx.clocks[cn]: M for cn in zctx.clock_names}
    return Z.extrapolate_max_bounds(bounds)


def clip_to_k_box(Z: "Federation", zctx: ZoneContext, bounds: Dict[str, int]) -> "Federation":
    """SPIKE ADDITION (Task A): intersect Z with 0 <= x_i <= k_i for each
    clock named in `bounds` (using the SAME k vector passed to
    extrapolate_max_bounds/normalize — pass {cn: M for cn in zctx.clock_names}
    for the standard case). Unlike normalize()/extrapolate_max_bounds (which
    only ever WIDENS/relaxes constraints beyond M to infinity), this is an
    exact intersection — it can only ever SHRINK the zone, and is what
    actually guarantees boundedness (extrapolation alone does not).
    """
    Z_new = Z
    for cname, k in bounds.items():
        clock = zctx.clocks[cname]
        Z_new = Z_new & (clock >= 0) & (clock <= k)
    return Z_new


def dominates(Z_old: "Federation", Z_new: "Federation") -> bool:
    """Z_old dominates Z_new iff Z_new ⊆ Z_old.

    If Z_old dominates Z_new, the old zone is at least as good (represents
    at least all the futures Z_new does, with no higher cost), so Z_new
    need not be explored further.
    From Lemma 4 of the UPTA theory: Z_old ⊑ Z_new  iff  Z_new ⊆ Z_old.
    """
    return Z_new <= Z_old


def is_bounded(dbm) -> bool:
    """SPIKE ADDITION (Task A): True iff every clock i has both bound(0,i)
    and bound(i,0) finite (i.e. the zone is a bounded polyhedron, not
    extending to infinity in any clock's direction).
    """
    dim = dbm.dimension
    for i in range(1, dim):
        if dbm.is_infinity(0, i) or dbm.is_infinity(i, 0):
            return False
    return True
