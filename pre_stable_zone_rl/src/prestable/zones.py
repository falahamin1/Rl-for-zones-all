"""Convex zones: a thin adapter over pyudbm. This is the only module that imports pyudbm.

pyudbm API mapping (verified against the installed pyudbm 0.0.1 source + experiments):

    our operation               -> pyudbm call
    --------------------------  ------------------------------------------------------------
    clock context               -> pyudbm.Context(clock_names); clock lookup ctx[name]
    atomic constraint l - r ~ c -> Federation(pyudbm.Constraint(cl, cr, c, strict)),
                                   with cl/cr = None for the zero clock "0"
                                   (the result already satisfies all clocks >= 0)
    zero()  {0}                 -> ctx.get_zero_federation()          (== Federation(ctx))
    universe()  clocks >= 0     -> Federation(ctx).set_init()         (set_init mutates in place)
    empty zone                  -> f - f
    intersect                   -> f & g
    up / down                   -> f.up() / f.down()                  (return copies)
    reset x := 0                -> f.update_value(clock, 0)           (alias f.reset_value(clock))
    free_x                      -> f.free_clock(clock)
    subtraction                 -> f - g exists, but `subtract` splits along constraints itself
                                   (UDBM's pieces can break K-boundedness; see `subtract`)
    inclusion / equality        -> f <= g / f == g                    (semantic, same Context required)
    emptiness, number of DBMs   -> f.is_empty(), f.get_size()
    canonical DBM matrix        -> f.to_dbm_list()[0].to_matrix("tuple" | "raw");
                                   cells also via DBM.bound / is_strict / is_infinity
                                   (DBMs are kept closed, i.e. canonical, by UDBM)
    extrapolation (one K)       -> f.extrapolate_max_bounds({clock: K for every clock})
                                   (all clocks must be given; matches the spec's definition,
                                   checked by tests against a matrix reference implementation)
    valuations                  -> v = FloatValuation(ctx); v["x"] = 0.5; f.contains(v)
    union / convex hull         -> f | g / f.convex_hull()  (matches the entry-wise maximum of the
                                   canonical DBMs; checked by tests against `_hull_by_matrix`)

There is no public "Federation from a DBM matrix" constructor, so a single DBM is turned
back into a zone by intersecting the atomic constraints of its finite matrix entries.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

from pyudbm import Constraint as _UConstraint
from pyudbm import Context as _UContext
from pyudbm import Federation as _Federation
from pyudbm import FloatValuation as _FloatValuation

from .model import ZERO, Constraint, PEdge

_INF = float("inf")
_RAW_INF = ((2 ** 31 - 1) >> 1) << 1  # UDBM's raw encoding of (<, ∞)
# A decoded DBM cell: (bound, strict); bound may be inf.
_Cell = tuple[float, bool]


class ZoneContext:
    """The clock set zones live over. Zones from different contexts never mix."""

    def __init__(self, clocks: Iterable[str]):
        self.clocks: tuple[str, ...] = tuple(clocks)
        if ZERO in self.clocks:
            raise ValueError(f"clock name {ZERO!r} is reserved for the zero clock")
        if len(set(self.clocks)) != len(self.clocks):
            raise ValueError("duplicate clock names")
        self._ctx = _UContext(self.clocks)
        # Matrix index i <-> name; index 0 is the zero clock.
        self._names: tuple[str, ...] = (ZERO,) + self.clocks
        self._universe = _Federation(self._ctx).set_init()

    def _clock(self, name: str):
        if name == ZERO:
            return None
        if name not in self.clocks:
            raise ValueError(f"unknown clock {name!r}")
        return self._ctx[name]

    def _atom(self, left: str, right: str, bound: int, strict: bool) -> _Federation:
        return _Federation(_UConstraint(self._clock(left), self._clock(right), int(bound), strict))

    def _fed_of(self, constraints: Iterable[Constraint]) -> _Federation:
        fed = self._universe
        for g in constraints:
            if g.left == g.right:
                raise ValueError(f"constraint {g} compares a clock with itself")
            fed = fed & self._atom(g.left, g.right, g.bound, g.strict)
        return fed

    def zero(self) -> "Zone":
        return Zone(self, self._ctx.get_zero_federation())

    def universe(self) -> "Zone":
        return Zone(self, self._universe)

    def empty(self) -> "Zone":
        return Zone(self, self._universe - self._universe)

    def from_guard(self, guard: Iterable[Constraint]) -> "Zone":
        """⟦guard⟧ ∩ universe for a conjunction of constraints."""
        return Zone(self, self._fed_of(guard))


class Zone:
    """An immutable convex zone (possibly empty) over a ZoneContext."""

    # _own, _desc and _constraints cache _own_bound, _description and constraints() (zones are never mutated).
    __slots__ = ("ctx", "_fed", "_key", "_own", "_desc", "_constraints")

    def __init__(self, ctx: ZoneContext, fed: _Federation):
        fed = fed & ctx._universe
        assert fed.get_size() <= 1, f"zone must be convex, got {fed.get_size()} DBMs: {fed}"
        self.ctx = ctx
        self._fed = fed
        # Canonical (closed) DBM matrix; None for the empty zone. Unique per nonempty zone.
        self._key: Optional[tuple[int, ...]] = (
            None if fed.is_empty() else tuple(v for row in fed.to_dbm_list()[0].to_matrix("raw") for v in row)
        )
        self._own: Optional[int] = None
        self._desc: Optional[tuple[tuple[int, int, int, bool], ...]] = None
        self._constraints: Optional[tuple[Constraint, ...]] = None

    @property
    def is_empty(self) -> bool:
        return self._key is None

    def key(self) -> tuple[int, ...]:
        """Hashable, totally ordered canonical key, for deterministic sorting only.

        The raw-encoded canonical DBM matrix (UDBM encodes (c, ≺) as 2c + [≺ is ≤]); the
        empty zone gets the key (), which sorts before every nonempty zone.
        """
        return () if self._key is None else self._key

    def _check(self, other: "Zone") -> None:
        if not isinstance(other, Zone) or other.ctx is not self.ctx:
            raise ValueError("zones must share the same ZoneContext")

    def __le__(self, other: "Zone") -> bool:
        self._check(other)
        return self._fed <= other._fed

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Zone) or other.ctx is not self.ctx:
            return NotImplemented
        return self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)

    def matrix(self) -> list[list[_Cell]]:
        """Canonical DBM as (bound, strict) cells; row/column 0 is the zero clock."""
        if self.is_empty:
            raise ValueError("the empty zone has no canonical DBM")
        return [[(float(b) if b == _INF else b, op == "<") for op, b in row]
                for row in self._fed.to_dbm_list()[0].to_matrix("tuple")]

    def constraints(self) -> tuple[Constraint, ...]:
        """Constraints whose conjunction (with clocks >= 0) is exactly this zone.

        The entries of `_description` (all constants within the zone's own bound K_z),
        minus the trivial lower bounds x >= 0. The empty zone gives (c <= -1,) for the first
        clock c, which is unsatisfiable together with c >= 0.
        """
        if self.is_empty:
            # Unsatisfiable together with clocks >= 0 (needs at least one clock).
            return (Constraint(self.ctx.clocks[0], ZERO, -1, False),)
        if self._constraints is None:
            names = self.ctx._names
            self._constraints = tuple(
                Constraint(names[i], names[j], c, strict)
                for i, j, c, strict in _description(self)
                if not (i == 0 and c == 0 and not strict)
            )
        return self._constraints

    def contains(self, point: Mapping[str, float]) -> bool:
        """Whether the valuation ``point`` (a value for every clock) lies in the zone."""
        v = _FloatValuation(self.ctx._ctx)
        for c in self.ctx.clocks:
            v[c] = float(point[c])
        return self._fed.contains(v)

    def __repr__(self) -> str:
        return f"Zone({self})"

    def __str__(self) -> str:
        if self.is_empty:
            return "false"
        m, names = self.matrix(), self.ctx._names
        parts: list[str] = []
        for i in range(1, len(names)):
            parts.extend(_render_range(names[i], _neg(m[0][i]), m[i][0], trivial_lower=(0, False)))
        for i in range(1, len(names)):
            for j in range(i + 1, len(names)):
                # Show a diagonal side only if the clock bounds do not already imply it.
                upper = m[i][j] if _lt(m[i][j], _add(m[i][0], m[0][j])) else (_INF, True)
                lower = _neg(m[j][i]) if _lt(m[j][i], _add(m[j][0], m[0][i])) else (-_INF, True)
                parts.extend(_render_range(f"{names[i]} - {names[j]}", lower, upper, trivial_lower=None))
        return " and ".join(parts) if parts else "true"


# --- zone operations --------------------------------------------------------

def intersect(z: Zone, w: Zone) -> Zone:
    z._check(w)
    return Zone(z.ctx, z._fed & w._fed)


def up(z: Zone) -> Zone:
    return Zone(z.ctx, z._fed.up())


def down(z: Zone) -> Zone:
    return Zone(z.ctx, z._fed.down())


def reset(z: Zone, R: Iterable[str]) -> Zone:
    """Set every clock in R to 0."""
    fed = z._fed
    for x in R:
        fed = fed.update_value(z.ctx._clock(x), 0)
    return Zone(z.ctx, fed)


def pre_edge(z_target: Zone, edge: PEdge, branch: int = 0) -> Zone:
    """{v | v ⊨ guard, v[R:=0] ∈ z_target} = ⟦guard⟧ ∩ free_R(z_target ∩ {x = 0 | x ∈ R}) ∩ universe,
    with R the resets of the given branch of the edge."""
    ctx = z_target.ctx
    fed = z_target._fed
    resets = edge.branches[branch].resets
    for x in resets:
        fed = fed & ctx._atom(x, ZERO, 0, False)
    for x in resets:
        fed = fed.free_clock(ctx._clock(x))
    return Zone(ctx, fed & ctx._fed_of(edge.guard))


def extrapolate(z: Zone, K: int) -> Zone:
    """Classical max-bound extrapolation with one bound K for all clocks.

    On the canonical DBM every entry (c, ≺) with c > K becomes ∞ and every entry with
    c < -K becomes (<, -K), diagonal entries included; the result is re-canonicalised.

    Uses UDBM's extrapolate_max_bounds, which tests/test_zones.py checks against the
    matrix definition (`_extrapolate_by_matrix`) on random zones.
    """
    return Zone(z.ctx, z._fed.extrapolate_max_bounds({c: int(K) for c in z.ctx.clocks}))


def _extrapolate_by_matrix(z: Zone, K: int) -> Zone:
    """Reference implementation of `extrapolate`, straight from the definition."""
    if z.is_empty:
        return z
    ctx, m = z.ctx, z.matrix()
    fed = ctx._universe
    for i, row in enumerate(m):
        for j, (c, strict) in enumerate(row):
            if i == j or c == _INF or c > K:
                continue
            if c < -K:
                c, strict = -K, True
            fed = fed & ctx._atom(ctx._names[i], ctx._names[j], int(c), strict)
    return Zone(ctx, fed)


def _zone_of_dbm(ctx: ZoneContext, rows) -> Zone:
    """Rebuild a zone from a DBM given as rows of (op, bound) cells (pyudbm "tuple" format)."""
    fed = ctx._universe
    for i, row in enumerate(rows):
        for j, (op, c) in enumerate(row):
            if i != j and c != _INF:
                fed = fed & ctx._atom(ctx._names[i], ctx._names[j], c, op == "<")
    return Zone(ctx, fed)


def _own_bound(w: Zone) -> int:
    """Smallest K such that w is K-bounded (cached on w).

    w is `top`-bounded for its largest finite |entry| `top`, and K-boundedness is monotone
    in K (a larger K extrapolates less), so a binary search over [0, top] finds it.
    """
    if w._own is None:
        lo = 0
        hi = max((abs(int(c)) for row in w.matrix() for c, _ in row if c != _INF), default=0)
        while lo < hi:
            mid = (lo + hi) // 2
            if is_k_bounded(w, mid):
                hi = mid
            else:
                lo = mid + 1
        w._own = lo
    return w._own


def _description(w: Zone) -> tuple[tuple[int, int, int, bool], ...]:
    """Constraints (i, j, c, strict) on matrix indices defining w, all with |c| <= _own_bound(w).

    These are the entries of w's canonical DBM after extrapolation with K = _own_bound(w)
    (which leaves w unchanged): entries above K are dropped as implied, entries below -K
    become (<, -K). Cached on w.
    """
    if w._desc is None:
        k, m = _own_bound(w), w.matrix()
        out = []
        for i, row in enumerate(m):
            for j, (c, strict) in enumerate(row):
                if i == j or c == _INF or c > k:
                    continue
                if c < -k:
                    c, strict = -k, True
                out.append((i, j, int(c), strict))
        w._desc = tuple(out)
    return w._desc


def subtract(z: Zone, w: Zone) -> list[Zone]:
    """Nonempty, pairwise disjoint convex zones whose union is exactly z \\ w.

    Splits z along the constraints c1..cn of `_description(w)`: the pieces are
    z ∧ c1 ∧ … ∧ c(i-1) ∧ ¬ci. Every constant used comes from z or has |c| <= K_w, so
    if z and w are K-bounded, so is every piece. (UDBM's own federation subtraction does
    not guarantee this; it may split along implied bounds larger than K.)
    """
    z._check(w)
    if intersect(z, w).is_empty:
        return [] if z.is_empty else [z]
    if z <= w:
        return []
    ctx, names = z.ctx, z.ctx._names
    pieces, running = [], z._fed
    for i, j, c, strict in _description(w):
        # ¬(x_i - x_j ≺ c)  ==  x_j - x_i ≺' -c, with ≺' = ≤ if ≺ is < and vice versa.
        piece = Zone(ctx, running & ctx._atom(names[j], names[i], -c, not strict))
        if not piece.is_empty:
            pieces.append(piece)
        running = running & ctx._atom(names[i], names[j], c, strict)
        if running.is_empty():
            break
    return pieces


def subtract_all(z: Zone, ws: Iterable[Zone]) -> list[Zone]:
    """Nonempty, pairwise disjoint convex zones whose union is exactly z \\ (⋃ ws).

    Built from `subtract`, so it inherits its K-boundedness guarantee.
    """
    pieces = [] if z.is_empty else [z]
    for w in ws:
        pieces = [q for p in pieces for q in subtract(p, w)]
    return pieces


def hull(zones: Iterable[Zone]) -> Zone:
    """Smallest zone containing all given zones (empty if they all are; at least one required).

    On canonical DBMs this is the entry-wise maximum (≤ beats < at equal bounds). Uses
    UDBM's convex_hull, which tests check against `_hull_by_matrix`.
    """
    zones = list(zones)
    if not zones:
        raise ValueError("hull() needs at least one zone (to know the context)")
    fed = zones[0]._fed
    for z in zones[1:]:
        zones[0]._check(z)
        fed = fed | z._fed
    return Zone(zones[0].ctx, fed.convex_hull())


def _hull_by_matrix(zones: Iterable[Zone]) -> Zone:
    """Reference implementation of `hull`: entry-wise maximum of the canonical DBMs."""
    zones = [z for z in zones if not z.is_empty]
    ctx = zones[0].ctx
    # In UDBM's raw encoding 2c + [≤], integer order is exactly the bound order.
    raw = [max(cells) for cells in zip(*(z.key() for z in zones))]
    dim = len(ctx._names)
    rows = [[_decode_raw(raw[i * dim + j]) for j in range(dim)] for i in range(dim)]
    return _zone_of_dbm(ctx, rows)


def _decode_raw(raw: int) -> tuple[str, float]:
    """UDBM raw cell 2c + [≤] -> (op, c), in pyudbm's "tuple" format."""
    if raw == _RAW_INF:
        return ("<", _INF)
    return ("<" if raw & 1 == 0 else "<=", raw >> 1)


def union_is_convex(zones: Iterable[Zone]) -> bool:
    """Whether the union of the zones is itself a zone, i.e. equals their hull."""
    zones = [z for z in zones if not z.is_empty]
    if not zones:
        return True
    return not subtract_all(hull(zones), zones)


def is_k_bounded(z: Zone, K: int) -> bool:
    return extrapolate(z, K) == z


# --- rendering helpers ------------------------------------------------------

def _neg(cell: _Cell) -> _Cell:
    return (-cell[0], cell[1])


def _add(a: _Cell, b: _Cell) -> _Cell:
    return (a[0] + b[0], a[1] or b[1])


def _lt(a: _Cell, b: _Cell) -> bool:
    """Whether upper bound a is strictly tighter than upper bound b."""
    return a[0] < b[0] or (a[0] == b[0] and a[1] and not b[1])


def _fmt(c: float) -> str:
    return str(int(c))


def _render_range(term: str, lower: _Cell, upper: _Cell, trivial_lower: Optional[_Cell]) -> list[str]:
    """Render lower ≺ term ≺ upper, where lower = (value, strict) means term > / >= value."""
    if not upper[1] and not lower[1] and upper[0] == lower[0]:
        return [f"{term} == {_fmt(upper[0])}"]
    has_upper = upper[0] != _INF
    has_lower = lower[0] != -_INF and lower != trivial_lower
    lo = f"{_fmt(lower[0])} {'<' if lower[1] else '<='} " if has_lower else ""
    hi = f" {'<' if upper[1] else '<='} {_fmt(upper[0])}" if has_upper else ""
    if not (lo or hi):
        return []
    if lo and not hi:
        return [f"{term} {'>' if lower[1] else '>='} {_fmt(lower[0])}"]
    return [f"{lo}{term}{hi}"]
