"""Zone: a small, self-contained wrapper around one canonical pyudbm DBM.

This module is independent of ``prob_jobshop_ext.zone_ops`` — it does not
import or modify it.  It exists to give the encoder study a clean, delta-free
geometric view of a single convex zone (exact half-space representation,
exact vertex enumeration, and a constraint digraph), plus a bounded
"k-box" clip that the raster views in :mod:`zoneviews.raster` build on.

A :class:`Zone` always wraps a single, non-empty, canonical DBM extracted
from a pyudbm ``Federation`` — never a multi-DBM federation. Clock index 0 is
the implicit pyudbm reference clock (always 0); user clocks are indices
``1..d`` in the order given by ``clock_names``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx

from pyudbm import Context, Federation

Number = Union[int, Fraction]

#: Scale factor used to encode a non-integer bound as a pyudbm-compatible
#: ``int``. pyudbm's ``Clock``/``VariableDifference`` comparisons only accept
#: plain Python ``int`` bounds (see ``pyudbm.binding.udbm._is_exact_int``);
#: this repository's job-shop domain only ever produces integer durations, so
#: this path is not exercised in normal use, but it is exercised directly by
#: tests and protects any future non-integer ``k`` value.
SCALE = 20

#: Maximum number of ``C(m, d)`` d-subsets :meth:`Zone.v_rep` will attempt to
#: solve before giving up. Each subset costs one Fraction Gauss-Jordan solve;
#: 200_000 keeps worst-case wall time bounded even for zones with many
#: non-redundant facets.
_VERTEX_ENUM_LIMIT = 200_000


class UnboundedZone(Exception):
    """Raised by :meth:`Zone.v_rep` when some clock has no finite bound in
    at least one direction (i.e. the zone is not confined to a box)."""


class VertexEnumerationLimit(Exception):
    """Raised by :meth:`Zone.v_rep` when ``C(m, d)`` (number of facets choose
    dimension) exceeds :data:`_VERTEX_ENUM_LIMIT`, making exact enumeration
    impractical."""


@dataclass(frozen=True)
class HalfSpace:
    """One non-redundant DBM-derived half-space constraint ``a . x <(=) b``.

    :ivar a: Length-``d`` integer coefficient vector over the real clocks
        (never involves the implicit reference clock, which is always 0).
    :ivar b: Integer bound.
    :ivar strict: Whether the inequality is strict (``<``) rather than
        non-strict (``<=``).
    """

    a: Tuple[int, ...]
    b: int
    strict: bool

    def satisfied_by(self, point: Sequence[Number], *, as_closure: bool = True) -> bool:
        """Return whether ``point`` satisfies this half-space.

        :param point: Length-``d`` point, ``int`` or ``Fraction`` entries.
        :param as_closure: If ``True`` (the default), strictness is ignored
            and the non-strict form ``a . x <= b`` is used instead — this is
            the topological closure of the half-space, which is what vertex
            geometry is defined over. If ``False``, strict half-spaces use
            ``a . x < b``.
        """
        lhs = sum(ai * xi for ai, xi in zip(self.a, point))
        if self.strict and not as_closure:
            return lhs < self.b
        return lhs <= self.b


def _ensure_pyudbm_int(value: Number) -> int:
    """Encode ``value`` as a plain pyudbm-compatible ``int``.

    Plain ``int`` values pass through unchanged. Non-integer values (``float``
    or non-integer ``Fraction``) are scaled by :data:`SCALE` and rounded; a
    :class:`ValueError` is raised if the scaled value is not itself exact,
    since silently truncating a bound would silently change the zone.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    frac = value if isinstance(value, Fraction) else Fraction(value)
    if frac.denominator == 1:
        return int(frac)
    scaled = frac * SCALE
    if scaled.denominator != 1:
        raise ValueError(
            f"{value!r} is not representable as an integer at SCALE={SCALE}; "
            "pyudbm bounds must be exact integers."
        )
    return int(scaled)


class _UnionFind:
    """Minimal union-find over ``0..n-1``, used only by :meth:`Zone.affine_dim`."""

    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[rx] = ry

    def num_components(self) -> int:
        return len({self.find(i) for i in range(len(self._parent))})


def _pinned_offsets(dbm, n: int) -> Tuple[List[int], List[Fraction]]:
    """Weighted union-find over the exact-pinning relation used by
    :meth:`Zone.affine_dim` (``bound(i,j) + bound(j,i) == 0``, both
    non-strict), additionally tracking the *exact offset* to each node's
    class representative rather than only which class it's in.

    Returns ``(root, offset)``, length-``n`` arrays such that
    ``x_i == x_{root[i]} + offset[i]`` exactly for every ``i`` -- in
    particular ``offset[root[i]] == 0``. Used by :meth:`Zone.v_rep` to
    collapse each pinning class (a zero-degrees-of-freedom direction) down
    to a single representative coordinate before vertex enumeration,
    instead of enumerating in the full (mostly redundant, for a
    non-full-dimensional zone) ambient clock space.
    """
    parent = list(range(n))
    offset = [Fraction(0)] * n  # offset[i] = x_i - x_{parent[i]}

    def find(i: int) -> Tuple[int, Fraction]:
        if parent[i] == i:
            return i, Fraction(0)
        root, off_to_parent = find(parent[i])
        total = offset[i] + off_to_parent
        parent[i] = root
        offset[i] = total
        return root, total

    def union(i: int, j: int, diff: Fraction) -> None:
        # diff = x_i - x_j, exact.
        ri, oi = find(i)  # x_i = x_ri + oi
        rj, oj = find(j)  # x_j = x_rj + oj
        if ri == rj:
            return
        # (x_ri + oi) - (x_rj + oj) = diff  =>  x_rj = x_ri - (diff - oi + oj)
        parent[rj] = ri
        offset[rj] = -(diff - oi + oj)

    for i in range(n):
        for j in range(i + 1, n):
            if dbm.is_infinity(i, j) or dbm.is_infinity(j, i):
                continue
            if dbm.is_strict(i, j) or dbm.is_strict(j, i):
                continue
            if dbm.bound(i, j) + dbm.bound(j, i) == 0:
                union(i, j, Fraction(dbm.bound(i, j)))

    roots = [0] * n
    offsets = [Fraction(0)] * n
    for i in range(n):
        r, o = find(i)
        roots[i] = r
        offsets[i] = o
    return roots, offsets


class Zone:
    """A single canonical, non-empty pyudbm DBM plus an associated k-box vector.

    :param federation: A pyudbm ``Federation`` that must contain exactly one
        DBM (i.e. be non-empty and convex). Federations with zero DBMs
        (empty) or more than one DBM (non-convex union) are rejected.
    :param clock_names: Names of the ``d`` real clocks, in the same order as
        ``federation.context.clocks``.
    :param k: Length-``d`` sequence of non-negative integer per-clock box
        bounds. This does not, by itself, clip the zone — it is metadata
        describing the intended encoder-facing box, applied by
        :meth:`clip_to_k_box`.

    Note on ``h_rep`` ordering: :meth:`h_rep` returns constraints sorted by
    ``(i, j)`` DBM index for reproducibility across calls on the *same* zone,
    but callers must not depend on this order being meaningful (e.g. stable
    across different zones or pyudbm versions) — it is an implementation
    detail exposed only so tests can be deterministic.
    """

    __slots__ = ("federation", "dbm", "clock_names", "d", "k", "_h_rep_cache", "_v_rep_cache")

    def __init__(self, federation: Federation, clock_names: Sequence[str], k: Sequence[int]):
        dbms = federation.to_dbm_list()
        if len(dbms) == 0:
            raise ValueError("Zone requires a non-empty Federation (got an empty one).")
        if len(dbms) != 1:
            raise ValueError(
                f"Zone requires a single-DBM (convex) Federation, got {len(dbms)} DBMs; "
                "reduce/convex_hull the federation first if a non-convex union is expected."
            )
        clock_names = tuple(clock_names)
        k = tuple(int(v) for v in k)
        if len(k) != len(clock_names):
            raise ValueError(f"k must have one entry per clock: len(k)={len(k)}, d={len(clock_names)}")
        if dbms[0].dimension != len(clock_names) + 1:
            raise ValueError(
                f"DBM dimension {dbms[0].dimension} does not match len(clock_names)+1={len(clock_names) + 1}"
            )

        self.federation = federation
        self.dbm = dbms[0]
        self.clock_names = clock_names
        self.d = len(clock_names)
        self.k = k
        self._h_rep_cache: Optional[List[HalfSpace]] = None
        self._v_rep_cache: Optional[List[Tuple[Fraction, ...]]] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_h_rep(
        cls,
        h_rep: Sequence[HalfSpace],
        clock_names: Sequence[str],
        k: Sequence[int],
        context_name: Optional[str] = None,
    ) -> "Zone":
        """Rebuild a :class:`Zone` from a half-space list produced by :meth:`h_rep`.

        Starts from the non-negative universe (``x_i >= 0`` for every clock,
        via pyudbm's ``set_init``) and intersects each half-space in turn.
        Every ``HalfSpace.a`` must be a DBM-style difference vector — at most
        one ``+1`` entry and at most one ``-1`` entry, all other entries
        zero — since that is the only form :meth:`h_rep` ever produces; a
        general half-space cannot be represented as a DBM constraint.

        :raises ValueError: If any half-space is not a difference constraint.
        """
        ctx = Context(list(clock_names), name=context_name)
        clocks = {name: getattr(ctx, name) for name in clock_names}
        Z = Federation(ctx).set_init()
        for hs in h_rep:
            Z = Z & _halfspace_to_federation(hs, clock_names, clocks)
        return cls(Z, clock_names, k)

    # ------------------------------------------------------------------
    # Half-space / vertex / graph views
    # ------------------------------------------------------------------

    def h_rep(self) -> List[HalfSpace]:
        """Return the non-redundant half-space representation.

        Every finite off-diagonal DBM entry ``(i, j)`` (``i != j``, both in
        ``0..d`` where ``0`` is the reference clock) yields a candidate
        constraint. Candidates are visited in ``(i, j)`` order and a
        candidate is dropped as redundant iff some third index ``k``
        (``k not in {i, j}``) has *both* ``(i, k)`` and ``(k, j)`` still
        surviving (not yet dropped) with ``bound(i,k) + bound(k,j) ==
        bound(i,j)`` and matching composed strictness (the composed hop is
        strict iff either half-hop is strict) — the standard single-hop DBM
        redundancy test.

        Checking survivors rather than the full closed set matters: DBMs
        with an equality cycle (e.g. two clocks pinned equal by a completed
        reset, each also box-bounded) have every one of those box bounds
        pairwise re-derivable from *some other* box bound plus the
        equality — dropping all of them in one pass would silently discard
        real information (the cycle would no longer reconstruct the same
        DBM). Greedily removing against the shrinking survivor set instead
        guarantees at least one witness of each derivable fact is kept.
        """
        if self._h_rep_cache is not None:
            return self._h_rep_cache

        dbm = self.dbm
        n = self.d + 1  # includes reference clock 0

        finite = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if not dbm.is_infinity(i, j):
                    finite[(i, j)] = (dbm.bound(i, j), dbm.is_strict(i, j))

        alive = dict(finite)
        for (i, j), (bound, strict) in sorted(finite.items()):
            redundant = False
            for kk in range(n):
                if kk == i or kk == j:
                    continue
                ik = alive.get((i, kk))
                kj = alive.get((kk, j))
                if ik is None or kj is None:
                    continue
                b_ik, s_ik = ik
                b_kj, s_kj = kj
                if b_ik + b_kj == bound and (s_ik or s_kj) == strict:
                    redundant = True
                    break
            if redundant:
                del alive[(i, j)]

        result: List[HalfSpace] = []
        for (i, j), (bound, strict) in sorted(finite.items()):
            if (i, j) not in alive:
                continue

            a = [0] * self.d
            if i != 0:
                a[i - 1] += 1
            if j != 0:
                a[j - 1] -= 1
            result.append(HalfSpace(a=tuple(a), b=bound, strict=strict))

        self._h_rep_cache = result
        return result

    def v_rep(self) -> List[Tuple[Fraction, ...]]:
        """Return exact vertices of the zone as tuples of :class:`Fraction`.

        Computes the affine dimension *first* and enumerates within that
        affine hull, not the full ``d``-dimensional ambient clock space:
        every clock is either pinned to a fixed constant (exact-equality
        chain back to the reference clock) or pinned to some other clock up
        to a known constant offset (:func:`_pinned_offsets`, a weighted
        union-find over the same exact-pinning relation
        :meth:`affine_dim` counts components of). Each non-reference pinning
        class collapses to exactly one free coordinate; every ``h_rep()``
        half-space is rewritten in terms of those free coordinates (a
        constraint between two clocks in the same class becomes a checked
        constant; one involving a fixed-class clock loses that term into
        the bound). Vertex enumeration then runs on this reduced system --
        ``C(m', affine_dim)`` subsets, not ``C(m, d)`` -- with exact
        :class:`Fraction` Gauss-Jordan elimination (never
        ``scipy.spatial.qhull``, which has produced wrong answers on this
        input class before), and each reduced solution is expanded back to
        a full ``d``-dimensional point before being returned.

        At ``affine_dim() == 0`` every clock is a fixed constant (its exact
        pinned offset from the reference clock) -- this is a single point,
        returned directly with no enumeration at all, since C(m, 0) would
        be nonsensical and C(m, d) (the pre-fix behavior) needlessly
        exploded on the ambient dimension even though the answer is O(1) to
        read straight off the DBM.

        :raises UnboundedZone: If any clock lacks a finite bound in both
            directions — call :meth:`clip_to_k_box` first.
        :raises VertexEnumerationLimit: If ``C(m', affine_dim) >
            200_000`` (``m'`` = number of non-trivial reduced constraints)
            — exact enumeration would be impractical.
        """
        if self._v_rep_cache is not None:
            return self._v_rep_cache

        if not self.is_bounded():
            raise UnboundedZone(
                "v_rep() requires every clock to have a finite bound in both directions; "
                "call clip_to_k_box() first."
            )

        n = self.d + 1
        roots, offs = _pinned_offsets(self.dbm, n)
        zero_root = roots[0]
        free_roots = sorted({r for r in roots} - {zero_root})
        dim = len(free_roots)
        assert dim == self.affine_dim(), (
            f"v_rep()/affine_dim() disagree on dimension ({dim} vs {self.affine_dim()}) -- "
            "this is an internal inconsistency, not a data problem."
        )

        if dim == 0:
            # Every clock is pinned straight back to the reference clock
            # (x_0 == 0 identically), so x_i == offs[i] exactly.
            point = tuple(offs[i] for i in range(1, n))
            self._v_rep_cache = [point]
            return self._v_rep_cache

        free_index = {r: idx for idx, r in enumerate(free_roots)}

        reduced: List[HalfSpace] = []
        for hs in self.h_rep():
            new_a = [0] * dim
            new_b = Fraction(hs.b)
            for real_idx in range(1, n):
                coeff = hs.a[real_idx - 1]
                if coeff == 0:
                    continue
                r = roots[real_idx]
                if r == zero_root:
                    new_b -= coeff * offs[real_idx]
                else:
                    new_a[free_index[r]] += coeff
                    new_b -= coeff * offs[real_idx]
            if all(c == 0 for c in new_a):
                continue  # constant after substitution -- automatically true given DBM consistency
            assert new_b.denominator == 1, f"non-integer reduced bound {new_b} -- offsets should always be exact integers here"
            reduced.append(HalfSpace(a=tuple(new_a), b=int(new_b), strict=hs.strict))

        m = len(reduced)
        if m < dim:
            self._v_rep_cache = []
            return self._v_rep_cache

        n_subsets = math.comb(m, dim)
        if n_subsets > _VERTEX_ENUM_LIMIT:
            raise VertexEnumerationLimit(
                f"C({m}, {dim}) = {n_subsets} exceeds the {_VERTEX_ENUM_LIMIT} exact-enumeration limit."
            )

        reduced_vertices: List[Tuple[Fraction, ...]] = []
        seen = set()
        for subset in combinations(reduced, dim):
            point = _solve_square_system(subset, dim)
            if point is None:
                continue
            if not all(hs.satisfied_by(point, as_closure=True) for hs in reduced):
                continue
            if point not in seen:
                seen.add(point)
                reduced_vertices.append(point)

        vertices: List[Tuple[Fraction, ...]] = []
        for yvec in reduced_vertices:
            full = []
            for real_idx in range(1, n):
                r = roots[real_idx]
                if r == zero_root:
                    full.append(Fraction(offs[real_idx]))
                else:
                    full.append(yvec[free_index[r]] + offs[real_idx])
            vertices.append(tuple(full))

        self._v_rep_cache = vertices
        return vertices

    def graph(self) -> "nx.DiGraph":
        """Return the constraint digraph: ``d + 1`` nodes, one edge per finite entry.

        Node ``0`` is the reference clock; nodes ``1..d`` are the real clocks
        in ``clock_names`` order. Every node carries a ``clock_index`` attribute
        equal to its own index (identity labeling; :func:`zoneviews.raster.graph_image`
        may permute *rendered positions* without touching this graph). Every
        edge ``i -> j`` corresponds to one finite DBM entry ``(i, j)`` and
        carries ``bound`` (int) and ``strict`` (bool) attributes. The diagonal
        is never included (it is always the trivial ``0 <= 0``).
        """
        g = nx.DiGraph()
        n = self.d + 1
        for i in range(n):
            g.add_node(i, clock_index=i)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if not self.dbm.is_infinity(i, j):
                    g.add_edge(i, j, bound=self.dbm.bound(i, j), strict=self.dbm.is_strict(i, j))
        return g

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def clip_to_k_box(self, k: Optional[Sequence[int]] = None) -> "Zone":
        """Return a new :class:`Zone` intersected with ``0 <= x_i <= k_i``.

        :param k: Override box bounds; defaults to ``self.k``.
        :return: A new, still-single-DBM, bounded ``Zone`` sharing ``self``'s
            pyudbm ``Context``.
        """
        k = tuple(int(v) for v in (k if k is not None else self.k))
        if len(k) != self.d:
            raise ValueError(f"k must have one entry per clock: len(k)={len(k)}, d={self.d}")

        ctx = self.dbm.context
        clocks = {name: getattr(ctx, name) for name in self.clock_names}
        Z = self.federation
        for name, ki in zip(self.clock_names, k):
            clock = clocks[name]
            Z = Z & (clock <= _ensure_pyudbm_int(ki)) & (clock >= 0)
        return Zone(Z, self.clock_names, k)

    # ------------------------------------------------------------------
    # Scalar/boolean properties
    # ------------------------------------------------------------------

    def affine_dim(self) -> int:
        """Return the affine dimension: ``(connected components) - 1`` under
        the union-find where clocks ``i, j`` (including the reference clock
        ``0``) are pinned together iff ``bound(i,j) + bound(j,i) == 0`` with
        both directions non-strict.
        """
        n = self.d + 1
        uf = _UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if self.dbm.is_infinity(i, j) or self.dbm.is_infinity(j, i):
                    continue
                if self.dbm.is_strict(i, j) or self.dbm.is_strict(j, i):
                    continue
                if self.dbm.bound(i, j) + self.dbm.bound(j, i) == 0:
                    uf.union(i, j)
        return uf.num_components() - 1

    def is_bounded(self) -> bool:
        """Return whether every real clock has a finite bound in both directions."""
        for i in range(1, self.d + 1):
            if self.dbm.is_infinity(i, 0) or self.dbm.is_infinity(0, i):
                return False
        return True

    def facet_count(self) -> int:
        """Return the number of non-redundant constraints (``len(h_rep())``)."""
        return len(self.h_rep())

    def contains(self, point: Union[Sequence[Number], Mapping[str, Number]]) -> bool:
        """Return whether ``point`` satisfies every DBM constraint exactly.

        :param point: Either a length-``d`` sequence (in ``clock_names``
            order) or a mapping from clock name to value. Values may be
            ``int`` or :class:`Fraction`; comparisons are exact.
        """
        if isinstance(point, Mapping):
            values: List[Number] = [point[name] for name in self.clock_names]
        else:
            values = list(point)
            if len(values) != self.d:
                raise ValueError(f"point must have {self.d} entries, got {len(values)}")

        full = [Fraction(0)] + [v if isinstance(v, Fraction) else Fraction(v) for v in values]
        n = self.d + 1
        for i in range(n):
            for j in range(n):
                if i == j or self.dbm.is_infinity(i, j):
                    continue
                diff = full[i] - full[j]
                bound = self.dbm.bound(i, j)
                if self.dbm.is_strict(i, j):
                    if not diff < bound:
                        return False
                else:
                    if not diff <= bound:
                        return False
        return True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _halfspace_to_federation(hs: HalfSpace, clock_names: Sequence[str], clocks: Dict[str, "object"]) -> Federation:
    """Turn one DBM-derived :class:`HalfSpace` back into a pyudbm ``Federation``."""
    nonzero = [(idx, coeff) for idx, coeff in enumerate(hs.a) if coeff != 0]
    b = _ensure_pyudbm_int(hs.b)

    if len(nonzero) == 0:
        raise ValueError("A zero coefficient vector is not a valid DBM half-space.")
    if len(nonzero) == 1:
        idx, coeff = nonzero[0]
        clock = clocks[clock_names[idx]]
        if coeff == 1:
            return clock < b if hs.strict else clock <= b
        if coeff == -1:
            return clock > -b if hs.strict else clock >= -b
        raise ValueError(f"Non-unit coefficient {coeff} is not a valid DBM half-space.")
    if len(nonzero) == 2:
        (idx_p, coeff_p), (idx_q, coeff_q) = nonzero
        if {coeff_p, coeff_q} != {1, -1}:
            raise ValueError("A two-clock DBM half-space must have coefficients +1 and -1.")
        if coeff_p == -1:
            (idx_p, coeff_p), (idx_q, coeff_q) = (idx_q, coeff_q), (idx_p, coeff_p)
        left = clocks[clock_names[idx_p]]
        right = clocks[clock_names[idx_q]]
        diff = left - right
        return diff < b if hs.strict else diff <= b
    raise ValueError(f"Half-space with {len(nonzero)} nonzero coefficients is not a DBM constraint.")


def _solve_square_system(constraints: Sequence[HalfSpace], d: int) -> Optional[Tuple[Fraction, ...]]:
    """Solve the ``d x d`` linear system ``a_1 . x = b_1, ...`` exactly.

    Returns ``None`` if the system is singular (rank < d).
    """
    # Augmented matrix, Fraction entries: rows = constraints, cols = d + 1 (last is RHS).
    rows = [[Fraction(ai) for ai in hs.a] + [Fraction(hs.b)] for hs in constraints]

    for col in range(d):
        pivot_row = None
        for r in range(col, d):
            if rows[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            return None
        rows[col], rows[pivot_row] = rows[pivot_row], rows[col]

        pivot_val = rows[col][col]
        rows[col] = [v / pivot_val for v in rows[col]]

        for r in range(d):
            if r == col:
                continue
            factor = rows[r][col]
            if factor != 0:
                rows[r] = [rv - factor * cv for rv, cv in zip(rows[r], rows[col])]

    return tuple(rows[i][d] for i in range(d))
