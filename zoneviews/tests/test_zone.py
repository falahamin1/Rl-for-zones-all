"""Tests for :mod:`zoneviews.zone`.

Covers required items 1, 2, 3, 6, and 9 from the encoder-study spec.
"""
from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from pyudbm import Context, Federation
from zoneviews.zone import HalfSpace, UnboundedZone, VertexEnumerationLimit, Zone


def _box_zone(bounds, names=None):
    """Build a Zone as an independent axis-aligned box: bounds[i] = (lo, hi)."""
    names = names or [f"x{i}" for i in range(len(bounds))]
    ctx = Context(names)
    Z = Federation(ctx).set_init()
    for name, (lo, hi) in zip(names, bounds):
        clock = getattr(ctx, name)
        Z = Z & (clock >= lo) & (clock <= hi)
    k = [hi + 1 for _, hi in bounds]
    return Zone(Z, names, k)


# ---------------------------------------------------------------------------
# Item 1: h_rep round trip
# ---------------------------------------------------------------------------

def _dbms_equal(z1: Zone, z2: Zone) -> bool:
    # Federation.__eq__ requires the *same* pyudbm Context object, which a
    # rebuilt zone never shares — compare the canonical matrix instead.
    return z1.dbm.to_matrix() == z2.dbm.to_matrix()


@pytest.mark.parametrize(
    "bounds",
    [
        [(2, 5)],
        [(2, 5), (1, 4)],
        [(0, 3), (0, 3), (0, 3)],
    ],
)
def test_h_rep_roundtrip_box(bounds):
    zone = _box_zone(bounds)
    rebuilt = Zone.from_h_rep(zone.h_rep(), zone.clock_names, zone.k)
    assert _dbms_equal(zone, rebuilt)


def test_h_rep_roundtrip_with_diagonal_constraint():
    ctx = Context(["x", "y"])
    Z = (ctx.x >= 2) & (ctx.x <= 5) & (ctx.y >= 1) & (ctx.y <= 4) & (ctx.x - ctx.y <= 3)
    zone = Zone(Z, ["x", "y"], k=[10, 10])
    rebuilt = Zone.from_h_rep(zone.h_rep(), zone.clock_names, zone.k)
    assert _dbms_equal(zone, rebuilt)


def test_h_rep_roundtrip_equality_cycle():
    """Regression for the cyclic-redundancy bug: two clocks pinned equal by
    a completed reset, each also box-bounded, must not lose all four box
    bounds to the naive one-shot redundancy test."""
    ctx = Context(["a", "b"])
    Z = ctx.a.context.get_zero_federation()
    Z = Z.up() & (ctx.a <= 5) & (ctx.b <= 5)
    zone = Zone(Z, ["a", "b"], k=[10, 10])
    assert zone.is_bounded(), "sanity: this fixture is meant to be a bounded diagonal segment"
    rebuilt = Zone.from_h_rep(zone.h_rep(), zone.clock_names, zone.k)
    assert _dbms_equal(zone, rebuilt)
    # And it must not have silently become the unbounded ray.
    assert rebuilt.is_bounded()


# ---------------------------------------------------------------------------
# Item 2: every vertex satisfies all h_rep constraints, equality on >= d of them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bounds",
    [
        [(2, 5)],
        [(2, 5), (1, 4)],
        [(0, 3), (0, 3), (0, 3)],
    ],
)
def test_vertices_satisfy_constraints_with_enough_equalities(bounds):
    zone = _box_zone(bounds)
    h = zone.h_rep()
    d = zone.d
    verts = zone.v_rep()
    assert len(verts) > 0
    for v in verts:
        assert all(hs.satisfied_by(v) for hs in h)
        n_equal = sum(1 for hs in h if sum(a * x for a, x in zip(hs.a, v)) == hs.b)
        assert n_equal >= d


def test_vertices_satisfy_constraints_diagonal_cut():
    ctx = Context(["x", "y"])
    Z = (ctx.x >= 2) & (ctx.x <= 5) & (ctx.y >= 1) & (ctx.y <= 4) & (ctx.x - ctx.y <= 3)
    zone = Zone(Z, ["x", "y"], k=[10, 10])
    h = zone.h_rep()
    verts = zone.v_rep()
    assert len(verts) == 5  # box minus one corner -> pentagon
    for v in verts:
        assert all(hs.satisfied_by(v) for hs in h)
        n_equal = sum(1 for hs in h if sum(a * x for a, x in zip(hs.a, v)) == hs.b)
        assert n_equal >= zone.d


def test_v_rep_raises_unbounded():
    ctx = Context(["x"])
    Z = ctx.x >= 0  # no upper bound
    zone = Zone(Z, ["x"], k=[10])
    with pytest.raises(UnboundedZone):
        zone.v_rep()


def test_v_rep_raises_vertex_enumeration_limit(monkeypatch):
    zone = _box_zone([(0, 1)])
    monkeypatch.setattr("zoneviews.zone._VERTEX_ENUM_LIMIT", 0)
    zone._h_rep_cache = None
    zone._v_rep_cache = None
    with pytest.raises(VertexEnumerationLimit):
        zone.v_rep()


# ---------------------------------------------------------------------------
# Item 3: interior points lie in the convex hull of v_rep() (LP check, not qhull)
# ---------------------------------------------------------------------------

def _point_in_hull_lp(point, vertices) -> bool:
    """LP feasibility check: is `point` a convex combination of `vertices`?

    Deliberately uses scipy.optimize.linprog (a generic LP solver), never
    scipy.spatial.qhull / ConvexHull / Delaunay.
    """
    from scipy.optimize import linprog

    n = len(vertices)
    d = len(point)
    # Variables: lambda_1..lambda_n >= 0.
    # Equality: sum(lambda_i * v_i) = point  (d equations)
    #           sum(lambda_i) = 1
    A_eq = np.zeros((d + 1, n))
    for j, v in enumerate(vertices):
        for i in range(d):
            A_eq[i, j] = float(v[i])
        A_eq[d, j] = 1.0
    b_eq = np.array([float(p) for p in point] + [1.0])
    c = np.zeros(n)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.success


@pytest.mark.parametrize(
    "bounds",
    [
        [(2, 5)],
        [(2, 5), (1, 4)],
        [(0, 3), (0, 3), (0, 3)],
    ],
)
def test_interior_points_in_convex_hull(bounds):
    zone = _box_zone(bounds)
    verts = zone.v_rep()
    rng = random.Random(1234)
    for _ in range(20):
        point = tuple(Fraction(rng.randint(2 * lo, 2 * hi), 2) for lo, hi in bounds)
        assert zone.contains(point)
        assert _point_in_hull_lp(point, verts)


def test_interior_points_in_convex_hull_diagonal_cut():
    ctx = Context(["x", "y"])
    Z = (ctx.x >= 2) & (ctx.x <= 5) & (ctx.y >= 1) & (ctx.y <= 4) & (ctx.x - ctx.y <= 3)
    zone = Zone(Z, ["x", "y"], k=[10, 10])
    verts = zone.v_rep()
    rng = random.Random(99)
    hits = 0
    while hits < 15:
        x = Fraction(rng.randint(4, 10), 2)
        y = Fraction(rng.randint(2, 8), 2)
        if not zone.contains((x, y)):
            continue
        hits += 1
        assert _point_in_hull_lp((x, y), verts)


# ---------------------------------------------------------------------------
# Item 6: graph() edge count and round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bounds",
    [
        [(2, 5)],
        [(2, 5), (1, 4)],
        [(0, 3), (0, 3), (0, 3)],
    ],
)
def test_graph_edge_count_matches_finite_entries(bounds):
    zone = _box_zone(bounds)
    g = zone.graph()
    n = zone.d + 1
    expected = sum(
        1
        for i in range(n)
        for j in range(n)
        if i != j and not zone.dbm.is_infinity(i, j)
    )
    assert g.number_of_nodes() == n
    assert g.number_of_edges() == expected


def test_graph_roundtrips_to_same_dbm():
    ctx = Context(["x", "y"])
    Z = (ctx.x >= 2) & (ctx.x <= 5) & (ctx.y >= 1) & (ctx.y <= 4) & (ctx.x - ctx.y <= 3)
    zone = Zone(Z, ["x", "y"], k=[10, 10])
    g = zone.graph()

    rebuilt_ctx = Context(list(zone.clock_names))
    rebuilt = Federation(rebuilt_ctx).set_init()
    clocks = {name: getattr(rebuilt_ctx, name) for name in zone.clock_names}
    for u, v, data in g.edges(data=True):
        a = [0] * zone.d
        if u != 0:
            a[u - 1] += 1
        if v != 0:
            a[v - 1] -= 1
        hs = HalfSpace(a=tuple(a), b=data["bound"], strict=data["strict"])
        from zoneviews.zone import _halfspace_to_federation

        rebuilt = rebuilt & _halfspace_to_federation(hs, zone.clock_names, clocks)

    assert zone.dbm.to_matrix() == rebuilt.to_dbm_list()[0].to_matrix()


# ---------------------------------------------------------------------------
# Item 9: regression — 5 hand-computed small zones
# ---------------------------------------------------------------------------

def test_regression_interval():
    """1 clock, x in [2, 5]: interval, 2 facets, 2 vertices, affine dim 1."""
    zone = _box_zone([(2, 5)], names=["x"])
    h = zone.h_rep()
    assert zone.facet_count() == 2
    assert {(hs.a, hs.b, hs.strict) for hs in h} == {
        ((1,), 5, False),
        ((-1,), -2, False),
    }
    assert sorted(zone.v_rep()) == [(Fraction(2),), (Fraction(5),)]
    assert zone.affine_dim() == 1


def test_regression_point():
    """1 clock, x == 3: a single point, affine dim 0."""
    ctx = Context(["x"])
    Z = ctx.x == 3
    zone = Zone(Z, ["x"], k=[10])
    assert zone.facet_count() == 2
    assert zone.affine_dim() == 0
    assert zone.v_rep() == [(Fraction(3),)]


def test_regression_unbounded_diagonal_ray():
    """2 clocks x == y, unbounded above: a 1-D ray from the origin."""
    ctx = Context(["x", "y"])
    Z = ctx.get_zero_federation().up()
    zone = Zone(Z, ["x", "y"], k=[10, 10])
    assert not zone.is_bounded()
    assert zone.affine_dim() == 1
    with pytest.raises(UnboundedZone):
        zone.v_rep()


def test_regression_half_open_interval_pinned_second_clock():
    """2 clocks: x in (0, 5] strict lower bound, y pinned to 1.

    Which single DBM entry ends up carrying the strict bit is an
    implementation-defined tie-break of the greedy redundancy elimination
    (the docstring on h_rep() explicitly warns consumers not to depend on
    that) — since y is pinned exactly to 1, the surviving strict witness
    may come from a clock-difference entry composed through y rather than
    the direct x-vs-reference entry. What must hold regardless is the
    *semantic* content: exactly one strict entry, x=0 excluded but x=5
    included, and the correct vertex/dimension geometry.
    """
    ctx = Context(["x", "y"])
    Z = (ctx.x > 0) & (ctx.x <= 5) & (ctx.y == 1)
    zone = Zone(Z, ["x", "y"], k=[10, 10])
    h = zone.h_rep()
    strict_entries = [hs for hs in h if hs.strict]
    assert len(strict_entries) == 1

    assert not zone.contains((Fraction(0), Fraction(1)))  # strict: x=0 excluded
    assert zone.contains((Fraction(1, 1000), Fraction(1)))  # just inside
    assert zone.contains((Fraction(5), Fraction(1)))  # non-strict upper bound included

    verts = zone.v_rep()
    # Closure of a half-open interval still has both endpoints as vertices.
    assert sorted(verts) == [(Fraction(0), Fraction(1)), (Fraction(5), Fraction(1))]
    assert zone.affine_dim() == 1

    rebuilt = Zone.from_h_rep(h, zone.clock_names, zone.k)
    assert zone.dbm.to_matrix() == rebuilt.dbm.to_matrix()


def test_regression_independent_3d_box():
    """3 independent clocks each in [0, 2]: a cube, 6 facets, 8 vertices, dim 3."""
    zone = _box_zone([(0, 2), (0, 2), (0, 2)])
    assert zone.facet_count() == 6
    assert zone.affine_dim() == 3
    verts = zone.v_rep()
    assert len(verts) == 8
    expected = {
        (Fraction(a), Fraction(b), Fraction(c))
        for a in (0, 2)
        for b in (0, 2)
        for c in (0, 2)
    }
    assert set(verts) == expected
