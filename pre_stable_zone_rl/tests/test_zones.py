import itertools
import random

import pytest

from prestable.model import Constraint, Edge, diff_le, eq, ge, gt, le, lt
from prestable.zones import (
    Zone,
    ZoneContext,
    _extrapolate_by_matrix,
    _hull_by_matrix,
    _own_bound,
    down,
    extrapolate,
    hull,
    intersect,
    is_k_bounded,
    pre_edge,
    reset,
    subtract,
    subtract_all,
    union_is_convex,
    up,
)


@pytest.fixture
def cx():
    return ZoneContext(["x"])


@pytest.fixture
def cxy():
    return ZoneContext(["x", "y"])


def edge(guard=(), resets=()):
    return Edge("l", "m", "a", guard=tuple(guard), resets=frozenset(resets))


# --- one clock --------------------------------------------------------------

def test_up_zero_is_universe(cx):
    assert up(cx.zero()) == cx.universe()


def test_pre_edge_with_reset(cx):
    assert pre_edge(cx.zero(), edge([ge("x", 1)], {"x"})) == cx.from_guard([ge("x", 1)])


def test_pre_edge_without_reset(cx):
    z = cx.from_guard([gt("x", 0), lt("x", 1)])
    assert pre_edge(z, edge([le("x", 2)])) == z


def test_pre_edge_reset_target_must_contain_zero(cx):
    # Resetting x lands at x = 0, which is not in x >= 1: nothing can take the edge.
    assert pre_edge(cx.from_guard([ge("x", 1)]), edge([], {"x"})).is_empty


def test_down(cx):
    assert down(cx.from_guard([ge("x", 1), le("x", 2)])) == cx.from_guard([le("x", 2)])


def test_extrapolate_upper_bound_dropped(cx):
    assert extrapolate(cx.from_guard([le("x", 2)]), 1) == cx.universe()


def test_extrapolate_lower_bound_clamped(cx):
    assert extrapolate(cx.from_guard([gt("x", 3)]), 2) == cx.from_guard([gt("x", 2)])


def test_extrapolate_point_beyond_k(cx):
    assert extrapolate(cx.from_guard(eq("x", 5)), 3) == cx.from_guard([gt("x", 3)])


def test_extrapolate_empty(cx):
    assert extrapolate(cx.empty(), 1).is_empty


# --- two clocks -------------------------------------------------------------

def test_up_zero_is_diagonal(cxy):
    assert up(cxy.zero()) == cxy.from_guard([diff_le("x", "y", 0), diff_le("y", "x", 0)])


def test_reset(cxy):
    z = cxy.from_guard([ge("x", 1), le("x", 2), le("y", 2)])
    assert reset(z, {"y"}) == cxy.from_guard([ge("x", 1), le("x", 2), *eq("y", 0)])


def test_reset_all_is_zero(cxy):
    assert reset(cxy.universe(), {"x", "y"}) == cxy.zero()


def test_pre_edge_two_clocks(cxy):
    # target: x - y == 2 and y <= 1; edge resets y under guard x >= 1.
    # v[y:=0] must satisfy x == 2, so pre = (x == 2) (y arbitrary, >= 0).
    target = cxy.from_guard([diff_le("x", "y", 2), diff_le("y", "x", -2), le("y", 1)])
    assert pre_edge(target, edge([ge("x", 1)], {"y"})) == cxy.from_guard(eq("x", 2))


def box(ctx, lo, hi):
    return ctx.from_guard([ge("x", lo), le("x", hi), ge("y", lo), le("y", hi)])


def open_box(ctx, xl, xh, yl, yh):
    return ctx.from_guard([gt("x", xl), lt("x", xh), gt("y", yl), lt("y", yh)])


def test_subtract_l_shape(cxy):
    big, hole = box(cxy, 0, 2), box(cxy, 1, 2)
    pieces = subtract(big, hole)
    assert pieces
    for p in pieces:
        assert not p.is_empty
        assert p <= big
        assert intersect(p, hole).is_empty
    for sample in (open_box(cxy, 0, 1, 0, 1), open_box(cxy, 0, 1, 1, 2), open_box(cxy, 1, 2, 0, 1)):
        assert any(sample <= p for p in pieces), sample


def grid(ctx, lo=-0.5, hi=4.0, step=0.25):
    n = int(round((hi - lo) / step)) + 1
    values = [lo + k * step for k in range(n)]
    for point in itertools.product(values, repeat=len(ctx.clocks)):
        yield dict(zip(ctx.clocks, point))


def test_subtract_union_is_exact(cxy):
    big, hole = box(cxy, 0, 2), box(cxy, 1, 2)
    pieces = subtract(big, hole)
    for v in grid(cxy):
        expected = big.contains(v) and not hole.contains(v)
        assert any(p.contains(v) for p in pieces) == expected, v


def test_subtract_edge_cases(cxy):
    z = box(cxy, 0, 2)
    assert subtract(z, z) == []
    assert subtract(z, cxy.universe()) == []
    assert subtract(cxy.empty(), z) == []
    assert subtract(z, cxy.empty()) == [z]
    disjoint = box(cxy, 5, 6)
    assert subtract(z, disjoint) == [z]


def test_is_k_bounded_with_implied_bound(cxy):
    z = cxy.from_guard([diff_le("x", "y", 1), le("y", 1)])
    # The canonical DBM contains x <= 2 (> K), yet extrapolation leaves z unchanged.
    m = z.matrix()
    assert m[1][0] == (2, False)
    assert is_k_bounded(z, 1)


def test_is_k_bounded_false(cxy):
    assert not is_k_bounded(cxy.from_guard([le("x", 2)]), 1)
    assert is_k_bounded(cxy.from_guard([le("x", 2)]), 2)


def test_equality_and_hash(cxy):
    a = cxy.from_guard([le("x", 1), le("y", 1)])
    b = cxy.from_guard([le("y", 1), diff_le("x", "y", 5), le("x", 1), le("x", 3)])
    c = intersect(cxy.from_guard([le("x", 1)]), cxy.from_guard([le("y", 1)]))
    assert a == b == c
    assert hash(a) == hash(b) == hash(c)
    assert len({a, b, c}) == 1
    assert a != cxy.from_guard([le("x", 1), le("y", 2)])


def test_up_zero_built_two_ways(cxy):
    a = up(cxy.zero())
    b = cxy.from_guard([diff_le("x", "y", 0), diff_le("y", "x", 0)])
    assert a == b and hash(a) == hash(b)


def test_empty_zones_equal(cxy):
    e1 = cxy.from_guard([le("x", 1), ge("x", 2)])
    e2 = intersect(cxy.zero(), cxy.from_guard([ge("y", 3)]))
    assert e1.is_empty and e2.is_empty
    assert e1 == e2 == cxy.empty()
    assert hash(e1) == hash(e2) == hash(cxy.empty())


def test_inclusion(cxy):
    small, big = box(cxy, 1, 2), box(cxy, 0, 2)
    assert small <= big
    assert not big <= small
    assert cxy.empty() <= small
    assert big <= cxy.universe()


def test_results_stay_nonnegative(cxy):
    # down() of a zone would leave the nonnegative orthant without the universe intersection.
    # Going back in time preserves x - y, so the diagonal band |x - y| <= 1 remains.
    d = down(box(cxy, 1, 2))
    assert d == cxy.from_guard([le("x", 2), le("y", 2), diff_le("x", "y", 1), diff_le("y", "x", 1)])
    assert not d.contains({"x": -0.5, "y": 0.0})


def test_zones_are_independent_of_inputs(cxy):
    z = box(cxy, 0, 2)
    before = str(z)
    up(z), down(z), reset(z, {"x"}), extrapolate(z, 0), subtract(z, box(cxy, 1, 2))
    assert str(z) == before


def test_contexts_do_not_mix(cxy):
    other = ZoneContext(["x", "y"])
    with pytest.raises(ValueError):
        intersect(cxy.universe(), other.universe())
    assert cxy.universe() != other.universe()


def test_unknown_clock_rejected(cxy):
    with pytest.raises(ValueError):
        cxy.from_guard([le("z", 1)])


# --- rendering --------------------------------------------------------------

@pytest.mark.parametrize(
    "guard, text",
    [
        ([gt("x", 0), lt("x", 1)], "0 < x < 1"),
        ([ge("x", 1), le("x", 2)], "1 <= x <= 2"),
        ([gt("x", 3)], "x > 3"),
        ([le("x", 4)], "x <= 4"),
        (eq("x", 2), "x == 2"),
        ([], "true"),
        ([le("x", 1), ge("x", 2)], "false"),
    ],
)
def test_str_one_clock(cx, guard, text):
    assert str(cx.from_guard(guard)) == text


def test_str_two_clocks(cxy):
    assert str(cxy.from_guard([diff_le("x", "y", 1)])) == "x - y <= 1"
    assert str(cxy.zero()) == "x == 0 and y == 0"
    assert str(up(cxy.zero())) == "x - y == 0"
    assert str(cxy.from_guard([diff_le("x", "y", 1), le("y", 1)])) == "x <= 2 and y <= 1 and x - y <= 1"


# --- extrapolation agrees with the definition --------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
def test_extrapolate_matches_matrix_definition(n):
    rng = random.Random(n)
    ctx = ZoneContext([f"c{i}" for i in range(n)])
    names = ("0",) + ctx.clocks
    checked = 0
    while checked < 400:
        guard = []
        for _ in range(rng.randint(1, 6)):
            left, right = rng.sample(names, 2)
            guard.append(Constraint(left, right, rng.randint(-10, 10), rng.random() < 0.5))
        z = ctx.from_guard(guard)
        if z.is_empty:
            continue
        K = rng.randint(0, 8)
        assert extrapolate(z, K) == _extrapolate_by_matrix(z, K), (str(z), K)
        checked += 1


# --- key, hull, subtract_all, union_is_convex ---------------------------------

def test_key_is_ordered_and_canonical(cxy):
    a = cxy.from_guard([le("x", 1), le("y", 1)])
    b = cxy.from_guard([le("y", 1), le("x", 1), diff_le("x", "y", 3)])
    assert a.key() == b.key()
    assert cxy.empty().key() == () and cxy.empty().key() < a.key()
    zs = [box(cxy, 0, 2), box(cxy, 1, 2), cxy.zero(), cxy.universe()]
    assert sorted(zs, key=Zone.key) == sorted(reversed(zs), key=Zone.key)


def test_hull_and_union_is_convex_one_clock(cx):
    a = cx.from_guard([gt("x", 0), lt("x", 1)])
    b = cx.from_guard([gt("x", 2), lt("x", 3)])
    assert hull([a, b]) == cx.from_guard([gt("x", 0), lt("x", 3)])
    assert not union_is_convex([a, b])
    c = cx.from_guard([ge("x", 0), lt("x", 1)])
    d = cx.from_guard([ge("x", 1), le("x", 2)])
    assert union_is_convex([c, d])
    assert hull([c, d]) == cx.from_guard([le("x", 2)])
    # (0,1) and (1,2) miss the point 1.
    assert not union_is_convex([a, cx.from_guard([gt("x", 1), lt("x", 2)])])


def test_hull_two_clocks_with_diagonal(cxy):
    # Two unit squares touching at a corner: hull is the band 0 <= x,y <= 2, |x - y| <= 1.
    lo, hi = box(cxy, 0, 1), box(cxy, 1, 2)
    h = hull([lo, hi])
    assert h == cxy.from_guard([le("x", 2), le("y", 2), diff_le("x", "y", 1), diff_le("y", "x", 1)])
    assert not union_is_convex([lo, hi])
    # Two halves of a diagonal band, split at x = 1, are convex together.
    band = [diff_le("x", "y", 1), diff_le("y", "x", 0), le("x", 2)]
    left = cxy.from_guard([*band, lt("x", 1)])
    right = cxy.from_guard([*band, ge("x", 1)])
    assert union_is_convex([left, right])
    assert hull([left, right]) == cxy.from_guard(band)


def test_hull_edge_cases(cxy):
    z = box(cxy, 0, 2)
    assert hull([z]) == z
    assert hull([z, cxy.empty()]) == z
    assert hull([cxy.empty()]).is_empty
    assert union_is_convex([]) and union_is_convex([z])
    with pytest.raises(ValueError):
        hull([])


@pytest.mark.parametrize("n", [1, 2, 3])
def test_hull_matches_entrywise_maximum(n):
    rng = random.Random(10 + n)
    ctx = ZoneContext([f"c{i}" for i in range(n)])
    names = ("0",) + ctx.clocks

    def rz():
        while True:
            guard = [Constraint(*rng.sample(names, 2), rng.randint(-6, 6), rng.random() < 0.5)
                     for _ in range(rng.randint(1, 5))]
            z = ctx.from_guard(guard)
            if not z.is_empty:
                return z

    for _ in range(300):
        zs = [rz() for _ in range(rng.randint(1, 4))]
        h = hull(zs)
        assert h == _hull_by_matrix(zs)
        assert all(z <= h for z in zs)


def test_subtract_all(cxy):
    big = box(cxy, 0, 3)
    ws = [box(cxy, 0, 1), box(cxy, 1, 2), cxy.from_guard([ge("x", 2)])]
    pieces = subtract_all(big, ws)
    for p in pieces:
        assert p <= big and all(intersect(p, w).is_empty for w in ws)
    for v in grid(cxy):
        expected = big.contains(v) and not any(w.contains(v) for w in ws)
        assert any(p.contains(v) for p in pieces) == expected, v
    assert subtract_all(big, []) == [big]
    assert subtract_all(big, [cxy.universe()]) == []


# --- _own_bound and constraints() (step 10.2) ---------------------------------------

def _own_bound_linear(w):
    """Reference: the original linear scan for the smallest K with w K-bounded."""
    top = max((abs(int(c)) for row in w.matrix() for c, _ in row if c != float("inf")), default=0)
    return next(k for k in range(top + 1) if is_k_bounded(w, k))


def _constraints_reference(w, k):
    """Reference: Zone.constraints() as before step 10, from a given own bound k."""
    names, out = w.ctx._names, []
    for i, row in enumerate(w.matrix()):
        for j, (c, strict) in enumerate(row):
            if i == j or c == float("inf") or c > k:
                continue
            if c < -k:
                c, strict = -k, True
            if not (i == 0 and c == 0 and not strict):
                out.append(Constraint(names[i], names[j], int(c), strict))
    return tuple(out)


def test_own_bound_binary_search_matches_linear_scan():
    rng = random.Random(1010)
    contexts = {n: ZoneContext([f"c{i}" for i in range(n)]) for n in (1, 2, 3)}
    checked = 0
    while checked < 2000:
        n = rng.choice((1, 2, 3))
        ctx = contexts[n]
        names = ("0",) + ctx.clocks
        K = rng.randint(0, 64)
        guard = [Constraint(*rng.sample(names, 2), rng.randint(-K, K), rng.random() < 0.5)
                 for _ in range(rng.randint(0, 6))]
        z = ctx.from_guard(guard)
        if z.is_empty:
            continue
        k = _own_bound_linear(z)
        assert _own_bound(z) == k, str(z)
        expected = _constraints_reference(z, k)
        assert z.constraints() == expected, str(z)
        assert z.constraints() is z.constraints()  # cached
        # A fresh, equal zone (no cache) gives the same answer.
        assert ctx.from_guard(guard).constraints() == expected
        checked += 1
