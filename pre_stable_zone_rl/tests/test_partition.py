import random

import pytest

from prestable.model import Constraint, diff_le, eq, ge, gt, le, lt
from prestable.partition import (
    _cut,
    _disjointify,
    _merge,
    _normalise,
    check_partition,
    partition,
)
from prestable.zones import Zone, ZoneContext, intersect, is_k_bounded, subtract_all

# Largest partition seen in the random tests, reported at the end of the run.
STATS = {"max_blocks": 0}


def Z(ctx, *constraints):
    flat = []
    for c in constraints:
        flat.extend(c if isinstance(c, tuple) else (c,))
    return ctx.from_guard(flat)


def run(F, G, ctx):
    """partition() plus the checks every result must pass."""
    F, G = list(F), list(G)
    P = partition(F, G, ctx)
    check_partition(P, F, G, ctx)
    assert partition(P, G, ctx) == P, "not idempotent"
    for b in P:
        assert partition(list(P) + [b], G, ctx) == P, "duplicate block changed the result"
    return P


@pytest.fixture
def cx():
    return ZoneContext(["x"])


@pytest.fixture
def cxy():
    return ZoneContext(["x", "y"])


# --- one clock --------------------------------------------------------------

def test_running_example_insert_l0(cx):
    F = [cx.zero(), cx.universe()]
    P = run(F, [ge("x", 1)], cx)
    assert P == {cx.zero(), Z(cx, gt("x", 0), lt("x", 1)), Z(cx, ge("x", 1))}


def test_running_example_later_insert_l0(cx):
    F = [cx.zero(), Z(cx, gt("x", 0), lt("x", 1)), Z(cx, ge("x", 1)), Z(cx, le("x", 2))]
    P = run(F, [ge("x", 1)], cx)
    assert P == {cx.zero(), Z(cx, gt("x", 0), lt("x", 1)), Z(cx, ge("x", 1), le("x", 2)), Z(cx, gt("x", 2))}


def test_phase2_split(cx):
    F = [Z(cx, gt("x", 0), le("x", 2)), Z(cx, gt("x", 0), lt("x", 1))]
    P = run(F, [], cx)
    assert P == {Z(cx, gt("x", 0), lt("x", 1)), Z(cx, ge("x", 1), le("x", 2))}


def test_nested_intervals(cx):
    F = [Z(cx, le("x", 3)), Z(cx, ge("x", 1), le("x", 2))]
    P = run(F, [], cx)
    assert P == {Z(cx, lt("x", 1)), Z(cx, ge("x", 1), le("x", 2)), Z(cx, gt("x", 2), le("x", 3))}


def test_empty_and_trivial_inputs(cx):
    assert run([], [ge("x", 1)], cx) == frozenset()
    assert run([cx.empty()], [], cx) == frozenset()
    assert run([cx.universe()], [], cx) == {cx.universe()}


def test_guard_atom_only_cuts_inside_cover(cx):
    # x >= 5 lies outside ⋃F, so it adds no blocks.
    P = run([Z(cx, le("x", 2))], [ge("x", 1), ge("x", 5)], cx)
    assert P == {Z(cx, lt("x", 1)), Z(cx, ge("x", 1), le("x", 2))}


# --- two clocks -------------------------------------------------------------

def square(ctx, lo, hi):
    return Z(ctx, ge("x", lo), le("x", hi), ge("y", lo), le("y", hi))


def test_l_shape(cxy):
    big, small = square(cxy, 0, 2), square(cxy, 1, 2)
    P = run([big, small], [], cxy)
    assert len(P) == 3 and small in P
    vertical = Z(cxy, lt("x", 1), le("y", 2))
    horizontal = Z(cxy, le("x", 2), lt("y", 1))
    for b in P - {small}:
        assert b <= vertical or b <= horizontal, b


def test_merge_required(cxy):
    big, small = square(cxy, 0, 2), square(cxy, 1, 2)
    P = run([big, small], [lt("y", 1)], cxy)
    assert P == {small, Z(cxy, le("x", 2), lt("y", 1)), Z(cxy, lt("x", 1), ge("y", 1), le("y", 2))}


def test_diagonal_guard(cxy):
    P = run([square(cxy, 0, 2)], [diff_le("x", "y", 0)], cxy)
    assert P == {Z(cxy, le("x", 2), le("y", 2), diff_le("x", "y", 0)),
                 Z(cxy, le("x", 2), le("y", 2), Constraint("y", "x", 0, True))}


# --- order independence -------------------------------------------------------

ORDER_CASES = [
    (["x"], lambda c: [c.zero(), Z(c, gt("x", 0), lt("x", 1)), Z(c, ge("x", 1)), Z(c, le("x", 2))], [ge("x", 1)]),
    (["x"], lambda c: [Z(c, le("x", 3)), Z(c, ge("x", 1), le("x", 2)), Z(c, ge("x", 2))], [gt("x", 1), le("x", 0)]),
    (["x", "y"], lambda c: [square(c, 0, 2), square(c, 1, 2), square(c, 0, 1)], [lt("y", 1), diff_le("x", "y", 0)]),
]


@pytest.mark.parametrize("clocks, make_F, G", ORDER_CASES)
def test_order_and_duplicates_do_not_matter(clocks, make_F, G):
    ctx = ZoneContext(clocks)
    F = make_F(ctx)
    expected = run(F, G, ctx)
    rng = random.Random(0)
    for _ in range(10):
        F2 = F + rng.sample(F, rng.randint(1, len(F)))
        # Also rebuild zones from scratch so equal-but-distinct objects are used.
        F2 = [intersect(z, ctx.universe()) for z in F2]
        rng.shuffle(F2)
        G2 = G + rng.sample(G, rng.randint(0, len(G)))
        rng.shuffle(G2)
        assert partition(F2, G2, ctx) == expected


# --- individual steps -------------------------------------------------------

def test_normalise(cx):
    a, b = Z(cx, le("x", 1)), Z(cx, ge("x", 1))
    zones, cutters = _normalise([b, cx.empty(), a, Z(cx, le("x", 1))], [ge("x", 1), le("x", 1)], cx)
    assert zones == sorted([a, b], key=Zone.key)
    assert cutters == zones  # the guard atoms equal zones of F and are deduplicated


def test_disjointify(cx):
    zones = [Z(cx, le("x", 2)), Z(cx, ge("x", 1), le("x", 3))]
    pieces = _disjointify(sorted(zones, key=Zone.key))
    for i, p in enumerate(pieces):
        assert all(intersect(p, q).is_empty for q in pieces[i + 1:])
    # Step 1 only promises a disjoint cover, not uniformity.
    assert not subtract_all(Z(cx, le("x", 3)), pieces)
    assert all(subtract_all(p, zones) == [] for p in pieces)


def test_cut_then_merge(cx):
    pieces = [Z(cx, le("x", 3))]
    cutters = [Z(cx, ge("x", 1)), Z(cx, le("x", 2))]
    cut = _cut(pieces, cutters)
    assert set(cut) == {Z(cx, lt("x", 1)), Z(cx, ge("x", 1), le("x", 2)), Z(cx, gt("x", 2), le("x", 3))}
    assert set(_merge(cut, cutters)) == set(cut)  # three different signatures
    assert _merge(cut, []) == [Z(cx, le("x", 3))]  # one signature, convex union


def test_check_partition_reports_violations(cx):
    a, b = Z(cx, le("x", 2)), Z(cx, ge("x", 1), le("x", 3))
    with pytest.raises(ValueError, match="overlap"):
        check_partition([a, b], [a, b], [], cx)
    with pytest.raises(ValueError, match="not covered by the blocks"):
        check_partition([a], [a, b], [], cx)
    with pytest.raises(ValueError, match="not covered by the zones"):
        check_partition([Z(cx, le("x", 3))], [a], [], cx)
    with pytest.raises(ValueError, match="neither inside nor disjoint"):
        check_partition([a], [a], [ge("x", 1)], cx)
    with pytest.raises(ValueError, match="empty"):
        check_partition([cx.empty(), a], [a], [], cx)


# --- random -----------------------------------------------------------------

def random_constraint(rng, names, k):
    left, right = rng.sample(names, 2)
    return Constraint(left, right, rng.randint(-k, k), rng.random() < 0.5)


def random_k_bounded_zone(rng, ctx, k):
    names = ["0", *ctx.clocks]
    while True:
        z = ctx.from_guard([random_constraint(rng, names, k) for _ in range(rng.randint(1, 4))])
        if not z.is_empty and is_k_bounded(z, k):
            return z


@pytest.mark.slow
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_random_partitions(n, k):
    rng = random.Random(100 * n + k)
    ctx = ZoneContext("xyz"[:n])
    names = ["0", *ctx.clocks]
    for _ in range(100):
        F = [random_k_bounded_zone(rng, ctx, k) for _ in range(rng.randint(1, 4))]
        G = [random_constraint(rng, names, k) for _ in range(rng.randint(0, 3))]
        P = partition(F, G, ctx)
        check_partition(P, F, G, ctx)
        assert partition(P, G, ctx) == P
        for b in P:
            assert is_k_bounded(b, k), f"block {b} not {k}-bounded; F={[str(z) for z in F]}, G={G}"
            assert any(b <= z for z in F), b
        STATS["max_blocks"] = max(STATS["max_blocks"], len(P))


def test_zz_report_max_blocks():
    """Runs after the random tests (file order); prints the size indicator with -s."""
    print(f"\nlargest partition in random tests: {STATS['max_blocks']} blocks")
