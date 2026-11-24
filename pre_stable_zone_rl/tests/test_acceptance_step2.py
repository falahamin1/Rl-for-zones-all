"""Acceptance tests for step 2 (partition / Pi_K).

These check what the algorithm phases will rely on:
1. The Phase 1 Insert sequence of the running example, including the
   "did the partition change?" check on line 33.
2. Inserting a zone only ever refines the old partition. Phase 1
   termination depends on this.
3. Inserting a union of existing blocks changes nothing, so no location is
   re-queued without reason.
4. Phase 2 split: Z- always survives as exactly one block.
5. Merging never coarsens the input, and does merge inside convex regions.
6. The new subtract: pieces are disjoint, exact and K-bounded.
"""

import random

import pytest

from prestable.model import Constraint, diff_ge, diff_lt, ge, gt, le, lt
from prestable.partition import check_partition, partition
from prestable.zones import (
    ZoneContext,
    extrapolate,
    hull,
    intersect,
    is_k_bounded,
    subtract,
    subtract_all,
    union_is_convex,
)


# ---------------------------------------------------------------- helpers


def _flatten(constraints):
    flat = []
    for c in constraints:
        if isinstance(c, (tuple, list)):
            flat.extend(c)
        else:
            flat.append(c)
    return tuple(flat)


def zone(ctx, *constraints):
    return ctx.from_guard(_flatten(constraints))


def insert(P, ctx, X, G, K):
    """Mirror of procedure Insert (lines 27-37). Returns (new P, changed)."""
    X = extrapolate(X, K)
    if X.is_empty:
        return P, False
    F = set(P) | {X}
    new = partition(F, G, ctx)
    check_partition(new, F, G, ctx)
    return new, new != frozenset(P)


CASES = [(n, k) for n in (1, 2, 3) for k in (1, 2, 3)]
ROUNDS = 100


def random_constraint(rng, clocks, k):
    left, right = rng.sample(["0", *clocks], 2)
    return Constraint(left, right, rng.randint(-k, k), rng.random() < 0.5)


def random_zone(rng, ctx, clocks, k):
    while True:
        cs = tuple(random_constraint(rng, clocks, k) for _ in range(rng.randint(1, 4)))
        z = ctx.from_guard(cs)
        if not z.is_empty:
            return z


def random_family(rng, ctx, clocks, k):
    F = [random_zone(rng, ctx, clocks, k) for _ in range(rng.randint(1, 3))]
    G = [random_constraint(rng, clocks, k) for _ in range(rng.randint(0, 3))]
    return F, G


def setup(n, k):
    clocks = tuple("xyz"[:n])
    return random.Random(7000 + 10 * n + k), clocks, ZoneContext(clocks)


# ------------------------------------------- 1. running example, Phase 1


def test_running_example_phase1_insert_sequence():
    ctx = ZoneContext(("x",))
    K = 2
    G0, G1 = [ge("x", 1)], [le("x", 2)]
    zero, universe = ctx.zero(), ctx.universe()
    open01 = zone(ctx, gt("x", 0), lt("x", 1))
    closed12 = zone(ctx, ge("x", 1), le("x", 2))
    from1 = zone(ctx, ge("x", 1))
    above2 = zone(ctx, gt("x", 2))
    P0, P1 = frozenset(), frozenset()

    # line 3 and first processing of l0
    P0, ch = insert(P0, ctx, zero, G0, K)
    assert ch and P0 == {zero}
    P0, ch = insert(P0, ctx, universe, G0, K)
    assert ch and P0 == {zero, open01, from1}
    P1, ch = insert(P1, ctx, zero, G1, K)
    assert ch and P1 == {zero}

    # second processing of l0: every insert is aligned
    for X in [universe, zone(ctx, gt("x", 0)), from1]:
        P0, ch = insert(P0, ctx, X, G0, K)
        assert not ch, str(X)
    P1, ch = insert(P1, ctx, zero, G1, K)
    assert not ch

    # processing l1
    P1, ch = insert(P1, ctx, universe, G1, K)
    assert ch and P1 == {zero, zone(ctx, gt("x", 0), le("x", 2)), above2}
    P0, ch = insert(P0, ctx, zone(ctx, le("x", 2)), G0, K)
    assert ch and P0 == {zero, open01, closed12, above2}

    # remaining inserts of the trace change nothing
    for X in [universe, zone(ctx, gt("x", 0)), above2]:
        P1, ch = insert(P1, ctx, X, G1, K)
        assert not ch, str(X)
    P0, ch = insert(P0, ctx, zone(ctx, gt("x", 0), le("x", 2)), G0, K)
    assert not ch
    P0, ch = insert(P0, ctx, intersect(above2, zone(ctx, le("x", 2))), G0, K)
    assert not ch  # empty successor: early return on line 29


# ------------------------------------------------------ 5. merge behaviour


def test_partition_never_coarsens_input():
    ctx = ZoneContext(("x",))
    a = zone(ctx, ge("x", 0), le("x", 1))
    b = zone(ctx, gt("x", 1), le("x", 2))
    assert partition([a, b], [], ctx) == {a, b}


def test_guard_regions_become_single_blocks():
    ctx = ZoneContext(("x", "y"))
    G = [le("x", 2), diff_lt("x", "y", 1)]
    P = partition([ctx.universe()], G, ctx)
    check_partition(P, [ctx.universe()], G, ctx)
    assert P == {
        zone(ctx, le("x", 2), diff_lt("x", "y", 1)),
        zone(ctx, le("x", 2), diff_ge("x", "y", 1)),
        zone(ctx, gt("x", 2), diff_lt("x", "y", 1)),
        zone(ctx, gt("x", 2), diff_ge("x", "y", 1)),
    }


# --------------------------------------------------- 2. inserts only refine


@pytest.mark.slow
@pytest.mark.parametrize("n,k", CASES)
def test_insert_only_refines(n, k):
    rng, clocks, ctx = setup(n, k)
    for _ in range(ROUNDS):
        F, G = random_family(rng, ctx, clocks, k)
        P = partition(F, G, ctx)
        X = random_zone(rng, ctx, clocks, k)
        P2 = partition(set(P) | {X}, G, ctx)
        check_partition(P2, set(P) | {X}, G, ctx)
        for b2 in P2:
            parents = [b for b in P if b2 <= b]
            if parents:
                assert len(parents) == 1
            else:
                assert all(intersect(b2, b).is_empty for b in P), str(b2)


# ------------------------------------------- 3. aligned inserts are no-ops


@pytest.mark.slow
@pytest.mark.parametrize("n,k", CASES)
def test_aligned_insert_changes_nothing(n, k):
    rng, clocks, ctx = setup(n, k)
    checked = 0
    for _ in range(ROUNDS):
        F, G = random_family(rng, ctx, clocks, k)
        P = partition(F, G, ctx)
        blocks = sorted(P, key=lambda z: z.key())
        subset = rng.sample(blocks, rng.randint(1, min(3, len(blocks))))
        if not union_is_convex(subset):
            continue
        X = hull(subset)
        assert partition(set(P) | {X}, G, ctx) == P, str(X)
        checked += 1
    assert checked > 0


# ------------------------------------------------------ 4. Phase 2 split


@pytest.mark.slow
@pytest.mark.parametrize("n,k", CASES)
def test_phase2_split_keeps_z_minus_as_one_block(n, k):
    rng, clocks, ctx = setup(n, k)
    checked = 0
    for _ in range(ROUNDS):
        Z = random_zone(rng, ctx, clocks, k)
        Z_minus = intersect(Z, random_zone(rng, ctx, clocks, k))
        if Z_minus.is_empty or Z_minus == Z:
            continue
        P = partition([Z, Z_minus], [], ctx)
        check_partition(P, [Z, Z_minus], [], ctx)
        assert Z_minus in P, f"Z={Z}, Z-={Z_minus}"
        assert len(P) >= 2
        assert all(b <= Z for b in P)
        checked += 1
    assert checked > 0


# ----------------------------------------------------- 6. new subtract


@pytest.mark.slow
@pytest.mark.parametrize("n,k", CASES)
def test_subtract_pieces_disjoint_exact_k_bounded(n, k):
    rng, clocks, ctx = setup(n, k)
    for _ in range(ROUNDS):
        z = random_zone(rng, ctx, clocks, k)
        w = random_zone(rng, ctx, clocks, k)
        pieces = subtract(z, w)
        for i, p in enumerate(pieces):
            assert not p.is_empty and p <= z
            assert intersect(p, w).is_empty
            assert is_k_bounded(p, k), f"z={z}, w={w}, piece={p}"
            for q in pieces[i + 1:]:
                assert intersect(p, q).is_empty
        assert subtract_all(z, [w, *pieces]) == []
