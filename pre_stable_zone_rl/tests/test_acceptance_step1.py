"""Acceptance tests for step 1 (model + zones).

These complement Claude Code's unit tests. Instead of re-checking single
operations, they check the properties the later algorithm relies on:

1. The operations reproduce the hand-traced running example (one clock, K = 2).
2. Strict vs non-strict bounds are handled exactly.
3. Pre_e works with diagonal guards and partial resets.
4. Subtraction is exact, and on the L-shape every piece lies in one of the two bars.
5. Extrapolation behaves as Extra_K should (contains input, idempotent, monotone in K).
6. K-bounded zones stay K-bounded under the operations whose results are NOT
   extrapolated before entering a partition: up, down, intersect, Pre_e and
   subtraction. Forward reset is deliberately excluded: it can break
   K-boundedness (see test_reset_can_break_k_boundedness), which is why
   Insert extrapolates every successor (line 28) before it reaches Pi_K.

Place in pre_stable_zone_rl/tests/ and run with pytest.
"""

import random

import pytest

from prestable.model import (
    Constraint,
    Edge,
    TimedAutomaton,
    diff_le,
    eq,
    ge,
    gt,
    le,
    lt,
)
from prestable.zones import (
    ZoneContext,
    down,
    extrapolate,
    intersect,
    is_k_bounded,
    pre_edge,
    reset,
    subtract,
    up,
)


# ---------------------------------------------------------------- helpers


def _flatten(constraints):
    flat = []
    for c in constraints:
        if isinstance(c, (tuple, list)):  # eq() returns two constraints
            flat.extend(c)
        else:
            flat.append(c)
    return tuple(flat)


def zone(ctx, *constraints):
    """Zone given by a conjunction of constraints."""
    return ctx.from_guard(_flatten(constraints))


def edge(guard=(), resets=(), source="a", target="b", action="t"):
    return Edge(
        source=source,
        target=target,
        action=action,
        guard=_flatten(guard),
        resets=frozenset(resets),
    )


def assert_exact_difference(z, w, pieces):
    """Every piece is nonempty, inside z and disjoint from w, and the pieces
    together cover z minus w (checked symbolically, no sampling)."""
    for p in pieces:
        assert not p.is_empty
        assert p <= z
        assert intersect(p, w).is_empty
    remaining = [z]
    for cut in [w, *pieces]:
        remaining = [r2 for r in remaining for r2 in subtract(r, cut)]
    assert not remaining, f"uncovered part of z minus w: {[str(r) for r in remaining]}"


def running_example():
    """l0 --(x >= 1, a, reset x)--> l1 --(x <= 2, b)--> l0, one clock x."""
    e1 = Edge(source="l0", target="l1", action="a",
              guard=(ge("x", 1),), resets=frozenset({"x"}))
    e2 = Edge(source="l1", target="l0", action="b",
              guard=(le("x", 2),), resets=frozenset())
    return TimedAutomaton(clocks=("x",), locations=("l0", "l1"),
                          initial="l0", edges=(e1, e2), rates={})


# ---------------------------------------------------- 1. running example


def test_running_example_model():
    ta = running_example()
    assert ta.max_constant() == 2
    assert ta.atomic_guards("l0") == frozenset({ge("x", 1)})
    assert ta.atomic_guards("l1") == frozenset({le("x", 2)})


def test_running_example_phase1_operations():
    ta = running_example()
    e1, e2 = ta.edges
    ctx = ZoneContext(ta.clocks)
    universe, zero = ctx.universe(), ctx.zero()

    # Processing l0: delay from {0}, then take e1 (guard x >= 1, reset x).
    zt = up(zero)
    assert zt == universe
    assert reset(intersect(zt, ctx.from_guard(e1.guard)), e1.resets) == zero

    # Processing l1: delay from {0}, then take e2 (guard x <= 2, no reset).
    succ = reset(intersect(up(zero), ctx.from_guard(e2.guard)), e2.resets)
    assert succ == zone(ctx, le("x", 2))
    assert extrapolate(succ, ta.max_constant()) == succ


def test_running_example_phase2_operations():
    ta = running_example()
    e1, e2 = ta.edges
    ctx = ZoneContext(ta.clocks)
    zero = ctx.zero()
    open01 = zone(ctx, gt("x", 0), lt("x", 1))
    closed12 = zone(ctx, ge("x", 1), le("x", 2))
    above2 = zone(ctx, gt("x", 2))

    assert pre_edge(zero, e1) == zone(ctx, ge("x", 1))
    assert pre_edge(open01, e2) == open01
    assert pre_edge(closed12, e2) == closed12
    assert pre_edge(above2, e2).is_empty

    # The one Phase 2 split: block (0,2] at l1 against target (0,1) at l0.
    block = zone(ctx, gt("x", 0), le("x", 2))
    z_minus = intersect(block, pre_edge(open01, e2))
    assert z_minus == open01 and z_minus != block
    assert subtract(block, z_minus) == [closed12]

    # Delay predecessors are downward closures.
    assert down(above2) == ctx.universe()
    assert down(closed12) == zone(ctx, le("x", 2))


# ----------------------------------------------------------- 2. strictness


def test_open_interval_excludes_its_endpoints():
    ctx = ZoneContext(("x",))
    closed = zone(ctx, ge("x", 0), le("x", 1))
    open_ = zone(ctx, gt("x", 0), lt("x", 1))
    pieces = subtract(closed, open_)
    assert set(pieces) == {zone(ctx, eq("x", 0)), zone(ctx, eq("x", 1))}
    assert_exact_difference(closed, open_, pieces)


# ------------------------------------------- 3. diagonals, partial resets


def test_pre_edge_with_diagonal_guard_and_partial_reset():
    ctx = ZoneContext(("x", "y"))
    e = edge(guard=(diff_le("x", "y", 1),), resets={"y"})

    # v[y := 0] lands in {x >= 2, y = 0} iff x >= 2; the guard adds x - y <= 1.
    target = zone(ctx, ge("x", 2), eq("y", 0))
    assert pre_edge(target, e) == zone(ctx, ge("x", 2), diff_le("x", "y", 1))

    # After resetting y it can never equal 1.
    assert pre_edge(zone(ctx, eq("y", 1)), e).is_empty


# ----------------------------------------------------------- 4. L-shape


def test_l_shape_subtraction_is_exact_and_pieces_lie_in_bars():
    ctx = ZoneContext(("x", "y"))
    big = zone(ctx, le("x", 2), le("y", 2))
    small = zone(ctx, ge("x", 1), ge("y", 1), le("x", 2), le("y", 2))
    pieces = subtract(big, small)
    assert_exact_difference(big, small, pieces)

    vertical_bar = zone(ctx, lt("x", 1), le("y", 2))
    horizontal_bar = zone(ctx, le("x", 2), lt("y", 1))
    for p in pieces:
        assert p <= vertical_bar or p <= horizontal_bar, str(p)


# ------------------------------------------------- random zone generators

CASES = [(n, k) for n in (1, 2, 3) for k in (1, 2, 3)]
ROUNDS = 200


def random_constraint(rng, clocks, k):
    left, right = rng.sample(["0", *clocks], 2)
    return Constraint(left, right, rng.randint(-k, k), rng.random() < 0.5)


def random_zone(rng, ctx, clocks, k):
    """Nonempty zone defined by 1-4 random constraints with |c| <= k."""
    while True:
        cs = tuple(random_constraint(rng, clocks, k) for _ in range(rng.randint(1, 4)))
        z = ctx.from_guard(cs)
        if not z.is_empty:
            return z


def random_edge(rng, clocks, k):
    guard = tuple(random_constraint(rng, clocks, k) for _ in range(rng.randint(0, 2)))
    resets = {c for c in clocks if rng.random() < 0.4}
    return edge(guard=guard, resets=resets)


def setup(n, k):
    clocks = tuple("xyz"[:n])
    return random.Random(1000 * n + k), clocks, ZoneContext(clocks)


# --------------------------------------------------------- 5. extrapolation


@pytest.mark.parametrize("n,k", CASES)
def test_extrapolation_properties(n, k):
    rng, clocks, ctx = setup(n, k)
    for _ in range(ROUNDS):
        z = random_zone(rng, ctx, clocks, k + 2)  # constants may exceed k
        ez = extrapolate(z, k)
        assert z <= ez, str(z)
        assert extrapolate(ez, k) == ez, str(z)
        assert is_k_bounded(ez, k), str(z)
        assert extrapolate(z, k + 1) <= ez, str(z)

        zk = random_zone(rng, ctx, clocks, k)  # already k-bounded
        assert extrapolate(zk, k) == zk, str(zk)


# ---------------------------------------------- 6. closure of K-bounded zones


@pytest.mark.parametrize("n,k", CASES)
def test_k_bounded_zones_closed_under_algorithm_operations(n, k):
    rng, clocks, ctx = setup(n, k)
    for _ in range(ROUNDS):
        z = random_zone(rng, ctx, clocks, k)
        w = random_zone(rng, ctx, clocks, k)
        e = random_edge(rng, clocks, k)
        results = {
            "up": [up(z)],
            "down": [down(z)],
            "intersect": [intersect(z, w)],
            "pre_edge": [pre_edge(z, e)],
            "subtract": subtract(z, w),
        }
        for op, zones_out in results.items():
            for r in zones_out:
                assert is_k_bounded(r, k), f"{op}: z={z}, w={w}, result={r}"


def test_reset_can_break_k_boundedness():
    """Counterexample found by the random test (n = 2, K = 1).

    z = {x < 1, x - y >= -1} is 1-bounded; it implies y < 2. Resetting x
    forgets the diagonal and leaves y < 2, which needs the constant 2 > K.
    In the algorithm this is harmless: a reset only happens on line 11, and
    Insert extrapolates the result before it reaches Pi_K.
    """
    ctx = ZoneContext(("x", "y"))
    z = zone(ctx, lt("x", 1), diff_le("y", "x", 1))
    assert is_k_bounded(z, 1)
    after = reset(z, {"x"})
    assert after == zone(ctx, eq("x", 0), lt("y", 2))
    assert not is_k_bounded(after, 1)
    assert is_k_bounded(extrapolate(after, 1), 1)


@pytest.mark.parametrize("n,k", CASES)
def test_random_subtraction_is_exact(n, k):
    rng, clocks, ctx = setup(n, k)
    for _ in range(ROUNDS):
        z = random_zone(rng, ctx, clocks, k)
        w = random_zone(rng, ctx, clocks, k)
        assert_exact_difference(z, w, subtract(z, w))