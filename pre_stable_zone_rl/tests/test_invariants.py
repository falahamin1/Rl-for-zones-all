import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from p3_example import pta_p3, zeta, zeta_blocks  # noqa: E402
from prestable.algorithm import DELAY, check_phase1, check_phase2, phase1, phase2, transitions  # noqa: E402
from prestable.graph import build_prestable_graph, reachable  # noqa: E402
from prestable.model import Constraint, Edge, TimedAutomaton, ge, gt, le, lt  # noqa: E402
from prestable.zones import ZoneContext, subtract_all  # noqa: E402
from test_phase1 import random_automaton  # noqa: E402


def Z(ctx, *constraints):
    return ctx.from_guard(constraints)


def run(ta):
    ctx = ZoneContext(ta.clocks)
    p1 = phase1(ta, ctx)
    check_phase1(ta, ctx, p1.store, p1.K)
    p2 = phase2(ta, ctx, p1.store)
    check_phase2(ta, ctx, p2.store, p1.K)
    return ctx, p1, p2


def test_invariant_caps_the_partition():
    ta = TimedAutomaton(("x",), ("l0", "l1"), "l0", (Edge("l0", "l1", "a", guard=(ge("x", 1),), resets={"x"}),),
                        invariants={"l0": (le("x", 2),)})
    ctx, p1, _ = run(ta)
    # The face x = 2 of the invariant is a block of its own (invariant atom x < 2).
    assert p1.store["l0"] == {ctx.zero(), Z(ctx, gt("x", 0), lt("x", 1)), Z(ctx, ge("x", 1), lt("x", 2)),
                              Z(ctx, ge("x", 2), le("x", 2))}


def test_pre_delay_with_invariant():
    ta = TimedAutomaton(("x",), ("l",), "l", (), invariants={"l": (le("x", 2),)})
    ctx = ZoneContext(ta.clocks)
    (delay,) = [t for t in transitions(ta) if t.label == DELAY]
    assert delay.pre(Z(ctx, ge("x", 1), le("x", 2))) == Z(ctx, le("x", 2))
    # The invariant is applied even to a target outside it.
    assert delay.pre(Z(ctx, ge("x", 3), le("x", 4))) == Z(ctx, le("x", 2))


def test_successor_cut_by_target_invariant():
    ta = TimedAutomaton(("x",), ("l0", "l1"), "l0", (Edge("l0", "l1", "a"),),
                        invariants={"l0": (le("x", 3),), "l1": (le("x", 1),)})
    ctx, p1, p2 = run(ta)
    for store in (p1.store, p2.store):
        cover = Z(ctx, le("x", 1))
        assert not subtract_all(cover, store["l1"])
        assert all(b <= cover for b in store["l1"])
    # Phase 2 splits l0 at x = 1: only x <= 1 can take a.
    assert any(b == Z(ctx, gt("x", 0), le("x", 1)) or b <= Z(ctx, le("x", 1)) for b in p2.store["l0"])
    assert all(b <= Z(ctx, le("x", 1)) or b <= Z(ctx, gt("x", 1)) for b in p2.store["l0"])


def test_initial_invariant_must_contain_zero():
    # validate() now rejects x < 0 at the initial location (test_model.py); phase1 keeps its own
    # check, exercised here on an automaton whose invariant is swapped in after validation.
    ta = TimedAutomaton(("x",), ("l0",), "l0", (), invariants={"l0": (le("x", 1),)})
    object.__setattr__(ta, "invariants", {"l0": (lt("x", 0),)})
    with pytest.raises(ValueError, match="initial"):
        phase1(ta, ZoneContext(ta.clocks))


def test_no_invariants_unchanged():
    # Same automaton with and without an empty invariants dict gives the same store.
    edges = (Edge("l0", "l1", "a", guard=(ge("x", 1),), resets={"x"}), Edge("l1", "l0", "b", guard=(le("x", 2),)))
    a = TimedAutomaton(("x",), ("l0", "l1"), "l0", edges)
    b = TimedAutomaton(("x",), ("l0", "l1"), "l0", edges, invariants={"l0": (), "l1": ()})
    ctx = ZoneContext(("x",))
    assert phase1(a, ctx).store == phase1(b, ctx).store


# --- P3 with real invariants ----------------------------------------------------

@pytest.fixture(scope="module")
def p3_inv():
    ta = pta_p3(invariants=True)
    r = build_prestable_graph(ta)
    ctx = r.graph.nodes[0].zone.ctx
    return ta, ctx, r


def test_p3_inv_checks(p3_inv):
    ta, ctx, r = p3_inv
    check_phase1(ta, ctx, r.phase1.store, r.phase1.K)
    check_phase2(ta, ctx, r.phase2.store, r.phase1.K)


def test_p3_inv_blocks_in_square(p3_inv):
    ta, ctx, r = p3_inv
    square = Z(ctx, le("x", 2), le("y", 2))
    for store in (r.phase1.store, r.phase2.store):
        for l, P in store.items():
            assert all(b <= square for b in P), l


def test_p3_inv_zeta(p3_inv):
    _, ctx, r = p3_inv
    for store in (r.phase1.store, r.phase2.store):
        zb = zeta_blocks(store["l1"], ctx)  # raises if a block straddles ζ
        assert all(not subtract_all(z, zb) for z in zeta(ctx))


def test_p3_inv_dead_blocks_are_exactly_the_l3_time_locks(p3_inv):
    """A reachable node is dead if no delay path leads to a node with an action edge.

    The spec expected none, but P3 with invariants has time-locks at l3: c (no guard,
    no reset) can enter l3 with x - y < -1, and there d needs x >= 1 while the invariant
    y <= 2 allows at most 2 - y more time, so x stays below x + 2 - y < 1. l2 has no
    such nodes because b enters it only with x, y <= 1.
    """
    _, ctx, r = p3_inv
    g = r.graph
    has_action = {e.source for e in g.edges if e.label != DELAY}
    dead = set()
    for n in reachable(g):
        seen, stack = {n}, [n]
        while stack:
            for e in g.successors(stack.pop()):
                if e.label == DELAY and e.target not in seen:
                    seen.add(e.target)
                    stack.append(e.target)
        if not seen & has_action:
            dead.add(n)
    lock = ctx.from_guard([Constraint("x", "y", -1, True)])  # x - y < -1
    expected = {n.id for n in g.nodes if n.location == "l3" and n.zone <= lock and n.id in reachable(g)}
    assert dead == expected
    assert len(dead) == 4


# --- random -----------------------------------------------------------------

def with_invariants(rng, ta):
    inv = {l: tuple(le(c, rng.randint(1, 3)) for c in ta.clocks) for l in ta.locations}
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, ta.edges, invariants=inv)


@pytest.mark.slow
@pytest.mark.parametrize("n_clocks", [1, 2, 3])
def test_random_automata_with_invariants(n_clocks):
    rng = random.Random(900 + n_clocks)
    for _ in range(30):
        ta = with_invariants(rng, random_automaton(rng, n_clocks))
        ctx, _, p2 = run(ta)
        for l, P in p2.store.items():
            inv = ctx.from_guard(ta.invariant(l))
            assert all(b <= inv for b in P), l
        build_prestable_graph(ta)
