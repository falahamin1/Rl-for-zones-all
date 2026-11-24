"""Acceptance tests for step 10: integer edge identity, fast cached _own_bound, validation."""

import json
import random
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from brg_examples import e1, e2, e3, e4, e5, e6, e7, r072  # noqa: E402
from p3_example import pta_p3  # noqa: E402
from prestable import algorithm, model  # noqa: E402
from prestable.block_vi import build_block_mdp, solve  # noqa: E402
from prestable.export import from_json, to_json  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import DELAY, Constraint, Edge, PEdge, TimedAutomaton, ge, le, lt  # noqa: E402
from prestable.zones import ZoneContext, _own_bound, extrapolate, intersect, is_k_bounded  # noqa: E402
from test_phase1 import running_example, two_clock_example  # noqa: E402

LAM = 0.5
TOL = 1e-9

FAST_MODELS = {
    "running": running_example, "two_clock": two_clock_example,
    "E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7, "R072": r072,
}
SLOW_MODELS = {"P3": lambda: pta_p3(invariants=False), "P3_inv": lambda: pta_p3(invariants=True)}
ALL_MODELS = [pytest.param(n, id=n) for n in FAST_MODELS] + \
             [pytest.param(n, id=n, marks=pytest.mark.slow) for n in SLOW_MODELS]


@lru_cache(maxsize=None)
def model_and_graph(name):
    ta = {**FAST_MODELS, **SLOW_MODELS}[name]()
    return ta, build_prestable_graph(ta).graph


def rebuilt(ta, edges=None):
    """An equal automaton with fresh Edge objects (or with the given edges)."""
    if edges is None:
        edges = tuple(PEdge(e.source, e.action, e.guard, e.cost, e.branches) for e in ta.edges)
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, edges, rates=ta.rates, invariants=ta.invariants)


def value(ta, graph, variant):
    return solve(ta, ["g"], LAM, variant=variant, allow_deadlocks=True, graph=graph).value


def solvable(ta):
    """The block MDP needs the target g and clocks bounded by invariants. The running and
    two-clock examples have neither (solve raises 'unknown target locations' for g and
    'no upper bound' for any other target), so for them only the graphs are compared."""
    return "g" in ta.locations


def graph_signature(ta, g):
    """Nodes as (location, zone key) and action edges as (src, TA edge value, tgt); delay edges
    as (src, tgt). Independent of node numbering and of the order of ta.edges."""
    node = {n.id: (n.location, n.zone.key()) for n in g.nodes}
    return (Counter(node.values()),
            Counter((node[e.source], e.label, None if e.edge is None else ta.edges[e.edge], node[e.target])
                    for e in g.edges))


# --- edge identity (10.1) ------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_MODELS)
def test_a1_edge_fields(name):
    ta, g = model_and_graph(name)
    for ge_ in g.edges:
        assert (ge_.edge is None) == (ge_.label == DELAY)
        if ge_.label == DELAY:
            assert ge_.edge is None and ge_.ta_edge is None
        else:
            assert isinstance(ge_.edge, int) and ta.edges[ge_.edge] is ge_.ta_edge


@pytest.mark.parametrize("name", ALL_MODELS)
def test_a2_edge_index_matches_index_lookup(name):
    ta, g = model_and_graph(name)
    if len(set(ta.edges)) != len(ta.edges):
        pytest.skip("model has value-identical edges")
    actions = [e for e in g.edges if e.label != DELAY]
    by_field = Counter((g.nodes[e.source].location, e.edge) for e in actions)
    by_lookup = Counter((g.nodes[e.source].location, ta.edges.index(e.ta_edge)) for e in actions)
    assert by_field == by_lookup


@pytest.mark.parametrize("name", ["E3", "E7", "R072", "two_clock"])
def test_a3_reversed_edge_order(name):
    ta, g = model_and_graph(name)
    ta_rev = rebuilt(ta, tuple(reversed(ta.edges)))
    g_rev = build_prestable_graph(ta_rev).graph
    if solvable(ta):
        for variant in ("boundary", "integer"):
            assert value(ta_rev, g_rev, variant) == pytest.approx(value(ta, g, variant), abs=TOL)
    else:
        assert graph_signature(ta_rev, g_rev) == graph_signature(ta, g)
    assert ta_rev.edges != ta.edges
    target = "g" if solvable(ta) else ta.locations[-1]  # an existing target, so the edge check is reached
    with pytest.raises(ValueError, match="another automaton"):
        build_block_mdp(ta_rev, g, [target], variant="boundary", allow_deadlocks=True)


def twin():
    return TimedAutomaton(("x",), ("l0", "g"), "l0",
                          (Edge("l0", "g", "a", guard=(ge("x", 1),), cost=1),
                           Edge("l0", "g", "a", guard=(ge("x", 1),), cost=1)),
                          rates={"l0": 1}, invariants={"l0": (le("x", 2),)})


def test_a4_value_identical_duplicates():
    ta, ta2 = twin(), twin()
    assert ta.edges[0] == ta.edges[1] and ta2 == ta and ta2.edges[0] is not ta.edges[0]
    g = build_prestable_graph(ta).graph
    assert {e.edge for e in g.edges if e.label != DELAY} == {0, 1}
    values = []
    for t in (ta, ta2):
        for variant in ("boundary", "integer"):
            build_block_mdp(t, g, ["g"], variant=variant)
            values.append(solve(t, ["g"], LAM, variant=variant, graph=g).value)
    assert values == pytest.approx([values[0]] * 4, abs=TOL)


@pytest.mark.parametrize("name", list(FAST_MODELS))
def test_a5_json_round_trip_with_rebuilt_automaton(name):
    ta, g = model_and_graph(name)
    ta2 = rebuilt(ta)
    back = from_json(json.loads(json.dumps(to_json(g, ta))), ta2)
    assert back.initial == g.initial
    assert [(n.id, n.location, n.zone.key()) for n in back.nodes] == \
           [(n.id, n.location, n.zone.key()) for n in g.nodes]
    assert [(e.source, e.label, e.target, e.edge, e.cost) for e in back.edges] == \
           [(e.source, e.label, e.target, e.edge, e.cost) for e in g.edges]
    for e in back.edges:
        assert e.ta_edge is None if e.edge is None else e.ta_edge == ta2.edges[e.edge]
    if not solvable(ta):
        return
    for variant in ("boundary", "integer"):
        assert value(ta2, back, variant) == pytest.approx(value(ta, g, variant), abs=TOL)


# --- zones (10.2) ----------------------------------------------------------------------

CONTEXTS = {n: ZoneContext([f"c{i}" for i in range(n)]) for n in (1, 2, 3)}


def random_guards(count, seed):
    """(ctx, guard) pairs with nonempty zones: 1-3 clocks, constants up to 64."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        ctx = CONTEXTS[rng.choice((1, 2, 3))]
        names = ("0",) + ctx.clocks
        K = rng.randint(0, 64)
        guard = [Constraint(*rng.sample(names, 2), rng.randint(-K, K), rng.random() < 0.5)
                 for _ in range(rng.randint(0, 6))]
        if not ctx.from_guard(guard).is_empty:
            out.append((ctx, guard))
    return out


def test_a6_own_bound_is_the_smallest_bound():
    for ctx, guard in random_guards(1000, seed=1006):
        w = ctx.from_guard(guard)
        b = _own_bound(w)
        assert is_k_bounded(w, b), str(w)
        assert b == 0 or not is_k_bounded(w, b - 1), str(w)


def test_a7_constraints_rebuild_the_zone():
    for ctx, guard in random_guards(1000, seed=1006):
        z = ctx.from_guard(guard)
        assert ctx.from_guard(z.constraints()) == z, str(z)


def test_a8_cache_consistency_across_routes():
    gs = random_guards(400, seed=1008)
    rng = random.Random(1008)
    for ctx, guard in gs:
        z = ctx.from_guard(guard)
        # intersect in either order, with a second random zone of the same context
        other = next(ctx.from_guard(g) for c, g in rng.sample(gs, len(gs)) if c is ctx)
        a, b = intersect(z, other), intersect(other, z)
        assert a == b and set(a.constraints()) == set(b.constraints())
        # from_guard vs extrapolate of a zone that is already K-bounded
        k = _own_bound(z) + rng.randint(0, 3)
        x = extrapolate(ctx.from_guard(guard), k)
        assert x == z and set(x.constraints()) == set(z.constraints())
        # from_guard(constraints()) vs the original, and repeated calls
        y = ctx.from_guard(z.constraints())
        assert set(y.constraints()) == set(z.constraints())
        assert z.constraints() == z.constraints() and y.constraints() == y.constraints()


# --- validation (10.3) -------------------------------------------------------------------

def test_a9_delay_name_is_shared():
    assert model.DELAY is algorithm.DELAY


def test_a10_reserved_action_name():
    def ta(action):
        return TimedAutomaton(("x",), ("l0", "l1"), "l0",
                              (Edge("l0", "l1", "a"), Edge("l1", "l0", action)))
    with pytest.raises(ValueError, match="reserved"):
        ta("delay")  # edge leaves l1, not the initial location
    for ok in ("Delay", "delay_1", "d"):
        ta(ok)


def test_a11_initial_invariant():
    def ta(inv):
        return TimedAutomaton(("x",), ("l0", "l1"), "l0", (), invariants=inv)
    ta({"l0": (le("x", 0),)})
    with pytest.raises(ValueError):
        ta({"l0": (lt("x", 0),)})
    ta({"l1": (le("x", 0),)})
