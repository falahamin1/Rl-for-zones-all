import random
import time
from collections import Counter

import pytest

from prestable.algorithm import DELAY, check_phase1, check_phase2, phase1, phase2, transitions
from prestable.graph import build_graph, build_prestable_graph, reachable
from prestable.model import Edge, TimedAutomaton, ge, gt, le, lt
from prestable.zones import ZoneContext, intersect
from test_phase1 import random_automaton, running_example, two_clock_example

STATS = {"max_graph": (0, 0), "max_graph_case": None, "slowest": 0.0, "slowest_case": None}


def Z(ctx, *constraints):
    return ctx.from_guard(constraints)


def run_phase2(ta, ctx=None, **kwargs):
    ctx = ctx or ZoneContext(ta.clocks)
    p1 = phase1(ta, ctx)
    p2 = phase2(ta, ctx, p1.store, **kwargs)
    check_phase2(ta, ctx, p2.store, p1.K)
    return ctx, p1, p2


def check_graph(ta, ctx, store, graph):
    """Every edge is justified, and there is exactly one edge per nonempty triple."""
    ts = transitions(ta)
    nonempty = 0
    for t in ts:
        for Zs in store[t.source]:
            for Zt in store[t.target]:
                if not intersect(Zs, t.pre(Zt)).is_empty:
                    nonempty += 1
    assert len(graph.edges) == nonempty
    for e in graph.edges:
        src, tgt = graph.nodes[e.source], graph.nodes[e.target]
        t = next(t for t in ts if t.edge is e.ta_edge and t.label == e.label and t.source == src.location)
        assert tgt.location == t.target
        assert src.zone <= t.pre(tgt.zone), (e, str(src.zone), str(tgt.zone))
        assert e.cost == (None if e.ta_edge is None else e.ta_edge.cost)
    assert graph.nodes[graph.initial].location == ta.initial
    assert graph.nodes[graph.initial].zone == ctx.zero()
    assert [n.id for n in graph.nodes] == list(range(len(graph.nodes)))


# --- running example ----------------------------------------------------------

def test_running_example_phase1_store_is_not_prestable():
    ta = running_example()
    ctx = ZoneContext(ta.clocks)
    p1 = phase1(ta, ctx)
    with pytest.raises(ValueError, match="not pre-stable"):
        check_phase2(ta, ctx, p1.store, p1.K)


def test_running_example_phase2():
    ta = running_example()
    ctx = ZoneContext(ta.clocks)
    p1 = phase1(ta, ctx)
    before = dict(p1.store)
    p2 = phase2(ta, ctx, p1.store, trace=True)
    check_phase2(ta, ctx, p2.store, p1.K)
    assert p1.store == before  # input not mutated

    open01, closed12 = Z(ctx, gt("x", 0), lt("x", 1)), Z(ctx, ge("x", 1), le("x", 2))
    assert p2.splits == 1 and len(p2.trace) == 1
    ev = p2.trace[0]
    assert ev.transition.label == "b" and ev.transition.source == "l1"
    assert ev.block == Z(ctx, gt("x", 0), le("x", 2))
    # Z′ is scanned in key() order, which puts [1,2] before (0,1) (key() starts with the
    # lower bound), so the first bad triple found targets [1,2], not (0,1).
    assert ev.target == closed12 and ev.z_minus == closed12
    assert ev.pieces == {open01, closed12}

    expected = {ctx.zero(), open01, closed12, Z(ctx, gt("x", 2))}
    assert p2.store == {"l0": expected, "l1": expected}


def test_running_example_graph():
    ta = running_example()
    r = build_prestable_graph(ta)
    g = r.graph
    ctx = g.nodes[0].zone.ctx
    check_graph(ta, ctx, r.phase2.store, g)
    assert len(g.nodes) == 8
    assert Counter(e.label for e in g.edges) == {DELAY: 20, "a": 2, "b": 3}
    assert g.initial == g.node_id("l0", ctx.zero())
    assert reachable(g) == set(range(8))

    zero, open01 = ctx.zero(), Z(ctx, gt("x", 0), lt("x", 1))
    closed12, above2 = Z(ctx, ge("x", 1), le("x", 2)), Z(ctx, gt("x", 2))
    nid = g.node_id

    def edges(label):
        return {(e.source, e.target) for e in g.edges if e.label == label}

    assert edges("a") == {(nid("l0", closed12), nid("l1", zero)), (nid("l0", above2), nid("l1", zero))}
    assert edges("b") == {(nid("l1", z), nid("l0", z)) for z in (zero, open01, closed12)}
    in_time = [zero, open01, closed12, above2]
    assert edges(DELAY) == {(nid(l, in_time[i]), nid(l, in_time[j]))
                            for l in ("l0", "l1") for i in range(4) for j in range(i, 4)}
    for e in g.edges:
        assert (e.ta_edge is None) == (e.label == DELAY)


def test_self_loop():
    ta = TimedAutomaton(("x",), ("l",), "l", (Edge("l", "l", "a", guard=(ge("x", 1),), resets={"x"}),))
    r = build_prestable_graph(ta)
    g = r.graph
    ctx = g.nodes[0].zone.ctx
    check_phase2(ta, ctx, r.phase2.store, r.phase1.K)
    check_graph(ta, ctx, r.phase2.store, g)
    assert r.phase2.splits == 0
    assert len(g.nodes) == 3
    assert Counter(e.label for e in g.edges) == {DELAY: 6, "a": 1}
    (a,) = [e for e in g.edges if e.label == "a"]
    assert (a.source, a.target) == (g.node_id("l", Z(ctx, ge("x", 1))), g.node_id("l", ctx.zero()))


def test_same_action_edges_stay_separate():
    e1 = Edge("l0", "l1", "a", guard=(ge("x", 1),), resets={"x"}, cost=1.0)
    e2 = Edge("l0", "l1", "a", guard=(ge("x", 2),), resets={"x"}, cost=5.0)
    ta = TimedAutomaton(("x",), ("l0", "l1"), "l0", (e1, e2))
    r = build_prestable_graph(ta)
    g = r.graph
    ctx = g.nodes[0].zone.ctx
    check_graph(ta, ctx, r.phase2.store, g)
    a_edges = [e for e in g.edges if e.label == "a"]
    assert {e.ta_edge for e in a_edges} == {e1, e2}
    assert r.phase2.store["l0"] == {ctx.zero(), Z(ctx, gt("x", 0), lt("x", 1)),
                                     Z(ctx, ge("x", 1), lt("x", 2)), Z(ctx, ge("x", 2))}
    # From x >= 2 both edges are enabled: two separate graph edges between the same nodes.
    src, tgt = g.node_id("l0", Z(ctx, ge("x", 2))), g.node_id("l1", ctx.zero())
    parallel = [e for e in a_edges if (e.source, e.target) == (src, tgt)]
    assert {(e.ta_edge, e.cost) for e in parallel} == {(e1, 1.0), (e2, 5.0)}
    for e in a_edges:
        assert g.nodes[e.source].zone <= ctx.from_guard(e.ta_edge.guard)


def test_two_clocks():
    ta = two_clock_example()
    start = time.perf_counter()
    r = build_prestable_graph(ta)
    elapsed = time.perf_counter() - start
    g = r.graph
    ctx = g.nodes[0].zone.ctx
    check_phase2(ta, ctx, r.phase2.store, r.phase1.K)
    check_graph(ta, ctx, r.phase2.store, g)
    labels = dict(sorted(Counter(e.label for e in g.edges).items()))
    print(f"\ntwo-clock example: splits={r.phase2.splits}, "
          f"blocks={ {l: len(P) for l, P in r.phase2.store.items()} }, nodes={len(g.nodes)}, "
          f"edges={len(g.edges)} {labels}, reachable={len(reachable(g))}, runtime={elapsed:.2f}s")


def test_deterministic():
    for ta in (running_example(), two_clock_example()):
        ctx = ZoneContext(ta.clocks)
        p1 = phase1(ta, ctx)
        a, b = phase2(ta, ctx, p1.store), phase2(ta, ctx, p1.store)
        assert a.store == b.store and a.splits == b.splits
        ga, gb = build_graph(ta, ctx, a.store), build_graph(ta, ctx, b.store)
        assert ga.nodes == gb.nodes and ga.edges == gb.edges and ga.initial == gb.initial


def test_max_splits():
    ta = running_example()
    ctx = ZoneContext(ta.clocks)
    p1 = phase1(ta, ctx)
    with pytest.raises(RuntimeError, match="max_splits"):
        phase2(ta, ctx, p1.store, max_splits=0)


def test_build_graph_requires_initial_zero_block():
    ta = running_example()
    ctx = ZoneContext(ta.clocks)
    store = {"l0": frozenset({ctx.universe()}), "l1": frozenset()}
    with pytest.raises(ValueError, match="initial"):
        build_graph(ta, ctx, store)


def test_phase2_keeps_phase1_properties():
    ta = two_clock_example()
    ctx, p1, p2 = run_phase2(ta)
    check_phase1(ta, ctx, p2.store, p1.K)
    # Phase 2 only refines: every new block lies in exactly one Phase 1 block.
    for l, P in p2.store.items():
        for b in P:
            assert sum(b <= old for old in p1.store[l]) == 1


# --- random -----------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("n_clocks", [1, 2, 3])
def test_random_automata(n_clocks):
    rng = random.Random(500 + n_clocks)  # same automata as the step 3 random test
    for i in range(30):
        ta = random_automaton(rng, n_clocks)
        start = time.perf_counter()
        r = build_prestable_graph(ta)
        elapsed = time.perf_counter() - start
        g = r.graph
        ctx = ZoneContext(ta.clocks) if not g.nodes else g.nodes[0].zone.ctx
        check_phase2(ta, ctx, r.phase2.store, r.phase1.K)
        check_graph(ta, ctx, r.phase2.store, g)
        assert g.initial in reachable(g)
        size = (len(g.nodes), len(g.edges))
        case = f"{n_clocks} clocks #{i}: nodes={size[0]}, edges={size[1]}, splits={r.phase2.splits}, " \
               f"K={r.phase1.K}, ta_edges={len(ta.edges)}, locations={len(ta.locations)}"
        if size > STATS["max_graph"]:
            STATS.update(max_graph=size, max_graph_case=case)
        if elapsed > STATS["slowest"]:
            STATS.update(slowest=elapsed, slowest_case=case)


def test_zz_report():
    """Runs last in this file; prints the random-test indicators with -s."""
    print(f"\nlargest graph: {STATS['max_graph_case']}")
    print(f"slowest build_prestable_graph: {STATS['slowest']:.2f}s ({STATS['slowest_case']})")


# --- breadth-first numbering ------------------------------------------------------

def bfs_distances(g):
    dist = {g.initial: 0}
    frontier = [g.initial]
    while frontier:
        nxt = []
        for u in frontier:
            for e in g.successors(u):
                if e.target not in dist:
                    dist[e.target] = dist[u] + 1
                    nxt.append(e.target)
        frontier = nxt
    return dist


def unreachable_example():
    return TimedAutomaton(
        clocks=("x",),
        locations=("l0", "l1", "island"),
        initial="l0",
        edges=(
            Edge("l0", "l1", "a", guard=(ge("x", 1),), resets={"x"}),
            Edge("l1", "l0", "b", guard=(le("x", 2),)),
            Edge("island", "l0", "c", guard=(le("x", 1),)),
        ),
    )


@pytest.mark.parametrize("make_ta", [running_example, two_clock_example, unreachable_example])
def test_bfs_numbering(make_ta):
    ta = make_ta()
    g = build_prestable_graph(ta).graph
    assert g.initial == 0
    reach = reachable(g)
    assert reach == set(range(len(reach)))
    dist = bfs_distances(g)
    assert [dist[i] for i in range(len(reach))] == sorted(dist[i] for i in range(len(reach)))
    # Edges sorted by source id; within a source, by transition order then target id.
    order = {t: i for i, t in enumerate(transitions(ta))}
    def t_index(e):
        src = g.nodes[e.source].location
        return next(i for t, i in order.items()
                    if t.edge is e.ta_edge and t.label == e.label and t.source == src)
    keys = [(e.source, t_index(e), e.target) for e in g.edges]
    assert keys == sorted(keys)
    # Same numbering on a second run.
    g2 = build_prestable_graph(make_ta()).graph
    assert [(n.id, n.location, str(n.zone)) for n in g.nodes] == [(n.id, n.location, str(n.zone)) for n in g2.nodes]
    assert [(e.source, e.label, e.target) for e in g.edges] == [(e.source, e.label, e.target) for e in g2.edges]


def test_bfs_numbering_running_example_order():
    g = build_prestable_graph(running_example()).graph
    ctx = g.nodes[0].zone.ctx
    # Initial first; its delay successors next, in target key() order (latest first).
    assert [str(n.zone) for n in g.nodes[:4]] == ["x == 0", "x > 2", "1 <= x <= 2", "0 < x < 1"]
    assert all(n.location == "l0" for n in g.nodes[:4])
    assert g.nodes[4].location == "l1" and g.nodes[4].zone == ctx.zero()


def test_two_clock_reference_numbers():
    r = build_prestable_graph(two_clock_example())
    g = r.graph
    assert r.phase2.splits == 12
    assert len(g.nodes) == 35 and len(g.edges) == 120
    assert Counter(e.label for e in g.edges) == {DELAY: 103, "a": 12, "b": 5}
