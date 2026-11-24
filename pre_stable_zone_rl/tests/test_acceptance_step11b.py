"""Acceptance tests for step 11b: Phases 1-3 per branch, on M1-M3 and the random probabilistic
batch of step 11b (random_probabilistic(200), seeds 2026 / 11; the batch tests are slow)."""

import itertools
import json
import re
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from prestable.algorithm import DELAY  # noqa: E402
from prestable.export import from_json, to_dot, to_json  # noqa: E402
from prestable.graph import ZoneGraph, build_prestable_graph, check_branches  # noqa: E402
from prestable.model import ZERO, Branch, Edge, PEdge, TimedAutomaton, ge, le  # noqa: E402
from pta_examples import MODELS, random_probabilistic  # noqa: E402

HALF = Fraction(1, 2)
SAMPLES = 50


# --- models and cached builds ------------------------------------------------------

@lru_cache(maxsize=None)
def kept_random():
    return tuple((n, ta) for n, ta in random_probabilistic(200) if ta is not None)


@lru_cache(maxsize=None)
def model(name):
    if name in MODELS:
        return MODELS[name]()
    return dict(kept_random())[name]


@lru_cache(maxsize=None)
def build(name):
    return build_prestable_graph(model(name))


def random_names():
    names = [n for n, _ in kept_random()]
    assert len(names) == 150, len(names)  # the 11b report: 150 kept
    return names


def run_all(check, names):
    """Run check(name) for every name; fail with every failing name (and the first message)."""
    failures = []
    for n in names:
        try:
            check(n)
        except AssertionError as exc:
            failures.append((n, str(exc)[:300]))
    assert not failures, f"{len(failures)} of {len(names)} models fail; first: {failures[0]}"


FAST = list(MODELS)


def nodes_view(g):
    return [(n.id, n.location, n.zone.key()) for n in g.nodes]


# --- C1: duplicate branches = deterministic edge ---------------------------------

def dup_pair(clocks, resets):
    inv = {l: tuple(le(c, 3) for c in clocks) for l in ("l0", "l1")}
    guard = (ge("x", 1),)
    b = (Edge("l1", "g", "b", guard=(ge("x", 2),)),)

    def ta(a):
        return TimedAutomaton(clocks, ("l0", "l1", "g"), "l0", (a,) + b, invariants=inv)
    return (ta(PEdge("l0", "a", guard, 0, (Branch(HALF, resets, "l1"), Branch(HALF, resets, "l1")))),
            ta(Edge("l0", "l1", "a", guard, resets)))


@pytest.mark.parametrize("clocks, resets", [(("x",), ()), (("x",), {"x"}), (("x", "y"), {"y"}), (("x", "y"), {"x", "y"})],
                         ids=["1c-none", "1c-x", "2c-y", "2c-xy"])
def test_c1_duplicate_branches_equal_deterministic_edge(clocks, resets):
    ta_dup, ta_det = dup_pair(clocks, resets)
    g_dup, g_det = build_prestable_graph(ta_dup).graph, build_prestable_graph(ta_det).graph
    assert nodes_view(g_dup) == nodes_view(g_det)
    delay = lambda g: [(e.source, e.target) for e in g.edges if e.label == DELAY]  # noqa: E731
    assert delay(g_dup) == delay(g_det)
    # edge 0 (a) appears once per branch; edge 1 (b) is deterministic in both automata
    det = sorted((e.source, e.label, e.target, e.edge, b) for e in g_det.edges if e.label != DELAY
                 for b in ((0, 1) if e.edge == 0 else (0,)))
    dup = sorted((e.source, e.label, e.target, e.edge, e.branch) for e in g_dup.edges if e.label != DELAY)
    assert dup == det
    assert all(e.branch == 0 for e in g_det.edges if e.label != DELAY)


# --- C2: probabilities don't shape the graph -------------------------------------

def reweighted(ta):
    def edge(e):
        if [b.prob for b in e.branches] != [HALF, HALF]:
            return e
        b0, b1 = e.branches
        return PEdge(e.source, e.action, e.guard, e.cost,
                     (Branch(Fraction(1, 4), b0.resets, b0.target), Branch(Fraction(3, 4), b1.resets, b1.target)))
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, tuple(map(edge, ta.edges)),
                          rates=ta.rates, invariants=ta.invariants)


P_LABEL = re.compile(r" p=\d+(/\d+)?")


def check_c2(name):
    ta = model(name)
    ta2 = reweighted(ta)
    assert ta2 != ta
    g, g2 = build(name).graph, build_prestable_graph(ta2).graph
    assert to_json(g2, ta2) == to_json(g, ta)
    d, d2 = to_dot(g, ta), to_dot(g2, ta2)
    if any(e.label != DELAY and not ta.edges[e.edge].is_deterministic for e in g.edges):
        assert "p=1/4" in d2 and "p=3/4" in d2 and "p=1/4" not in d
    else:  # no multi-branch edge is ever enabled, so no p= label is drawn
        assert " p=" not in d2 and d2 == d
    assert P_LABEL.sub("", d2) == P_LABEL.sub("", d)


@pytest.mark.parametrize("name", FAST)
def test_c2_probabilities_do_not_shape_the_graph(name):
    check_c2(name)


@pytest.mark.slow
def test_c2_random_batch():
    run_all(check_c2, [n for n in random_names()
                       if any(not e.is_deterministic for e in model(n).edges)])


# --- C3: branch = separate edge (main test) ---------------------------------------

def split_ta(ta):
    """Every multi-branch edge, in place, becomes one deterministic edge per branch; returns the
    TA and the map (edge index, branch) -> new edge index."""
    edges, index = [], {}
    for k, e in enumerate(ta.edges):
        for j, b in enumerate(e.branches):
            index[(k, j)] = len(edges)
            edges.append(Edge(e.source, b.target, e.action, e.guard, b.resets, e.cost))
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, tuple(edges), rates=ta.rates,
                          invariants=ta.invariants), index


def check_c3(name):
    ta = model(name)
    ta_split, index = split_ta(ta)
    r, rs = build(name), build_prestable_graph(ta_split)
    assert nodes_view(rs.graph) == nodes_view(r.graph), "nodes differ"
    assert rs.phase2.splits == r.phase2.splits, (rs.phase2.splits, r.phase2.splits)
    mapped = [(e.source, e.label, e.target) + ((None, None) if e.label == DELAY else (index[(e.edge, e.branch)], 0))
              for e in r.graph.edges]
    actual = [(e.source, e.label, e.target, e.edge, e.branch) for e in rs.graph.edges]
    assert sorted(mapped, key=str) == sorted(actual, key=str), "edges differ"
    assert mapped == actual, "same edges, different order"


@pytest.mark.parametrize("name", FAST)
def test_c3_branch_is_a_separate_edge(name):
    check_c3(name)


@pytest.mark.slow
def test_c3_random_batch():
    run_all(check_c3, random_names())


# --- C4: semantic successor check -------------------------------------------------

def holds(constraints, nu):
    def val(c):
        return Fraction(0) if c == ZERO else nu[c]
    for g in constraints:
        d = val(g.left) - val(g.right)
        if not (d < g.bound if g.strict else d <= g.bound):
            return False
    return True


def in_zone(zone, nu):
    return all(v >= 0 for v in nu.values()) and holds(zone.constraints(), nu)


def reset(nu, R):
    return {c: Fraction(0) if c in R else v for c, v in nu.items()}


@lru_cache(maxsize=None)
def grid(clocks, K):
    axis = [Fraction(i, 4) for i in range(4 * (K + 1) + 1)]
    return [dict(zip(clocks, p)) for p in itertools.product(axis, repeat=len(clocks))]


def samples(zone, clocks, K):
    pts = [nu for nu in grid(clocks, K) if in_zone(zone, nu)]
    step = max(1, len(pts) // SAMPLES)
    return pts[::step][:SAMPLES]


def check_c4(name):
    ta, g = model(name), build(name).graph
    K = g.K
    checked = 0
    for n in g.nodes:
        pts = samples(n.zone, ta.clocks, K)
        for k, e in enumerate(ta.edges):
            if e.source != n.location:
                continue
            succ = dict(g.action_successors(n.id, k))
            for nu in pts:
                if succ:
                    assert sorted(succ) == list(range(len(e.branches))), (n.id, k, succ)
                    assert holds(e.guard, nu), f"node {n.id}, edge {k}: {nu} has successors but fails the guard"
                    for j, b in enumerate(e.branches):
                        target = g.nodes[succ[j]]
                        assert target.location == b.target
                        nu2 = reset(nu, b.resets)
                        assert in_zone(target.zone, nu2), \
                            f"node {n.id} ({n.zone}), edge {k}, branch {j}: {nu} -> {nu2} not in {target.zone}"
                else:
                    enabled = holds(e.guard, nu) and all(
                        holds(ta.invariant(b.target), reset(nu, b.resets)) for b in e.branches)
                    assert not enabled, f"node {n.id} ({n.zone}), edge {k}: no successors, but {nu} is enabled"
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("name", FAST)
def test_c4_semantic_successors(name):
    check_c4(name)


@pytest.mark.slow
def test_c4_random_batch():
    run_all(check_c4, random_names())


# --- C5: mutation -----------------------------------------------------------------

def mutants(ta, g):
    """(dropped edge, graph without it) for every action edge of a multi-branch TA edge."""
    for e in g.edges:
        if e.label != DELAY and not ta.edges[e.edge].is_deterministic:
            yield e, ZoneGraph(nodes=g.nodes, edges=tuple(x for x in g.edges if x is not e), initial=g.initial, K=g.K)


def check_c5(name, limit=None):
    ta, g = model(name), build(name).graph
    count = 0
    for dropped, broken in itertools.islice(mutants(ta, g), limit):
        with pytest.raises(AssertionError, match=rf"^node {dropped.source} \(.*\), edge {dropped.edge} \("):
            check_branches(broken, ta)
        count += 1
    return count


@pytest.mark.parametrize("name", FAST)
def test_c5_mutation(name):
    assert check_c5(name) > 0


@pytest.mark.slow
def test_c5_random_batch():
    # up to 5 mutants per model
    run_all(lambda n: check_c5(n, limit=5), [n for n in random_names()
                                              if any(not e.is_deterministic for e in model(n).edges)])


# --- C6: JSON ---------------------------------------------------------------------

def check_c6(name):
    ta, g = model(name), build(name).graph
    ctx = g.nodes[0].zone.ctx
    data = json.loads(json.dumps(to_json(g, ta)))
    back = from_json(data, ta, ctx)
    assert back.edges == g.edges and nodes_view(back) == nodes_view(g)
    for e in data["edges"]:
        e.pop("branch", None)
    old = from_json(data, ta, ctx)
    assert [e.branch for e in old.edges] == [None if e.label == DELAY else 0 for e in g.edges]
    assert [(e.source, e.label, e.target, e.edge) for e in old.edges] == \
           [(e.source, e.label, e.target, e.edge) for e in g.edges]


@pytest.mark.parametrize("name", FAST)
def test_c6_json(name):
    ta = model(name)
    assert any(not e.is_deterministic for e in ta.edges)
    assert any("branch" in e for e in to_json(build(name).graph, ta)["edges"])
    check_c6(name)


@pytest.mark.slow
def test_c6_random_batch():
    run_all(check_c6, random_names())
