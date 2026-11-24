"""Step 11b: Phases 1-3 per branch. A branch is a transition: every (edge, branch) pair is
handled exactly like a deterministic edge."""

import json
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from brg_examples import EXAMPLES  # noqa: E402
from p3_example import pta_p3  # noqa: E402
from prestable.algorithm import DELAY, check_phase1, check_phase2, transitions  # noqa: E402
from prestable.export import from_json, to_dot, to_json  # noqa: E402
from prestable.graph import build_prestable_graph, check_branches  # noqa: E402
from prestable.model import Branch, Edge, PEdge, TimedAutomaton, ge, le  # noqa: E402
from prestable.zones import ZoneContext, pre_edge  # noqa: E402
from pta_examples import MODELS, random_probabilistic  # noqa: E402
from test_phase1 import running_example, two_clock_example  # noqa: E402

DETERMINISTIC = {"running": running_example, "two_clock": two_clock_example, "P3": lambda: pta_p3(False),
                 "P3_inv": lambda: pta_p3(True), **{n: make for n, (make, _) in EXAMPLES.items()}}


@lru_cache(maxsize=None)
def built(name):
    ta = MODELS[name]()
    return ta, build_prestable_graph(ta)


def check_all(ta, r):
    ctx = r.graph.nodes[0].zone.ctx
    check_phase1(ta, ctx, r.phase1.store, r.phase1.K)
    check_phase2(ta, ctx, r.phase2.store, r.phase1.K)
    check_branches(r.graph, ta)


# --- deterministic models are unchanged ------------------------------------------

@pytest.mark.parametrize("name", [n for n in DETERMINISTIC if not n.startswith("P3")])
def test_deterministic_transitions(name):
    ta = DETERMINISTIC[name]()
    for t in transitions(ta):
        if t.label == DELAY:
            assert (t.edge, t.edge_index, t.branch_index) == (None, None, None)
        else:
            assert t.branch_index == 0 and t.edge is ta.edges[t.edge_index] and t.target == t.edge.target


def test_pre_edge_default_branch():
    ctx = ZoneContext(("x", "y"))
    e = PEdge("l0", "a", (ge("x", 1),), branches=(Branch("1/2", {"x"}, "l1"), Branch("1/2", {"y"}, "l1")))
    z = ctx.from_guard([le("x", 0), le("y", 2)])
    assert pre_edge(z, e) == pre_edge(z, e, 0) == pre_edge(z, Edge("l0", "l1", "a", (ge("x", 1),), {"x"}))
    assert pre_edge(z, e, 1) != pre_edge(z, e, 0)


# --- M1-M3 ----------------------------------------------------------------------

@pytest.mark.parametrize("name", list(MODELS))
def test_phase_checks(name):
    check_all(*built(name))


@pytest.mark.parametrize("name", list(MODELS))
def test_two_successors_from_enabled_l0_nodes(name):
    ta, r = built(name)
    g = r.graph
    ctx = g.nodes[0].zone.ctx
    enabled = [n for n in g.nodes if n.location == "l0" and n.zone <= ctx.from_guard([ge("x", 1)])]
    assert enabled
    for n in g.nodes:
        if n.location == "l0":
            succ = g.action_successors(n.id, 0)
            assert [b for b, _ in succ] == ([0, 1] if n in enabled else [])


@pytest.mark.parametrize("name", list(MODELS))
def test_json_round_trip_keeps_branch(name):
    ta, r = built(name)
    g = r.graph
    data = json.loads(json.dumps(to_json(g, ta)))
    assert {e["branch"] for e in data["edges"] if e["ta_edge"] == 0} == {0, 1}
    assert all("branch" not in e for e in data["edges"] if e["ta_edge"] not in (None, 0))  # deterministic
    back = from_json(data, ta, g.nodes[0].zone.ctx)
    assert back.edges == g.edges


def test_json_without_branch_key_loads_as_branch_0():
    ta = running_example()
    g = build_prestable_graph(ta).graph
    back = from_json(to_json(g, ta), ta)
    assert [e.branch for e in back.edges] == [e.branch for e in g.edges]
    assert {e.branch for e in back.edges if e.label != DELAY} == {0}


@pytest.mark.parametrize("name", list(MODELS))
def test_dot_shows_probabilities(name):
    ta, r = built(name)
    dot = to_dot(r.graph, ta)
    assert "a p=1/2" in dot
    assert "b p=" not in dot  # deterministic labels unchanged


def test_m2_duplicate_branches_share_the_target():
    ta, r = built("M2")
    g = r.graph
    ctx = g.nodes[0].zone.ctx
    nodes = [n for n in g.nodes if n.location == "l0" and n.zone <= ctx.from_guard([ge("x", 1)])]
    assert nodes
    for n in nodes:
        (b0, t0), (b1, t1) = g.action_successors(n.id, 0)
        assert (b0, b1) == (0, 1) and t0 == t1


def test_m1_blocks():
    """The blocks reported for step 11b (see pta_examples.py); g's partition splits l1."""
    ta, r = built("M1")
    blocks = {l: sorted(str(n.zone) for n in r.graph.nodes if n.location == l) for l in ta.locations}
    assert blocks == {
        "l0": sorted(["x == 0", "0 < x < 1", "1 <= x < 3", "x == 3"]),
        "l1": sorted(["x == 0", "0 < x < 2", "2 <= x < 3", "x == 3"]),
        "l2": sorted(["1 <= x < 3", "x == 3"]),
        "g": sorted(["2 <= x < 3", "x == 3", "x > 3"]),
    }


# --- check_branches catches broken graphs ---------------------------------------

def test_check_branches_rejects_missing_branch():
    ta, r = built("M1")
    g = r.graph
    dropped = next(e for e in g.edges if e.edge == 0 and e.branch == 1)
    broken = type(g)(nodes=g.nodes, edges=tuple(e for e in g.edges if e is not dropped), initial=g.initial)
    with pytest.raises(AssertionError, match=f"node {dropped.source} .*edge 0 .*branch 1"):
        check_branches(broken, ta)


# --- TimedAutomaton equality -----------------------------------------------------

def test_equality_compares_rates_and_invariants():
    def ta(rates, inv):
        return TimedAutomaton(("x",), ("l0", "l1"), "l0", (Edge("l0", "l1", "a"),), rates=rates, invariants=inv)
    base = ta({"l0": 1}, {"l0": (le("x", 2),)})
    assert base == ta({"l0": 1}, {"l0": (le("x", 2),)})
    assert base != ta({"l0": 2}, {"l0": (le("x", 2),)})
    assert base != ta({"l0": 1}, {"l0": (le("x", 3),)})
    assert base != ta({"l0": 1, "l1": 0}, {"l0": (le("x", 2),)})  # mappings compared as given
    assert hash(base) == hash(ta({"l0": 2}, {"l0": (le("x", 3),)}))  # still not hashed


# --- random batch ----------------------------------------------------------------

@lru_cache(maxsize=None)
def random_batch():
    return random_probabilistic(200)


@pytest.mark.slow
def test_random_probabilistic_models():
    batch = random_batch()
    assert len(batch) == 200
    kept = [(n, ta) for n, ta in batch if ta is not None]
    assert any(not e.is_deterministic for _, ta in kept for e in ta.edges)
    for name, ta in kept:
        try:
            check_all(ta, build_prestable_graph(ta))
        except Exception as exc:
            raise AssertionError(f"{name}: {exc}\n{ta!r}") from exc
