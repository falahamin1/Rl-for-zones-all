"""Acceptance tests for step 11a: PEdge/Branch construction, equality, exact probabilities,
well-formedness edge cases, and repr round trips."""

import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from brg_examples import e1, e2, e3, e4, e5, e6, e7, r072  # noqa: E402
from p3_example import pta_p3  # noqa: E402
from prestable.algorithm import check_well_formed  # noqa: E402
from prestable.block_vi import solve  # noqa: E402
from prestable.brg import build_brg, value_iteration  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import (Branch, Constraint, Edge, MultiBranchError, PEdge, TimedAutomaton,  # noqa: E402
                             ge, gt, le, lt, require_deterministic)
from prestable.zones import ZoneContext  # noqa: E402
from test_phase1 import running_example, two_clock_example  # noqa: E402

LAM = 0.5
MODELS = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7, "R072": r072,
          "running": running_example, "two_clock": two_clock_example}


# --- B1 ---------------------------------------------------------------------------

def explicit(ta):
    edges = tuple(PEdge(e.source, e.action, e.guard, e.cost, (Branch(1, e.resets, e.target),)) for e in ta.edges)
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, edges, rates=ta.rates, invariants=ta.invariants)


def graph_view(g):
    return ([(n.id, n.location, n.zone.key()) for n in g.nodes],
            [(e.source, e.label, e.target, e.edge, e.cost, e.ta_edge) for e in g.edges], g.initial)


def values(ta, g):
    out = {v: solve(ta, ["g"], LAM, variant=v, allow_deadlocks=True, graph=g).value for v in ("boundary", "integer")}
    brg = build_brg(ta, ["g"], allow_deadlocks=True)
    out["BRG"] = value_iteration(brg, LAM)[0][brg.initial]
    return out


@pytest.mark.parametrize("name", list(MODELS))
def test_b1_explicit_construction(name):
    ta = MODELS[name]()
    ta2 = explicit(ta)
    assert ta2 == ta and ta2.edges == ta.edges
    g, g2 = build_prestable_graph(ta).graph, build_prestable_graph(ta2).graph
    assert graph_view(g2) == graph_view(g)
    if "g" in ta.locations:  # the running and two-clock examples have no target
        assert values(ta2, g2) == values(ta, g)


# --- B2 ---------------------------------------------------------------------------

def test_b2_factory_equals_explicit():
    a, b = Edge("l0", "l1", "a"), PEdge("l0", "a", (), 0.0, (Branch(1, frozenset(), "l1"),))
    assert a == b and hash(a) == hash(b)


def test_b2_branch_order_is_identity():
    b1, b2 = Branch("1/2", (), "l1"), Branch("1/2", {"x"}, "l2")
    e, f = PEdge("l0", "a", branches=(b1, b2)), PEdge("l0", "a", branches=(b2, b1))
    assert e != f
    assert e == PEdge("l0", "a", branches=(Branch(Fraction(1, 2), [], "l1"), Branch(0.5, ["x"], "l2")))


# --- B3 ---------------------------------------------------------------------------

def one_location_ta(*bs):
    return TimedAutomaton(("x",), ("l0", "l1"), "l0", (PEdge("l0", "a", branches=bs),))


def test_b3_ten_tenths():
    ta = one_location_ta(*(Branch(0.1, (), "l1") for _ in range(10)))
    assert all(b.prob == Fraction(1, 10) for b in ta.edges[0].branches)


def test_b3_decimal_and_fraction_strings():
    assert Branch("0.5", (), "l").prob == Branch("1/2", (), "l").prob == Fraction(1, 2)


def test_b3_three_tenths_twice_and_four_tenths():
    one_location_ta(Branch(0.3, (), "l1"), Branch(0.3, (), "l1"), Branch(0.4, (), "l1"))


# --- B4 ---------------------------------------------------------------------------

def b4_ta(clocks, inv0, guard, inv1, resets):
    edge = PEdge("l0", "a", guard, 0, (Branch("1/2", resets, "l1"), Branch("1/2", (), "l2")))
    return TimedAutomaton(clocks, ("l0", "l1", "l2"), "l0", (edge,), invariants={"l0": inv0, "l1": inv1})


B4 = {
    "a": (("x",), (le("x", 2),), (ge("x", 1),), (le("x", 2),), (), True),
    "b": (("x",), (le("x", 3),), (ge("x", 1),), (lt("x", 3),), (), False),
    "c": (("x", "y"), (le("y", 3),), (ge("x", 1), le("y", 1)), (le("y", 1),), {"x"}, True),
    "d": (("x", "y"), (le("y", 3),), (ge("x", 1),), (le("y", 1),), {"x"}, False),
    "e": (("x",), (le("x", 3),), (ge("x", 5),), (le("x", 0),), (), True),
}


@pytest.mark.parametrize("case", list(B4))
def test_b4_well_formedness(case):
    *args, ok = B4[case]
    ta = b4_ta(*args)
    if ok:
        check_well_formed(ta, ZoneContext(ta.clocks))
    else:
        with pytest.raises(ValueError, match="edge 0 .*branch 0 "):
            check_well_formed(ta, ZoneContext(ta.clocks))


# --- B5 ---------------------------------------------------------------------------

CATALOGUE = {**MODELS, "P3": lambda: pta_p3(invariants=False), "P3_inv": lambda: pta_p3(invariants=True)}


def test_b5_multibranch_error_is_value_error():
    assert issubclass(MultiBranchError, ValueError)


@pytest.mark.parametrize("name", list(CATALOGUE))
def test_b5_catalogue_is_deterministic(name):
    assert require_deterministic(CATALOGUE[name](), "test") is None


# --- B6 ---------------------------------------------------------------------------

def test_b6_repr_round_trip():
    ta = TimedAutomaton(
        ("x", "y"), ("l0", "l1", "l2"), "l0",
        (PEdge("l0", "a", (ge("x", 1), gt("y", 0)), 1.5,
               (Branch("1/3", {"x"}, "l1"), Branch(Fraction(2, 3), {"x", "y"}, "l2"))),
         Edge("l1", "l0", "b", (lt("x", 2),), {"y"}, 2)),
        rates={"l0": 1, "l1": 2}, invariants={"l0": (le("x", 3),), "l1": (le("x", 2), le("y", 2))},
    )
    scope = {"Fraction": Fraction, "Branch": Branch, "PEdge": PEdge, "TimedAutomaton": TimedAutomaton,
             "Constraint": Constraint, "le": le, "lt": lt, "ge": ge, "gt": gt}
    back = eval(repr(ta), scope)
    assert back == ta and back.rates == ta.rates and back.invariants == ta.invariants
