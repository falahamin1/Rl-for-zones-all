"""Step 11a: probabilistic edges in the model (Branch, PEdge, the Edge factory), their
validation, well-formedness, and the "not yet" guard of the solvers."""

from fractions import Fraction

import pytest

from prestable.algorithm import check_well_formed, phase1, transitions
from prestable.block_vi import build_block_mdp
from prestable.brg import build_brg
from prestable.graph import build_prestable_graph
from prestable.model import Branch, Edge, MultiBranchError, PEdge, TimedAutomaton, ge, le
from prestable.zones import ZoneContext



# --- Branch -----------------------------------------------------------------

@pytest.mark.parametrize("given, expected", [
    (0.1, Fraction(1, 10)), ("1/3", Fraction(1, 3)), (1, Fraction(1)), (Fraction(2, 5), Fraction(2, 5)),
])
def test_branch_prob_coercion(given, expected):
    b = Branch(given, (), "l")
    assert b.prob == expected and type(b.prob) is Fraction


@pytest.mark.parametrize("bad", [object(), None, True, [1]])
def test_branch_prob_rejects_other_types(bad):
    with pytest.raises(TypeError):
        Branch(bad, (), "l")


def test_branch_resets_list_becomes_frozenset():
    b = Branch(1, ["x", "y", "x"], "l")
    assert b.resets == frozenset({"x", "y"}) and isinstance(b.resets, frozenset)


# --- Edge factory -------------------------------------------------------------

def test_edge_factory_positional_and_keyword():
    g = (ge("x", 1),)
    pos = Edge("l0", "l1", "a", g, {"x"}, 2)
    kw = Edge(source="l0", target="l1", action="a", guard=g, resets={"x"}, cost=2)
    for e in (pos, kw):
        assert isinstance(e, PEdge) and e.is_deterministic
        assert e.branches == (Branch(Fraction(1), frozenset({"x"}), "l1"),)
        assert (e.source, e.target, e.action, e.guard, e.resets, e.cost) == ("l0", "l1", "a", g, {"x"}, 2)
    assert pos == kw and hash(pos) == hash(kw) and pos is not kw
    assert pos != Edge("l0", "l1", "a", g, {"x"}, 3)


def test_edge_factory_defaults():
    e = Edge("l0", "l1", "a")
    assert (e.guard, e.resets, e.cost) == ((), frozenset(), 0.0)
    assert isinstance(e.guard, tuple) and isinstance(e.resets, frozenset)
    assert Edge("l0", "l1", "a", [ge("x", 1)]).guard == (ge("x", 1),)  # guard coerced as before


# --- multi-branch PEdge -------------------------------------------------------

def two_branch(b0=(Fraction(1, 2), (), "l1"), b1=(Fraction(1, 2), (), "l2"), guard=(ge("x", 1),)):
    return PEdge("l0", "a", guard, 0, (Branch(*b0), Branch(*b1)))


def test_multi_branch_target_and_resets_raise():
    e = two_branch()
    assert not e.is_deterministic
    for attr in ("target", "resets"):
        with pytest.raises(MultiBranchError, match="l0->l1\\|l2"):
            getattr(e, attr)
    assert issubclass(MultiBranchError, ValueError)


# --- validation ---------------------------------------------------------------

def automaton(*edges, invariants=None):
    return TimedAutomaton(("x",), ("l0", "l1", "l2"), "l0", edges, invariants=invariants or {})


def branches(*probs):
    return tuple(Branch(p, (), "l1") for p in probs)


@pytest.mark.parametrize("bs, match", [
    ((), "no branches"),
    (branches(0, 1), "probability 0"),
    (branches(Fraction(-1, 2), Fraction(3, 2)), "probability -1/2"),
    (branches(0.333, 0.333, 0.333), "sum to 999/1000"),
    ((Branch(1, (), "nowhere"),), "unknown location 'nowhere'"),
    ((Branch(1, {"z"}, "l1"),), "resets unknown clocks \\['z'\\]"),
])
def test_validation_errors(bs, match):
    with pytest.raises(ValueError, match=match):
        automaton(PEdge("l0", "a", branches=bs))


def test_validation_accepts_exact_thirds_and_duplicate_branches():
    automaton(PEdge("l0", "a", branches=branches(Fraction(1, 3), Fraction(1, 3), Fraction(1, 3))))
    automaton(PEdge("l0", "a", branches=branches("1/3", "1/3", "1/3")))


# --- well-formedness ----------------------------------------------------------

INV = {"l0": (le("x", 3),), "l1": (le("x", 2),)}  # l2 has no invariant


def test_well_formed_rejects_branch_outside_target_invariant():
    ta = automaton(two_branch(), invariants=INV)
    ctx = ZoneContext(ta.clocks)
    with pytest.raises(ValueError, match="edge 0 .*branch 0 .*outside the invariant of l1: 2 < x <= 3"):
        check_well_formed(ta, ctx)
    with pytest.raises(ValueError, match="edge 0 .*branch 0"):
        phase1(ta, ctx)


def test_well_formed_accepts_reset_branch():
    ta = automaton(two_branch(b0=(Fraction(1, 2), {"x"}, "l1")), invariants=INV)
    check_well_formed(ta, ZoneContext(ta.clocks))


def test_well_formed_skips_deterministic_edges():
    ta = automaton(Edge("l0", "l1", "a", guard=(ge("x", 1),)), invariants=INV)
    ctx = ZoneContext(ta.clocks)
    check_well_formed(ta, ctx)
    store = phase1(ta, ctx).store
    # As before: the edge is disabled where x > 2, so l1 is reached with 1 <= x <= 2 only.
    assert store["l1"] and all(Z <= ctx.from_guard([ge("x", 1), le("x", 2)]) for Z in store["l1"])


# --- not yet ------------------------------------------------------------------

@pytest.fixture
def probabilistic():
    return automaton(two_branch(b0=(Fraction(1, 2), {"x"}, "l1")), invariants=INV)


def test_phase1_accepts_probabilistic_edges(probabilistic):  # step 11b lifted the guard
    store = phase1(probabilistic, ZoneContext(probabilistic.clocks)).store
    assert store["l1"] and store["l2"]


def test_transitions_one_per_branch(probabilistic):  # step 11b lifted the guard
    ts = [(t.label, t.target, t.edge_index, t.branch_index) for t in transitions(probabilistic)]
    assert ts == [("delay", "l0", None, None), ("a", "l1", 0, 0), ("a", "l2", 0, 1),
                  ("delay", "l1", None, None), ("delay", "l2", None, None)]


def test_build_brg_accepts_probabilistic_edges(probabilistic):  # step 11d lifted the guard
    brg = build_brg(probabilistic, ["l2"], allow_deadlocks=True)
    assert any(len(a.successors) == 2 for acts in brg.actions for a in acts)


def test_build_block_mdp_accepts_probabilistic_edges(probabilistic):  # step 11c lifted the guard
    graph = build_prestable_graph(probabilistic).graph
    mdp = build_block_mdp(probabilistic, graph, ["l2"], variant="boundary", allow_deadlocks=True)
    assert any(len(a.successors) == 2 for acts in mdp.actions for a in acts)
