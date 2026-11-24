import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from brg_examples import ALLOW_DEADLOCKS, e1, e2, e3, e4, e5, e6, e7, r072  # noqa: E402
from compare_values import compare_model, example_models, format_row, p3_model, random_models  # noqa: E402
from prestable.block_vi import build_block_mdp, check_block_mdp, in_closure, solve, wait_interval  # noqa: E402
from prestable.brg import build_brg, value_iteration  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import Edge, PEdge, TimedAutomaton, ge, le, lt  # noqa: E402
from prestable.zones import ZoneContext  # noqa: E402

TOL = 1e-6

# The λ values used in test_brg.py.
CASES = [("E1", e1, 0.3), ("E1", e1, 0.9), ("E2", e2, 0.3), ("E2", e2, 0.9),
         ("E3", e3, 0.5), ("E3", e3, 0.8), ("E3", e3, 0.9), ("E4", e4, 0.25), ("E4", e4, 0.5),
         ("E5", e5, 0.5), ("E5", e5, 0.25),
         ("E6", e6, 0.5), ("E6", e6, 0.9), ("E7", e7, 0.5), ("E7", e7, 0.9), ("R072", r072, 0.5), ("R072", r072, 0.9)]


@pytest.mark.parametrize("name, make, lam", CASES)
def test_examples_agree_with_brg(name, make, lam):
    ta = make()
    ad = name in ALLOW_DEADLOCKS or name == "R072"
    brg = build_brg(ta, ["g"], allow_deadlocks=ad)
    v_brg = value_iteration(brg, lam)[0][brg.initial]
    graph = build_prestable_graph(ta).graph
    for variant in ("integer", "boundary"):
        r = solve(ta, ["g"], lam, variant=variant, allow_deadlocks=ad, graph=graph)
        check_block_mdp(ta, graph, r.mdp)
        assert r.value == pytest.approx(v_brg, abs=TOL), (variant, r.run)


def test_wait_interval_and_closure():
    ctx = ZoneContext(("x",))
    z = ctx.from_guard([ge("x", 1), lt("x", 3)])  # closure [1, 3]
    assert wait_interval(z, (0,), ("x",)) == (1, 3)
    assert wait_interval(z, (2,), ("x",)) == (0, 1)
    assert in_closure(z, (3,), ("x",)) and not in_closure(z, (4,), ("x",))
    with pytest.raises(ValueError, match="no upper bound"):
        wait_interval(ctx.from_guard([ge("x", 1)]), (0,), ("x",))


def test_integer_variant_has_more_actions():
    ta = e3()
    graph = build_prestable_graph(ta).graph
    mi = build_block_mdp(ta, graph, ["g"], variant="integer")
    mb = build_block_mdp(ta, graph, ["g"], variant="boundary")
    assert sum(map(len, mi.actions)) > sum(map(len, mb.actions))
    with pytest.raises(ValueError):
        build_block_mdp(ta, graph, ["g"], variant="corner")


def _check_rows(models):
    for m in models:
        rows = compare_model(m)  # also runs check_brg and check_block_mdp on both variants
        for r in rows or []:
            assert r.lower_bounds_hold, f"lower bound violated: {format_row(r)}\n{m.ta!r}"


def test_lower_bounds_examples_and_p3():
    _check_rows(example_models() + [p3_model()])


@pytest.mark.slow
def test_lower_bounds_random_models():
    _check_rows(random_models())


@pytest.mark.parametrize("name, make, expected", [
    ("E6", e6, lambda lam: 0.0), ("E7", e7, lambda lam: 2 * lam), ("R072", r072, lambda lam: 0.0),
])
@pytest.mark.parametrize("lam", [0.5, 0.9])
def test_lower_corner_examples_values(name, make, expected, lam):
    ta = make()
    graph = build_prestable_graph(ta).graph
    for variant in ("integer", "boundary"):
        r = solve(ta, ["g"], lam, variant=variant, allow_deadlocks=True, graph=graph)
        assert r.value == pytest.approx(expected(lam), abs=TOL), (name, variant, r.run)


def test_random_generator_tags():
    from compare_values import random_models, structurally_non_zeno
    models = random_models(40)
    assert sum(len(m.ta.clocks) == 2 for m in models) >= 20
    assert {m.tag for m in models} <= {"zeno", "non_zeno"}
    assert all((m.tag == "non_zeno") == structurally_non_zeno(m.ta) for m in models)
    # E7 has no cycles at all, so it is structurally non-Zeno; R072's zero-reset loop is not.
    assert structurally_non_zeno(e7()) and not structurally_non_zeno(r072())


# --- edge identity (step 10.1) -----------------------------------------------------

def _rebuilt(ta, edges=None):
    """An equal automaton with fresh Edge objects (or the given edges)."""
    edges = edges if edges is not None else tuple(
        PEdge(e.source, e.action, e.guard, e.cost, e.branches) for e in ta.edges)
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, edges, rates=ta.rates, invariants=ta.invariants)


@pytest.mark.parametrize("make", [e3, e7, r072])
def test_mdp_from_rebuilt_automaton(make):
    # Before step 10 this raised KeyError: block_vi looked edges up by id().
    ta = make()
    ta2 = _rebuilt(ta)
    assert ta2 == ta and all(a is not b for a, b in zip(ta.edges, ta2.edges))
    graph = build_prestable_graph(ta).graph
    for variant in ("integer", "boundary"):
        r1 = solve(ta, ["g"], 0.5, variant=variant, allow_deadlocks=True, graph=graph)
        r2 = solve(ta2, ["g"], 0.5, variant=variant, allow_deadlocks=True, graph=graph)
        assert r2.value == r1.value and r2.mdp.states == r1.mdp.states and r2.mdp.actions == r1.mdp.actions


def test_mdp_rejects_mismatched_automaton():
    ta = e3()
    graph = build_prestable_graph(ta).graph
    swapped = _rebuilt(ta, (ta.edges[1], ta.edges[0], ta.edges[2]))  # same edges, other order
    changed = _rebuilt(ta, (ta.edges[0], ta.edges[1], Edge("l1", "g", "c", guard=(ge("x", 2),), cost=0.5)))
    for bad in (swapped, changed):
        with pytest.raises(ValueError, match="built from another automaton"):
            build_block_mdp(bad, graph, ["g"], variant="boundary")
    fewer = _rebuilt(ta, ta.edges[:2])
    with pytest.raises(ValueError, match="edge index 2"):
        build_block_mdp(fewer, graph, ["g"], variant="boundary")
