"""Uniform constant scaling (step 11-pre): graph isomorphism (P1), boundary MDP isomorphism
(P2) and value scaling (P3) under multiplying every guard and invariant bound by c."""

import sys
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from prestable.graph import build_prestable_graph  # noqa: E402
from scaling_uniform import (SOLVERS, catalogue, check_p1, check_p2, close, scale_ta,  # noqa: E402
                             solve_values)

FAST = [(m, c) for m in ("E1", "E3", "E4", "E7") for c in (1, 4)]
SLOW = [(m, c) for m in catalogue() for c in (1, 2, 4, 8, 16) if (m, c) not in FAST]
CASES = [pytest.param(m, c, id=f"{m}-c{c}") for m, c in FAST] + \
        [pytest.param(m, c, id=f"{m}-c{c}", marks=pytest.mark.slow) for m, c in SLOW]


@lru_cache(maxsize=None)
def spec(name):
    return catalogue()[name]


@lru_cache(maxsize=None)
def graph(name, c, cost_div=1):
    return build_prestable_graph(scale_ta(spec(name).ta, c, cost_div))


@lru_cache(maxsize=None)
def solved(name, c, solver, cost_div=1):
    ta = scale_ta(spec(name).ta, c, cost_div)
    return solve_values(spec(name), ta, None if solver == "BRG" else graph(name, c, cost_div).graph, solver)


def test_scale_ta_shape():
    ta = spec("R072").ta
    s = scale_ta(ta, 3, cost_div=2)
    assert (s.clocks, s.locations, s.initial, s.rates) == (ta.clocks, ta.locations, ta.initial, ta.rates)
    assert s.max_constant() == 3 * ta.max_constant()
    for e, f in zip(ta.edges, s.edges):
        assert (e.source, e.action, e.branches) == (f.source, f.action, f.branches)
        assert [g.bound * 3 for g in e.guard] == [g.bound for g in f.guard]
        assert f.cost * 2 == e.cost
    assert scale_ta(ta, 1) == ta


@pytest.mark.parametrize("name, c", CASES)
def test_p1_graph_isomorphism(name, c):
    check_p1(graph(name, 1), graph(name, c), c)


@pytest.mark.parametrize("name, c", CASES)
def test_p2_boundary_mdp_isomorphism(name, c):
    if not spec(name).values:
        pytest.skip("graph checks only")
    phi, _ = check_p1(graph(name, 1), graph(name, c), c)
    check_p2(solved(name, 1, "boundary").mdp, solved(name, c, "boundary").mdp, phi, c)


@pytest.mark.parametrize("name, c", CASES)
def test_p3_value_scaling(name, c):
    if not spec(name).values:
        pytest.skip("graph checks only")
    values = {}
    for solver in SOLVERS:
        v, ref = solved(name, c, solver).values, solved(name, 1, solver, cost_div=c).values
        assert all(close(a, c * b) for a, b in zip(v, ref)), (solver, v, [c * b for b in ref])
        values[solver] = v
    for solver in SOLVERS[1:]:
        assert all(close(a, b) for a, b in zip(values[solver], values["BRG"])), values
