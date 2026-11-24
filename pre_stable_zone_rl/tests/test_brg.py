import ast
import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from brg_examples import e1, e2, e3, e4, e5  # noqa: E402
from prestable.algorithm import phase1  # noqa: E402
from prestable.brg import (  # noqa: E402
    build_brg,
    check_brg,
    in_closure,
    is_thin,
    optimal_run,
    region_of,
    representative,
    reset,
    satisfies,
    time_successors,
    value_iteration,
)
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import Constraint, Edge, TimedAutomaton, diff_le, diff_lt, ge, le, lt  # noqa: E402
from prestable.zones import ZoneContext  # noqa: E402

TOL = 1e-6


def solve(ta, lam, **kwargs):
    brg = build_brg(ta, ["g"], **kwargs)
    check_brg(ta, brg)
    V, policy, _ = value_iteration(brg, lam)
    return brg, V[brg.initial], optimal_run(ta, brg, policy)


# --- examples -----------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.3, 0.9])
def test_e1(lam):
    _, v, run = solve(e1(), lam)
    assert v == pytest.approx(2, abs=TOL)
    assert run[0][2:] == ("a", 1, "thin")


@pytest.mark.parametrize("lam", [0.3, 0.9])
def test_e2_infimum_not_attained(lam):
    _, v, run = solve(e2(), lam)
    assert v == pytest.approx(2, abs=TOL)
    assert run[0][2:] == ("a", 1, "after")


@pytest.mark.parametrize("lam, value, first", [(0.5, 2.25, ("b", 1)), (0.8, 3.0, None), (0.9, 3.0, ("a", 3))])
def test_e3(lam, value, first):
    _, v, run = solve(e3(), lam)
    assert v == pytest.approx(min(3, 1 + 2.5 * lam), abs=TOL)
    assert v == pytest.approx(value, abs=TOL)
    if first:
        assert run[0][2:4] == first


def test_e4_has_deadlocks():
    # a at t in (2, 3] reaches l1 with x > 2, y = 0: b needs y >= 1, i.e. x > 3 (invariant).
    with pytest.raises(ValueError, match="without actions"):
        build_brg(e4(), ["g"])
    brg = build_brg(e4(), ["g"], allow_deadlocks=True)
    assert brg.deadlocks
    for s in brg.deadlocks:
        loc, nu, sig = brg.states[s]
        assert loc == "l1" and nu[1] == 0 and representative(sig)[0] > 2


@pytest.mark.parametrize("lam, value, steps", [
    (0.25, 2.25, [("a", 0)]),
    (0.5, 3.5, [("a", 2), ("b", 1)]),
])
def test_e4(lam, value, steps):
    _, v, run = solve(e4(), lam, allow_deadlocks=True)
    expected = min(t + 3 * lam * max(1, 3 - t) for t in [k / 100 for k in range(201)])  # feasible t ∈ [0, 2]
    assert v == pytest.approx(expected, abs=TOL)
    assert v == pytest.approx(value, abs=TOL)
    assert [r[2:4] for r in run[:len(steps)]] == steps


# --- properties ---------------------------------------------------------------

@pytest.mark.parametrize("make, kwargs", [(e1, {}), (e2, {}), (e3, {}), (e4, {"allow_deadlocks": True}), (e5, {})])
def test_brg_properties(make, kwargs):
    ta = make()
    brg = build_brg(ta, ["g"], **kwargs)
    check_brg(ta, brg)
    K = ta.max_constant()
    for _, nu, _ in brg.states:
        assert all(isinstance(n, int) and 0 <= n <= K for n in nu)
    assert len(build_brg(ta, ["g"], **kwargs).states) == len(brg.states)


@pytest.mark.parametrize("make, kwargs", [(e3, {}), (e4, {"allow_deadlocks": True})])
def test_value_nondecreasing_in_discount(make, kwargs):
    brg = build_brg(make(), ["g"], **kwargs)
    values = [value_iteration(brg, k / 10)[0][brg.initial] for k in range(1, 10)]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:])), values


def test_missing_invariant_rejected():
    ta = TimedAutomaton(("x",), ("l0", "g"), "l0", (Edge("l0", "g", "a", guard=(ge("x", 1),)),))
    with pytest.raises(ValueError, match="does not bound"):
        build_brg(ta, ["g"])
    # Targets need no invariant.
    ta = TimedAutomaton(("x",), ("l0", "g"), "l0", (Edge("l0", "g", "a"),), invariants={"l0": (le("x", 1),)})
    build_brg(ta, ["g"])


def test_max_states():
    with pytest.raises(RuntimeError, match="max_states"):
        build_brg(e3(), ["g"], max_states=2)


def test_brg_does_not_import_zone_code():
    src = (Path(__file__).resolve().parents[1] / "src" / "prestable" / "brg.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    forbidden = {"zones", "partition", "algorithm", "graph", "pyudbm"}
    assert not {m for m in imported if m.split(".")[-1] in forbidden or m.startswith("pyudbm")}, imported


# --- regions ------------------------------------------------------------------

def test_region_of_representative_round_trip():
    for val in [(0, 0), (F(1, 3), F(2, 3)), (F(5, 2), 1), (F(1, 2), F(1, 2)), (F(7, 4), F(1, 4), 2)]:
        sig = region_of(val)
        assert region_of(representative(sig)) == sig
    assert region_of((F(1, 3), F(2, 3))) == ((0, 0), (), ((0,), (1,)))
    assert region_of((F(1, 2), F(3, 2))) == ((0, 1), (), ((0, 1),))
    assert region_of((2, F(1, 5))) == ((2, 0), (0,), ((1,),))


def test_time_successors_diagonal():
    chain = time_successors(region_of((0, 0)), 2)
    assert [is_thin(r) for r in chain] == [True, False, True, False, True]
    reps = [representative(r) for r in chain]
    assert reps == [(0, 0), (F(1, 2), F(1, 2)), (1, 1), (F(3, 2), F(3, 2)), (2, 2)]


def test_time_successors_off_diagonal():
    chain = time_successors(region_of((F(1, 3), F(2, 3))), 1)
    # 0 < x < y < 1, then y = 1 (thin); the next region has y > 1 = K, so the chain stops.
    assert [r for r in chain] == [region_of((F(1, 3), F(2, 3))), region_of((F(1, 2), 1))]


def test_satisfies_diagonal_and_reset():
    clocks = ("x", "y")
    r = region_of((F(1, 3), F(2, 3)))  # 0 < x < y < 1
    assert satisfies(r, [diff_lt("x", "y", 0)], clocks)
    assert not satisfies(r, [diff_le("y", "x", 0)], clocks)
    assert satisfies(r, [lt("x", 1), lt("y", 1), ge("x", 0)], clocks)
    assert reset(r, {"y"}, clocks) == region_of((F(1, 2), 0))
    assert reset(r, {"x", "y"}, clocks) == region_of((0, 0))


def test_in_closure():
    r = region_of((F(1, 3), F(2, 3)))  # 0 < x < y < 1
    assert in_closure((0, 0), r) and in_closure((0, 1), r) and in_closure((1, 1), r)
    assert not in_closure((1, 0), r)  # needs frac(x) <= frac(y)
    assert in_closure((2, 1), region_of((2, F(1, 2)))) and not in_closure((3, 1), region_of((2, F(1, 2))))


# --- model --------------------------------------------------------------------

def test_invariant_validation():
    def ta(inv):
        return TimedAutomaton(("x", "y"), ("l0",), "l0", (), invariants=inv)
    ta({"l0": (le("x", 3), lt("y", 2))})
    for bad in [{"nowhere": (le("x", 1),)},
                {"l0": (le("z", 1),)},
                {"l0": (ge("x", 1),)},  # lower bound
                {"l0": (diff_le("x", "y", 1),)},  # diagonal
                {"l0": (Constraint("0", "x", 3, False),)}]:
        with pytest.raises(ValueError):
            ta(bad)


def test_max_constant_includes_invariants():
    t = TimedAutomaton(("x",), ("l0",), "l0", (Edge("l0", "l0", "a", guard=(ge("x", 1),)),),
                       invariants={"l0": (le("x", 7),)})
    assert t.max_constant() == 7
    assert t.invariant("l0") == (le("x", 7),) and t.invariant("other") == ()
    hash(t)


def test_zone_algorithm_accepts_invariants():
    # Step 7 replaced the NotImplementedError of step 6 by invariant support.
    ta = e1()
    ctx = ZoneContext(ta.clocks)
    p1 = phase1(ta, ctx)
    assert all(b <= ctx.from_guard([le("x", 3)]) for b in p1.store["l0"])
    build_prestable_graph(ta)


# --- E5: the infimum is approached from before the boundary ---------------------

@pytest.mark.parametrize("lam, value, steps", [
    (0.5, 2.0, [("a", 2, "before"), ("b", 0, "thin")]),
    (0.25, 1.5, [("a", 0, "thin"), ("b", 2, "thin")]),
])
def test_e5(lam, value, steps):
    _, v, run = solve(e5(), lam)
    assert v == pytest.approx(min(t + 3 * lam * (2 - t) for t in [k / 1000 for k in range(2001)]), abs=1e-3)
    assert v == pytest.approx(value, abs=TOL)
    assert [r[2:] for r in run] == steps


# --- step 9: lower corner of a thick region -------------------------------------

from brg_examples import e6, e7, r072  # noqa: E402


@pytest.mark.parametrize("lam", [0.5, 0.9])
def test_e6_lower_corner(lam):
    _, v, run = solve(e6(), lam, allow_deadlocks=True)
    assert v == pytest.approx(0, abs=TOL)
    assert [r[2:4] for r in run] == [("a", 0), ("b", 0)]
    # The step 6 BRG (without the fix) only offered b after a delay of 1: 5λ.
    _, v_old, _ = solve(e6(), lam, allow_deadlocks=True, _lower_corner_fix=False)
    assert v_old == pytest.approx(5 * lam, abs=TOL)


@pytest.mark.parametrize("lam, value", [(0.5, 1.0), (0.9, 1.8)])
def test_e7_fix_is_sound(lam, value):
    # An unsound fix (acting in the thin region x = 0 the run has already left) gives 0.
    _, v, run = solve(e7(), lam)
    assert v == pytest.approx(value, abs=TOL)
    assert [r[2] for r in run] == ["a", "c"]


@pytest.mark.parametrize("lam", [0.5, 0.9])
def test_r072_zero(lam):
    _, v, _ = solve(r072(), lam, allow_deadlocks=True)
    assert v == pytest.approx(0, abs=TOL)
    _, v_old, _ = solve(r072(), lam, allow_deadlocks=True, _lower_corner_fix=False)
    assert v_old == pytest.approx(2 * lam, abs=TOL)
