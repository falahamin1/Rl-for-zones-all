"""Step 11d: the probabilistic boundary region graph."""

import json
import math
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "examples"))

from compare_values import example_models, p3_model, random_models  # noqa: E402
from prestable.block_vi import solve  # noqa: E402
from prestable.brg import BRGAction, build_brg, check_brg, optimal_run, value_iteration  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import Branch, Edge, MultiBranchError, PEdge, TimedAutomaton, ge, le  # noqa: E402
from pta_examples import MODELS, split_batch  # noqa: E402
from scaling_uniform import Spec, check_p1, check_p2, close, scale_ta, solve_values  # noqa: E402

INF = math.inf
TOL = 1e-9
HALF = Fraction(1, 2)
BASELINE = json.loads((HERE / "data" / "brg_baseline_11d.json").read_text())  # saved before step 11d


def with_rates(ta, rate=1, edges=None):
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, ta.edges if edges is None else edges,
                          rates={l: rate for l in ta.locations if l != "g"}, invariants=ta.invariants)


def value(ta, lam, targets=("g",)):
    brg = build_brg(ta, targets, allow_deadlocks=True)
    check_brg(ta, brg)
    return value_iteration(brg, lam)[0][brg.initial]


# --- action record ----------------------------------------------------------------

def test_action_target():
    a = BRGAction(0, 1, "thin", Fraction(1), ((Fraction(1), 7),))
    assert a.target == 7 and a.is_deterministic
    b = BRGAction(0, 1, "thin", Fraction(1), ((HALF, 7), (HALF, 8)))
    assert not b.is_deterministic
    with pytest.raises(MultiBranchError):
        b.target


# --- baseline (deterministic models) -----------------------------------------------

def baseline_models():
    out = {m.name: m for m in example_models() + [p3_model()]}
    rnd = []
    for m in random_models(1000, 2026):
        if len(rnd) == 100:
            break
        if build_brg(m.ta, m.targets, allow_deadlocks=True).targets:
            rnd.append(m)
    out.update({m.name: m for m in rnd})
    return out


def entry(m):
    brg = build_brg(m.ta, m.targets, allow_deadlocks=m.allow_deadlocks)
    V, policy, _ = value_iteration(brg, 0.5)
    a = policy[brg.initial]
    first = None if a is None else {"edge": a.edge, "delay": a.delay, "kind": a.kind, "cost": str(a.cost),
                                    "successors": [[str(p), s] for p, s in a.successors]}
    return {"states": len(brg.states), "actions": sum(map(len, brg.actions)), "value": repr(V[brg.initial]),
            "first": first}


CATALOGUE = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "R072"]


def test_baseline_catalogue():
    models = baseline_models()
    for name in CATALOGUE:
        assert entry(models[name]) == BASELINE[name], name


@pytest.mark.slow
def test_baseline_p3_and_random():
    models = baseline_models()
    names = sorted(set(BASELINE) - set(CATALOGUE))
    assert len(names) == 101
    for name in names:
        assert entry(models[name]) == BASELINE[name], name


# --- values ------------------------------------------------------------------------

def m1_asymmetric():
    ta = MODELS["M1"]()
    c = ta.edges[2]
    return with_rates(ta, edges=ta.edges[:2] + (PEdge(c.source, c.action, c.guard, 1, c.branches),))


def d6(with_b=True, split=False):
    if split:
        edges = (Edge("l0", "g", "a", guard=(ge("x", 1),)), Edge("l0", "d", "a", guard=(ge("x", 1),)))
    else:
        edges = (PEdge("l0", "a", (ge("x", 1),), 0, (Branch(HALF, (), "g"), Branch(HALF, (), "d"))),)
        if with_b:
            edges += (Edge("l0", "g", "b", guard=(ge("x", 2),)),)
    return TimedAutomaton(("x",), ("l0", "d", "g"), "l0", edges, rates={"l0": 1, "d": 1},
                          invariants={"l0": (le("x", 3),), "d": (le("x", 3),)})


VALUE_CASES = {
    "M1": (lambda: with_rates(MODELS["M1"]()), lambda lam: 1 + 2 * lam),
    "M1_asym": (m1_asymmetric, lambda lam: 1 + 2.5 * lam),
    "D6_b": (d6, lambda lam: 2.0),
    "D6_no_b": (lambda: d6(with_b=False), lambda lam: INF),
    "D6_split": (lambda: d6(split=True), lambda lam: 1.0),
}


@pytest.mark.parametrize("lam", [0.3, 0.5, 0.9])
@pytest.mark.parametrize("case", list(VALUE_CASES))
def test_values(case, lam):
    make, expected = VALUE_CASES[case]
    v = value(make(), lam)
    assert v == expected(lam) if expected(lam) == INF else v == pytest.approx(expected(lam), abs=TOL)


def test_optimal_run_rejects_probabilistic_action():
    ta = with_rates(MODELS["M1"]())
    brg = build_brg(ta, ["g"])
    _, policy, _ = value_iteration(brg, 0.5)
    with pytest.raises(ValueError, match="probabilistic action"):
        optimal_run(ta, brg, policy)


def test_ill_formed_branch_raises():
    ta = TimedAutomaton(("x",), ("l0", "l1", "l2"), "l0",
                        (PEdge("l0", "a", (ge("x", 1),), 0, (Branch(HALF, (), "l1"), Branch(HALF, (), "l2"))),),
                        invariants={"l0": (le("x", 3),), "l1": (le("x", 2),)})
    with pytest.raises(ValueError, match=r"edge 0 .*branch 0 .*region"):
        build_brg(ta, ["l2"], allow_deadlocks=True)


# --- Bellman optimality and exact policy evaluation -----------------------------------

def q(a, V, lam):
    return float(a.cost) + lam * sum(float(p) * V[s] for p, s in a.successors)


def solve_exact(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        inv = 1 / M[col][col]
        M[col] = [x * inv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def check_bellman_and_exact(ta, lam):
    brg = build_brg(ta, ["g"], allow_deadlocks=True)
    V, policy, _ = value_iteration(brg, lam)
    for s, acts in enumerate(brg.actions):
        if s in brg.targets:
            assert V[s] == 0 and not acts
        elif not acts:
            assert s in brg.deadlocks and V[s] == INF
        else:
            best = min(q(a, V, lam) for a in acts)
            assert V[s] == best == INF or abs(V[s] - best) <= TOL, (s, V[s], best)
            chosen = q(policy[s], V, lam)
            assert chosen == best == INF or abs(chosen - best) <= TOL, (s, chosen, best)
    assert V[brg.initial] < INF
    order, seen, stack = [], {brg.initial}, [brg.initial]
    while stack:
        s = stack.pop()
        if s in brg.targets:
            continue
        order.append(s)
        for _, j in policy[s].successors:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    col = {s: i for i, s in enumerate(order)}
    L = Fraction(repr(lam))
    A = [[Fraction(0)] * len(order) for _ in order]
    c = [Fraction(0)] * len(order)
    for s, i in col.items():
        A[i][i] += 1
        c[i] = Fraction(policy[s].cost)
        for p, j in policy[s].successors:
            if j in col:
                A[i][col[j]] -= L * p
    X = solve_exact(A, c)
    for s, i in col.items():
        assert abs(float(X[i]) - V[s]) <= 1e-8, (s, float(X[i]), V[s])


@pytest.mark.parametrize("lam", [0.5, 0.9])
@pytest.mark.parametrize("name", list(MODELS))
def test_bellman_and_exact_m(name, lam):
    check_bellman_and_exact(with_rates(MODELS[name]()), lam)


@lru_cache(maxsize=None)
def batch():
    return split_batch(200)


@lru_cache(maxsize=None)
def kept():
    out = tuple((n, ta) for n, ta, err in batch() if err is None)
    assert len(out) == 150
    return out


@lru_cache(maxsize=None)
def small_random():
    """The first 10 kept models with finite BRG values (λ 0.5, 0.9), at most 100 BRG states and
    at least one BRG action with two successors."""
    out = []
    for name, ta in kept():
        brg = build_brg(ta, ["g"], allow_deadlocks=True)
        if len(brg.states) <= 100 and any(len(a.successors) > 1 for acts in brg.actions for a in acts) and \
                all(value_iteration(brg, lam)[0][brg.initial] < INF for lam in (0.5, 0.9)):
            out.append((name, ta))
        if len(out) == 10:
            break
    assert len(out) == 10
    return tuple(out)


@pytest.mark.parametrize("lam", [0.5, 0.9])
@pytest.mark.parametrize("i", range(10))
def test_bellman_and_exact_random(i, lam):
    name, ta = small_random()[i]
    check_bellman_and_exact(ta, lam)


# --- random batch -------------------------------------------------------------------

@pytest.mark.slow
def test_random_batch_brg_below_integer():
    failures = []
    for name, ta in kept():
        brg = build_brg(ta, ["g"], allow_deadlocks=True)
        check_brg(ta, brg)
        g = build_prestable_graph(ta).graph
        for lam in (0.5, 0.9):
            vb = value_iteration(brg, lam)[0][brg.initial]
            vi = solve(ta, ["g"], lam, variant="integer", allow_deadlocks=True, graph=g).value
            if not (vb <= vi + TOL):  # inf <= inf holds; inf > finite fails
                failures.append((name, lam, vb, vi))
    assert not failures, failures[:5]


# --- well-formedness ----------------------------------------------------------------

def brg_well_formedness_error(ta):
    try:
        build_brg(ta, ["g"], allow_deadlocks=True)
    except ValueError as exc:
        if "is not well-formed" in str(exc):
            return exc
        raise
    return None


@pytest.mark.slow
def test_well_formedness_agrees():
    assert len(batch()) == 200
    for name, ta, err in batch():
        brg_err = brg_well_formedness_error(ta)
        if err is None:
            assert brg_err is None, (name, brg_err)
        if brg_err is not None:
            assert err is not None, name


# --- scaling ------------------------------------------------------------------------

def scaling_models():
    from test_acceptance_step11c import random_selection
    names = set(random_selection())
    return [(n, with_rates(MODELS[n]())) for n in MODELS] + [(n, ta) for n, ta in kept() if n in names]


@pytest.mark.slow
def test_scaling_p1_p3():
    models = scaling_models()
    assert len(models) == 33
    failures = []
    for name, ta in models:
        spec = Spec(name, ta, ("g",), (0.5, 0.9), True)
        r1 = build_prestable_graph(ta)
        m1 = solve_values(spec, ta, r1.graph, "boundary").mdp
        for c in (1, 4):
            try:
                ta_c = scale_ta(ta, c)
                rc = build_prestable_graph(ta_c)
                phi, _ = check_p1(r1, rc, c)
                ta_div = scale_ta(ta, 1, cost_div=c)
                g_div = build_prestable_graph(ta_div).graph
                for solver in ("BRG", "integer", "boundary"):
                    res = solve_values(spec, ta_c, rc.graph, solver)
                    ref = solve_values(spec, ta_div, g_div, solver)
                    assert all(close(v, c * r) for v, r in zip(res.values, ref.values)), (solver, res.values, ref.values)
                    if solver == "boundary":
                        check_p2(m1, res.mdp, phi, c)
            except AssertionError as exc:
                failures.append((name, c, str(exc)[:300]))
    assert not failures, failures[:3]
