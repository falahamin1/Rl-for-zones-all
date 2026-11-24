"""Acceptance tests for step 11c: the block MDP with probabilistic actions.

Models: M1-M3 (with rate 1 in every non-target location, so values are not trivially 0) and 30
of the 150 kept random probabilistic models of step 11b: the first 30, in batch order, with
finite values (both variants, λ ∈ {0.5, 0.9}), at most 200 boundary states and at least one
action with two successors. The random-batch tests are slow.
"""

import random
import sys
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from prestable.algorithm import DELAY  # noqa: E402
from prestable.block_vi import (build_block_mdp, discounted_cost, sample_run, solve,  # noqa: E402
                                wait_interval)
from prestable.brg import _frac  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import Branch, Edge, PEdge, TimedAutomaton, ge, le  # noqa: E402
from pta_examples import MODELS, random_probabilistic  # noqa: E402

VARIANTS = ("boundary", "integer")
LAMS = (0.5, 0.9)
INF = float("inf")
HALF = Fraction(1, 2)


def with_rates(ta, rate=1, edges=None):
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, ta.edges if edges is None else edges,
                          rates={l: rate for l in ta.locations if l != "g"}, invariants=ta.invariants)


# --- models -------------------------------------------------------------------------

@lru_cache(maxsize=None)
def random_selection():
    out = []
    for name, ta in random_probabilistic(200):
        if ta is None:
            continue
        g = build_prestable_graph(ta).graph
        results = [solve(ta, ["g"], lam, variant=v, allow_deadlocks=True, graph=g) for v in VARIANTS for lam in LAMS]
        if all(r.value < INF for r in results) and results[0].states <= 200 and \
                any(len(a.successors) > 1 for acts in results[0].mdp.actions for a in acts):
            out.append(name)
        if len(out) == 30:
            break
    assert len(out) == 30
    return tuple(out)


@lru_cache(maxsize=None)
def model(name):
    if name in MODELS:
        return with_rates(MODELS[name]())
    return dict((n, ta) for n, ta in random_probabilistic(200) if ta is not None)[name]


@lru_cache(maxsize=None)
def graph(name):
    return build_prestable_graph(model(name)).graph


@lru_cache(maxsize=None)
def solved(name, variant, lam):
    return solve(model(name), ["g"], lam, variant=variant, allow_deadlocks=True, graph=graph(name))


FAST = [pytest.param(m, v, lam, id=f"{m}-{v}-{lam}") for m in MODELS for v in VARIANTS for lam in LAMS]


def run_random(check):
    failures = []
    names = random_selection()
    for name in names:
        for v in VARIANTS:
            for lam in LAMS:
                try:
                    check(name, v, lam)
                except AssertionError as exc:
                    failures.append((name, v, lam, str(exc)[:400]))
    assert not failures, f"{len(failures)} of {len(names) * 4} cases fail; first: {failures[0]}"


# --- D1: asymmetric M1 ----------------------------------------------------------------

def m1_asymmetric():
    ta = MODELS["M1"]()
    c = ta.edges[2]
    assert c.action == "c"
    return with_rates(ta, edges=ta.edges[:2] + (PEdge(c.source, c.action, c.guard, 1, c.branches),))


@pytest.mark.parametrize("lam", LAMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_d1_asymmetric_m1(variant, lam):
    ta = m1_asymmetric()
    r = solve(ta, ["g"], lam, variant=variant)
    assert r.value == pytest.approx(1 + 2.5 * lam, abs=1e-9)
    rng = random.Random(2026)
    n, costs, to_l1 = 4000, [], 0
    for _ in range(n):
        steps, end = sample_run(ta, r.graph, r.mdp, r.policy, rng)
        assert end == "target"
        to_l1 += r.mdp.states[steps[0][2]][0] == "l1"
        costs.append(discounted_cost(steps, lam))
    assert abs(to_l1 / n - 0.5) <= 0.05 and abs((n - to_l1) / n - 0.5) <= 0.05
    assert abs(sum(costs) / n - r.value) <= 0.05


# --- D2: Bellman optimality ------------------------------------------------------------

def q(a, V, lam):
    return float(a.cost) + lam * sum(float(p) * V[s] for p, s in a.successors)


def check_d2(name, variant, lam):
    r = solved(name, variant, lam)
    mdp, V = r.mdp, r.values
    for s in range(len(mdp.states)):
        acts = mdp.actions[s]
        if s in mdp.targets:
            assert V[s] == 0 and not acts
            continue
        if not acts:
            assert s in mdp.deadlocks and V[s] == INF
            continue
        best = min(q(a, V, lam) for a in acts)
        assert V[s] == best == INF or abs(V[s] - best) <= 1e-9, (s, V[s], best)
        assert r.policy[s] in acts
        chosen = q(r.policy[s], V, lam)
        assert chosen == best == INF or abs(chosen - best) <= 1e-9, (s, chosen, best)


@pytest.mark.parametrize("name, variant, lam", FAST)
def test_d2_bellman_optimality(name, variant, lam):
    check_d2(name, variant, lam)


@pytest.mark.slow
def test_d2_random():
    run_random(check_d2)


# --- D3: exact policy evaluation -------------------------------------------------------

def solve_exact(A, b):
    """Gaussian elimination over Fractions (A square, nonsingular)."""
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


def check_d3(name, variant, lam):
    r = solved(name, variant, lam)
    mdp, policy = r.mdp, r.policy
    assert r.value < INF
    # states reachable under the policy, targets excluded (V = 0 there)
    order, seen, stack = [], {mdp.initial}, [mdp.initial]
    while stack:
        s = stack.pop()
        if s in mdp.targets:
            continue
        order.append(s)
        assert policy[s] is not None, f"finite value but deadlock {s} reachable under the policy"
        for _, j in policy[s].successors:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    col = {s: i for i, s in enumerate(order)}
    L = Fraction(repr(lam))
    A = [[Fraction(0)] * len(order) for _ in order]
    c = [Fraction(0)] * len(order)
    for s, i in col.items():
        a = policy[s]
        A[i][i] += 1
        c[i] = Fraction(a.cost)
        for p, j in a.successors:
            if j in col:
                A[i][col[j]] -= L * p
    V = solve_exact(A, c)
    for s, i in col.items():
        assert abs(float(V[i]) - r.values[s]) <= 1e-8, (s, float(V[i]), r.values[s])


@pytest.mark.parametrize("name, variant, lam", FAST)
def test_d3_exact_policy_evaluation(name, variant, lam):
    check_d3(name, variant, lam)


@pytest.mark.slow
def test_d3_random():
    run_random(check_d3)


# --- D4: merge bookkeeping by brute force ----------------------------------------------

def raw_choices(ta, g, mdp, s, variant):
    """(t, distribution) -> [(cost, edge, via)] for every raw choice from state s."""
    loc, nu, n = mdp.states[s]
    index = {st: i for i, st in enumerate(mdp.states)}
    rate = _frac(ta.rates.get(loc, 0))
    groups = defaultdict(list)
    for d in g.successors(n):
        if d.label != DELAY:
            continue
        via = d.target
        for k in range(len(ta.edges)):
            pairs = g.action_successors(via, k)
            if not pairs:
                continue
            lo, hi = wait_interval(g.nodes[via].zone, nu, ta.clocks)
            times = {lo, hi} if variant == "boundary" else set(range(lo, hi + 1))
            e = ta.edges[k]
            for t in times:
                dist = defaultdict(Fraction)
                for b, n2 in pairs:
                    br = e.branches[b]
                    nu2 = tuple(0 if x in br.resets else v + t for x, v in zip(ta.clocks, nu))
                    dist[index[(br.target, nu2, n2)]] += br.prob
                groups[(t, tuple(sorted((j, p) for j, p in dist.items())))].append(
                    (rate * t + _frac(e.cost), k, via))
    return groups


def check_d4(name, variant, lam=None):
    ta, g = model(name), graph(name)
    mdp = solved(name, variant, LAMS[0]).mdp
    for s in range(len(mdp.states)):
        if s in mdp.targets:
            assert not mdp.actions[s]
            continue
        groups = raw_choices(ta, g, mdp, s, variant)
        actions = {(a.delay, tuple((j, p) for p, j in a.successors)): a for a in mdp.actions[s]}
        assert len(actions) == len(mdp.actions[s]), f"state {s}: two actions with the same (delay, successors)"
        assert set(actions) == set(groups), (s, sorted(set(groups) ^ set(actions)))
        for key, cands in groups.items():
            a = actions[key]
            cmin = min(c for c, _, _ in cands)
            minimal = [(k, v) for c, k, v in cands if c == cmin]
            assert a.cost == cmin, (s, key, a.cost, cmin)
            assert (a.edge, a.via) == min(minimal), (s, key, (a.edge, a.via), min(minimal))
            assert a.labels == tuple(sorted((v, k) for k, v in minimal)), (s, key, a.labels)
            assert a.dominated == len(cands) - len(minimal), (s, key, a.dominated)


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("variant", VARIANTS)
def test_d4_merge_bookkeeping(name, variant):
    check_d4(name, variant)


@pytest.mark.slow
def test_d4_random():
    failures = []
    for name in random_selection():
        for v in VARIANTS:
            try:
                check_d4(name, v)
            except AssertionError as exc:
                failures.append((name, v, str(exc)[:300]))
    assert not failures, f"{len(failures)} of 60 cases fail; first: {failures[0]}"


# --- D5: probabilities don't shape the structure ---------------------------------------

def reweighted(ta):
    def edge(e):
        if [b.prob for b in e.branches] != [HALF, HALF]:
            return e
        b0, b1 = e.branches
        return PEdge(e.source, e.action, e.guard, e.cost,
                     (Branch(Fraction(1, 4), b0.resets, b0.target), Branch(Fraction(3, 4), b1.resets, b1.target)))
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, tuple(map(edge, ta.edges)),
                          rates=ta.rates, invariants=ta.invariants)


def structure(mdp):
    return [[(a.edge, a.delay, a.via, a.labels, tuple(j for _, j in a.successors)) for a in acts]
            for acts in mdp.actions]


def check_d5(name, variant, lam):
    ta = model(name)
    ta2 = reweighted(ta)
    assert ta2 != ta
    r = solved(name, variant, lam)
    r2 = solve(ta2, ["g"], lam, variant=variant, allow_deadlocks=True)
    assert r2.mdp.states == r.mdp.states
    assert structure(r2.mdp) == structure(r.mdp)


@pytest.mark.parametrize("name, variant, lam", FAST)
def test_d5_probabilities_do_not_shape_the_structure(name, variant, lam):
    check_d5(name, variant, lam)
    if name in ("M1", "M3"):  # distinct branch successors: the probabilities and values do change
        r = solved(name, variant, lam)
        r2 = solve(reweighted(model(name)), ["g"], lam, variant=variant, allow_deadlocks=True)
        probs = lambda m: [[tuple(p for p, _ in a.successors) for a in acts] for acts in m.actions]  # noqa: E731
        assert probs(r2.mdp) != probs(r.mdp)


def partition(ta, g, mdp, s, variant):
    """The raw choices (t, edge, via) of state s, grouped as the merge groups them."""
    return {frozenset((key[0], k, v) for _, k, v in cands)
            for key, cands in raw_choices(ta, g, mdp, s, variant).items()}


def check_d5_by_state(name, variant, lam):
    """D5 on every state whose raw choices merge the same way under both distributions; returns
    the states where they do not (two choices with the same successor states but branches
    mapped to them in a different order: one action at (1/2, 1/2), two at (1/4, 3/4))."""
    ta = model(name)
    ta2 = reweighted(ta)
    r = solved(name, variant, lam)
    r2 = solve(ta2, ["g"], lam, variant=variant, allow_deadlocks=True)
    assert r2.mdp.states == r.mdp.states
    st, st2 = structure(r.mdp), structure(r2.mdp)
    collisions = []
    for s in range(len(r.mdp.states)):
        if partition(ta, r.graph, r.mdp, s, variant) == partition(ta2, r2.graph, r2.mdp, s, variant):
            assert st2[s] == st[s], (s, r.mdp.states[s])
        else:
            collisions.append(s)
    return collisions


@pytest.mark.slow
def test_d5_random_by_state():
    """The literal D5 fails on R0017 only (see the step 11c acceptance report); here it is
    checked on every state without a merge collision, and the collisions are counted."""
    collisions = {}
    failures = []
    for name in random_selection():
        for v in VARIANTS:
            for lam in LAMS:
                try:
                    c = check_d5_by_state(name, v, lam)
                except AssertionError as exc:
                    failures.append((name, v, lam, str(exc)[:300]))
                    continue
                if c:
                    collisions[(name, v, lam)] = c
    assert not failures, failures[0]
    assert {n for n, _, _ in collisions} == {"R0017"}, collisions


# --- D6: deadlock mixtures --------------------------------------------------------------

def d6(with_b=True, split=False):
    a_branches = (Branch(HALF, (), "g"), Branch(HALF, (), "d"))
    if split:
        edges = (Edge("l0", "g", "a", guard=(ge("x", 1),)), Edge("l0", "d", "a", guard=(ge("x", 1),)))
    else:
        edges = (PEdge("l0", "a", (ge("x", 1),), 0, a_branches),)
        if with_b:
            edges += (Edge("l0", "g", "b", guard=(ge("x", 2),)),)
    return TimedAutomaton(("x",), ("l0", "d", "g"), "l0", edges, rates={"l0": 1, "d": 1},
                          invariants={"l0": (le("x", 3),), "d": (le("x", 3),)})


@pytest.mark.parametrize("lam", LAMS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case, expected", [("with b", 2.0), ("without b", INF), ("split", 1.0)])
def test_d6_deadlock_mixtures(case, expected, variant, lam):
    ta = {"with b": d6(), "without b": d6(with_b=False), "split": d6(split=True)}[case]
    r = solve(ta, ["g"], lam, variant=variant, allow_deadlocks=True)
    assert r.value == expected if expected == INF else r.value == pytest.approx(expected, abs=1e-9)
