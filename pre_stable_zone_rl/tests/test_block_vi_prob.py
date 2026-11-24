"""Step 11c: the block MDP with probabilistic actions."""

import json
import random
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "examples"))

from compare_values import example_models, p3_model, random_models  # noqa: E402
from prestable.block_vi import (BlockAction, build_block_mdp, check_block_mdp, discounted_cost,  # noqa: E402
                                optimal_run, sample_run, solve)
from prestable.brg import BRGAction, build_brg  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import Edge, MultiBranchError, PEdge, TimedAutomaton, ge  # noqa: E402
from pta_examples import m1, m2, random_probabilistic  # noqa: E402

VARIANTS = ("boundary", "integer")
TOL = 1e-9
# Saved before step 11c; step 12 (invariant atoms) regenerated its states, actions and node/state ids,
# with every value and first action (edge, delay, cost, probabilities) unchanged.
BASELINE = json.loads((HERE / "data" / "block_baseline_11c.json").read_text())


def with_rates(ta, rate=1):
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, ta.edges,
                          rates={l: rate for l in ta.locations if l != "g"}, invariants=ta.invariants)


# --- action record ----------------------------------------------------------------

def test_action_target_and_successors():
    a = BlockAction(0, 1, 2, Fraction(1), ((Fraction(1), 5),), ((2, 0),))
    assert a.target == 5 and a.is_deterministic
    b = BlockAction(0, 1, 2, Fraction(1), ((Fraction(1, 2), 5), (Fraction(1, 2), 6)), ((2, 0),))
    with pytest.raises(MultiBranchError):
        b.target
    assert BRGAction(0, 1, "thin", Fraction(1), ((Fraction(1), 7),)).target == 7


# --- baseline (deterministic models) -------------------------------------------------

def baseline_models():
    models = {m.name: m for m in example_models() + [p3_model()]}
    rnd = []
    for m in random_models(1000, 2026):
        if len(rnd) == 100:
            break
        if build_brg(m.ta, m.targets, allow_deadlocks=True).targets:
            rnd.append(m)
    models.update({m.name: m for m in rnd})
    return models


def entry(m, variant):
    r = solve(m.ta, m.targets, 0.5, variant=variant, allow_deadlocks=m.allow_deadlocks)
    a = r.policy[r.mdp.initial]
    first = None if a is None else {"edge": a.edge, "delay": a.delay, "via": a.via, "cost": str(a.cost),
                                    "successors": [[str(p), s] for p, s in a.successors]}
    return {"states": r.states, "actions": r.actions, "value": repr(r.value), "first": first}


def check_baseline(names):
    models = baseline_models()
    for name in names:
        for v in VARIANTS:
            assert entry(models[name], v) == BASELINE[f"{name}/{v}"], (name, v)


CATALOGUE = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "R072"]


def test_baseline_catalogue():
    check_baseline(CATALOGUE)


@pytest.mark.slow
def test_baseline_p3_and_random():
    names = sorted({k.split("/")[0] for k in BASELINE} - set(CATALOGUE))
    assert len(names) == 101  # P3 (with invariants) and 100 random models
    check_baseline(names)


# --- M1, M2 -----------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.3, 0.5, 0.9])
@pytest.mark.parametrize("variant", VARIANTS)
def test_m1_value(variant, lam):
    ta = with_rates(m1())
    r = solve(ta, ["g"], lam, variant=variant)
    assert r.value == pytest.approx(1 + 2 * lam, abs=TOL)
    first = r.policy[r.mdp.initial]
    assert (first.delay, ta.edges[first.edge].action) == (1, "a")
    assert len(first.successors) == 2 and r.run is None
    check_block_mdp(ta, r.graph, r.mdp)


@pytest.mark.parametrize("variant", VARIANTS)
def test_m2_equals_deterministic_edge(variant):
    ta = with_rates(m2())
    det = TimedAutomaton(ta.clocks, ta.locations, ta.initial,
                         (Edge("l0", "l1", "a", guard=(ge("x", 1),)), ta.edges[1]),
                         rates=ta.rates, invariants=ta.invariants)
    r, rd = solve(ta, ["g"], 0.5, variant=variant), solve(det, ["g"], 0.5, variant=variant)
    assert r.value == rd.value
    l0_actions = [a for s, acts in enumerate(r.mdp.actions) for a in acts
                  if r.mdp.states[s][0] == "l0" and ta.edges[a.edge].action == "a"]
    assert l0_actions and all(len(a.successors) == 1 and a.successors[0][0] == 1 for a in l0_actions)
    check_block_mdp(ta, r.graph, r.mdp)


def test_optimal_run_rejects_probabilistic_path():
    ta = with_rates(m1())
    r = solve(ta, ["g"], 0.5, variant="boundary")
    with pytest.raises(ValueError, match="sample_run"):
        optimal_run(ta, r.graph, r.mdp, r.policy)


def test_sample_run_m1():
    lam = 0.5
    ta = with_rates(m1())
    r = solve(ta, ["g"], lam, variant="boundary")
    rng = random.Random(11)
    costs, ends, branches = [], set(), set()
    for _ in range(1000):
        steps, end = sample_run(ta, r.graph, r.mdp, r.policy, rng)
        ends.add(end)
        costs.append(discounted_cost(steps, lam))
        branches.add(r.mdp.states[steps[0][2]][0])
    assert ends == {"target"}
    assert branches == {"l1", "l2"}  # both branches are sampled
    assert abs(sum(costs) / len(costs) - r.value) <= 0.05


# --- random batch ------------------------------------------------------------------

def split_ta(ta):
    """One deterministic edge per branch, in place (acceptance test C3 of step 11b)."""
    edges = [Edge(e.source, b.target, e.action, e.guard, b.resets, e.cost) for e in ta.edges for b in e.branches]
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, tuple(edges), rates=ta.rates,
                          invariants=ta.invariants)


@lru_cache(maxsize=None)
def kept_random():
    kept = tuple((n, ta) for n, ta in random_probabilistic(200) if ta is not None)
    assert len(kept) == 150
    return kept


def check_random(name, ta):
    g = build_prestable_graph(ta).graph
    gs = build_prestable_graph(split_ta(ta)).graph
    for v in VARIANTS:
        mdp = build_block_mdp(ta, g, ["g"], variant=v, allow_deadlocks=True)
        check_block_mdp(ta, g, mdp)
    for lam in (0.5, 0.9):
        V = {v: solve(ta, ["g"], lam, variant=v, allow_deadlocks=True, graph=g).value for v in VARIANTS}
        assert V["integer"] <= V["boundary"] + TOL, (name, lam, V)
        for v in VARIANTS:
            vs = solve(split_ta(ta), ["g"], lam, variant=v, allow_deadlocks=True, graph=gs).value
            assert vs <= V[v] + TOL, (name, lam, v, vs, V[v])


@pytest.mark.slow
def test_random_batch():
    failures = []
    for name, ta in kept_random():
        try:
            check_random(name, ta)
        except (AssertionError, ValueError) as exc:
            failures.append((name, str(exc)[:300]))
    assert not failures, f"{len(failures)} of 150 fail; first: {failures[0]}"
