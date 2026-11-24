import random
import time

import pytest

from prestable.algorithm import check_phase1, phase1
from prestable.model import Constraint, Edge, TimedAutomaton, ge, gt, le, lt
from prestable.zones import ZoneContext, is_k_bounded

# Size/speed indicators from the random test, printed at the end with -s.
STATS = {"max_blocks": 0, "max_blocks_case": None, "slowest": 0.0, "slowest_case": None}


def Z(ctx, *constraints):
    return ctx.from_guard(constraints)


def run(ta, K=None, ctx=None, **kwargs):
    ctx = ctx or ZoneContext(ta.clocks)
    result = phase1(ta, ctx, K, **kwargs)
    check_phase1(ta, ctx, result.store, result.K)
    return ctx, result


def running_example(extra_locations=()):
    return TimedAutomaton(
        clocks=("x",),
        locations=("l0", "l1", *extra_locations),
        initial="l0",
        edges=(
            Edge("l0", "l1", "a", guard=(ge("x", 1),), resets={"x"}),
            Edge("l1", "l0", "b", guard=(le("x", 2),)),
        ),
    )


def test_running_example():
    ctx, r = run(running_example())
    assert r.K == 2
    assert r.store["l0"] == {ctx.zero(), Z(ctx, gt("x", 0), lt("x", 1)), Z(ctx, ge("x", 1), le("x", 2)),
                             Z(ctx, gt("x", 2))}
    assert r.store["l1"] == {ctx.zero(), Z(ctx, gt("x", 0), le("x", 2)), Z(ctx, gt("x", 2))}
    assert r.locations_processed == 5
    assert r.changed_inserts == 5
    # Initial insert + per processed block: one delay insert and one per outgoing edge.
    # l0 is processed with 1, 3 and 4 blocks, l1 with 1 and 3; each has one edge.
    assert r.inserts == 1 + 2 * (1 + 3 + 4 + 1 + 3)
    assert r.trace is None


def test_running_example_trace():
    ctx, plain = run(running_example())
    _, r = run(running_example(), ctx=ctx, trace=True)
    assert r.store == plain.store
    assert len(r.trace) == r.inserts
    first = r.trace[0]
    assert (first.location, first.zone, first.extrapolated, first.changed, first.blocks_after) == (
        "l0", ctx.zero(), ctx.zero(), True, 1)
    assert sum(ev.changed for ev in r.trace) == r.changed_inserts


def test_self_loop():
    ta = TimedAutomaton(("x",), ("l",), "l", (Edge("l", "l", "a", guard=(ge("x", 1),), resets={"x"}),))
    ctx, r = run(ta)
    assert r.store["l"] == {ctx.zero(), Z(ctx, gt("x", 0), lt("x", 1)), Z(ctx, ge("x", 1))}


def test_unreachable_location():
    ta = TimedAutomaton(
        clocks=("x",),
        locations=("l0", "l1", "island"),
        initial="l0",
        edges=(
            Edge("l0", "l1", "a", guard=(ge("x", 1),), resets={"x"}),
            Edge("l1", "l0", "b", guard=(le("x", 2),)),
            Edge("island", "l0", "c", guard=(le("x", 1),)),
        ),
    )
    ctx, r = run(ta)
    assert r.store["island"] == frozenset()
    _, base = run(running_example(), ctx=ctx)
    assert {l: r.store[l] for l in ("l0", "l1")} == base.store


def two_clock_example():
    return TimedAutomaton(
        clocks=("x", "y"),
        locations=("l0", "l1"),
        initial="l0",
        edges=(
            Edge("l0", "l1", "a", guard=(ge("x", 1),), resets={"x"}),
            Edge("l1", "l0", "b", guard=(ge("y", 2), le("x", 1)), resets={"y"}),
        ),
    )


def test_two_clocks():
    ta = two_clock_example()
    ctx, r = run(ta)
    assert r.K == 2
    for l, P in r.store.items():
        assert P, l
        assert all(is_k_bounded(b, r.K) for b in P)
    print(f"\ntwo-clock example: blocks={ {l: len(P) for l, P in r.store.items()} }, "
          f"locations_processed={r.locations_processed}, inserts={r.inserts}, "
          f"changed_inserts={r.changed_inserts}")


def test_deterministic():
    for ta in (running_example(), two_clock_example()):
        ctx = ZoneContext(ta.clocks)
        a, b = phase1(ta, ctx), phase1(ta, ctx)
        assert a.store == b.store
        assert (a.locations_processed, a.inserts, a.changed_inserts) == (
            b.locations_processed, b.inserts, b.changed_inserts)
        # A fresh context gives the same blocks too (compared by printed form).
        c = phase1(ta, ZoneContext(ta.clocks))
        assert {l: sorted(map(str, P)) for l, P in a.store.items()} == {
            l: sorted(map(str, P)) for l, P in c.store.items()}


def test_max_iterations():
    ta = running_example()
    with pytest.raises(RuntimeError, match="max_iterations"):
        phase1(ta, ZoneContext(ta.clocks), max_iterations=1)


def test_explicit_k():
    # A larger K than needed still gives a valid (finer) result.
    ctx, r = run(running_example(), K=4)
    assert r.K == 4
    assert len(r.store["l0"]) >= 4


# --- random -----------------------------------------------------------------

def random_constraint(rng, clocks, c_max=2):
    left, right = rng.sample(["0", *clocks], 2)
    return Constraint(left, right, rng.randint(-c_max, c_max), rng.random() < 0.5)


def random_automaton(rng, n_clocks):
    clocks = tuple("xyz"[:n_clocks])
    locations = tuple(f"l{i}" for i in range(rng.randint(2, 3)))
    edges = []
    for i in range(rng.randint(1, 4)):
        guard = tuple(random_constraint(rng, clocks) for _ in range(rng.randint(0, 2)))
        resets = frozenset(c for c in clocks if rng.random() < 0.4)
        edges.append(Edge(rng.choice(locations), rng.choice(locations), f"e{i}", guard=guard, resets=resets))
    return TimedAutomaton(clocks, locations, locations[0], tuple(edges))


@pytest.mark.slow
@pytest.mark.parametrize("n_clocks", [1, 2, 3])
def test_random_automata(n_clocks):
    rng = random.Random(500 + n_clocks)
    for i in range(30):
        ta = random_automaton(rng, n_clocks)
        start = time.perf_counter()
        ctx, r = run(ta)
        elapsed = time.perf_counter() - start
        total = sum(len(P) for P in r.store.values())
        case = f"{n_clocks} clocks #{i}: blocks={ {l: len(P) for l, P in r.store.items()} }, K={r.K}, " \
               f"edges={len(ta.edges)}, processed={r.locations_processed}, inserts={r.inserts}"
        if total > STATS["max_blocks"]:
            STATS.update(max_blocks=total, max_blocks_case=case)
        if elapsed > STATS["slowest"]:
            STATS.update(slowest=elapsed, slowest_case=case)


def test_zz_report():
    """Runs last in this file; prints the random-test indicators with -s."""
    print(f"\nlargest store: {STATS['max_blocks']} blocks ({STATS['max_blocks_case']})")
    print(f"slowest run (incl. check_phase1): {STATS['slowest']:.2f}s ({STATS['slowest_case']})")
