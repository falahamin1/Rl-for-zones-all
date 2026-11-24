"""Small probabilistic timed automata (edges with several branches) for step 11.

Every non-target location has invariant x <= 3 (and y <= 3 with two clocks); the target g
has none. Costs and rates are 0.

    python examples/pta_examples.py      # blocks per location and branch successors
"""

import random
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # run without installing the package
sys.path.insert(0, str(HERE))

from prestable.algorithm import check_well_formed  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import Branch, Edge, PEdge, TimedAutomaton, ge, le  # noqa: E402
from prestable.zones import ZoneContext  # noqa: E402

HALF = Fraction(1, 2)


def _inv(clocks, locations):
    return {l: tuple(le(c, 3) for c in clocks) for l in locations}


def m1() -> TimedAutomaton:
    """a: l0 -x >= 1-> {1/2: reset x -> l1, 1/2: -> l2}; b: l1 -> g (x >= 2); c: l2 -> g (x >= 3)"""
    return TimedAutomaton(
        ("x",), ("l0", "l1", "l2", "g"), "l0",
        (PEdge("l0", "a", (ge("x", 1),), branches=(Branch(HALF, {"x"}, "l1"), Branch(HALF, (), "l2"))),
         Edge("l1", "g", "b", guard=(ge("x", 2),)),
         Edge("l2", "g", "c", guard=(ge("x", 3),))),
        invariants=_inv(("x",), ("l0", "l1", "l2")),
    )


def m2() -> TimedAutomaton:
    """Duplicate branches: a: l0 -x >= 1-> {1/2: -> l1, 1/2: -> l1}; b: l1 -> g (x >= 2)"""
    return TimedAutomaton(
        ("x",), ("l0", "l1", "g"), "l0",
        (PEdge("l0", "a", (ge("x", 1),), branches=(Branch(HALF, (), "l1"), Branch(HALF, (), "l1"))),
         Edge("l1", "g", "b", guard=(ge("x", 2),))),
        invariants=_inv(("x",), ("l0", "l1")),
    )


def m3() -> TimedAutomaton:
    """a: l0 -x >= 1-> {1/2: reset x -> l1, 1/2: reset y -> l1}; b: l1 -> g (x >= 2, y >= 1)"""
    return TimedAutomaton(
        ("x", "y"), ("l0", "l1", "g"), "l0",
        (PEdge("l0", "a", (ge("x", 1),), branches=(Branch(HALF, {"x"}, "l1"), Branch(HALF, {"y"}, "l1"))),
         Edge("l1", "g", "b", guard=(ge("x", 2), ge("y", 1)))),
        invariants=_inv(("x", "y"), ("l0", "l1")),
    )


MODELS = {"M1": m1, "M2": m2, "M3": m3}


def split_edges(ta: TimedAutomaton, rng: random.Random) -> TimedAutomaton:
    """Each edge, with probability 1/2, becomes {1/2: its branch, 1/2: a random reset subset to a
    random location}. May raise ValueError (validate)."""
    edges = []
    for e in ta.edges:
        if rng.random() < 0.5:
            extra = Branch(HALF, {c for c in ta.clocks if rng.random() < 0.5}, rng.choice(ta.locations))
            e = PEdge(e.source, e.action, e.guard, e.cost, (Branch(HALF, e.resets, e.target), extra))
        edges.append(e)
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, tuple(edges), rates=ta.rates, invariants=ta.invariants)


def split_batch(n: int = 200, seed: int = 2026, split_seed: int = 11):
    """(name, split TA, error) for the first n runnable random models of compare_values.py, each
    with its edges split by split_edges; error is the ValueError of validate or
    check_well_formed, or None if the split model is well-formed."""
    from compare_values import random_models
    from prestable.brg import build_brg

    rng = random.Random(split_seed)
    out = []
    for m in random_models(1000, seed):
        if len(out) == n:
            break
        if not build_brg(m.ta, m.targets, allow_deadlocks=True).targets:
            continue
        ta, error = split_edges(m.ta, rng), None
        try:
            check_well_formed(ta, ZoneContext(ta.clocks))
        except ValueError as exc:
            error = exc
        out.append((m.name, ta, error))
    return out


def random_probabilistic(n: int = 200, seed: int = 2026, split_seed: int = 11):
    """(name, TA or None) as split_batch, with None for the models that are not well-formed."""
    return [(name, None if error else ta) for name, ta, error in split_batch(n, seed, split_seed)]


def main() -> None:
    for name, make in MODELS.items():
        ta = make()
        g = build_prestable_graph(ta).graph
        print(f"{name}: {len(g.nodes)} nodes, {len(g.edges)} edges")
        for l in ta.locations:
            print(f"  {l}: {', '.join(str(n.zone) for n in g.nodes if n.location == l)}")
        for n in g.nodes:
            for k, e in enumerate(ta.edges):
                if e.source == n.location and not e.is_deterministic:
                    for b, t in g.action_successors(n.id, k):
                        print(f"    node {n.id} ({n.zone}) --{e.action}[{b}]--> node {t} "
                              f"({g.nodes[t].location}, {g.nodes[t].zone})")


if __name__ == "__main__":
    main()
