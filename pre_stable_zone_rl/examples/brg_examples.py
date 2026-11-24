"""Small priced timed automata for the boundary region graph (BRG) reference solver.

Every non-target location has invariant x <= 3 (and y <= 3 with two clocks); the target
location is g. Edge costs are 0 unless stated; rates are 0 unless stated.

    python examples/brg_examples.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # run without installing the package

from prestable.brg import build_brg, check_brg, optimal_run, value_iteration  # noqa: E402
from prestable.model import Edge, TimedAutomaton, ge, gt, le, lt  # noqa: E402

TARGETS = ("g",)


def _inv(clocks, locations):
    return {l: tuple(le(c, 3) for c in clocks) for l in locations}


def e1() -> TimedAutomaton:
    """l0 (rate 2) --a: x >= 1--> g"""
    return TimedAutomaton(("x",), ("l0", "g"), "l0", (Edge("l0", "g", "a", guard=(ge("x", 1),)),),
                          rates={"l0": 2}, invariants=_inv(("x",), ("l0",)))


def e2() -> TimedAutomaton:
    """l0 (rate 2) --a: x > 1--> g"""
    return TimedAutomaton(("x",), ("l0", "g"), "l0", (Edge("l0", "g", "a", guard=(gt("x", 1),)),),
                          rates={"l0": 2}, invariants=_inv(("x",), ("l0",)))


def e3() -> TimedAutomaton:
    """l0, l1 rate 1; a: l0 -> g (x >= 3); b: l0 -> l1 (x >= 1); c: l1 -> g (x >= 3, cost 0.5)"""
    return TimedAutomaton(
        ("x",), ("l0", "l1", "g"), "l0",
        (Edge("l0", "g", "a", guard=(ge("x", 3),)),
         Edge("l0", "l1", "b", guard=(ge("x", 1),)),
         Edge("l1", "g", "c", guard=(ge("x", 3),), cost=0.5)),
        rates={"l0": 1, "l1": 1}, invariants=_inv(("x",), ("l0", "l1")),
    )


def e4() -> TimedAutomaton:
    """l0 rate 1, l1 rate 3; a: l0 -> l1 (reset y); b: l1 -> g (x >= 3, y >= 1)"""
    return TimedAutomaton(
        ("x", "y"), ("l0", "l1", "g"), "l0",
        (Edge("l0", "l1", "a", resets={"y"}),
         Edge("l1", "g", "b", guard=(ge("x", 3), ge("y", 1)))),
        rates={"l0": 1, "l1": 3}, invariants=_inv(("x", "y"), ("l0", "l1")),
    )


def e5() -> TimedAutomaton:
    """l0 rate 1, l1 rate 3; a: l0 -> l1 (x < 2); b: l1 -> g (x >= 2)"""
    return TimedAutomaton(
        ("x",), ("l0", "l1", "g"), "l0",
        (Edge("l0", "l1", "a", guard=(lt("x", 2),)),
         Edge("l1", "g", "b", guard=(ge("x", 2),))),
        rates={"l0": 1, "l1": 3}, invariants=_inv(("x",), ("l0", "l1")),
    )


def e6() -> TimedAutomaton:
    """Lower corner: l0, l1 rate 5; a: l0 -> l1 (x > 0); b: l1 -> g (x < 1). V = 0.

    a and b can both be taken just after time 0. The step 6 BRG gave 5λ: after a it sat at
    the lower corner of 0 < x < 1 and only offered b after a delay of 1 (the next thin
    region), i.e. at the "before" side of x = 1.
    """
    return TimedAutomaton(
        ("x",), ("l0", "l1", "g"), "l0",
        (Edge("l0", "l1", "a", guard=(gt("x", 0),)),
         Edge("l1", "g", "b", guard=(lt("x", 1),))),
        rates={"l0": 5, "l1": 5}, invariants={l: (le("x", 2),) for l in ("l0", "l1")},
    )


def e7() -> TimedAutomaton:
    """Soundness of the lower-corner fix: l0, l1 rate 1; a: l0 -> l1 (x > 0);
    b: l1 -> g (x <= 0); c: l1 -> g (x >= 2). V = 2λ.

    After a, x > 0 for good, so b is never enabled; acting in the thin region x = 0 (which the
    run is already past) would wrongly give V = 0.
    """
    return TimedAutomaton(
        ("x",), ("l0", "l1", "g"), "l0",
        (Edge("l0", "l1", "a", guard=(gt("x", 0),)),
         Edge("l1", "g", "b", guard=(le("x", 0),)),
         Edge("l1", "g", "c", guard=(ge("x", 2),))),
        rates={"l0": 1, "l1": 1}, invariants={l: (le("x", 2),) for l in ("l0", "l1")},
    )


def r072() -> TimedAutomaton:
    """Random model R072 from the step 8 comparison (compare_values.py writes such gap models to
    examples/out/gaps/, which is git-ignored and regenerated on demand; this is the saved copy):
    the zero-cost self-loop e2 (x > 0) can be taken infinitely often with vanishing delays, so V = 0."""
    return TimedAutomaton(
        ("x",), ("l0", "l1", "g"), "l0",
        (Edge("l0", "l1", "e0", cost=0),
         Edge("l1", "g", "e1", resets={"x"}, cost=2),
         Edge("l0", "l0", "e2", guard=(gt("x", 0),), cost=0),
         Edge("l0", "l0", "e3", cost=2),
         Edge("l0", "l1", "e4", guard=(ge("x", 0), gt("x", -2)), resets={"x"}, cost=0)),
        rates={"l0": 3, "l1": 3}, invariants={l: (le("x", 2),) for l in ("l0", "l1")},
    )


EXAMPLES = {"E1": (e1, (0.3, 0.9)), "E2": (e2, (0.3, 0.9)), "E3": (e3, (0.5, 0.8, 0.9)), "E4": (e4, (0.25, 0.5)),
            "E5": (e5, (0.5, 0.25)), "E6": (e6, (0.5, 0.9)), "E7": (e7, (0.5, 0.9)), "R072": (r072, (0.5, 0.9))}
# E4 has deadlocks: taking a at t in (2, 3] reaches l1 with x > 2, y = 0, and b (y >= 1)
# can never be taken before the invariant x <= 3 runs out.
ALLOW_DEADLOCKS = {"E4", "E6"}  # E6: reaching l1 with x >= 1 leaves b (x < 1) disabled for good


def main() -> None:
    for name, (make, discounts) in EXAMPLES.items():
        ta = make()
        brg = build_brg(ta, TARGETS, allow_deadlocks=name in ALLOW_DEADLOCKS)
        check_brg(ta, brg)
        n_actions = sum(len(a) for a in brg.actions)
        if brg.deadlocks:
            print(f"{name}: {len(brg.deadlocks)} deadlock states (value +inf)")
        for lam in discounts:
            V, policy, iters = value_iteration(brg, lam)
            print(f"{name} λ={lam}: states={len(brg.states)}, actions={n_actions}, "
                  f"VI iterations={iters}, V(initial)={V[brg.initial]:.6f}")
            for step in optimal_run(ta, brg, policy):
                if step[0] == "cycle":  # the greedy run repeats from step k
                    print(f"    ... repeats from step {step[1]}")
                    continue
                loc, nu, edge, delay, kind = step
                print(f"    ({loc}, {nu}, {edge}, delay {delay}, {kind})")


if __name__ == "__main__":
    main()
