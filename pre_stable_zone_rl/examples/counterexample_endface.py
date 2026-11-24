"""End-face counterexamples: optima that wait to an interior integer point of a block.

Both: clocks x, y; start in s at (0, 0); all edge costs 0; target g has no invariant.

CE1 (max objective, solved as min with negated rates; the max value is minus the min value):

    s (rate 0.5, x <= 2, y <= 2) --a: reset {y}--> m (rate 1, x <= 3, y <= 2) --e: reset {x, y}--> g

    Waiting t in s and then until the invariant of m runs out gives 0.5·t + λ·min(3 − t, 2):
    at λ = 0.9 this is 1.8, 2.3, 1.9 at t = 0, 1, 2.

CE2 (min objective, nonnegative rates):

    s (rate 0.1, x <= 2, y <= 2) --a: reset {y}--> m (rate 0, x <= 3, y <= 2)
      --e: reset {y}--> n (rate 1, x <= 5, y <= 5) --f: x >= 5, reset {x, y}--> g

    Waiting t in s and then until the invariant of m runs out gives 0.1·t + λ²·(5 − min(3, t + 2)):
    at λ = 0.9 this is 2.43, 1.72, 1.82 at t = 0, 1, 2.

In both the optimum t = 1 is not an end of the waiting interval [0, 2] of the diagonal block at
s. Without invariant atoms the face x = 3 of m is no block boundary, so the boundary variant
misses it (see partition_atoms in model.py).

    python examples/counterexample_endface.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # run without installing the package

from prestable.block_vi import solve  # noqa: E402
from prestable.brg import build_brg, check_brg, optimal_run, value_iteration  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import Edge, TimedAutomaton, ge, le  # noqa: E402

TARGETS = ("g",)
# (name, model, λ, sign): the value in the model's own form is sign · (min value)
RUNS = (("CE1", lambda: endface(0.5, 1), 0.9, -1), ("CE1", lambda: endface(0.5, 2), 0.9, -1),
        ("CE2", lambda: endface2(), 0.9, 1), ("CE2", lambda: endface2(), 0.5, 1))


def endface(rate_s=0.5, rate_m=1) -> TimedAutomaton:
    """CE1 with min-form (negated) rates."""
    return TimedAutomaton(
        ("x", "y"), ("s", "m", "g"), "s",
        (Edge("s", "m", "a", resets={"y"}),
         Edge("m", "g", "e", resets={"x", "y"})),
        rates={"s": -rate_s, "m": -rate_m},
        invariants={"s": (le("x", 2), le("y", 2)), "m": (le("x", 3), le("y", 2))},
    )


def endface2() -> TimedAutomaton:
    """CE2 (min form)."""
    return TimedAutomaton(
        ("x", "y"), ("s", "m", "n", "g"), "s",
        (Edge("s", "m", "a", resets={"y"}),
         Edge("m", "n", "e", resets={"y"}),
         Edge("n", "g", "f", guard=(ge("x", 5),), resets={"x", "y"})),
        rates={"s": 0.1, "m": 0, "n": 1},
        invariants={"s": (le("x", 2), le("y", 2)), "m": (le("x", 3), le("y", 2)), "n": (le("x", 5), le("y", 5))},
    )


def max_value_by_wait(t, lam, rate_s=0.5, rate_m=1):
    """CE1, max form: wait t in s, then until the invariant of m runs out."""
    return rate_s * t + lam * rate_m * min(3 - t, 2)


def ce2_value_by_wait(t, lam):
    """CE2: wait t in s, then until the invariant of m runs out, then until x = 5 in n."""
    return 0.1 * t + lam ** 2 * (5 - min(3, t + 2))


def brg_value(ta, lam, sign=-1):
    """(value in the model's own form, first step of the optimal run)."""
    brg = build_brg(ta, TARGETS)
    check_brg(ta, brg)
    V, policy, _ = value_iteration(brg, lam)
    return sign * V[brg.initial], optimal_run(ta, brg, policy)[0]


def block_value(ta, lam, variant, graph=None, sign=-1, invariant_atoms=True):
    """(value in the model's own form, first step of the optimal run, BlockResult)."""
    if graph is None:
        graph = build_prestable_graph(ta, invariant_atoms=invariant_atoms).graph
    r = solve(ta, TARGETS, lam, variant=variant, graph=graph)
    return sign * r.value, r.run[0], r


def main() -> None:
    for name, make in (("CE1", endface), ("CE2", endface2)):
        ta = make()
        for atoms in (False, True):
            res = build_prestable_graph(ta, invariant_atoms=atoms)
            print(f"{name}, invariant_atoms={atoms}: Phase 2 splits {res.phase2.splits}")
            for l in ta.locations:
                print(f"  blocks at {l}: {sorted(str(Z) for Z in res.phase2.store[l])}")
    for name, make, lam, sign in RUNS:
        ta = make()
        rates = {l: sign * r for l, r in ta.rates.items()}
        v, step = brg_value(ta, lam, sign)
        print(f"\n{name}, λ={lam}, rates {rates} ({'max' if sign < 0 else 'min'} form)")
        print(f"  BRG {v:.6f}, first step {step}")
        for atoms in (False, True):
            graph = build_prestable_graph(ta, invariant_atoms=atoms).graph
            for variant in ("integer", "boundary"):
                v, step, _ = block_value(ta, lam, variant, graph, sign)
                print(f"  invariant_atoms={atoms!s:5} {variant:8} {v:.6f}, first step {step}")


if __name__ == "__main__":
    main()
