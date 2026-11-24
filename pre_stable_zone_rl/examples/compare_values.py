"""Compare discounted optimal values: BRG (reference) vs the block-corner MDP on the
pre-stable graph, in the "integer" and "boundary" variants.

One row per (model, λ):  model | λ | V_BRG | V_integer | V_boundary | states/actions.
Flags: "integer gap" if V_integer > V_BRG + 1e-6, "boundary gap" if V_boundary > V_integer
+ 1e-6. For each flagged row the optimal runs are printed and the model is saved as a
Python snippet in examples/out/gaps/.

    python examples/compare_values.py            # E1-E7, R072, P3 and 1000 random models
    python examples/compare_values.py --quick    # E1-E7, R072 and P3 only
"""

import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # run without installing the package
sys.path.insert(0, str(HERE))

from brg_examples import ALLOW_DEADLOCKS, EXAMPLES  # noqa: E402
from p3_example import pta_p3  # noqa: E402
from prestable.block_vi import build_block_mdp, check_block_mdp, solve  # noqa: E402
from prestable.brg import build_brg, check_brg, optimal_run, value_iteration  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import Branch, Constraint, Edge, PEdge, TimedAutomaton, le  # noqa: E402

OUT = HERE / "out" / "gaps"
TOL = 1e-6
INF = float("inf")


@dataclass
class Row:
    model: str
    lam: float
    v_brg: float
    v_int: float
    v_bnd: float
    brg_size: tuple[int, int]
    int_size: tuple[int, int]
    bnd_size: tuple[int, int]
    runs: dict = field(default_factory=dict)

    @property
    def integer_gap(self) -> bool:
        return self.v_int > self.v_brg + TOL

    @property
    def boundary_gap(self) -> bool:
        return self.v_bnd > self.v_int + TOL

    @property
    def lower_bounds_hold(self) -> bool:
        return self.v_brg <= self.v_int + TOL and self.v_int <= self.v_bnd + TOL


@dataclass
class Model:
    name: str
    ta: TimedAutomaton
    targets: tuple[str, ...]
    lams: tuple[float, ...]
    allow_deadlocks: bool = True
    tag: str = ""  # "zeno" / "non_zeno" for random models


def compare_model(m: Model) -> Optional[list[Row]]:
    """Rows for every λ of the model; None if the target is unreachable (no target BRG state)."""
    brg = build_brg(m.ta, m.targets, allow_deadlocks=m.allow_deadlocks)
    check_brg(m.ta, brg)
    if not brg.targets:
        return None
    graph = build_prestable_graph(m.ta).graph
    mdps = {v: build_block_mdp(m.ta, graph, m.targets, variant=v, allow_deadlocks=m.allow_deadlocks)
            for v in ("integer", "boundary")}
    for mdp in mdps.values():
        check_block_mdp(m.ta, graph, mdp)
    rows = []
    for lam in m.lams:
        V, policy, _ = value_iteration(brg, lam)
        res = {v: solve(m.ta, m.targets, lam, variant=v, graph=graph, mdp=mdp) for v, mdp in mdps.items()}
        row = Row(
            m.name, lam, V[brg.initial], res["integer"].value, res["boundary"].value,
            (len(brg.states), sum(len(a) for a in brg.actions)),
            (res["integer"].states, res["integer"].actions),
            (res["boundary"].states, res["boundary"].actions),
        )
        if row.integer_gap or row.boundary_gap:
            row.runs = {
                "BRG": optimal_run(m.ta, brg, policy),
                "integer": res["integer"].run,
                "boundary": res["boundary"].run,
            }
        rows.append(row)
    return rows


# --- models -----------------------------------------------------------------

def example_models() -> list[Model]:
    return [Model(name, make(), ("g",), lams, name in ALLOW_DEADLOCKS) for name, (make, lams) in EXAMPLES.items()]


def p3_priced() -> TimedAutomaton:
    """P3 with real invariants, rates l0=1, l1=2, l2=1, l3=0, edge cost c=3, others 0."""
    base = pta_p3(invariants=True)
    edges = tuple(PEdge(e.source, e.action, e.guard, 3 if e.action == "c" else 0, e.branches) for e in base.edges)
    return TimedAutomaton(base.clocks, base.locations, base.initial, edges,
                          rates={"l0": 1, "l1": 2, "l2": 1, "l3": 0}, invariants=base.invariants)


def p3_model() -> Model:
    return Model("P3", p3_priced(), ("l3",), (0.3, 0.6, 0.9), allow_deadlocks=True)


def _random_edge(rng: random.Random, clocks, source: str, target: str, name: str) -> Edge:
    names = ["0", *clocks]
    guard = []
    for _ in range(rng.randint(0, 2)):
        left, right = rng.sample(names, 2)
        guard.append(Constraint(left, right, rng.randint(-2, 2), rng.random() < 0.5))
    resets = frozenset(c for c in clocks if rng.random() < 0.4)
    return Edge(source, target, name, guard=tuple(guard), resets=resets, cost=rng.choice((0, 1, 2)))


def random_model(rng: random.Random, i: int) -> Model:
    """1 or 2 clocks (2 for even i, so at least half the batch), 2-4 locations plus the target
    g; each non-target location gets an edge to g with probability 0.7, plus 2-5 random edges."""
    clocks = ("x", "y") if i % 2 == 0 else ("x",)
    locations = tuple(f"l{k}" for k in range(rng.randint(2, 4)))
    edges = []
    for l in locations:
        if rng.random() < 0.7:
            edges.append(_random_edge(rng, clocks, l, "g", f"e{len(edges)}"))
    for _ in range(rng.randint(2, 5)):
        edges.append(_random_edge(rng, clocks, rng.choice(locations), rng.choice(locations + ("g",)),
                                  f"e{len(edges)}"))
    ta = TimedAutomaton(
        clocks, locations + ("g",), "l0", tuple(edges),
        rates={l: rng.choice((0, 1, 2, 3)) for l in locations},
        invariants={l: tuple(le(c, rng.choice((2, 3))) for c in clocks) for l in locations},
    )
    return Model(f"R{i:04d}", ta, ("g",), (0.5, 0.9), allow_deadlocks=True,
                 tag="non_zeno" if structurally_non_zeno(ta) else "zeno")


def random_models(n: int = 1000, seed: int = 2026) -> list[Model]:
    rng = random.Random(seed)
    return [random_model(rng, i) for i in range(n)]


def simple_cycles(ta: TimedAutomaton) -> list[list[tuple[PEdge, Branch]]]:
    """Every simple cycle of the location graph, as a list of (edge, branch) steps (multi-edges
    and branches distinct)."""
    order = {l: k for k, l in enumerate(ta.locations)}
    cycles = []

    def dfs(start: str, loc: str, path: list[tuple[PEdge, Branch]], visited: set[str]) -> None:
        for e in ta.outgoing(loc):
            for b in e.branches:
                if b.target == start:
                    cycles.append(path + [(e, b)])
                elif order[b.target] > order[start] and b.target not in visited:
                    dfs(start, b.target, path + [(e, b)], visited | {b.target})

    for start in ta.locations:
        dfs(start, start, [], {start})
    return cycles


def structurally_non_zeno(ta: TimedAutomaton) -> bool:
    """Every simple cycle has a clock that is reset on the cycle and tested on the cycle with
    a lower bound >= 1 (x >= c or x > c with c >= 1), so each round takes at least 1 time unit."""
    for cycle in simple_cycles(ta):
        reset = set().union(*(b.resets for _, b in cycle))
        tested = {g.right for e, _ in cycle for g in e.guard if g.left == "0" and g.right != "0" and -g.bound >= 1}
        if not reset & tested:
            return False
    return True


# --- reporting --------------------------------------------------------------

def _fmt(v: float) -> str:
    return "inf" if math.isinf(v) else f"{v:.6f}"


def format_row(r: Row) -> str:
    flags = [f for f, on in (("integer gap", r.integer_gap), ("boundary gap", r.boundary_gap)) if on]
    return (f"{r.model:5} | {r.lam:<4} | {_fmt(r.v_brg):>9} | {_fmt(r.v_int):>9} | {_fmt(r.v_bnd):>9} | "
            f"{r.brg_size[0]}/{r.brg_size[1]} | {r.int_size[0]}/{r.int_size[1]} | {r.bnd_size[0]}/{r.bnd_size[1]}"
            + (f"  <-- {', '.join(flags)}" if flags else ""))


HEADER = "model | λ    |     V_BRG | V_integer | V_boundary | BRG s/a | integer s/a | boundary s/a"


def save_gap(m: Model, r: Row) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{m.name}_lam{r.lam}.py"
    runs = "\n".join(f"#   {side}: {run}" for side, run in r.runs.items())
    path.write_text(
        f'"""Gap found by compare_values.py: {format_row(r)}"""\n'
        "from fractions import Fraction\n\n"
        "from prestable.model import Branch, Constraint, PEdge, TimedAutomaton\n\n"
        f"ta = {m.ta!r}\n"
        f"targets = {m.targets!r}\n"
        f"discount = {r.lam!r}\n"
        f"allow_deadlocks = {m.allow_deadlocks!r}\n"
        f"# V_BRG={r.v_brg!r}, V_integer={r.v_int!r}, V_boundary={r.v_bnd!r}\n"
        f"# optimal runs:\n{runs}\n"
    )
    return path


def run_models(models: list[Model], *, verbose: bool = True):
    rows, skipped = [], []
    for m in models:
        out = compare_model(m)
        if out is None:
            skipped.append(m)
            continue
        for r in out:
            rows.append((m, r))
            if verbose:
                print(format_row(r))
                if r.runs:
                    for side, run in r.runs.items():
                        print(f"        {side:8}: {run}")
                    print(f"        saved {save_gap(m, r).relative_to(HERE.parent)}")
    return rows, skipped


def main() -> None:
    quick = "--quick" in sys.argv
    print(HEADER)
    start = time.perf_counter()
    run_models(example_models() + [p3_model()])
    if quick:
        return
    print("\nrandom models:")
    models = random_models()
    rows, skipped = run_models(models, verbose=False)
    gaps_int = [(m, r) for m, r in rows if r.integer_gap]
    gaps_bnd = [(m, r) for m, r in rows if r.boundary_gap]
    for m, r in gaps_int + gaps_bnd:
        save_gap(m, r)
    ran = {m.name: m for m, _ in rows}
    print(f"generated {len(models)}; run: {len(ran)} models ({len(rows)} rows); "
          f"skipped (target unreachable): {len(skipped)}")
    print(f"{'split':16} {'run':>5} {'skipped':>8} {'int gaps':>9} {'bnd gaps':>9}")
    for label, pick in [("zeno", lambda m: m.tag == "zeno"), ("non_zeno", lambda m: m.tag == "non_zeno"),
                        ("1 clock", lambda m: len(m.ta.clocks) == 1), ("2 clocks", lambda m: len(m.ta.clocks) == 2)]:
        print(f"{label:16} {sum(map(pick, ran.values())):>5} {sum(map(pick, skipped)):>8} "
              f"{sum(pick(m) for m, _ in gaps_int):>9} {sum(pick(m) for m, _ in gaps_bnd):>9}")
    print(f"integer gaps: {len(gaps_int)} rows in {len({m.name for m, _ in gaps_int})} models; "
          f"boundary gaps: {len(gaps_bnd)} rows in {len({m.name for m, _ in gaps_bnd})} models")
    bad = [(m, r) for m, r in rows if not r.lower_bounds_hold]
    print(f"lower-bound violations: {len(bad)}")
    for m, r in bad:
        print("  " + format_row(r))
    for kind, gaps in (("integer", gaps_int), ("boundary", gaps_bnd)):
        if gaps:
            m, r = min(gaps, key=lambda g: (len(g[0].ta.clocks), len(g[0].ta.locations), len(g[0].ta.edges),
                                            g[1].brg_size[0]))
            print(f"\nsmallest {kind}-gap model: {m.name} ({m.tag}), λ={r.lam}")
            print(f"  {m.ta!r}")
            print("  " + format_row(r))
            for side, run in r.runs.items():
                print(f"  {side:8}: {run}")
    print("\naverage states per row (BRG / integer / boundary):")
    for n_clocks in (1, 2):
        sel = [r for m, r in rows if len(m.ta.clocks) == n_clocks]
        if sel:
            avg = [sum(r.brg_size[0] for r in sel) / len(sel), sum(r.int_size[0] for r in sel) / len(sel),
                   sum(r.bnd_size[0] for r in sel) / len(sel)]
            print(f"  {n_clocks} clock(s): {avg[0]:.1f} / {avg[1]:.1f} / {avg[2]:.1f}  ({len(sel)} rows)")
    print(f"total time: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
