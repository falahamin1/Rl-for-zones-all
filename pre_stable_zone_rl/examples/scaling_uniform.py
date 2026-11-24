"""Uniform constant scaling: multiply every guard and invariant bound by c and compare.

Predictions checked for each model and factor c (g1 = graph at c = 1, gc = graph at c):
    P1  same number of Phase 2 splits; (l, Z) -> (l, c·Z) is a bijection from g1's nodes to
        gc's nodes that maps the edges (source, label, edge index, target) exactly
    P2  the boundary MDPs are isomorphic under (l, nu, n) -> (l, c·nu, phi(n)), with
        delay -> c·delay and the same edge, mapped via and mapped target
    P3  V(scale_ta(ta, c)) == c · V(scale_ta(ta, 1, cost_div=c)) for BRG, integer and
        boundary (1e-9 relative), and the three solvers agree at every c
and the growth of states, actions and times is recorded (P4).

Value iteration runs with eps = 1e-12 (not the default 1e-9), since its absolute stopping
threshold would otherwise dominate a 1e-9 relative comparison of values that scale with c.

Writes examples/out/scaling_uniform.csv, scaling_uniform.png and scaling_uniform_summary.txt.

    python examples/scaling_uniform.py                  # catalogue + 100 random models, c up to 64
    python examples/scaling_uniform.py --random 0 --cs 1 2 4 --timeout 30
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # run without installing the package
sys.path.insert(0, str(HERE))

from brg_examples import ALLOW_DEADLOCKS, EXAMPLES  # noqa: E402
from compare_values import p3_priced, random_models  # noqa: E402
from p3_example import pta_p3  # noqa: E402
from prestable.block_vi import _tie_key, build_block_mdp  # noqa: E402
from prestable.brg import _frac, build_brg, value_iteration  # noqa: E402
from prestable.graph import PrestableResult, build_prestable_graph  # noqa: E402
from prestable.model import Constraint, PEdge, TimedAutomaton  # noqa: E402
from prestable.zones import Zone, ZoneContext  # noqa: E402

OUT = HERE / "out"
CS = (1, 2, 4, 8, 16, 32, 64)
SOLVERS = ("BRG", "integer", "boundary")
EPS = 1e-12
REL = 1e-9
TIMEOUT = 120.0
RANDOM_SEED = 2026


# --- scaling ------------------------------------------------------------------

def _scale(cs, c: int) -> tuple[Constraint, ...]:
    return tuple(Constraint(g.left, g.right, g.bound * c, g.strict) for g in cs)


def scale_ta(ta: TimedAutomaton, c: int, cost_div: int = 1) -> TimedAutomaton:
    """Every guard and invariant bound times c; edge costs divided by cost_div (exactly);
    branches copied unchanged."""
    edges = tuple(
        PEdge(e.source, e.action, _scale(e.guard, c), e.cost if cost_div == 1 else _frac(e.cost) / cost_div,
              e.branches)
        for e in ta.edges
    )
    return TimedAutomaton(ta.clocks, ta.locations, ta.initial, edges, rates=ta.rates,
                          invariants={l: _scale(cs, c) for l, cs in ta.invariants.items()})


def scale_zone(z: Zone, c: int, ctx_c: ZoneContext) -> Zone:
    return ctx_c.from_guard(_scale(z.constraints(), c))


# --- models -------------------------------------------------------------------

@dataclass
class Spec:
    name: str
    ta: TimedAutomaton
    targets: tuple[str, ...]
    lams: tuple[float, ...]
    allow_deadlocks: bool
    values: bool = True  # False: graph checks only (P1)
    group: str = "catalogue"


def catalogue() -> dict[str, Spec]:
    out = {name: Spec(name, make(), ("g",), lams, name in ALLOW_DEADLOCKS) for name, (make, lams) in EXAMPLES.items()}
    out["P3"] = Spec("P3", pta_p3(invariants=False), (), (), False, values=False)  # no invariants: no solver applies
    out["P3_inv"] = Spec("P3_inv", p3_priced(), ("l3",), (0.3, 0.6, 0.9), True)  # the catalogue's P3 (compare_values)
    return out


def random_specs(n: int, seed: int = RANDOM_SEED, pool: int = 1000) -> list[Spec]:
    """The first n random models of compare_values.py whose target is reachable (λ = 0.5)."""
    out = []
    for m in random_models(pool, seed):
        if len(out) == n:
            break
        if build_brg(m.ta, m.targets, allow_deadlocks=True).targets:
            out.append(Spec(m.name, m.ta, m.targets, (0.5,), True, group="random"))
    return out


def spec_by_key(key: tuple[str, str]) -> Spec:
    group, name = key
    if group == "catalogue":
        return catalogue()[name]
    i = int(name[1:])
    m = random_models(i + 1, RANDOM_SEED)[i]
    return Spec(m.name, m.ta, m.targets, (0.5,), True, group="random")


# --- checks -------------------------------------------------------------------

def check_p1(r1: PrestableResult, rc: PrestableResult, c: int) -> tuple[dict[int, int], bool]:
    """Raise AssertionError unless P1 holds; return phi (node id map) and whether ids coincide."""
    g1, gc = r1.graph, rc.graph
    assert r1.phase2.splits == rc.phase2.splits, f"splits: {r1.phase2.splits} at c=1, {rc.phase2.splits} at c={c}"
    ctx_c = gc.nodes[0].zone.ctx
    phi = {}
    for n in g1.nodes:
        z = scale_zone(n.zone, c, ctx_c)
        try:
            phi[n.id] = gc.node_id(n.location, z)
        except KeyError:
            raise AssertionError(f"node {n.id} ({n.location}, {n.zone}) scales to {z}, not a node at c={c}") from None
    assert len(set(phi.values())) == len(phi) == len(gc.nodes), \
        f"{len(g1.nodes)} nodes at c=1 map onto {len(set(phi.values()))} of the {len(gc.nodes)} nodes at c={c}"
    mapped = Counter((phi[e.source], e.label, e.edge, phi[e.target]) for e in g1.edges)
    actual = Counter((e.source, e.label, e.edge, e.target) for e in gc.edges)
    if mapped != actual:
        raise AssertionError(f"edges at c={c}: missing {list((mapped - actual).elements())[:3]}, "
                             f"extra {list((actual - mapped).elements())[:3]}")
    return phi, all(k == v for k, v in phi.items())


def check_p2(m1, mc, phi: dict[int, int], c: int) -> None:
    """Raise AssertionError unless the boundary MDPs m1 (c = 1) and mc are isomorphic."""
    def image(s):
        loc, nu, n = s
        return loc, tuple(c * v for v in nu), phi[n]

    assert len(m1.states) == len(mc.states), f"{len(m1.states)} states at c=1, {len(mc.states)} at c={c}"
    index = {s: j for j, s in enumerate(mc.states)}
    assert image(m1.states[m1.initial]) == mc.states[mc.initial]
    for i, s in enumerate(m1.states):
        j = index.get(image(s))
        assert j is not None, f"state {s} maps to {image(s)}, not a state at c={c}"
        mapped = Counter((a.edge, c * a.delay, phi[a.via],
                          tuple(sorted((image(m1.states[k]), p) for p, k in a.successors))) for a in m1.actions[i])
        actual = Counter((a.edge, a.delay, a.via, tuple(sorted((mc.states[k], p) for p, k in a.successors)))
                         for a in mc.actions[j])
        assert mapped == actual, (f"state {s} -> {mc.states[j]}: missing {list((mapped - actual).elements())[:3]}, "
                                  f"extra {list((actual - mapped).elements())[:3]}")


def close(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= REL * max(abs(a), abs(b)) + 1e-12


@dataclass
class Solved:
    values: list[float]  # V(initial) per λ
    mdp: object
    states: int
    actions: int
    build_s: float
    vi_s: float


def solve_values(spec: Spec, ta: TimedAutomaton, graph, solver: str) -> Solved:
    t0 = time.perf_counter()
    if solver == "BRG":
        mdp, key = build_brg(ta, spec.targets, allow_deadlocks=spec.allow_deadlocks), None
    else:
        mdp = build_block_mdp(ta, graph, spec.targets, variant=solver, allow_deadlocks=spec.allow_deadlocks)
        key = _tie_key
    t1 = time.perf_counter()
    values = [value_iteration(mdp, lam, eps=EPS, tie_key=key)[0][mdp.initial] for lam in spec.lams]
    return Solved(values, mdp, len(mdp.states), sum(len(a) for a in mdp.actions), t1 - t0, time.perf_counter() - t1)


# --- experiment ---------------------------------------------------------------

class CellTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    def fire(*_):
        raise CellTimeout()
    old = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


FIELDS = ("model", "group", "clocks", "c", "K", "solver", "status", "reason", "states", "actions", "build_s",
          "vi_s", "graph_s", "graph_nodes", "graph_edges", "splits", "p1", "ids_coincide", "p2", "p3",
          "p3_max_rel", "agree", "values")


def run_model(key: tuple[str, str], cs=CS, timeout: float = TIMEOUT) -> list[dict]:
    """All rows of one model. A cell that times out or exceeds max_states is skipped, and so
    are the cells of that solver at every larger c."""
    spec = spec_by_key(key)
    solvers = SOLVERS if spec.values else ()
    rows: list[dict] = []
    dead: dict[str, str] = {}
    r1 = build_prestable_graph(spec.ta)
    m1 = solve_values(spec, spec.ta, r1.graph, "boundary").mdp if spec.values else None
    for c in cs:
        ta_c = scale_ta(spec.ta, c)
        base = dict(model=spec.name, group=spec.group, clocks=len(spec.ta.clocks), c=c, K=ta_c.max_constant())
        rc, phi = None, None
        if "graph" not in dead:
            try:
                t0 = time.perf_counter()
                with time_limit(timeout):
                    rc = build_prestable_graph(ta_c)
                base.update(graph_s=round(time.perf_counter() - t0, 4), graph_nodes=len(rc.graph.nodes),
                            graph_edges=len(rc.graph.edges), splits=rc.phase2.splits)
            except CellTimeout:
                dead["graph"] = f"graph > {timeout:.0f} s at c={c}"
        if rc is not None:
            try:
                phi, same_ids = check_p1(r1, rc, c)
                base.update(p1="ok", ids_coincide=same_ids)
            except AssertionError as err:
                base.update(p1=f"FAIL: {err}")
        cell_rows = {}
        for solver in solvers or ("-",):
            row = dict(base, solver=solver)
            cell_rows[solver] = row
            if solver == "-":
                row.update(status="n/a", reason="graph only")
                continue
            if solver in dead or (solver != "BRG" and rc is None):
                row.update(status="skipped", reason=dead.get(solver) or dead["graph"])
                continue
            try:
                with time_limit(timeout):
                    res = solve_values(spec, ta_c, None if rc is None else rc.graph, solver)
            except CellTimeout:
                dead[solver] = f"{solver} > {timeout:.0f} s at c={c}"
                row.update(status="skipped", reason=dead[solver])
                continue
            except RuntimeError as err:  # max_states
                dead[solver] = f"c={c}: {err}"
                row.update(status="skipped", reason=dead[solver])
                continue
            row.update(status="ok", states=res.states, actions=res.actions, build_s=round(res.build_s, 4),
                       vi_s=round(res.vi_s, 4), values=";".join(repr(v) for v in res.values))
            # reference: the unscaled guards with costs / c, at c = 1 size
            ta_div = scale_ta(spec.ta, 1, cost_div=c)
            g_div = None if solver == "BRG" else build_prestable_graph(ta_div).graph
            ref = solve_values(spec, ta_div, g_div, solver).values
            rel = max((0.0 if math.isinf(v) else abs(v - c * r) / max(abs(v), abs(c * r), 1e-300)
                       for v, r in zip(res.values, ref) if not (v == 0 and r == 0)), default=0.0)
            ok = all(close(v, c * r) for v, r in zip(res.values, ref))
            row.update(p3="ok" if ok else f"FAIL: V={res.values} c·V_div={[c * r for r in ref]}", p3_max_rel=rel)
            row["_values"] = res.values
            if solver == "boundary" and phi is not None:
                try:
                    check_p2(m1, res.mdp, phi, c)
                    row["p2"] = "ok"
                except AssertionError as err:
                    row["p2"] = f"FAIL: {err}"
        done = [r for r in cell_rows.values() if "_values" in r]
        if len(done) >= 2:
            agree = all(close(a, b) for r in done for a, b in zip(r["_values"], done[0]["_values"]))
            for r in done:
                r["agree"] = agree
        for r in cell_rows.values():
            r.pop("_values", None)
            rows.append(r)
    return rows


# --- outputs ------------------------------------------------------------------

def slope(points: list[tuple[int, int]]) -> Optional[float]:
    """Least-squares slope of log(states) against log(c)."""
    if len(points) < 2:
        return None
    xs = [math.log(c) for c, _ in points]
    ys = [math.log(s) for _, s in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if n == 0 else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def summarise(rows: list[dict]) -> str:
    lines = []
    fails = [r for r in rows if any(str(r.get(k, "")).startswith("FAIL") for k in ("p1", "p2", "p3"))]
    lines.append(f"P1-P3 failures: {len(fails)}")
    for r in fails:
        lines.append(f"  {r['model']} c={r['c']} {r['solver']}: " +
                     "; ".join(f"{k}={r[k]}" for k in ("p1", "p2", "p3") if str(r.get(k, "")).startswith("FAIL")))
    dis = sorted({(r["model"], r["c"]) for r in rows if r.get("agree") is False})
    lines.append(f"(model, c) cells where the solvers disagree: {len(dis)}")
    by_model: dict[str, list] = {}
    for m, c in dis:
        by_model.setdefault(m, []).append(c)
    for m, c_list in by_model.items():
        lines.append(f"  {m}: c = {c_list}")
    ids = Counter(r.get("ids_coincide") for r in rows if r["solver"] in ("BRG", "-") and r.get("p1") == "ok" and r["c"] > 1)
    lines.append(f"node ids coincide (cells with c > 1): {ids.get(True, 0)} yes, {ids.get(False, 0)} no")

    lines.append("\nCatalogue: c | states BRG / integer / boundary | graph nodes | graph build s")
    for m in dict.fromkeys(r["model"] for r in rows if r["group"] == "catalogue"):
        lines.append(f"  {m}")
        for c in CS:
            cell = {r["solver"]: r for r in rows if r["model"] == m and r["c"] == c}
            if not cell:
                continue
            any_row = next(iter(cell.values()))
            st = " / ".join(str(cell[s]["states"]) if s in cell and cell[s].get("status") == "ok" else "-"
                            for s in SOLVERS)
            lines.append(f"    c={c:<3} {st:>28} | {any_row.get('graph_nodes', '-')!s:>6} | "
                         f"{any_row.get('graph_s', 'skipped')}")

    lines.append("\nRandom batch: per-model slope of log(states) vs log(c); median [min, max] (models)")
    rnd = [r for r in rows if r["group"] == "random" and r.get("status") == "ok"]
    for k in (1, 2):
        for s in SOLVERS:
            per_model: dict[str, list] = {}
            for r in rnd:
                if r["clocks"] == k and r["solver"] == s:
                    per_model.setdefault(r["model"], []).append((r["c"], r["states"]))
            slopes = [v for v in (slope(p) for p in per_model.values()) if v is not None]
            reached = max((c for p in per_model.values() for c, _ in p), default=None)
            full = sum(1 for p in per_model.values() if len(p) == len(CS))
            if slopes:
                lines.append(f"  {k} clock(s) {s:8}: {median(slopes):.3f} [{min(slopes):.3f}, {max(slopes):.3f}] "
                             f"({len(slopes)} models); largest c {reached}; all c reached: {full}")
    skipped = [r for r in rows if r.get("status") == "skipped"]
    lines.append(f"\nSkipped cells: {len(skipped)}")
    first = {}
    for r in skipped:
        first.setdefault((r["model"], r["solver"]), r)
    reasons = Counter(r["reason"].split(" at ")[0].split(": ")[-1] if "max_states" not in r["reason"]
                      else f"{r['solver']} max_states" for r in first.values())
    for reason, n in reasons.most_common():
        lines.append(f"  first skip, {reason}: {n} (model, solver) pairs")
    for r in first.values():
        if r["group"] == "catalogue":
            lines.append(f"  {r['model']} {r['solver']}: {r['reason']}")
    return "\n".join(lines)


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"BRG": "#2a78d6", "integer": "#eb6834", "boundary": "#1baf7a"}  # categorical slots 1-3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, k in zip(axes, (1, 2)):
        for s in SOLVERS:
            xs, med, lo, hi, ns = [], [], [], [], []
            for c in CS:
                v = [r["states"] for r in rows if r["group"] == "random" and r["clocks"] == k and r["solver"] == s
                     and r["c"] == c and r.get("status") == "ok"]
                if v:
                    xs.append(c), med.append(median(v)), lo.append(min(v)), hi.append(max(v)), ns.append(len(v))
            ax.fill_between(xs, lo, hi, color=colors[s], alpha=0.15, linewidth=0)
            ax.plot(xs, med, color=colors[s], linewidth=2, marker="o", markersize=5, label=s)
            if xs:
                ax.annotate(f"n={ns[-1]}", (xs[-1], med[-1]), textcoords="offset points", xytext=(4, 4),
                            fontsize=8, color="#52514e")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(CS, [str(c) for c in CS])
        ax.set_title(f"{k} clock{'s' if k > 1 else ''}", fontsize=11)
        ax.set_xlabel("scaling factor c")
        ax.grid(True, which="major", color="#e5e4df", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("MDP states (median, band = min–max)")
    axes[0].legend(frameon=False)
    fig.suptitle("States vs uniform constant scaling, random models (λ = 0.5)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", type=int, nargs="+", default=list(CS))
    ap.add_argument("--random", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=TIMEOUT)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--models", nargs="*", help="catalogue models only (default: all)")
    args = ap.parse_args()
    keys = [("catalogue", n) for n in (args.models or catalogue())]
    keys += [("random", s.name) for s in random_specs(args.random)]
    keys.sort(key=lambda k: -len(spec_by_key(k).ta.clocks))  # two-clock models first: they run longest
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_model, k, tuple(args.cs), args.timeout): k for k in keys}
        for i, f in enumerate(as_completed(futures), 1):
            rows.extend(f.result())
            print(f"[{time.perf_counter() - start:7.1f}s] {i}/{len(keys)} {futures[f][1]}", flush=True)
    order = {k[1]: i for i, k in enumerate(sorted(keys, key=lambda k: (k[0] != "catalogue", k[1])))}
    rows.sort(key=lambda r: (order[r["model"]], r["c"], SOLVERS.index(r["solver"]) if r["solver"] in SOLVERS else 0))
    OUT.mkdir(exist_ok=True)
    with open(OUT / "scaling_uniform.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    plot(rows, OUT / "scaling_uniform.png")
    text = summarise(rows) + f"\n\ntotal time: {time.perf_counter() - start:.1f} s"
    (OUT / "scaling_uniform_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
