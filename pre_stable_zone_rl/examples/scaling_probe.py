"""How the pipeline scales with the constant K, on a two-clock toy family.

Toy family T(K): clocks x, y; locations l0 and the target g; l0 has rate 1 and invariant
x <= K, y <= K; edges
    f: l0 -> g   guard x >= K
    r: l0 -> l0  guard y >= 1, reset {y}
(all edge costs 0).

For each K: Phase 1/2/3 times (uninstrumented run), then a second, instrumented run
(monkeypatched in this script only; nothing in src/ is changed) counting Phase 2 splits,
_find_bad calls, triples examined (and how many were re-examinations of a triple already
found good in an earlier scan), time inside _find_bad, and Pre-cache hits/misses in
Phase 2 and Phase 3. Then graph sizes, block-MDP and BRG sizes and values (λ = 0.5).
Finally a cProfile of build_prestable_graph at the largest K.

    python examples/scaling_probe.py              # K = 2, 4, 8, 16 (+ 32 if K = 16 builds in < 60 s)
    python examples/scaling_probe.py 2 4          # chosen K values, no automatic K = 32
"""

import cProfile
import io
import pstats
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # run without installing the package

import prestable.algorithm as alg  # noqa: E402
from prestable.block_vi import build_block_mdp, solve  # noqa: E402
from prestable.brg import build_brg, value_iteration  # noqa: E402
from prestable.graph import build_graph, build_prestable_graph  # noqa: E402
from prestable.model import DELAY, Edge, TimedAutomaton, ge, le  # noqa: E402
from prestable.zones import ZoneContext  # noqa: E402

LAMBDA = 0.5
TARGETS = ("g",)


def toy(K: int) -> TimedAutomaton:
    return TimedAutomaton(
        ("x", "y"), ("l0", "g"), "l0",
        (Edge("l0", "g", "f", guard=(ge("x", K),)),
         Edge("l0", "l0", "r", guard=(ge("y", 1),), resets={"y"})),
        rates={"l0": 1}, invariants={"l0": (le("x", K), le("y", K))},
    )


def timed_phases(ta: TimedAutomaton):
    ctx = ZoneContext(ta.clocks)
    t0 = time.perf_counter()
    p1 = alg.phase1(ta, ctx)
    t1 = time.perf_counter()
    p2 = alg.phase2(ta, ctx, p1.store)
    t2 = time.perf_counter()
    g = build_graph(ta, ctx, p2.store, K=p1.K)
    t3 = time.perf_counter()
    return (t1 - t0, t2 - t1, t3 - t2), p2, g


class Instrument:
    """Counters around algorithm._find_bad, algorithm._triples and _PreCache.__call__."""

    def __init__(self):
        self.find_bad_calls = 0
        self.find_bad_time = 0.0
        self.triples = 0
        self.reexamined = 0
        self.seen_good: set = set()
        self.cache = Counter()  # (phase, "hit"/"miss") -> n
        self.phase = "phase2"

    def __enter__(self):
        self._orig = (alg._find_bad, alg._triples, alg._PreCache.__call__)
        orig_find_bad, orig_triples, orig_call = self._orig
        inst = self

        def triples(ts, store):
            for item in orig_triples(ts, store):
                inst.triples += 1
                key = (item[0], item[2], item[3])
                if key in inst.seen_good:
                    inst.reexamined += 1
                yield item
                inst.seen_good.add(key)  # only reached if the scan continued: the triple was good

        def find_bad(ts, store, pre):
            inst.find_bad_calls += 1
            t0 = time.perf_counter()
            try:
                return orig_find_bad(ts, store, pre)
            finally:
                inst.find_bad_time += time.perf_counter() - t0

        def call(self, i, target):
            inst.cache[(inst.phase, "hit" if (i, target) in self.cache else "miss")] += 1
            return orig_call(self, i, target)

        alg._find_bad, alg._triples, alg._PreCache.__call__ = find_bad, triples, call
        return self

    def __exit__(self, *exc):
        alg._find_bad, alg._triples, alg._PreCache.__call__ = self._orig


def instrumented(ta: TimedAutomaton):
    ctx = ZoneContext(ta.clocks)
    p1 = alg.phase1(ta, ctx)
    with Instrument() as inst:
        t0 = time.perf_counter()
        p2 = alg.phase2(ta, ctx, p1.store)
        p2_time = time.perf_counter() - t0
        inst.phase = "phase3"
        build_graph(ta, ctx, p2.store, K=p1.K)
    return inst, p2, p2_time


def measure(K: int) -> dict:
    ta = toy(K)
    (t1, t2, t3), p2, g = timed_phases(ta)
    inst, p2b, p2_inst_time = instrumented(ta)
    assert p2b.splits == p2.splits
    kinds = Counter("delay" if e.label == DELAY else "action" for e in g.edges)
    row = dict(K=K, t1=t1, t2=t2, t3=t3, splits=p2.splits, find_bad=inst.find_bad_calls,
               triples=inst.triples, reexamined=inst.reexamined,
               find_bad_share=inst.find_bad_time / p2_inst_time if p2_inst_time else 0.0,
               p2_hit=inst.cache[("phase2", "hit")], p2_miss=inst.cache[("phase2", "miss")],
               p3_hit=inst.cache[("phase3", "hit")], p3_miss=inst.cache[("phase3", "miss")],
               nodes=len(g.nodes), delay=kinds["delay"], action=kinds["action"])
    for v in ("boundary", "integer"):
        mdp = build_block_mdp(ta, g, TARGETS, variant=v, allow_deadlocks=True)
        res = solve(ta, TARGETS, LAMBDA, variant=v, graph=g, mdp=mdp)
        row[v] = (res.states, res.actions, res.value)
    brg = build_brg(ta, TARGETS, allow_deadlocks=True)
    row["brg"] = (len(brg.states), sum(map(len, brg.actions)), value_iteration(brg, LAMBDA)[0][brg.initial])
    return row


def print_tables(rows: list[dict]) -> None:
    print("\n## Zone graph (times from an uninstrumented run)\n")
    print("| K | phase1 s | phase2 s | phase3 s | total s | splits | _find_bad calls | triples examined "
          "| of which re-examined | _find_bad share of phase2 | Pre cache p2 hit/miss | Pre cache p3 hit/miss "
          "| nodes | delay edges | action edges |")
    print("|" + "---|" * 15)
    for r in rows:
        total = r["t1"] + r["t2"] + r["t3"]
        print(f"| {r['K']} | {r['t1']:.2f} | {r['t2']:.2f} | {r['t3']:.2f} | {total:.2f} | {r['splits']} "
              f"| {r['find_bad']} | {r['triples']} | {r['reexamined']} ({r['reexamined'] / max(r['triples'], 1):.0%}) "
              f"| {r['find_bad_share']:.0%} | {r['p2_hit']}/{r['p2_miss']} | {r['p3_hit']}/{r['p3_miss']} "
              f"| {r['nodes']} | {r['delay']} | {r['action']} |")
    print(f"\n## MDPs (λ = {LAMBDA}): states / actions / V(initial)\n")
    print("| K | boundary | integer | BRG |")
    print("|---|---|---|---|")
    for r in rows:
        cells = [f"{s} / {a} / {v:.6f}" for s, a, v in (r["boundary"], r["integer"], r["brg"])]
        print(f"| {r['K']} | " + " | ".join(cells) + " |")


def profile(K: int) -> None:
    ta = toy(K)
    pr = cProfile.Profile()
    pr.enable()
    build_prestable_graph(ta)
    pr.disable()
    for key in ("cumulative", "tottime"):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats(key).print_stats(15)
        print(f"\n## cProfile of build_prestable_graph(T({K})), top 15 by {key}\n")
        print("```")
        print(s.getvalue().strip())
        print("```")


def main() -> None:
    Ks = [int(a) for a in sys.argv[1:]] or [2, 4, 8, 16]
    auto = len(sys.argv) == 1
    rows = []
    for K in Ks:
        rows.append(measure(K))
        r = rows[-1]
        print(f"K={K} done: phase1/2/3 = {r['t1']:.2f}/{r['t2']:.2f}/{r['t3']:.2f} s", flush=True)
    last = rows[-1]
    if auto and last["t1"] + last["t2"] + last["t3"] < 60:
        rows.append(measure(32))
        r = rows[-1]
        print(f"K=32 done: phase1/2/3 = {r['t1']:.2f}/{r['t2']:.2f}/{r['t3']:.2f} s", flush=True)
    print_tables(rows)
    profile(rows[-1]["K"])


if __name__ == "__main__":
    main()
