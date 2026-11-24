"""PTA P3 (Jovanović et al. 2017, Fig. 7): run the algorithm and plot the partitions.

The invariant 0 <= x, y <= 2 of every location is encoded by adding x <= 2 and y <= 2 to
the guard of every edge leaving that location.

Writes into examples/out/:
    p3_phase1.png, p3_phase1_blocks.txt     Phase 1 store and its block ids
    p3_final.png, p3_final_blocks.txt       final (pre-stable) store and its block ids
    p3_phase2_splits.txt                    one line per Phase 2 split
    p3_graph.dot, p3_graph.png, p3_summary.txt
and the same files with prefix p3_inv_ for the version with real location invariants.

    python examples/p3_example.py
"""

import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # run without installing the package

from prestable.algorithm import phase1  # noqa: E402
from prestable.export import render, summary, write_dot  # noqa: E402
from prestable.graph import build_prestable_graph, reachable  # noqa: E402
from prestable.model import Edge, TimedAutomaton, ge, gt, le  # noqa: E402
from prestable.zones import Zone, ZoneContext, intersect, subtract_all  # noqa: E402

OUT = HERE / "out"


def pta_p3(invariants: bool = False) -> TimedAutomaton:
    """PTA P3. invariants=False encodes the invariant 0 <= x, y <= 2 in the guards of the
    edges leaving each location; invariants=True uses real location invariants instead."""
    inv = (le("x", 2), le("y", 2))
    extra = () if invariants else inv
    return TimedAutomaton(
        clocks=("x", "y"),
        locations=("l0", "l1", "l2", "l3"),
        initial="l0",
        edges=(
            Edge("l0", "l1", "a1", guard=extra, resets={"x"}),
            Edge("l0", "l1", "a2", guard=extra, resets={"y"}),
            Edge("l1", "l2", "b", guard=(le("x", 1), le("y", 1), *extra)),
            Edge("l1", "l3", "c", guard=extra),
            Edge("l2", "l2", "d", guard=(ge("x", 1), *extra), resets={"x", "y"}),
            Edge("l3", "l3", "d", guard=(ge("x", 1), *extra), resets={"x", "y"}),
        ),
        invariants={l: inv for l in ("l0", "l1", "l2", "l3")} if invariants else {},
    )


def square(ctx: ZoneContext) -> Zone:
    """[0, 2]²"""
    return ctx.from_guard([le("x", 2), le("y", 2)])


def zeta(ctx: ZoneContext) -> list[Zone]:
    """ζ = {v ∈ [0,2]² : x > 1 or y > 1}, as two zones."""
    return [ctx.from_guard([le("x", 2), le("y", 2), gt("x", 1)]),
            ctx.from_guard([le("x", 2), le("y", 2), le("x", 1), gt("y", 1)])]


def inside(z: Zone, cover: list[Zone]) -> bool:
    return not subtract_all(z, cover)


def disjoint(z: Zone, cover: list[Zone]) -> bool:
    return all(intersect(z, w).is_empty for w in cover)


def zeta_blocks(blocks, ctx: ZoneContext) -> list[Zone]:
    """The blocks inside ζ; raises if some block straddles ζ."""
    Z = zeta(ctx)
    out = []
    for b in sorted(blocks, key=Zone.key):
        if inside(b, Z):
            out.append(b)
        elif not disjoint(b, Z):
            raise AssertionError(f"block {b} at l1 is neither inside nor disjoint from ζ")
    return out


def block_ids(store) -> dict[str, dict[Zone, str]]:
    return {l: {z: f"{l}.{i}" for i, z in enumerate(sorted(P, key=Zone.key))} for l, P in store.items()}


def write_blocks(ids, path: Path) -> None:
    lines = [f"{name}: {z}" for l in ids for z, name in ids[l].items()]
    path.write_text("\n".join(lines) + "\n")


def plot(store, ctx, name: str):
    """Plot `store` with block ids and ζ highlighted at l1; returns the ids. Needs matplotlib."""
    from prestable.plot2d import plot_store

    ids = block_ids(store)
    write_blocks(ids, OUT / f"{name}_blocks.txt")
    plot_store(store, OUT / f"{name}.png", labels=ids, highlight={"l1": zeta_blocks(store["l1"], ctx)})
    return ids


def run(prefix: str, invariants: bool) -> None:
    print(f"=== {prefix} (invariants={invariants}) ===")
    ta = pta_p3(invariants)
    ctx = ZoneContext(ta.clocks)

    # 1. Phase 1
    p1 = phase1(ta, ctx, trace=True)
    plot(p1.store, ctx, f"{prefix}_phase1")
    print("after Phase 1:", {l: len(P) for l, P in p1.store.items()})

    # 2. Full pipeline (own context; the zones of the two runs are compared by str only)
    start = time.perf_counter()
    r = build_prestable_graph(ta, trace=True)
    elapsed = time.perf_counter() - start
    g, final = r.graph, r.phase2.store
    fctx = g.nodes[0].zone.ctx
    ids = plot(final, fctx, f"{prefix}_final")
    write_dot(g, ta, OUT / f"{prefix}_graph.dot", delay_edges="covering")
    if shutil.which("dot"):
        render(g, ta, OUT / f"{prefix}_graph.png", delay_edges="covering")
    (OUT / f"{prefix}_summary.txt").write_text(summary(g))

    # 3. Phase 2 splits
    lines = [
        f"{ev.transition.source}  {ev.transition.label}  block: {ev.block}  target: {ev.target}  "
        f"Z-: {ev.z_minus}  pieces: {' | '.join(sorted(map(str, ev.pieces)))}"
        for ev in r.phase2.trace
    ]
    (OUT / f"{prefix}_phase2_splits.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    labels = sorted({e.label for e in g.edges})
    print("after Phase 2:", {l: len(P) for l, P in final.items()}, f"| splits: {r.phase2.splits}")
    print(f"graph: {len(g.nodes)} nodes, {len(g.edges)} edges",
          {lab: sum(e.label == lab for e in g.edges) for lab in labels},
          f"| reachable: {len(reachable(g))} | runtime: {elapsed:.2f}s")

    # 4. ζ check (zeta_blocks raises if a block straddles ζ)
    for name, store, c in (("Phase 1", p1.store, ctx), ("final", final, fctx)):
        zb = zeta_blocks(store["l1"], c)
        print(f"ζ blocks at l1 ({name}): {len(zb)} of {len(store['l1'])}")
    print("ζ blocks at l1 (final store):")
    for z in zeta_blocks(final["l1"], fctx):
        print(f"  {ids['l1'][z]}: {z}")

    # 5. dead blocks: l1 blocks outside [0,2]² (no action enabled)
    dead = [z for z in sorted(final["l1"], key=Zone.key) if disjoint(z, [square(fctx)])]
    print(f"dead blocks at l1 (final store): {len(dead)}")
    for z in dead:
        print(f"  {ids['l1'][z]}: {z}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    run("p3", invariants=False)
    run("p3_inv", invariants=True)


if __name__ == "__main__":
    main()
