"""Build the pre-stable zone graph of two small automata and write it out.

Writes into examples/out/: <name>.dot, <name>.json, <name>.txt (summary) and <name>.png
(only if Graphviz `dot` is installed).

    python examples/render_examples.py
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # run without installing the package

from prestable.export import render, summary, to_json, write_dot  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import Edge, TimedAutomaton, ge, le  # noqa: E402

OUT = HERE / "out"


def running_example() -> TimedAutomaton:
    """l0 --(x >= 1, a, reset {x})--> l1 --(x <= 2, b)--> l0"""
    return TimedAutomaton(
        clocks=("x",),
        locations=("l0", "l1"),
        initial="l0",
        edges=(
            Edge("l0", "l1", "a", guard=(ge("x", 1),), resets={"x"}),
            Edge("l1", "l0", "b", guard=(le("x", 2),)),
        ),
    )


def two_clock_example() -> TimedAutomaton:
    """l0 --(x >= 1, a, reset {x})--> l1 --(y >= 2 and x <= 1, b, reset {y})--> l0"""
    return TimedAutomaton(
        clocks=("x", "y"),
        locations=("l0", "l1"),
        initial="l0",
        edges=(
            Edge("l0", "l1", "a", guard=(ge("x", 1),), resets={"x"}),
            Edge("l1", "l0", "b", guard=(ge("y", 2), le("x", 1)), resets={"y"}),
        ),
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    have_dot = shutil.which("dot") is not None
    for name, ta in [("running_example", running_example()), ("two_clock_example", two_clock_example())]:
        graph = build_prestable_graph(ta).graph
        write_dot(graph, ta, OUT / f"{name}.dot")
        (OUT / f"{name}.json").write_text(json.dumps(to_json(graph, ta), indent=2) + "\n")
        (OUT / f"{name}.txt").write_text(summary(graph))
        written = [f"{name}.dot", f"{name}.json", f"{name}.txt"]
        if have_dot:
            render(graph, ta, OUT / f"{name}.png")
            written.append(f"{name}.png")
        print(f"{name}: {len(graph.nodes)} nodes, {len(graph.edges)} edges -> {', '.join(written)}")
    if not have_dot:
        print("Graphviz 'dot' not found: skipped PNG rendering")


if __name__ == "__main__":
    main()
