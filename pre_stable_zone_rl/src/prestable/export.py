"""Export and inspection of pre-stable zone graphs: DOT, Graphviz rendering, JSON, text summary.

`delay_edges` in `to_dot` only changes what is drawn; the graph itself is never pruned.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

from .algorithm import DELAY
from .graph import GraphEdge, Node, ZoneGraph, reachable
from .model import Constraint, TimedAutomaton
from .zones import ZoneContext

DELAY_MODES = ("all", "no_self_loops", "covering")


# --- DOT --------------------------------------------------------------------

def _q(text: str) -> str:
    """A DOT double-quoted string."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _drawn_delay_edges(graph: ZoneGraph, mode: str) -> list[GraphEdge]:
    delay = [e for e in graph.edges if e.label == DELAY]
    if mode == "all":
        return delay
    proper = [e for e in delay if e.source != e.target]
    if mode == "no_self_loops":
        return proper
    # "covering": drop u→v when some third node w has delay edges u→w and w→v.
    # Delay edges stay within one location, so w is automatically at the same location.
    succ: dict[int, set[int]] = {}
    for e in proper:
        succ.setdefault(e.source, set()).add(e.target)
    return [
        e for e in proper
        if not any(e.target in succ.get(w, ()) for w in succ[e.source] if w not in (e.source, e.target))
    ]


def _action_label(e: GraphEdge) -> str:
    label = e.label if not e.cost else f"{e.label} / {e.cost:g}"
    if e.ta_edge is not None and not e.ta_edge.is_deterministic:
        label += f" p={e.ta_edge.branches[e.branch].prob}"
    return label


def to_dot(
    graph: ZoneGraph,
    ta: TimedAutomaton,
    *,
    delay_edges: str = "covering",
    include_unreachable: bool = True,
) -> str:
    if delay_edges not in DELAY_MODES:
        raise ValueError(f"delay_edges must be one of {DELAY_MODES}, got {delay_edges!r}")
    reach = reachable(graph)
    shown = {n.id for n in graph.nodes if include_unreachable or n.id in reach}
    lines = ["digraph prestable {", "  rankdir=LR;", "  node [shape=box, fontname=Helvetica];",
             "  edge [fontname=Helvetica];"]
    by_location: dict[str, list[Node]] = {l: [] for l in ta.locations}
    for n in graph.nodes:
        by_location[n.location].append(n)
    for l in ta.locations:
        nodes = [n for n in by_location[l] if n.id in shown]
        lines.append(f"  subgraph {_q('cluster_' + l)} {{")
        lines.append(f"    label={_q(l)};")
        for n in nodes:
            attrs = [f"label={_q(f'{n.id}: {n.zone}')}"]
            if n.id == graph.initial:
                attrs.append("peripheries=2")
            if n.id not in reach:
                attrs.append('style=dashed, color=grey, fontcolor=grey')
            lines.append(f"    n{n.id} [{', '.join(attrs)}];")
        lines.append("  }")
    for e in graph.edges:
        if e.label != DELAY and e.source in shown and e.target in shown:
            lines.append(f"  n{e.source} -> n{e.target} [label={_q(_action_label(e))}];")
    for e in _drawn_delay_edges(graph, delay_edges):
        if e.source in shown and e.target in shown:
            lines.append(f"  n{e.source} -> n{e.target} [style=dashed];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_dot(graph: ZoneGraph, ta: TimedAutomaton, path: Union[str, Path], **kwargs: Any) -> None:
    Path(path).write_text(to_dot(graph, ta, **kwargs))


def render(graph: ZoneGraph, ta: TimedAutomaton, path: Union[str, Path], fmt: str = "png", **kwargs: Any) -> None:
    """Render with the Graphviz `dot` binary (must be on PATH)."""
    dot = shutil.which("dot")
    if dot is None:
        raise RuntimeError("Graphviz 'dot' is not installed or not on PATH; install Graphviz to render graphs")
    subprocess.run([dot, f"-T{fmt}", "-o", str(path)], input=to_dot(graph, ta, **kwargs),
                   text=True, check=True, capture_output=True)


# --- JSON -------------------------------------------------------------------

def to_json(graph: ZoneGraph, ta: TimedAutomaton) -> dict:
    return {
        "clocks": list(ta.clocks),
        "K": graph.K,
        "initial": graph.initial,
        "nodes": [
            {
                "id": n.id,
                "location": n.location,
                "zone": str(n.zone),
                "constraints": [[c.left, c.right, c.bound, c.strict] for c in n.zone.constraints()],
            }
            for n in graph.nodes
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "label": e.label,
                "ta_edge": e.edge,  # index into ta.edges; None for delay edges
                "cost": e.cost,
                # Written only for edges with several branches (absent means branch 0), so the
                # JSON of deterministic models is unchanged.
                **({"branch": e.branch} if e.ta_edge is not None and not e.ta_edge.is_deterministic else {}),
            }
            for e in graph.edges
        ],
    }


def from_json(data: dict, ta: TimedAutomaton, ctx: Optional[ZoneContext] = None) -> ZoneGraph:
    """Rebuild a graph from `to_json` output. Pass `ctx` to get zones comparable with an
    existing graph's zones (zones from different contexts never compare equal)."""
    if list(data["clocks"]) != list(ta.clocks):
        raise ValueError(f"JSON clocks {data['clocks']} differ from automaton clocks {list(ta.clocks)}")
    ctx = ctx or ZoneContext(ta.clocks)
    nodes = tuple(
        Node(n["id"], n["location"], ctx.from_guard(Constraint(l, r, b, s) for l, r, b, s in n["constraints"]))
        for n in data["nodes"]
    )
    edges = tuple(
        GraphEdge(
            source=e["source"],
            label=e["label"],
            target=e["target"],
            ta_edge=None if e["ta_edge"] is None else ta.edges[e["ta_edge"]],
            cost=e["cost"],
            edge=e["ta_edge"],
            branch=None if e["ta_edge"] is None else e.get("branch", 0),
        )
        for e in data["edges"]
    )
    return ZoneGraph(nodes=nodes, edges=edges, initial=data["initial"], K=data.get("K"))


# --- text -------------------------------------------------------------------

def summary(graph: ZoneGraph) -> str:
    lines = []
    locations: dict[str, list[Node]] = {}
    for n in graph.nodes:  # locations in order of their first node
        locations.setdefault(n.location, []).append(n)
    reach = reachable(graph)
    for l, nodes in locations.items():
        lines.append(f"location {l} ({len(nodes)} nodes):")
        for n in nodes:
            mark = "" if n.id in reach else "  [unreachable]"
            initial = "  [initial]" if n.id == graph.initial else ""
            lines.append(f"  {n.id}: {n.zone}{initial}{mark}")
    counts = Counter(e.label for e in graph.edges)
    lines.append(f"edges: {len(graph.edges)} ({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))})")
    lines.append(f"reachable: {len(reach)} of {len(graph.nodes)} nodes")
    return "\n".join(lines) + "\n"
