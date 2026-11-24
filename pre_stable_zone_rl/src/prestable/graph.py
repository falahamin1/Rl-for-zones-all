"""Phase 3: the pre-stable zone graph, and the end-to-end driver.

    N ← {(l, Z) | Z ∈ P_l};  E′ ← ∅
    for (τ, Z, Z′) ∈ Cand(P):
        if Z ∩ Pre_τ(Z′) ≠ ∅: add edge ((Src τ, Z), Lab τ, (Tgt τ, Z′))

Delay edges are kept literally (self-loops and jumps over several blocks included).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .algorithm import Phase1Result, Phase2Result, _PreCache, _triples, phase1, phase2, transitions
from .model import PEdge, TimedAutomaton
from .zones import Zone, ZoneContext, intersect, pre_edge


@dataclass(frozen=True)
class Node:
    id: int
    location: str
    zone: Zone


@dataclass(frozen=True)
class GraphEdge:
    source: int
    label: str
    target: int
    ta_edge: Optional[PEdge]  # None for delay edges
    cost: Optional[float]  # ta_edge.cost; None for delay edges
    edge: Optional[int] = None  # index of ta_edge in ta.edges; None for delay edges
    branch: Optional[int] = None  # index into ta_edge.branches; None for delay edges


@dataclass
class ZoneGraph:
    nodes: tuple[Node, ...]
    edges: tuple[GraphEdge, ...]
    initial: int
    K: Optional[int] = None  # extrapolation bound used to build the graph, if known
    _ids: dict[tuple[str, Zone], int] = field(init=False, repr=False, compare=False)
    _succ: dict[int, list[GraphEdge]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._ids = {(n.location, n.zone): n.id for n in self.nodes}
        self._succ = {n.id: [] for n in self.nodes}
        for e in self.edges:
            self._succ[e.source].append(e)

    def successors(self, node_id: int) -> list[GraphEdge]:
        return list(self._succ[node_id])

    def action_successors(self, node_id: int, edge_index: int) -> tuple[tuple[int, int], ...]:
        """(branch, target node) of the action edges from `node_id` for TA edge `edge_index`, by branch."""
        return tuple(sorted((e.branch, e.target) for e in self._succ[node_id] if e.edge == edge_index))

    def node_id(self, location: str, zone: Zone) -> int:
        try:
            return self._ids[(location, zone)]
        except KeyError:
            raise KeyError(f"no node for zone {zone} at {location}") from None


def build_graph(
    ta: TimedAutomaton, ctx: ZoneContext, store: dict[str, frozenset[Zone]], *, K: Optional[int] = None
) -> ZoneGraph:
    """Phase 3. Node ids follow breadth-first order from the initial node (id 0), visiting
    each node's outgoing edges in Cand order (transition order, then target key() order);
    unreachable nodes follow in ta.locations order, then key() order. Edges are sorted by
    (source id, transition order, target id)."""
    keys = [(l, Z) for l in ta.locations for Z in sorted(store[l], key=Zone.key)]
    initial = (ta.initial, ctx.zero())
    if initial not in set(keys):
        raise ValueError(f"{{0}} is not a block at the initial location {ta.initial}")

    # Edges in Cand order, on (location, zone) keys; i is the transition index.
    ts = transitions(ta)
    pre = _PreCache(ts)
    raw: list[tuple[int, tuple[str, Zone], tuple[str, Zone]]] = []
    for i, t, Z, Zp in _triples(ts, store):
        if not intersect(Z, pre(i, Zp)).is_empty:
            raw.append((i, (t.source, Z), (t.target, Zp)))
    out: dict[tuple[str, Zone], list[tuple[str, Zone]]] = {k: [] for k in keys}
    for _, src, tgt in raw:
        out[src].append(tgt)

    ids: dict[tuple[str, Zone], int] = {initial: 0}
    queue = deque([initial])
    while queue:
        for tgt in out[queue.popleft()]:
            if tgt not in ids:
                ids[tgt] = len(ids)
                queue.append(tgt)
    for k in keys:
        if k not in ids:
            ids[k] = len(ids)

    nodes = sorted((Node(i, l, Z) for (l, Z), i in ids.items()), key=lambda n: n.id)
    edges = sorted(
        (
            (ids[src], i, ids[tgt],
             GraphEdge(source=ids[src], label=ts[i].label, target=ids[tgt], ta_edge=ts[i].edge,
                       cost=None if ts[i].edge is None else ts[i].edge.cost, edge=ts[i].edge_index,
                       branch=ts[i].branch_index))
            for i, src, tgt in raw
        ),
        key=lambda x: x[:3],
    )
    return ZoneGraph(nodes=tuple(nodes), edges=tuple(e for *_, e in edges), initial=0, K=K)


def check_branches(graph: ZoneGraph, ta: TimedAutomaton) -> None:
    """Raise AssertionError unless every node has, for every TA edge e out of its location and
    every branch b of e, exactly one b-successor if b is enabled on the whole block and none if
    b is disabled on it (a block partly enabled for b contradicts pre-stability).

    Branch b is enabled at v when v ⊨ guard and v[R_b := 0] ⊨ Inv(target_b), i.e. on
    pre_edge(Inv(target_b), e, b). For a well-formed edge with several branches this is
    ⟦guard⟧ on every block (blocks lie in Inv(source)); a deterministic edge is also disabled
    where its target invariant fails.
    """
    if not graph.nodes:
        return
    ctx = graph.nodes[0].zone.ctx
    inv = {l: ctx.from_guard(ta.invariant(l)) for l in ta.locations}
    enabled: dict[tuple[int, int], Zone] = {}
    for n in graph.nodes:
        for k, e in enumerate(ta.edges):
            if e.source != n.location:
                continue
            pairs = graph.action_successors(n.id, k)
            succ = dict(pairs)
            if len(succ) != len(pairs):
                raise AssertionError(f"node {n.id} ({n.location}, {n.zone}), edge {k} ({e.action!r}): "
                                     f"several successors for one branch: {pairs}")
            for j, b in enumerate(e.branches):
                if (k, j) not in enabled:
                    enabled[(k, j)] = pre_edge(inv[b.target], e, j)
                E = enabled[(k, j)]
                if n.zone <= E:
                    if j not in succ:
                        raise AssertionError(f"node {n.id} ({n.location}, {n.zone}), edge {k} ({e.action!r}), "
                                             f"branch {j}: enabled on the whole block but has no successor")
                elif intersect(n.zone, E).is_empty:
                    if j in succ:
                        raise AssertionError(f"node {n.id} ({n.location}, {n.zone}), edge {k} ({e.action!r}), "
                                             f"branch {j}: disabled on the block but has successor {succ[j]}")
                else:
                    raise AssertionError(f"node {n.id} ({n.location}, {n.zone}), edge {k} ({e.action!r}), "
                                         f"branch {j}: enabled on only part of the block, {intersect(n.zone, E)}")
            extra = set(succ) - set(range(len(e.branches)))
            if extra:
                raise AssertionError(f"node {n.id}, edge {k}: successors for unknown branches {sorted(extra)}")


def reachable(graph: ZoneGraph) -> set[int]:
    """Node ids reachable from the initial node."""
    seen = {graph.initial}
    stack = [graph.initial]
    while stack:
        for e in graph.successors(stack.pop()):
            if e.target not in seen:
                seen.add(e.target)
                stack.append(e.target)
    return seen


@dataclass
class PrestableResult:
    graph: ZoneGraph
    phase1: Phase1Result
    phase2: Phase2Result


def build_prestable_graph(
    ta: TimedAutomaton, K: Optional[int] = None, *, trace: bool = False, invariant_atoms: bool = True
) -> PrestableResult:
    """Phases 1-3; `invariant_atoms` selects G_l (see algorithm.py)."""
    ctx = ZoneContext(ta.clocks)
    p1 = phase1(ta, ctx, K, trace=trace, invariant_atoms=invariant_atoms)
    p2 = phase2(ta, ctx, p1.store, trace=trace)
    graph = build_graph(ta, ctx, p2.store, K=p1.K)
    check_branches(graph, ta)
    return PrestableResult(graph=graph, phase1=p1, phase2=p2)
