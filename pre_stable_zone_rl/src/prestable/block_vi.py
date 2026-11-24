"""Discounted optimal cost on the pre-stable zone graph: a finite MDP over block corners.

States are (location, nu, node): an integer valuation nu in the closure of the node's
zone Z, meaning "at some valuation of Z arbitrarily close to nu". From (l, nu, n), for
every delay edge n -> n_a (the self-loop included) and every TA edge e enabled at n_a
(graph.action_successors(n_a, e) nonempty), the waiting times t that put the run into the
closure of Z_a form an interval [lo, hi], and

    variant "boundary":  t ∈ {lo, hi}
    variant "integer":   t ∈ {lo, lo + 1, …, hi}

each giving an action with cost rate(l)·t + cost(e) that moves, for every branch
(p, C, l′) of e with successor node n′, to (l′, (nu + t)[C := 0], n′) with probability p.
Actions with the same (delay, successors) are merged: the cheapest (ties by (edge, via))
represents them, `labels` lists every (via, edge) reaching that cost, `dominated` counts the
costlier ones.

Cost and discounting exactly as in brg.py (per step, no (1-λ) factor); targets are
absorbing with value 0 and not expanded; deadlocks are optional with value +inf. Value
iteration is brg.value_iteration itself.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional

from .algorithm import DELAY
from .brg import _frac, value_iteration
from .graph import ZoneGraph, build_prestable_graph
from .model import ZERO, MultiBranchError, TimedAutomaton
from .zones import Zone

VARIANTS = ("boundary", "integer")
BlockState = tuple[str, tuple[int, ...], int]  # (location, nu, node id)


@dataclass(frozen=True)
class BlockAction:
    edge: int  # index into ta.edges (of the representative)
    delay: int
    via: int  # node n_a: the block the run waits into before taking the edge (representative)
    cost: Fraction
    successors: tuple[tuple[Fraction, int], ...]  # (probability, state index), by state index; sum 1
    labels: tuple[tuple[int, int], ...] = ()  # every (via, edge) giving (delay, successors) at this cost
    dominated: int = 0  # (via, edge) pairs giving (delay, successors) at a higher cost

    @property
    def target(self) -> int:
        """The successor state of a deterministic action."""
        if len(self.successors) != 1:
            raise MultiBranchError(f"action {self.edge}@{self.via} (delay {self.delay}) has "
                                   f"{len(self.successors)} successors, not one")
        return self.successors[0][1]

    @property
    def is_deterministic(self) -> bool:
        return len(self.successors) == 1


@dataclass
class BlockMDP:
    variant: str
    states: list[BlockState]
    actions: list[list[BlockAction]]  # per state; [] for target and deadlock states
    initial: int
    targets: frozenset[int]
    deadlocks: frozenset[int] = frozenset()


def _closed_holds(zone: Zone, nu: tuple[int, ...], clocks: tuple[str, ...], only_diagonal: bool = False) -> bool:
    """Whether nu satisfies the closed (non-strict) form of the zone's constraints."""
    index = {c: i for i, c in enumerate(clocks)}

    def val(name: str) -> int:
        return 0 if name == ZERO else nu[index[name]]

    for g in zone.constraints():
        if only_diagonal and ZERO in (g.left, g.right):
            continue
        if val(g.left) - val(g.right) > g.bound:
            return False
    return True


def in_closure(zone: Zone, nu: tuple[int, ...], clocks: tuple[str, ...]) -> bool:
    """Whether nu lies in the closure of a nonempty zone (closure = closed constraints)."""
    return not zone.is_empty and _closed_holds(zone, nu, clocks)


def wait_interval(zone: Zone, nu: tuple[int, ...], clocks: tuple[str, ...]) -> tuple[int, int]:
    """[lo, hi]: the waiting times t >= 0 with nu + t in the closure of `zone`.

    Diagonal constraints do not change with time and must already hold at nu.
    lo > hi means no such t (callers decide whether that is an error).
    """
    if not _closed_holds(zone, nu, clocks, only_diagonal=True):
        raise AssertionError(f"valuation {nu} violates a diagonal constraint of {zone}")
    index = {c: i for i, c in enumerate(clocks)}
    lo, hi = 0, None
    for g in zone.constraints():
        if g.right == ZERO and g.left != ZERO:  # x <= c
            bound = g.bound - nu[index[g.left]]
            hi = bound if hi is None else min(hi, bound)
        elif g.left == ZERO and g.right != ZERO:  # x >= -c
            lo = max(lo, -g.bound - nu[index[g.right]])
    if hi is None:
        raise ValueError(f"zone {zone} has no upper bound: clocks must be bounded by invariants")
    return lo, hi


def _check_graph_edges(ta: TimedAutomaton, graph: ZoneGraph) -> None:
    """Raise ValueError unless every action edge's index points at an equal edge of `ta`."""
    for ge in graph.edges:
        if ge.label == DELAY:
            continue
        if ge.edge is None or not 0 <= ge.edge < len(ta.edges) or ta.edges[ge.edge] != ge.ta_edge:
            found = ta.edges[ge.edge] if ge.edge is not None and 0 <= ge.edge < len(ta.edges) else None
            raise ValueError(
                f"graph edge {ge.source}->{ge.target} ({ge.label}) has edge index {ge.edge} and TA edge "
                f"{ge.ta_edge}, but the automaton has {found} there: the graph was built from another automaton"
            )


def build_block_mdp(
    ta: TimedAutomaton,
    graph: ZoneGraph,
    targets: Iterable[str],
    *,
    variant: str,
    allow_deadlocks: bool = False,
    max_states: int = 200_000,
) -> BlockMDP:
    """The reachable block-corner MDP, by BFS from (initial, 0, graph.initial)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    targets = frozenset(targets)
    unknown = targets - set(ta.locations)
    if unknown:
        raise ValueError(f"unknown target locations {sorted(unknown)}")
    _check_graph_edges(ta, graph)
    clocks = ta.clocks
    zero = tuple(0 for _ in clocks)
    init: BlockState = (ta.initial, zero, graph.initial)
    states: list[BlockState] = [init]
    index = {init: 0}
    actions: list[list[BlockAction]] = []
    stuck = []
    queue = deque([0])

    def state_index(succ: BlockState) -> int:
        if succ not in index:
            if len(states) >= max_states:
                raise RuntimeError(f"build_block_mdp exceeded max_states={max_states}")
            index[succ] = len(states)
            states.append(succ)
            queue.append(index[succ])
        return index[succ]

    while queue:
        s = queue.popleft()
        loc, nu, n = states[s]
        out: list[BlockAction] = []
        if loc not in targets:
            rate = _frac(ta.rates.get(loc, 0))
            # (delay, successors) -> [(cost, edge, via)], keys in order of first appearance
            found: dict[tuple[int, tuple[tuple[Fraction, int], ...]], list[tuple[Fraction, int, int]]] = {}
            for d in graph.successors(n):
                if d.label != DELAY:
                    continue
                n_a = d.target
                enabled = sorted({e.edge for e in graph.successors(n_a) if e.label != DELAY})
                if not enabled:
                    continue
                Z_a = graph.nodes[n_a].zone
                lo, hi = wait_interval(Z_a, nu, clocks)
                if lo > hi:
                    raise AssertionError(
                        f"empty waiting interval [{lo}, {hi}] from state {(loc, nu, n)} "
                        f"(zone {graph.nodes[n].zone}) via delay edge {n}->{n_a} into {Z_a}; "
                        f"enabled edges {enabled}"
                    )
                delays = (lo, hi) if variant == "boundary" else range(lo, hi + 1)
                for k in enabled:
                    e = ta.edges[k]
                    pairs = graph.action_successors(n_a, k)
                    for t in sorted(set(delays)):
                        succ = _successors(e, pairs, nu, t, clocks, state_index)
                        found.setdefault((t, succ), []).append((rate * t + _frac(e.cost), k, n_a))
            for (t, succ), cands in found.items():
                cost, k, n_a = min(cands)
                labels = tuple(sorted((v, j) for c, j, v in cands if c == cost))
                out.append(BlockAction(k, t, n_a, cost, succ, labels, sum(c > cost for c, _, _ in cands)))
            if not out:
                stuck.append(s)
        while len(actions) <= s:
            actions.append([])
        actions[s] = out
    if stuck and not allow_deadlocks:
        raise ValueError(f"reachable non-target states without actions: {[states[s] for s in stuck]}")
    return BlockMDP(
        variant=variant,
        states=states,
        actions=actions,
        initial=0,
        targets=frozenset(i for i, (l, _, _) in enumerate(states) if l in targets),
        deadlocks=frozenset(stuck),
    )


def _successors(e, pairs, nu, t, clocks, state_index) -> tuple[tuple[Fraction, int], ...]:
    """((p, state index), ...) of edge e taken after delay t from nu, one entry per successor
    state (probabilities of equal states summed), sorted by state index."""
    probs: dict[int, Fraction] = {}
    for j, n2 in pairs:
        b = e.branches[j]
        nu2 = tuple(0 if x in b.resets else v + t for x, v in zip(clocks, nu))
        i = state_index((b.target, nu2, n2))
        probs[i] = probs.get(i, Fraction(0)) + b.prob
    return tuple((p, i) for i, p in sorted(probs.items()))


def check_block_mdp(ta: TimedAutomaton, graph: ZoneGraph, mdp: BlockMDP) -> None:
    """Raise ValueError unless every nu is integral and in the closure of its node's zone, and
    every action is consistent: probabilities > 0 summing to exactly 1, nu + delay in the
    closure of Z_via, every successor valuation in the closure of its node's zone, and every
    labelled (via, edge) enabled at via and reproducing the same successors and cost."""
    index = {st: i for i, st in enumerate(mdp.states)}

    def lookup(st: BlockState) -> int:
        if st not in index:
            raise ValueError(f"successor {st} is not a state of the MDP")
        return index[st]

    for i, (loc, nu, n) in enumerate(mdp.states):
        node = graph.nodes[n]
        if node.location != loc:
            raise ValueError(f"state {i}: node {n} is at {node.location}, not {loc}")
        if not all(isinstance(v, int) for v in nu):
            raise ValueError(f"state {i}: valuation {nu} is not integral")
        if not in_closure(node.zone, nu, ta.clocks):
            raise ValueError(f"state {i}: valuation {nu} is not in the closure of {node.zone} (node {n})")
        if i in mdp.targets and mdp.actions[i]:
            raise ValueError(f"target state {i} has actions")
        rate = _frac(ta.rates.get(loc, 0))
        for a in mdp.actions[i]:
            if a.delay < 0:
                raise ValueError(f"state {i}: action {a} has negative delay")
            if any(p <= 0 for p, _ in a.successors) or sum(p for p, _ in a.successors) != 1:
                raise ValueError(f"state {i}: action {a} has probabilities {[p for p, _ in a.successors]}")
            if [j for _, j in a.successors] != sorted({j for _, j in a.successors}):
                raise ValueError(f"state {i}: successors of action {a} are not sorted by distinct state index")
            for _, j in a.successors:
                loc2, nu2, n2 = mdp.states[j]
                if not in_closure(graph.nodes[n2].zone, nu2, ta.clocks):
                    raise ValueError(
                        f"state {i} {(loc, nu, n)}: successor valuation {nu2} of action {a} is not in the "
                        f"closure of {graph.nodes[n2].zone} (node {n2})"
                    )
            if (a.via, a.edge) not in a.labels:
                raise ValueError(f"state {i}: action {a} does not list its representative in its labels")
            lifted = tuple(v + a.delay for v in nu)
            for via, k in a.labels:
                if not in_closure(graph.nodes[via].zone, lifted, ta.clocks):
                    raise ValueError(f"state {i}: {lifted} = nu + delay of action {a} is not in the closure "
                                     f"of {graph.nodes[via].zone} (node {via})")
                pairs = graph.action_successors(via, k)
                if not pairs:
                    raise ValueError(f"state {i}: edge {k} is not enabled at node {via} (action {a})")
                if _successors(ta.edges[k], pairs, nu, a.delay, ta.clocks, lookup) != a.successors:
                    raise ValueError(f"state {i}: label {(via, k)} of action {a} gives other successors")
                if rate * a.delay + _frac(ta.edges[k].cost) != a.cost:
                    raise ValueError(f"state {i}: label {(via, k)} of action {a} has another cost")


def _tie_key(a: BlockAction):
    return (a.delay, a.edge, a.via, a.successors)


def optimal_run(ta: TimedAutomaton, graph: ZoneGraph, mdp: BlockMDP, policy):
    """Steps (location, nu, zone of n_a, edge action, delay) of the greedy run until a target
    or deadlock; ends with ("cycle", k) if the run repeats from step k (see brg.optimal_run).
    Only for runs whose actions are all deterministic; otherwise use sample_run."""
    steps: list[tuple] = []
    seen: dict[int, int] = {}
    s = mdp.initial
    while policy[s] is not None:
        if s in seen:
            steps.append(("cycle", seen[s]))
            return steps
        seen[s] = len(steps)
        a = policy[s]
        if not a.is_deterministic:
            raise ValueError(f"the optimal run reaches a probabilistic action at state {mdp.states[s]}: "
                             f"use sample_run")
        loc, nu, _ = mdp.states[s]
        steps.append((loc, nu, str(graph.nodes[a.via].zone), ta.edges[a.edge].action, a.delay))
        s = a.target
    return steps


def sample_run(ta: TimedAutomaton, graph: ZoneGraph, mdp: BlockMDP, policy, rng, max_steps: int = 1000):
    """Follow the policy from the initial state, sampling successors with rng.random().
    Returns (steps, end): steps are (state, action, successor) with state indices, end is
    "target", "deadlock" (a non-target state without actions) or "max_steps"."""
    steps: list[tuple[int, BlockAction, int]] = []
    s = mdp.initial
    while len(steps) < max_steps:
        if s in mdp.targets:
            return steps, "target"
        a = policy[s]
        if a is None:
            return steps, "deadlock"
        r, acc, nxt = rng.random(), Fraction(0), a.successors[-1][1]
        for p, j in a.successors:
            acc += p
            if r < acc:
                nxt = j
                break
        steps.append((s, a, nxt))
        s = nxt
    return steps, "target" if s in mdp.targets else "max_steps"


def discounted_cost(steps, discount: float) -> float:
    """Σ_i λ^i · cost of the i-th step of a sample_run."""
    return sum(float(a.cost) * discount ** i for i, (_, a, _) in enumerate(steps))


@dataclass
class BlockResult:
    value: float  # V(initial)
    values: list[float]
    policy: list[Optional[BlockAction]]
    run: Optional[list[tuple]]  # optimal_run for deterministic automata, else None (see sample_run)
    states: int
    actions: int
    iterations: int
    mdp: BlockMDP
    graph: ZoneGraph


def solve(
    ta: TimedAutomaton,
    targets: Iterable[str],
    discount: float,
    *,
    variant: str,
    K: Optional[int] = None,
    allow_deadlocks: bool = False,
    max_states: int = 200_000,
    graph: Optional[ZoneGraph] = None,
    mdp: Optional[BlockMDP] = None,
) -> BlockResult:
    """Build the pre-stable graph (unless given), the block MDP (unless given), and run VI."""
    graph = graph or build_prestable_graph(ta, K).graph
    mdp = mdp or build_block_mdp(ta, graph, targets, variant=variant, allow_deadlocks=allow_deadlocks,
                                 max_states=max_states)
    values, policy, iterations = value_iteration(mdp, discount, tie_key=_tie_key)
    return BlockResult(
        value=values[mdp.initial],
        values=values,
        policy=policy,
        run=optimal_run(ta, graph, mdp, policy) if all(e.is_deterministic for e in ta.edges) else None,
        states=len(mdp.states),
        actions=sum(len(a) for a in mdp.actions),
        iterations=iterations,
        mdp=mdp,
        graph=graph,
    )
