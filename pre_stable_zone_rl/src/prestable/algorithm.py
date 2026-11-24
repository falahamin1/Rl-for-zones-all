"""The pre-stable zone graph algorithm: Phase 1 (forward closure) and Phase 2 (pre-stability).

Phase 3 (building the graph) lives in graph.py.

Phase 1 (pseudocode lines 1-14, Insert = lines 27-37):

    P_l ← ∅ for every location l;  Q ← ∅
    Insert(l0, {0})
    while Q ≠ ∅:
        choose l ∈ Q; Q ← Q \\ {l}
        B ← P_l                              # snapshot
        for Z in B:
            Zt ← Z↑
            Insert(l, Zt)
            for e = (l, g, a, branches) in outgoing(l):
                for (p, R, l') in branches:          # a branch is a transition
                    Insert(l', reset(Zt ∩ ⟦g⟧, R))

    Insert(l, Z):
        X ← extrapolate(Z, K)
        if X empty: return
        P_new ← partition(P_l ∪ {X}, G_l)    # G_l = ta.partition_atoms(l), or ta.atomic_guards(l)
        if P_new ≠ P_l: P_l ← P_new; Q ← Q ∪ {l}

Phase 2:

    while ∃(τ, Z, Z′) ∈ Cand(P) with Z ∩ Pre_τ(Z′) ≠ ∅ and ≠ Z:
        choose such a triple; l ← Src(τ)
        Z⁻ ← Z ∩ Pre_τ(Z′)
        P_l ← (P_l \\ {Z}) ∪ partition({Z, Z⁻}, ∅)

Transitions T = every (TA edge, branch) pair plus one delay transition δ_l per location;
Pre_(e,b) is `pre_edge(·, e, b)`, Pre_δ is `down`. Cand(P) = all (τ, Z, Z′) with Z ∈ P_Src(τ), Z′ ∈ P_Tgt(τ).

G_l is ta.partition_atoms(l) (guard atoms plus invariant faces) with invariant_atoms=True, the
default, and ta.atomic_guards(l) with invariant_atoms=False. Phase 2 splits with no atoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model import DELAY, Constraint, PEdge, TimedAutomaton  # DELAY: label of delay transitions/edges
from .partition import check_partition, partition
from .zones import (Zone, ZoneContext, down, extrapolate, intersect, is_k_bounded, pre_edge, reset, subtract,
                    subtract_all, up)


@dataclass(frozen=True)
class InsertEvent:
    location: str
    zone: Zone  # the zone passed to Insert
    extrapolated: Zone  # X after extrapolation
    changed: bool
    blocks_after: int  # |P_l| after the call


@dataclass
class Phase1Result:
    store: dict[str, frozenset[Zone]]  # every location present; unreached → empty
    K: int
    locations_processed: int  # iterations of the while loop
    inserts: int  # Insert calls, including the initial one
    changed_inserts: int  # Insert calls that changed a partition
    trace: Optional[list[InsertEvent]]  # None unless trace=True


class _Phase1:
    """Mutable state of one Phase 1 run."""

    def __init__(self, ta: TimedAutomaton, ctx: ZoneContext, K: int, trace: bool, invariant_atoms: bool):
        self.ta, self.ctx, self.K = ta, ctx, K
        self.store: dict[str, frozenset[Zone]] = {l: frozenset() for l in ta.locations}
        self.guards = {l: _atoms(ta, l, invariant_atoms) for l in ta.locations}
        self.inv = {l: ctx.from_guard(ta.invariant(l)) for l in ta.locations}
        self.queue: set[str] = set()
        self.inserts = self.changed_inserts = 0
        self.trace: Optional[list[InsertEvent]] = [] if trace else None

    def insert(self, l: str, Z: Zone) -> None:
        self.inserts += 1
        # The intersection keeps every block inside Inv(l) even if extrapolation relaxes a
        # bound; it stays K-bounded because K includes the invariant constants.
        X = intersect(extrapolate(Z, self.K), self.inv[l])
        changed = False
        if not X.is_empty:
            old = self.store[l]
            new = partition(old | {X}, self.guards[l], self.ctx)
            if new != old:
                self.store[l] = new
                self.queue.add(l)
                self.changed_inserts += 1
                changed = True
        if self.trace is not None:
            self.trace.append(InsertEvent(l, Z, X, changed, len(self.store[l])))

    def choose(self) -> str:
        l = next(l for l in self.ta.locations if l in self.queue)
        self.queue.remove(l)
        return l


def _atoms(ta: TimedAutomaton, l: str, invariant_atoms: bool) -> frozenset[Constraint]:
    """G_l: the atoms the partition of l is made uniform with."""
    return ta.partition_atoms(l) if invariant_atoms else ta.atomic_guards(l)


def check_well_formed(ta: TimedAutomaton, ctx: ZoneContext) -> None:
    """Raise ValueError unless every branch of every multi-branch edge lands inside its
    target's invariant: reset(⟦guard⟧ ∩ Inv(source), b.resets) ⊆ ⟦Inv(b.target)⟧.

    Deterministic edges are not checked: where the target invariant fails they are simply
    disabled (Phase 1 intersects their successors with it).
    """
    for k, e in enumerate(ta.edges):
        if e.is_deterministic:
            continue
        Z = intersect(ctx.from_guard(e.guard), ctx.from_guard(ta.invariant(e.source)))
        for i, b in enumerate(e.branches):
            succ, inv = reset(Z, b.resets), ctx.from_guard(ta.invariant(b.target))
            if not succ <= inv:
                raise ValueError(
                    f"edge {k} ({e.action!r} from {e.source}), branch {i} (to {b.target}) is not well-formed: "
                    f"its successor zone has a part outside the invariant of {b.target}: {subtract(succ, inv)[0]}"
                )


def phase1(
    ta: TimedAutomaton,
    ctx: ZoneContext,
    K: Optional[int] = None,
    *,
    trace: bool = False,
    max_iterations: int = 10_000,
    invariant_atoms: bool = True,
) -> Phase1Result:
    """Forward closure: a K-bounded, guard-uniform partition per location, inside the
    location's invariant, closed under delay and edges, containing {0} at the initial location."""
    if set(ctx.clocks) != set(ta.clocks):
        raise ValueError(f"context clocks {ctx.clocks} differ from automaton clocks {ta.clocks}")
    check_well_formed(ta, ctx)
    if not ctx.zero() <= ctx.from_guard(ta.invariant(ta.initial)):
        raise ValueError(f"the zero valuation violates the invariant of the initial location {ta.initial}")
    K = ta.max_constant() if K is None else K
    s = _Phase1(ta, ctx, K, trace, invariant_atoms)
    s.insert(ta.initial, ctx.zero())
    iterations = 0
    while s.queue:
        if iterations >= max_iterations:
            sizes = {l: len(P) for l, P in s.store.items()}
            raise RuntimeError(
                f"phase1 exceeded max_iterations={max_iterations}; "
                f"queue={sorted(s.queue)}, partition sizes={sizes}"
            )
        iterations += 1
        l = s.choose()
        for Z in sorted(s.store[l], key=Zone.key):  # snapshot B
            Zt = intersect(up(Z), s.inv[l])
            s.insert(l, Zt)
            for e in ta.outgoing(l):
                enabled = intersect(Zt, ctx.from_guard(e.guard))
                for b in e.branches:
                    s.insert(b.target, intersect(reset(enabled, b.resets), s.inv[b.target]))
    return Phase1Result(
        store=dict(s.store),
        K=K,
        locations_processed=iterations,
        inserts=s.inserts,
        changed_inserts=s.changed_inserts,
        trace=s.trace,
    )


def check_phase1(ta: TimedAutomaton, ctx: ZoneContext, store: dict[str, frozenset[Zone]], K: int, *,
                 invariant_atoms: bool = True) -> None:
    """Raise ValueError unless `store` has the properties Phase 1 guarantees (exact checks)."""
    missing = set(ta.locations) - set(store)
    if missing:
        raise ValueError(f"store lacks locations {sorted(missing)}")
    inv = {l: ctx.from_guard(ta.invariant(l)) for l in ta.locations}
    for l in ta.locations:
        P = store[l]
        try:
            check_partition(P, P, _atoms(ta, l, invariant_atoms), ctx)
        except ValueError as exc:
            raise ValueError(f"location {l}: {exc}") from exc
        for Z in P:
            if not is_k_bounded(Z, K):
                raise ValueError(f"location {l}: block {Z} is not {K}-bounded")
            if not Z <= inv[l]:
                raise ValueError(f"location {l}: block {Z} is not inside the invariant {inv[l]}")
    if ctx.zero() not in store[ta.initial]:
        raise ValueError(f"{{0}} is not a block at the initial location {ta.initial}")
    for l in ta.locations:
        for Z in sorted(store[l], key=Zone.key):
            Zt = intersect(up(Z), inv[l])
            rest = subtract_all(Zt, store[l])
            if rest:
                raise ValueError(f"location {l}: delay of block {Z} leaves the partition: {[str(r) for r in rest]}")
            for e in ta.outgoing(l):
                enabled = intersect(Zt, ctx.from_guard(e.guard))
                for j, b in enumerate(e.branches):
                    succ = intersect(reset(enabled, b.resets), inv[b.target])
                    rest = subtract_all(succ, store[b.target])
                    if rest:
                        raise ValueError(
                            f"edge {e.action} {l}->{b.target} (branch {j}): successor of block {Z} leaves "
                            f"the partition of {b.target}: {[str(r) for r in rest]}"
                        )


# --- Phase 2 ----------------------------------------------------------------

@dataclass(frozen=True)
class Transition:
    source: str
    target: str
    label: str  # action, or DELAY
    edge: Optional[PEdge]  # None for a delay transition
    invariant: tuple[Constraint, ...] = ()  # Inv(source), used by delay transitions
    edge_index: Optional[int] = None  # index of `edge` in ta.edges; None for a delay transition
    branch_index: Optional[int] = None  # index into edge.branches; None for a delay transition

    def pre(self, z: Zone) -> Zone:
        """Pre_δ(Z′) = down(Z′) ∩ Inv(l); Pre_e = pre_edge.

        For upper-bound invariants down(Z′) ∩ Inv(l) equals down(Z′) whenever Z′ ⊆ Inv(l)
        (which holds for all blocks), but the definition is stated explicitly. Edge
        transitions need no invariant: source blocks lie in Inv(l), target blocks in Inv(l′).
        """
        if self.edge is not None:
            return pre_edge(z, self.edge, self.branch_index)
        return intersect(down(z), z.ctx.from_guard(self.invariant))


def transitions(ta: TimedAutomaton) -> list[Transition]:
    """Per location in ta.locations order: the delay transition, then ta.outgoing(l) in order,
    one transition per branch (in branch order), going to the branch's target."""
    out = []
    for l in ta.locations:
        out.append(Transition(l, l, DELAY, None, ta.invariant(l)))
        # Same edges and order as ta.outgoing(l), with their indices in ta.edges.
        out.extend(Transition(e.source, b.target, e.action, e, edge_index=k, branch_index=j)
                   for k, e in enumerate(ta.edges) if e.source == l for j, b in enumerate(e.branches))
    return out


@dataclass(frozen=True)
class SplitEvent:
    transition: Transition
    block: Zone  # Z
    target: Zone  # Z′
    z_minus: Zone
    pieces: frozenset[Zone]


@dataclass
class Phase2Result:
    store: dict[str, frozenset[Zone]]
    splits: int
    trace: Optional[list[SplitEvent]]  # None unless trace=True


class _PreCache:
    """Pre_τ(Z′) keyed by (transition index, Z′); blocks are immutable, so never stale."""

    def __init__(self, ts: list[Transition]):
        self.ts = ts
        self.cache: dict[tuple[int, Zone], Zone] = {}

    def __call__(self, i: int, target: Zone) -> Zone:
        key = (i, target)
        if key not in self.cache:
            self.cache[key] = self.ts[i].pre(target)
        return self.cache[key]


def _triples(ts: list[Transition], store: dict[str, frozenset[Zone]]):
    """Cand(P) in the deterministic order: transitions, then Z, then Z′ in key() order."""
    for i, t in enumerate(ts):
        targets = sorted(store[t.target], key=Zone.key)
        for Z in sorted(store[t.source], key=Zone.key):
            for Zp in targets:
                yield i, t, Z, Zp


def _find_bad(ts, store, pre) -> Optional[tuple[int, Transition, Zone, Zone, Zone]]:
    for i, t, Z, Zp in _triples(ts, store):
        z_minus = intersect(Z, pre(i, Zp))
        if not z_minus.is_empty and z_minus != Z:
            return i, t, Z, Zp, z_minus
    return None


def phase2(
    ta: TimedAutomaton,
    ctx: ZoneContext,
    store: dict[str, frozenset[Zone]],
    *,
    trace: bool = False,
    max_splits: int = 100_000,
) -> Phase2Result:
    """Refine `store` (not mutated) until every (τ, Z, Z′) is all-or-nothing."""
    store = dict(store)
    ts = transitions(ta)
    pre = _PreCache(ts)
    events: Optional[list[SplitEvent]] = [] if trace else None
    splits = 0
    while True:
        bad = _find_bad(ts, store, pre)
        if bad is None:
            break
        if splits >= max_splits:
            sizes = {l: len(P) for l, P in store.items()}
            raise RuntimeError(f"phase2 exceeded max_splits={max_splits}; partition sizes={sizes}")
        _, t, Z, Zp, z_minus = bad
        pieces = partition({Z, z_minus}, (), ctx)
        store[t.source] = (store[t.source] - {Z}) | pieces
        splits += 1
        if events is not None:
            events.append(SplitEvent(t, Z, Zp, z_minus, pieces))
    return Phase2Result(store=store, splits=splits, trace=events)


def check_phase2(ta: TimedAutomaton, ctx: ZoneContext, store: dict[str, frozenset[Zone]], K: int, *,
                 invariant_atoms: bool = True) -> None:
    """Raise ValueError unless `store` is pre-stable and still passes check_phase1."""
    ts = transitions(ta)
    pre = _PreCache(ts)
    for i, t, Z, Zp in _triples(ts, store):
        z_minus = intersect(Z, pre(i, Zp))
        if not z_minus.is_empty and z_minus != Z:
            raise ValueError(
                f"not pre-stable: transition {t.label} {t.source}->{t.target}, block {Z} at {t.source} "
                f"meets Pre of {Zp} at {t.target} only in {z_minus}"
            )
    check_phase1(ta, ctx, store, K, invariant_atoms=invariant_atoms)
