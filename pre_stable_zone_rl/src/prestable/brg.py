"""Boundary region graph (BRG) and discounted value iteration: a reference solver.

Follows the boundary region graph of Jurdziński, Kwiatkowska, Norman, Trivedi
(CONCUR 2009), restricted to deterministic edges and linear prices. Independent of the
zone algorithm: this module imports only `model`, and all clock values are exact
`Fraction`s.

Cost of a run (t1,a1),(t2,a2),... (wait ti, then take edge ai):
    sum_i λ^(i-1) · (rate(l_i)·t_i + cost(a_i)),   λ ∈ (0,1).

Regions. Clocks are bounded by K = ta.max_constant() (every reachable non-target
location bounds every clock by its invariant), so no "> K" class is needed. A region
signature is a hashable triple over clock indices (clock order of the automaton):

    (ints, zero, groups)
    ints:   integer part of each clock
    zero:   sorted indices of the clocks with fractional part 0
    groups: the clocks with nonzero fractional part, grouped by equal fraction, in
            increasing order of the fraction; each group a sorted tuple of indices
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Iterable, Optional, Sequence

from .model import ZERO, Constraint, MultiBranchError, TimedAutomaton

Sig = tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]
State = tuple[str, tuple[int, ...], Sig]

KINDS = ("thin", "after", "before")  # tie-break order for the greedy policy


# --- regions ----------------------------------------------------------------

def region_of(valuation: Sequence) -> Sig:
    """Region signature of a valuation (one nonnegative rational per clock)."""
    vals = [Fraction(v) for v in valuation]
    if any(v < 0 for v in vals):
        raise ValueError(f"negative clock value in {valuation}")
    ints = tuple(math.floor(v) for v in vals)
    fracs = [v - i for v, i in zip(vals, ints)]
    zero = tuple(i for i, f in enumerate(fracs) if f == 0)
    by_frac: dict[Fraction, list[int]] = {}
    for i, f in enumerate(fracs):
        if f != 0:
            by_frac.setdefault(f, []).append(i)
    groups = tuple(tuple(by_frac[f]) for f in sorted(by_frac))
    return ints, zero, groups


def representative(sig: Sig) -> tuple[Fraction, ...]:
    """A canonical point of the region: fraction (i+1)/(g+1) for the i-th of g nonzero groups."""
    ints, _, groups = sig
    vals = [Fraction(n) for n in ints]
    for i, group in enumerate(groups):
        for c in group:
            vals[c] += Fraction(i + 1, len(groups) + 1)
    return tuple(vals)


def is_thin(sig: Sig) -> bool:
    """Some clock has fractional part 0."""
    return bool(sig[1])


def _value(point: Sequence[Fraction], name: str, index: dict[str, int]) -> Fraction:
    return Fraction(0) if name == ZERO else point[index[name]]


def satisfies(sig: Sig, constraints: Iterable[Constraint], clocks: Sequence[str]) -> bool:
    """Whether the region satisfies a conjunction of integer-constant constraints.

    Evaluated on the representative; exact because regions are uniform for such constraints.
    """
    point, index = representative(sig), {c: i for i, c in enumerate(clocks)}
    for g in constraints:
        d = _value(point, g.left, index) - _value(point, g.right, index)
        if not (d < g.bound if g.strict else d <= g.bound):
            return False
    return True


def reset(sig: Sig, R: Iterable[str], clocks: Sequence[str]) -> Sig:
    """Region of the representative with the clocks in R set to 0."""
    point = list(representative(sig))
    for x in R:
        point[clocks.index(x)] = Fraction(0)
    return region_of(point)


def _inside(sig: Sig, K: int) -> bool:
    """All clocks <= K."""
    ints, zero, _ = sig
    return all(n < K or (n == K and i in zero) for i, n in enumerate(ints))


def time_successors(sig: Sig, K: int) -> list[Sig]:
    """Regions visited by representative + t, t >= 0: sig itself first, alternating thin and
    thick, up to and including the last region with all clocks <= K."""
    chain = []
    cur = sig
    while _inside(cur, K):
        chain.append(cur)
        ints, zero, groups = cur
        if zero:  # thin: the integral clocks start moving, with the smallest fraction
            cur = (ints, (), (zero,) + groups)
        else:  # thick: the clocks with the largest fraction reach the next integer
            top = groups[-1]
            cur = (tuple(n + 1 if i in top else n for i, n in enumerate(ints)), top, groups[:-1])
    return chain


def in_closure(nu: Sequence[int], sig: Sig) -> bool:
    """Whether the integer valuation nu lies in the closure of the region.

    With f_c = nu_c - int_c: clocks with fraction 0 need f = 0, the others f ∈ {0, 1},
    equal within a group and nondecreasing along the group order.
    """
    ints, zero, groups = sig
    f = [n - i for n, i in zip(nu, ints)]
    if any(f[c] != 0 for c in zero):
        return False
    prev = 0
    for group in groups:
        values = {f[c] for c in group}
        if len(values) != 1:
            return False
        (v,) = values
        if v not in (0, 1) or v < prev:
            return False
        prev = v
    return True


# --- the boundary region graph ----------------------------------------------

@dataclass(frozen=True)
class BRGAction:
    edge: int  # index into ta.edges
    delay: int
    kind: str  # "thin" | "after" | "before"
    cost: Fraction
    successors: tuple[tuple[Fraction, int], ...]  # (probability, state index), by state index; sum 1

    @property
    def target(self) -> int:
        """The successor state of a deterministic action."""
        if len(self.successors) != 1:
            raise MultiBranchError(f"BRG action (edge {self.edge}, delay {self.delay}, {self.kind}) has "
                                   f"{len(self.successors)} successors, not one")
        return self.successors[0][1]

    @property
    def is_deterministic(self) -> bool:
        return len(self.successors) == 1


@dataclass
class BRG:
    states: list[State]  # (location, nu, sig)
    actions: list[list[BRGAction]]  # per state; [] for target states
    initial: int
    targets: frozenset[int]
    # Non-target states without actions; only nonempty if build_brg(allow_deadlocks=True).
    # Their value is +inf (the target cannot be reached from them).
    deadlocks: frozenset[int] = frozenset()


def _frac(x) -> Fraction:
    """Exact cost/rate: floats are read by their decimal repr (0.1 -> 1/10)."""
    return Fraction(repr(x)) if isinstance(x, float) else Fraction(x)


def _check_bounded(ta: TimedAutomaton, loc: str) -> None:
    bounded = {c.left for c in ta.invariant(loc)}
    missing = [x for x in ta.clocks if x not in bounded]
    if missing:
        raise ValueError(
            f"location {loc!r} is reachable and not a target, but its invariant does not bound "
            f"clocks {missing} (bounded clocks are required)"
        )


def _branch_successors(ta: TimedAutomaton, k: int, lifted: tuple[int, ...], zeta: Sig, *, check: bool = True):
    """[(p, (l′, lifted[C := 0], reset(zeta, C)))] for each branch (p, C, l′) of edge k, taken in
    region zeta at the integer valuation `lifted`. A multi-branch edge must be well-formed:
    with `check`, a branch whose successor region violates its target invariant raises
    ValueError. (A deterministic edge is not checked: its successor then has no moves.)"""
    clocks, e = ta.clocks, ta.edges[k]
    out = []
    for j, b in enumerate(e.branches):
        sig2 = reset(zeta, b.resets, clocks)
        if check and not e.is_deterministic and not satisfies(sig2, ta.invariant(b.target), clocks):
            raise ValueError(
                f"edge {k} ({e.action!r} from {e.source}), branch {j} (to {b.target}) is not well-formed: "
                f"taken in region {zeta}, it leads to region {sig2}, outside the invariant of {b.target}"
            )
        out.append((b.prob, (b.target, tuple(0 if x in b.resets else n for x, n in zip(clocks, lifted)), sig2)))
    return out


def _summed(dist) -> tuple[tuple[Fraction, State], ...]:
    """Probabilities of equal successor states summed, in order of first appearance."""
    acc: dict[State, Fraction] = {}
    for p, st in dist:
        acc[st] = acc.get(st, Fraction(0)) + p
    return tuple((p, st) for st, p in acc.items())


def _moves(ta: TimedAutomaton, loc: str, nu: tuple[int, ...], sig: Sig, K: int, lower_corner_fix: bool = True):
    """(edge index, delay, kind, cost, successor distribution) for every BRG action, deduplicated
    by (edge, delay, distribution); the distribution is ((p, state), ...) with equal states summed."""
    clocks, inv = ta.clocks, ta.invariant(loc)
    chain = time_successors(sig, K)
    walked = []
    for r in chain:
        if not satisfies(r, inv, clocks):
            break
        walked.append(r)
    rate = _frac(ta.rates.get(loc, 0))
    best: dict[tuple, tuple[Fraction, str, tuple]] = {}

    def offer(k: int, t: int, kind: str, cost: Fraction, lifted: tuple[int, ...], zeta: Sig) -> None:
        dist = _summed(_branch_successors(ta, k, lifted, zeta))
        key = (k, t, tuple(sorted((st, p) for p, st in dist)))
        if key not in best or cost < best[key][0]:
            best[key] = (cost, kind, dist)

    for i, theta in enumerate(walked):
        if not is_thin(theta):
            continue
        c = theta[1][0]
        t = theta[0][c] - nu[c]
        assert t >= 0, (loc, nu, sig, theta)
        after = chain[i + 1] if i + 1 < len(chain) else None
        before = chain[i - 1] if i >= 1 else None
        candidates = [("thin", theta), ("after", after), ("before", before)]
        lifted = tuple(n + t for n in nu)
        for k, e in enumerate(ta.edges):
            if e.source != loc:
                continue
            cost = rate * t + _frac(e.cost)
            for kind, zeta in candidates:
                if zeta is None or not satisfies(zeta, inv, clocks) or not satisfies(zeta, e.guard, clocks):
                    continue
                offer(k, t, kind, cost, lifted, zeta)

    # Lower corner of a thick region. If sig is thick and nu lies in the closure of the thin
    # region θ0 just before sig in time (where sig's smallest-fraction group has fraction 0),
    # the state stands for valuations of sig arbitrarily close to its lower corner nu. Acting
    # in sig itself after a delay ≈ 0 is then possible, at cost ≈ e.cost. Definition 13 of the
    # paper read literally only considers thin regions in the future of sig (the first one here
    # needs delay >= 1 from nu), so this move would be missing. θ0 itself is NOT a candidate,
    # and neither is a "before" region around it: the run is already past θ0.
    if lower_corner_fix and _lower_corner(sig, nu) and satisfies(sig, inv, clocks):
        for k, e in enumerate(ta.edges):
            if e.source != loc or not satisfies(sig, e.guard, clocks):
                continue
            offer(k, 0, "after", _frac(e.cost), nu, sig)
    for (k, t, _), (cost, kind, dist) in best.items():
        yield k, t, kind, cost, dist


def _indexed(dist, index: dict[State, int]) -> tuple[tuple[Fraction, int], ...]:
    """((p, state index), ...) sorted by state index."""
    return tuple((p, i) for i, p in sorted((index[st], p) for p, st in dist))


def _lower_corner(sig: Sig, nu: tuple[int, ...]) -> bool:
    """sig is thick and nu lies in the closure of the thin region just before it in time."""
    ints, zero, groups = sig
    return not zero and in_closure(nu, (ints, groups[0], groups[1:]))


def _action_regions(ta: TimedAutomaton, nu: tuple[int, ...], sig: Sig, K: int, delay: int, kind: str) -> list[Sig]:
    """The regions in which an action (delay, kind) from (nu, sig) can take its edge."""
    chain = time_successors(sig, K)
    out = []
    for i, theta in enumerate(chain):
        if is_thin(theta) and theta[0][theta[1][0]] - nu[theta[1][0]] == delay:
            j = {"thin": i, "after": i + 1, "before": i - 1}[kind]
            if 0 <= j < len(chain):
                out.append(chain[j])
    if kind == "after" and delay == 0 and _lower_corner(sig, nu):
        out.append(sig)
    return out


def build_brg(
    ta: TimedAutomaton,
    targets: Iterable[str],
    *,
    max_states: int = 200_000,
    allow_deadlocks: bool = False,
    _lower_corner_fix: bool = True,  # False reproduces the step 6 BRG (for tests only)
) -> BRG:
    """The reachable boundary region graph, by BFS from (initial, 0, region of 0).

    A reachable non-target state without actions raises ValueError, unless
    `allow_deadlocks` is set: then such states are kept (listed in `BRG.deadlocks`) and
    get value +inf in value_iteration.
    """
    targets = frozenset(targets)
    unknown = targets - set(ta.locations)
    if unknown:
        raise ValueError(f"unknown target locations {sorted(unknown)}")
    K = ta.max_constant()
    zero = tuple(0 for _ in ta.clocks)
    init: State = (ta.initial, zero, region_of(zero))
    states: list[State] = [init]
    index = {init: 0}
    actions: list[list[BRGAction]] = []
    stuck = []
    queue = deque([0])
    while queue:
        s = queue.popleft()
        loc, nu, sig = states[s]
        out: list[BRGAction] = []
        if loc not in targets:
            _check_bounded(ta, loc)
            for k, t, kind, cost, dist in _moves(ta, loc, nu, sig, K, _lower_corner_fix):
                for _, succ in dist:
                    if succ not in index:
                        if len(states) >= max_states:
                            raise RuntimeError(f"build_brg exceeded max_states={max_states}")
                        index[succ] = len(states)
                        states.append(succ)
                        queue.append(index[succ])
                out.append(BRGAction(k, t, kind, cost, _indexed(dist, index)))
            if not out:
                stuck.append(s)
        while len(actions) <= s:
            actions.append([])
        actions[s] = out
    if stuck and not allow_deadlocks:
        raise ValueError(f"reachable non-target states without actions: {[states[s] for s in stuck]}")
    return BRG(states=states, actions=actions, initial=0,
               targets=frozenset(i for i, (l, _, _) in enumerate(states) if l in targets),
               deadlocks=frozenset(stuck))


def check_brg(ta: TimedAutomaton, brg: BRG) -> None:
    """Raise ValueError unless every nu is integral and in closure(sig), every delay is >= 0,
    target states have no actions, and every action's distribution is consistent: probabilities
    > 0 summing exactly to 1, and equal to one entry per branch (p, location, nu′, signature) of
    its edge, taken in a region the action's (delay, kind) allows, summed and indexed."""
    index = {st: i for i, st in enumerate(brg.states)}
    K = ta.max_constant()
    for i, (loc, nu, sig) in enumerate(brg.states):
        if not all(isinstance(n, int) for n in nu):
            raise ValueError(f"state {i}: valuation {nu} is not integral")
        if not in_closure(nu, sig):
            raise ValueError(f"state {i}: valuation {nu} is not in the closure of region {sig}")
        if i in brg.targets and brg.actions[i]:
            raise ValueError(f"target state {i} has actions")
        if i not in brg.targets and not brg.actions[i] and i not in brg.deadlocks:
            raise ValueError(f"non-target state {i} has no actions and is not a declared deadlock")
        for a in brg.actions[i]:
            if a.delay < 0:
                raise ValueError(f"state {i}: action {a} has negative delay")
            probs = [p for p, _ in a.successors]
            if any(p <= 0 for p in probs) or sum(probs) != 1:
                raise ValueError(f"state {i}: action {a} has probabilities {probs}")
            if ta.edges[a.edge].source != loc:
                raise ValueError(f"state {i}: action {a} takes an edge leaving {ta.edges[a.edge].source}")
            lifted = tuple(n + a.delay for n in nu)
            expected = []
            for zeta in _action_regions(ta, nu, sig, K, a.delay, a.kind):
                dist = _branch_successors(ta, a.edge, lifted, zeta, check=False)
                if all(st in index for _, st in dist):
                    expected.append(_indexed(_summed(dist), index))
            if a.successors not in expected:
                raise ValueError(f"state {i}: successors of action {a} match no region of its (delay, kind)")


# --- value iteration ----------------------------------------------------------

def _q(a, V: list[float], discount: float) -> float:
    """cost(a) + λ·Σ p·V(s′) over a.successors."""
    return float(a.cost) + discount * _expect(_succ(a), V)


def _succ(a) -> tuple[tuple[float, int], ...]:
    return tuple((float(p), s) for p, s in a.successors)


def _expect(succ: tuple[tuple[float, int], ...], V: list[float]) -> float:
    if len(succ) == 1 and succ[0][0] == 1.0:  # deterministic: exactly V[s], as before
        return V[succ[0][1]]
    return sum(p * V[s] for p, s in succ)


def value_iteration(
    brg: BRG,
    discount: float,
    *,
    eps: float = 1e-9,
    max_iter: int = 1_000_000,
    tie_key: Optional[Callable[[Any], Any]] = None,
) -> tuple[list[float], list[Optional[BRGAction]], int]:
    """V <- min_a [cost(a) + λ·Σ p·V(s′)], V = 0 on targets, V = +inf on deadlocks.

    Stops when max |V_new - V_old| < eps·(1-λ)/λ, so the result is within eps of the fixed
    point. Returns (values, greedy policy, iterations); ties in the policy (within 1e-9) are
    broken by the smallest (delay, edge index, kind in KINDS order), or by `tie_key`.

    Works on any MDP with the attributes states, actions (objects with .cost and .successors,
    a tuple of (probability, state index)), targets and deadlocks; block_vi reuses it.
    """
    if not 0 < discount < 1:
        raise ValueError(f"discount must be in (0, 1), got {discount}")
    inf = float("inf")
    V = [inf if i in brg.deadlocks else 0.0 for i in range(len(brg.states))]
    threshold = eps * (1 - discount) / discount
    # (float cost, successors with float probabilities), converted once
    flat = [[(float(a.cost), _succ(a)) for a in acts] for acts in brg.actions]
    for it in range(1, max_iter + 1):
        new = [min((c + discount * _expect(succ, V) for c, succ in acts), default=V[i])
               for i, acts in enumerate(flat)]
        delta = max((0.0 if x == y else abs(x - y) for x, y in zip(new, V)), default=0.0)
        V = new
        if delta < threshold:
            break
    else:
        raise RuntimeError(f"value_iteration did not converge in {max_iter} iterations")
    policy: list[Optional[BRGAction]] = []
    for acts in brg.actions:
        if not acts:
            policy.append(None)
            continue
        best = min(_q(a, V, discount) for a in acts)
        tied = [a for a in acts if _q(a, V, discount) <= best + 1e-9 * max(1.0, abs(best))]
        policy.append(min(tied, key=tie_key or (lambda a: (a.delay, a.edge, KINDS.index(a.kind)))))
    return V, policy, it


def optimal_run(ta: TimedAutomaton, brg: BRG, policy: list[Optional[BRGAction]]):
    """Steps (location, valuation, edge action, delay, kind) of the greedy run until a target
    or deadlock. Discounted cost is finite even for runs that never reach a target, so the
    greedy run may cycle: then it ends with ("cycle", k), meaning it repeats from step k."""
    steps: list[tuple] = []
    seen: dict[int, int] = {}
    s = brg.initial
    while policy[s] is not None:
        if s in seen:
            steps.append(("cycle", seen[s]))
            return steps
        seen[s] = len(steps)
        a = policy[s]
        if not a.is_deterministic:
            raise ValueError(f"the optimal run reaches a probabilistic action at state {brg.states[s]}")
        loc, nu, _ = brg.states[s]
        steps.append((loc, nu, ta.edges[a.edge].action, a.delay, a.kind))
        s = a.target
    return steps
