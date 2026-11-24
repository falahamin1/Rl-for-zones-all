"""Timed automaton data classes.

A clock constraint is a difference constraint ``left - right < bound`` (strict)
or ``left - right <= bound`` (non-strict). The reserved clock name ``"0"``
denotes the zero clock, so simple bounds such as ``x <= 3`` or ``x >= 1`` are
the special cases ``x - 0 <= 3`` and ``0 - x <= -1``.

Costs and rates are carried along for later phases; nothing here uses them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable

ZERO = "0"
DELAY = "delay"  # reserved: the label of delay transitions/edges, not allowed as an action name


@dataclass(frozen=True)
class Constraint:
    """``left - right < bound`` if ``strict`` else ``left - right <= bound``."""

    left: str
    right: str
    bound: int
    strict: bool

    def __str__(self) -> str:
        op = "<" if self.strict else "<="
        if self.right == ZERO:
            return f"{self.left} {op} {self.bound}"
        if self.left == ZERO:
            op = ">" if self.strict else ">="
            return f"{self.right} {op} {-self.bound}"
        return f"{self.left} - {self.right} {op} {self.bound}"


# --- helper constructors ----------------------------------------------------

def le(clock: str, c: int) -> Constraint:
    """clock <= c"""
    return Constraint(clock, ZERO, c, False)


def lt(clock: str, c: int) -> Constraint:
    """clock < c"""
    return Constraint(clock, ZERO, c, True)


def ge(clock: str, c: int) -> Constraint:
    """clock >= c"""
    return Constraint(ZERO, clock, -c, False)


def gt(clock: str, c: int) -> Constraint:
    """clock > c"""
    return Constraint(ZERO, clock, -c, True)


def eq(clock: str, c: int) -> tuple[Constraint, Constraint]:
    """clock == c, as (clock <= c, clock >= c)"""
    return (le(clock, c), ge(clock, c))


def diff_le(x: str, y: str, c: int) -> Constraint:
    """x - y <= c"""
    return Constraint(x, y, c, False)


def diff_lt(x: str, y: str, c: int) -> Constraint:
    """x - y < c"""
    return Constraint(x, y, c, True)


def diff_ge(x: str, y: str, c: int) -> Constraint:
    """x - y >= c, i.e. y - x <= -c"""
    return Constraint(y, x, -c, False)


def diff_gt(x: str, y: str, c: int) -> Constraint:
    """x - y > c, i.e. y - x < -c"""
    return Constraint(y, x, -c, True)


# --- automaton --------------------------------------------------------------

class MultiBranchError(ValueError):
    """`.target` or `.resets` read on an edge with more than one branch."""


def _prob(x) -> Fraction:
    """Exact probability: floats are read by their decimal repr (0.1 -> 1/10), as brg._frac does."""
    if isinstance(x, bool) or not isinstance(x, (Fraction, int, str, float)):
        raise TypeError(f"branch probability must be a Fraction, int, str or float, got {type(x).__name__}")
    return Fraction(repr(x)) if isinstance(x, float) else Fraction(x)


@dataclass(frozen=True)
class Branch:
    prob: Fraction
    resets: frozenset[str]
    target: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "prob", _prob(self.prob))
        object.__setattr__(self, "resets", frozenset(self.resets))


@dataclass(frozen=True)
class PEdge:
    """An edge with a guard and a cost, leading to branch b with probability b.prob."""

    source: str
    action: str
    guard: tuple[Constraint, ...] = ()  # conjunction
    cost: float = 0.0
    branches: tuple[Branch, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "guard", tuple(self.guard))
        object.__setattr__(self, "branches", tuple(self.branches))

    @property
    def is_deterministic(self) -> bool:
        return len(self.branches) == 1

    def _single(self) -> Branch:
        if not self.is_deterministic:
            raise MultiBranchError(f"edge {_edge_name(self)} has {len(self.branches)} branches, not one")
        return self.branches[0]

    @property
    def target(self) -> str:
        return self._single().target

    @property
    def resets(self) -> frozenset[str]:
        return self._single().resets


def Edge(source: str, target: str, action: str, guard: tuple[Constraint, ...] = (),
         resets: frozenset[str] = frozenset(), cost: float = 0.0) -> PEdge:
    """A deterministic edge: one branch with probability 1."""
    return PEdge(source, action, guard, cost, (Branch(1, resets, target),))


def _edge_name(e: PEdge) -> str:
    targets = "|".join(b.target for b in e.branches) or "?"
    return f"{e.source}->{targets} ({e.action!r})"


def require_deterministic(ta: "TimedAutomaton", where: str) -> None:
    """Raise NotImplementedError if some edge of `ta` has more than one branch (until step 11b).
    `where` names the caller, for reading call sites; the message itself is fixed."""
    if any(not e.is_deterministic for e in ta.edges):
        raise NotImplementedError("probabilistic edges are not supported yet (step 11b)")


@dataclass(frozen=True)
class TimedAutomaton:
    clocks: tuple[str, ...]
    locations: tuple[str, ...]
    initial: str
    edges: tuple[PEdge, ...]
    # Cost rate per location; may be empty. Excluded from hashing (dicts are unhashable).
    rates: dict[str, float] = field(default_factory=dict, hash=False)
    # Invariant per location: upper bounds x <= c / x < c on single clocks; may be empty.
    invariants: dict[str, tuple[Constraint, ...]] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "clocks", tuple(self.clocks))
        object.__setattr__(self, "locations", tuple(self.locations))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "rates", dict(self.rates))
        object.__setattr__(self, "invariants", {l: tuple(cs) for l, cs in self.invariants.items()})
        self.validate()

    def outgoing(self, loc: str) -> list[PEdge]:
        return [e for e in self.edges if e.source == loc]

    def atomic_guards(self, loc: str) -> frozenset[Constraint]:
        """All constraints appearing in guards of edges leaving ``loc``."""
        return frozenset(g for e in self.outgoing(loc) for g in e.guard)

    def partition_atoms(self, loc: str) -> frozenset[Constraint]:
        """The atoms a partition of ``loc`` is made uniform with: ``atomic_guards(loc)`` plus
        the strict atom ``x < b`` for every non-strict invariant bound ``x <= b`` of ``loc``.

        Blocks lie inside the invariant, so uniformity w.r.t. ``x <= b`` itself separates
        nothing; ``x < b`` makes the face ``x = b`` a block boundary, as in the canonical
        decomposition. A strict bound ``x < b`` has no face inside the invariant and adds nothing.
        """
        faces = (Constraint(c.left, c.right, c.bound, True) for c in self.invariant(loc) if not c.strict)
        return self.atomic_guards(loc) | frozenset(faces)

    def invariant(self, loc: str) -> tuple[Constraint, ...]:
        """The invariant of ``loc`` (a conjunction of upper bounds); empty if none."""
        return self.invariants.get(loc, ())

    def max_constant(self) -> int:
        """Max |bound| over all guard and invariant constraints (diagonals included); 0 if none."""
        bounds = [g.bound for e in self.edges for g in e.guard]
        bounds += [c.bound for cs in self.invariants.values() for c in cs]
        return max((abs(b) for b in bounds), default=0)

    def validate(self) -> None:
        _check_unique("clock", self.clocks)
        _check_unique("location", self.locations)
        if ZERO in self.clocks:
            raise ValueError(f"clock name {ZERO!r} is reserved for the zero clock")
        clocks = set(self.clocks)
        locations = set(self.locations)
        if self.initial not in locations:
            raise ValueError(f"initial location {self.initial!r} is not a location")
        for e in self.edges:
            if e.action == DELAY:
                raise ValueError(f"edge {_edge_name(e)} uses the reserved action name {DELAY!r}")
            if e.source not in locations:
                raise ValueError(f"edge {e.action!r} uses unknown location {e.source!r}")
            for g in e.guard:
                _check_constraint(g, clocks)
            if not e.branches:
                raise ValueError(f"edge {_edge_name(e)} has no branches")
            for i, b in enumerate(e.branches):
                if b.target not in locations:
                    raise ValueError(f"edge {e.action!r} uses unknown location {b.target!r}")
                unknown = b.resets - clocks
                if unknown:
                    raise ValueError(f"edge {e.action!r} resets unknown clocks {sorted(unknown)}")
                if b.prob <= 0:
                    raise ValueError(f"edge {_edge_name(e)}: branch {i} has probability {b.prob} <= 0")
            total = sum((b.prob for b in e.branches), Fraction(0))
            if total != 1:
                raise ValueError(f"edge {_edge_name(e)}: branch probabilities sum to {total}, not 1")
        unknown = set(self.rates) - locations
        if unknown:
            raise ValueError(f"rates given for unknown locations {sorted(unknown)}")
        unknown = set(self.invariants) - locations
        if unknown:
            raise ValueError(f"invariants given for unknown locations {sorted(unknown)}")
        for loc, cs in self.invariants.items():
            for c in cs:
                _check_constraint(c, clocks)
                if c.right != ZERO or c.left == ZERO:
                    raise ValueError(f"invariant {c} of {loc!r} is not an upper bound x <= c or x < c on one clock")
        # All invariants are upper bounds x - 0 ≺ c, so at the zero valuation each reads 0 ≺ c.
        for c in self.invariant(self.initial):
            if not (0 < c.bound if c.strict else 0 <= c.bound):
                raise ValueError(
                    f"the zero valuation violates the invariant {c} of the initial location {self.initial!r}"
                )


def _check_unique(kind: str, names: Iterable[str]) -> None:
    seen: set[str] = set()
    for n in names:
        if n in seen:
            raise ValueError(f"duplicate {kind} name {n!r}")
        seen.add(n)


def _check_constraint(g: Constraint, clocks: set[str]) -> None:
    for c in (g.left, g.right):
        if c != ZERO and c not in clocks:
            raise ValueError(f"constraint {g} uses unknown clock {c!r}")
    if g.left == g.right:
        raise ValueError(f"constraint {g} compares a clock with itself")
    if not isinstance(g.bound, int) or isinstance(g.bound, bool):
        raise ValueError(f"constraint {g} has a non-integer bound")
