"""The partition operator Π_K.

`partition(F, G, ctx)` turns a family of zones F and a set of guard atoms G into a
disjoint family of zones P with these guaranteed properties:

1. Cover: ⋃P = ⋃F exactly.
2. Disjoint: blocks are pairwise disjoint and nonempty.
3. Uniform: every block is inside or disjoint from every zone of F and every ⟦g⟧, g ∈ G.
4. Idempotent: if F is already a G-uniform partition, the output equals F.
5. Deterministic: the result depends only on the set F (semantic equality) and the set G.

Unlike the paper we do not compute inclusion-maximal blocks; pieces of one Venn region
are merged greedily (whole region if convex, else pairwise while the union is convex).
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

from .model import Constraint
from .zones import Zone, ZoneContext, hull, intersect, subtract, subtract_all, union_is_convex


def partition(F: Iterable[Zone], G: Iterable[Constraint], ctx: ZoneContext) -> frozenset[Zone]:
    zones, cutters = _normalise(F, G, ctx)
    pieces = _disjointify(zones)
    pieces = _cut(pieces, cutters)
    return frozenset(_merge(pieces, cutters))


def _sorted(zones: Iterable[Zone]) -> list[Zone]:
    """Nonempty zones, semantically deduplicated, in canonical order."""
    return sorted({z for z in zones if not z.is_empty}, key=Zone.key)


def _normalise(F: Iterable[Zone], G: Iterable[Constraint], ctx: ZoneContext) -> tuple[list[Zone], list[Zone]]:
    """Step 0: the zones of F, and the cutters (F plus one zone per guard atom)."""
    zones = _sorted(F)
    for z in zones:
        if z.ctx is not ctx:
            raise ValueError("all zones must belong to the given ZoneContext")
    cutters = _sorted([*zones, *(ctx.from_guard((g,)) for g in G)])
    return zones, cutters


def _add_disjoint(pieces: list[Zone], z: Zone) -> None:
    """Append to `pieces` disjoint pieces covering exactly z minus the current pieces.

    subtract_all only promises a cover, so each remainder is re-checked against the
    pieces collected so far (including earlier remainders) before it is accepted.
    """
    todo = [z]
    while todo:
        r = todo.pop(0)
        if all(intersect(r, p).is_empty for p in pieces):
            pieces.append(r)
        else:
            todo = sorted(subtract_all(r, pieces), key=Zone.key) + todo


def _disjointify(zones: list[Zone]) -> list[Zone]:
    """Step 1: disjoint pieces covering ⋃zones; each new zone minus everything so far."""
    pieces: list[Zone] = []
    for z in zones:
        _add_disjoint(pieces, z)
    return sorted(pieces, key=Zone.key)


def _cut(pieces: list[Zone], cutters: list[Zone]) -> list[Zone]:
    """Step 2: split every piece into its part inside and outside each cutter."""
    for c in cutters:
        new: list[Zone] = []
        for p in pieces:
            inside = intersect(p, c)
            if not inside.is_empty:
                new.append(inside)
            outside: list[Zone] = []
            for q in sorted(subtract(p, c), key=Zone.key):
                _add_disjoint(outside, q)
            new.extend(outside)
        pieces = sorted(new, key=Zone.key)
    return pieces


def _signature(p: Zone, cutters: list[Zone]) -> tuple[bool, ...]:
    return tuple(p <= c for c in cutters)


def _merge(pieces: list[Zone], cutters: list[Zone]) -> list[Zone]:
    """Step 3: merge pieces of the same Venn region (same signature) into convex blocks."""
    groups: dict[tuple[bool, ...], list[Zone]] = {}
    for p in sorted(pieces, key=Zone.key):
        groups.setdefault(_signature(p, cutters), []).append(p)
    blocks: list[Zone] = []
    for sig in sorted(groups):
        blocks.extend(_merge_group(groups[sig]))
    return sorted(blocks, key=Zone.key)


def _merge_group(group: list[Zone]) -> list[Zone]:
    group = sorted(group, key=Zone.key)
    if union_is_convex(group):
        return [hull(group)]
    while True:
        for a, b in combinations(group, 2):
            if union_is_convex((a, b)):
                group = sorted([z for z in group if z is not a and z is not b] + [hull((a, b))], key=Zone.key)
                break
        else:
            return group


def check_partition(P: Iterable[Zone], F: Iterable[Zone], G: Iterable[Constraint], ctx: ZoneContext) -> None:
    """Raise ValueError unless P is a disjoint, G- and F-uniform cover of ⋃F (properties 1-3)."""
    blocks, zones = list(P), _sorted(F)
    for b in blocks:
        if b.is_empty:
            raise ValueError("partition contains an empty block")
    for a, b in combinations(blocks, 2):
        if a == b or not intersect(a, b).is_empty:
            raise ValueError(f"blocks overlap: {a} and {b}")
    for b in blocks:
        if subtract_all(b, zones):
            raise ValueError(f"block {b} is not covered by the zones of F")
    for z in zones:
        rest = subtract_all(z, blocks)
        if rest:
            raise ValueError(f"zone {z} of F is not covered by the blocks; uncovered: {[str(r) for r in rest]}")
    atoms = [(f"guard atom {g}", ctx.from_guard((g,))) for g in G]
    for name, c in [*((f"zone {z} of F", z) for z in zones), *atoms]:
        for b in blocks:
            if not (b <= c or intersect(b, c).is_empty):
                raise ValueError(f"block {b} is neither inside nor disjoint from {name}")
