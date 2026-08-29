"""READ-ONLY diagnostic: how much room does a SOUND state abstraction have
in this domain, before building another one?

Does not modify env.py, dp.py, or the model -- this module only reads
dp.explore_and_solve's output and zones.py's existing zone construction,
and adds new, self-contained analysis on top (Measurement 3's "smallest
DBM containing a set of points" is new machinery, built directly on
pyudbm, not a change to zone_ops.py/zones.py).

Every measurement excludes terminal (DONE/FAILED) and deadlocked (empty
optimal_actions -- see dp.py) states from partition statistics; counts are
reported explicitly, not silently dropped.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from pyudbm import Context, Federation, IntValuation

from .dp import DPResult, explore_and_solve
from .env import GridState, is_terminal, peek
from .instance import DELTA, GridInstance
from .zones import build_state_zones

CellKey = object


def usable_states(dp_result: DPResult) -> List[GridState]:
    """Non-terminal, non-deadlocked reachable states -- the population
    every measurement below operates on."""
    return [s for s in dp_result.states if not is_terminal(s) and dp_result.optimal_actions.get(s)]


def exclusion_counts(dp_result: DPResult) -> dict:
    n_terminal = sum(1 for s in dp_result.states if is_terminal(s))
    n_deadlock = sum(1 for s in dp_result.states if not is_terminal(s) and not dp_result.optimal_actions.get(s))
    return {
        "n_states_total": len(dp_result.states),
        "n_terminal_excluded": n_terminal,
        "n_deadlock_excluded": n_deadlock,
        "n_usable": len(dp_result.states) - n_terminal - n_deadlock,
    }


# ---------------------------------------------------------------------------
# Measurement 1 -- the ceiling
# ---------------------------------------------------------------------------

def measurement1_ceiling(dp_result: DPResult, states: List[GridState]) -> dict:
    classes: Dict[frozenset, List[GridState]] = defaultdict(list)
    for s in states:
        classes[frozenset(dp_result.optimal_actions[s])].append(s)
    sizes = [len(v) for v in classes.values()]
    n_singleton = sum(1 for sz in sizes if sz == 1)
    return {
        "n_usable_states": len(states),
        "n_classes": len(classes),
        "class_size_mean": statistics.mean(sizes),
        "class_size_median": statistics.median(sizes),
        "class_size_max": max(sizes),
        "singleton_class_fraction": n_singleton / len(classes),
        "classes": classes,
    }


# ---------------------------------------------------------------------------
# Measurement 2 -- candidate observation models
# ---------------------------------------------------------------------------

def partition_stats(states: List[GridState], dp_result: DPResult, key_fn) -> dict:
    cells: Dict[CellKey, List[GridState]] = defaultdict(list)
    for s in states:
        cells[key_fn(s)].append(s)

    sizes = [len(v) for v in cells.values()]
    n_sound = 0
    for members in cells.values():
        common = None
        for s in members:
            acts = set(dp_result.optimal_actions[s])
            common = acts if common is None else (common & acts)
            if not common:
                break
        if common:
            n_sound += 1

    n_cells = len(cells)
    return {
        "n_cells": n_cells,
        "soundness_fraction": n_sound / n_cells,
        "mean_cell_size": statistics.mean(sizes),
        "max_cell_size": max(sizes),
        "singleton_fraction": sum(1 for sz in sizes if sz == 1) / n_cells,
        "cells": cells,
    }


def _activated_bitmask(s: GridState) -> Tuple[int, ...]:
    # A switch clock equals y identically iff it has NEVER been reset (see
    # module docstring in instance.py: both start at 0 and advance
    # together until the first touch; after any reset the gap between the
    # clock and y is fixed and positive forever after, since y strictly
    # increases and the clock was just set to 0 < y at that instant).
    return tuple(1 if c != s.y else 0 for c in s.clocks)


def _gate_passable_bits(instance: GridInstance, s: GridState) -> Tuple[int, ...]:
    bits = []
    for i in range(instance.n_switches):
        a, b = instance.gate_windows[i]
        bits.append(1 if a <= s.clocks[i] <= b else 0)
    return tuple(bits)


def _gate_reachable_bits(instance: GridInstance, s: GridState) -> Tuple[int, ...]:
    """The exact test decision_relevance.py's _is_gate_boundary_flip uses
    (peek legality for some allowed delay), generalized to every gate, not
    just ones implicated in a differing action. Convention (documented,
    not silently chosen): if the gate isn't adjacent to s.pos at all,
    there is no move that enters it from here in one step, so the bit is
    False regardless of window timing."""
    bits = []
    for i, gate_pos in enumerate(instance.gates):
        move = None
        for m, (dr, dc) in DELTA.items():
            if (s.pos[0] + dr, s.pos[1] + dc) == gate_pos:
                move = m
                break
        if move is None:
            bits.append(0)
            continue
        reachable = any(peek(instance, s, f"d{d}_{move}").legal for d in instance.allowed_delays)
        bits.append(1 if reachable else 0)
    return tuple(bits)


def _window_relative(instance: GridInstance, s: GridState) -> Tuple[int, ...]:
    cats = []
    for i in range(instance.n_switches):
        a, b = instance.gate_windows[i]
        v = s.clocks[i]
        cats.append(0 if v < a else (1 if v <= b else 2))
    return tuple(cats)


def build_candidates(instance: GridInstance, dp_result: DPResult, states: List[GridState]):
    _dp2, _zctx, zones = build_state_zones(instance, dp_result)

    candidates = {
        "a_concrete_state": lambda s: s,
        "b_location_plus_activated": lambda s: (s.pos, _activated_bitmask(s)),
        "c_current_zone_construction": lambda s: (s.pos, str(zones[s])),
        "d_gate_passability_bits": lambda s: (s.pos, _gate_passable_bits(instance, s)),
        "e_gate_reachability_bits": lambda s: (s.pos, _gate_reachable_bits(instance, s)),
        "f_discretized_clocks_g1": lambda s: (s.pos, tuple(c // 1 for c in s.clocks), s.y // 1),
        "f_discretized_clocks_g2": lambda s: (s.pos, tuple(c // 2 for c in s.clocks), s.y // 2),
        "f_discretized_clocks_g4": lambda s: (s.pos, tuple(c // 4 for c in s.clocks), s.y // 4),
        "g_window_relative_position": lambda s: (s.pos, _window_relative(instance, s)),
    }
    return candidates


def measurement2_candidates(instance: GridInstance, dp_result: DPResult, states: List[GridState]) -> dict:
    candidates = build_candidates(instance, dp_result, states)
    results = {}
    for name, key_fn in candidates.items():
        results[name] = partition_stats(states, dp_result, key_fn)
    return results


# ---------------------------------------------------------------------------
# Measurement 3 -- is the ceiling zone-expressible?
# ---------------------------------------------------------------------------

def smallest_federation_containing(clock_names: List[str], points: List[Tuple[int, ...]]):
    """The tightest DBM (entrywise-max of pairwise differences, then
    pyudbm's own canonicalization on intersection) containing every given
    integer point -- the standard "DBM hull" of a finite point set."""
    ctx = Context(list(clock_names))
    clocks = {name: getattr(ctx, name) for name in clock_names}
    n = len(clock_names) + 1

    def coord(pt, i):
        return 0 if i == 0 else pt[i - 1]

    Z = Federation(ctx).set_init()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            b = max(coord(pt, i) - coord(pt, j) for pt in points)
            if i == 0:
                Z = Z & (clocks[clock_names[j - 1]] >= -b)
            elif j == 0:
                Z = Z & (clocks[clock_names[i - 1]] <= b)
            else:
                Z = Z & ((clocks[clock_names[i - 1]] - clocks[clock_names[j - 1]]) <= b)
    return ctx, Z


def _valuation(ctx, clock_names: List[str], point: Tuple[int, ...]) -> IntValuation:
    val = IntValuation(ctx)
    for name, v in zip(clock_names, point):
        val[name] = v
    return val


def measurement3_zone_expressibility(instance: GridInstance, classes: Dict[frozenset, List[GridState]],
                                      states: List[GridState], top_n: int = 10) -> dict:
    clock_names = list(instance.clock_names) + ["y"]
    ranked = sorted(classes.items(), key=lambda kv: -len(kv[1]))[:top_n]

    per_class = []
    for acts, members in ranked:
        points = [tuple(s.clocks) + (s.y,) for s in members]
        ctx, Z = smallest_federation_containing(clock_names, points)

        for pt in points:
            assert Z.contains(_valuation(ctx, clock_names, pt)), (
                "constructed hull DBM does not contain its own defining point -- internal bug, not a data issue"
            )

        member_set = set(members)
        contamination = 0
        contaminating_examples = []
        for s in states:
            if s in member_set:
                continue
            pt = tuple(s.clocks) + (s.y,)
            if Z.contains(_valuation(ctx, clock_names, pt)):
                contamination += 1
                if len(contaminating_examples) < 3:
                    contaminating_examples.append(str(s))

        per_class.append(
            {
                "class_size": len(members),
                "action_set": sorted(acts),
                "contamination_count": contamination,
                "clean": contamination == 0,
                "contaminating_examples": contaminating_examples,
            }
        )

    return {"per_class": per_class, "n_clean_of_top": sum(1 for r in per_class if r["clean"]), "top_n": len(ranked)}


# ---------------------------------------------------------------------------
# Measurement 4 -- what does each candidate already determine?
# ---------------------------------------------------------------------------

def is_refinement_of_action_partition(cells: Dict[CellKey, List[GridState]], dp_result: DPResult) -> Tuple[bool, float]:
    n_fail = 0
    for members in cells.values():
        acts0 = frozenset(dp_result.optimal_actions[members[0]])
        if any(frozenset(dp_result.optimal_actions[s]) != acts0 for s in members[1:]):
            n_fail += 1
    n_cells = len(cells)
    return (n_fail == 0), (n_fail / n_cells if n_cells else float("nan"))


def measurement4_refinement(dp_result: DPResult, m2_results: dict) -> dict:
    out = {}
    for name, r in m2_results.items():
        is_ref, fail_frac = is_refinement_of_action_partition(r["cells"], dp_result)
        out[name] = {
            "is_refinement_of_action_partition": is_ref,
            "cells_failing_fraction": fail_frac,
            "mean_cell_size": r["mean_cell_size"],
            "abstracts_at_all": r["mean_cell_size"] > 1.0,
        }
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(w: float = 0.25) -> dict:
    instance = GridInstance(w=w)
    dp_result = explore_and_solve(instance)
    states = usable_states(dp_result)
    excl = exclusion_counts(dp_result)

    m1 = measurement1_ceiling(dp_result, states)
    m2 = measurement2_candidates(instance, dp_result, states)
    m3 = measurement3_zone_expressibility(instance, m1["classes"], states)
    m4 = measurement4_refinement(dp_result, m2)

    n_locations = len({s.pos for s in states})

    return {
        "instance_w": w,
        "exclusions": excl,
        "measurement1": {k: v for k, v in m1.items() if k != "classes"},
        "n_locations": n_locations,
        "measurement2": {name: {k: v for k, v in r.items() if k != "cells"} for name, r in m2.items()},
        "measurement3": m3,
        "measurement4": m4,
    }


if __name__ == "__main__":
    import json

    r = run()
    print(json.dumps(r, indent=2, default=str))
    with open("/tmp/partition_results.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("\nWrote /tmp/partition_results.json")
