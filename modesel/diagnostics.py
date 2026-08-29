"""Step 3 -- THE PARTITION DIAGNOSTIC, ported (not imported) from
gridtimed/diagnostics.py to this domain. Same methodology: group concrete
states by optimal-action SET (the ceiling), test candidate observation
models for soundness/coarseness, and test whether the ceiling's largest
classes are zone-expressible (smallest-DBM-hull containment check).
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from pyudbm import Context, Federation, IntValuation

from .dp import DPResult
from .env import ModeSelState, is_terminal
from .instance import ModeSelInstance

CellKey = object


def usable_states(instance: ModeSelInstance, dp_result: DPResult) -> List[ModeSelState]:
    return [s for s in dp_result.states if not is_terminal(instance, s) and dp_result.optimal_actions.get(s)]


def exclusion_counts(instance: ModeSelInstance, dp_result: DPResult) -> dict:
    n_terminal = sum(1 for s in dp_result.states if is_terminal(instance, s))
    n_deadlock = sum(
        1 for s in dp_result.states
        if not is_terminal(instance, s) and not dp_result.optimal_actions.get(s)
    )
    return {
        "n_states_total": len(dp_result.states),
        "n_terminal_excluded": n_terminal,
        "n_deadlock_excluded": n_deadlock,
        "n_usable": len(dp_result.states) - n_terminal - n_deadlock,
    }


# ---------------------------------------------------------------------------
# Measurement 1 -- the ceiling
# ---------------------------------------------------------------------------

def measurement1_ceiling(dp_result: DPResult, states: List[ModeSelState]) -> dict:
    classes: Dict[frozenset, List[ModeSelState]] = defaultdict(list)
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

def partition_stats(states: List[ModeSelState], dp_result: DPResult, key_fn) -> dict:
    cells: Dict[CellKey, List[ModeSelState]] = defaultdict(list)
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


def _location_key(instance: ModeSelInstance, s: ModeSelState):
    return tuple((js.cursor, js.active_mode, js.failed) for js in s.jobs)


def remaining_slow_work(instance: ModeSelInstance, job_idx: int, s: ModeSelState) -> float:
    job = instance.jobs[job_idx]
    js = s.jobs[job_idx]
    if js.failed or js.cursor == job.n_tasks:
        return 0.0
    total = 0.0
    if js.active_mode is not None:
        total += js.active_duration - js.c
        start = js.cursor + 1
    else:
        start = js.cursor
    total += sum(t.d_slow for t in job.tasks[start:])
    return total


def slack(instance: ModeSelInstance, job_idx: int, s: ModeSelState) -> float:
    job = instance.jobs[job_idx]
    return (job.deadline - s.g) - remaining_slow_work(instance, job_idx, s)


def _job_status(instance: ModeSelInstance, job_idx: int, s: ModeSelState) -> str:
    js = s.jobs[job_idx]
    if js.failed:
        return "FAILED"
    if js.cursor == instance.jobs[job_idx].n_tasks:
        return "DONE"
    return "ACTIVE"


def _slack_bucket_key(instance: ModeSelInstance, s: ModeSelState, b: int):
    loc = _location_key(instance, s)
    buckets = []
    for i in range(instance.n_jobs):
        st = _job_status(instance, i, s)
        buckets.append(st if st != "ACTIVE" else int(slack(instance, i, s) // b))
    return (loc, tuple(buckets))


def _slack_ordering_key(instance: ModeSelInstance, s: ModeSelState):
    loc = _location_key(instance, s)
    active = [i for i in range(instance.n_jobs) if _job_status(instance, i, s) == "ACTIVE"]
    ranking = tuple(sorted(active, key=lambda i: slack(instance, i, s)))
    return (loc, ranking)


def _g_bucket_key(instance: ModeSelInstance, s: ModeSelState, b: int):
    return (_location_key(instance, s), int(s.g // b))


def build_candidates(instance: ModeSelInstance):
    candidates = {
        "a_concrete_state": lambda s: s,
        "b_location_only": lambda s: _location_key(instance, s),
        "c_slack_bucket_b1": lambda s: _slack_bucket_key(instance, s, 1),
        "c_slack_bucket_b2": lambda s: _slack_bucket_key(instance, s, 2),
        "c_slack_bucket_b4": lambda s: _slack_bucket_key(instance, s, 4),
        "d_slack_ordering": lambda s: _slack_ordering_key(instance, s),
        "e_discretized_g_b1": lambda s: _g_bucket_key(instance, s, 1),
        "e_discretized_g_b2": lambda s: _g_bucket_key(instance, s, 2),
        "e_discretized_g_b4": lambda s: _g_bucket_key(instance, s, 4),
    }
    return candidates


def measurement2_candidates(instance: ModeSelInstance, dp_result: DPResult, states: List[ModeSelState]) -> dict:
    candidates = build_candidates(instance)
    return {name: partition_stats(states, dp_result, key_fn) for name, key_fn in candidates.items()}


# ---------------------------------------------------------------------------
# Measurement 3 -- is the ceiling zone-expressible?
# ---------------------------------------------------------------------------

def _state_point(instance: ModeSelInstance, s: ModeSelState) -> Tuple[int, ...]:
    """One integer coordinate per job (its active-task elapsed clock c,
    0 if not active) plus g -- the clocks a zone construction (Step 4)
    would actually track."""
    return tuple(int(round(js.c)) for js in s.jobs) + (int(round(s.g)),)


def smallest_federation_containing(clock_names: List[str], points: List[Tuple[int, ...]]):
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


def measurement3_zone_expressibility(instance: ModeSelInstance, classes: Dict[frozenset, List[ModeSelState]],
                                      states: List[ModeSelState], top_n: int = 10) -> dict:
    clock_names = [f"c{i}" for i in range(instance.n_jobs)] + ["g"]
    ranked = sorted(classes.items(), key=lambda kv: -len(kv[1]))[:top_n]

    per_class = []
    for acts, members in ranked:
        points = [_state_point(instance, s) for s in members]
        ctx, Z = smallest_federation_containing(clock_names, points)

        for pt in points:
            assert Z.contains(_valuation(ctx, clock_names, pt)), (
                "constructed hull DBM does not contain its own defining point -- internal bug"
            )

        member_set = set(members)
        contamination = 0
        contaminating_examples = []
        for s in states:
            if s in member_set:
                continue
            pt = _state_point(instance, s)
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
# Orchestration
# ---------------------------------------------------------------------------

def run_diagnostic(instance: ModeSelInstance, dp_result: DPResult) -> dict:
    states = usable_states(instance, dp_result)
    excl = exclusion_counts(instance, dp_result)

    m1 = measurement1_ceiling(dp_result, states)
    m2 = measurement2_candidates(instance, dp_result, states)
    m3 = measurement3_zone_expressibility(instance, m1["classes"], states)

    n_locations = len({_location_key(instance, s) for s in states})

    return {
        "label": instance.label,
        "exclusions": excl,
        "measurement1": {k: v for k, v in m1.items() if k != "classes"},
        "n_locations": n_locations,
        "measurement2": {name: {k: v for k, v in r.items() if k != "cells"} for name, r in m2.items()},
        "measurement3": m3,
    }
