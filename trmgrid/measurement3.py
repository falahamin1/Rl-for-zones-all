"""Measurement 3 -- PARTITION DIAGNOSTIC. Only meaningful if Measurement 2
passed. Ported (re-derived from the same measurement definitions, NOT
imported) from gridtimed/diagnostics.py + gridtimed/decision_relevance.py.

Every measurement excludes terminal (DONE/FAILED) and deadlocked (empty
optimal_actions) states from partition statistics; counts are reported
explicitly, not silently dropped.
"""
from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

from .dp import DPResult, explore_and_solve
from .env import TRMState, all_actions, initial_state, is_terminal, peek
from .instance import TRMGridInstance
from .zones import build_state_zones, distinct_zones, validate_containment

CellKey = object


# ---------------------------------------------------------------------
# Uniform-random-policy rollout diagnostic -- a calibration sanity check,
# not part of the DP-based measurements above. If EVERY random episode
# succeeds or EVERY one fails, the guards/deadline aren't actually being
# exercised by ordinary play and the parameter setting is miscalibrated,
# regardless of what the DP-based numbers say.
# ---------------------------------------------------------------------

def random_policy_outcomes(instance: TRMGridInstance, n_episodes: int = 500, seed: int = 0,
                            max_steps: int = 500) -> dict:
    rng = random.Random(seed)
    actions = all_actions(instance)
    n_success = 0
    n_guard_pair = 0
    n_guard_goal = 0
    n_deadline = 0
    n_horizon = 0  # hit max_steps without terminating -- reported, not silently dropped

    for _ep in range(n_episodes):
        s = initial_state(instance)
        for _t in range(max_steps):
            aid = rng.choice(actions)
            res = peek(instance, s, aid)
            s = res.next_state
            if res.newly_done:
                n_success += 1
                break
            if res.newly_failed:
                if res.fail_reason == "guard_pair":
                    n_guard_pair += 1
                elif res.fail_reason == "guard_goal":
                    n_guard_goal += 1
                else:
                    n_deadline += 1
                break
        else:
            n_horizon += 1

    n = n_episodes
    return {
        "n_episodes": n,
        "success_rate": n_success / n,
        "guard_pair_fail_rate": n_guard_pair / n,
        "guard_goal_fail_rate": n_guard_goal / n,
        "deadline_fail_rate": n_deadline / n,
        "total_fail_rate": (n_guard_pair + n_guard_goal + n_deadline) / n,
        "horizon_truncated_rate": n_horizon / n,
        "degenerate": (n_success == 0 or n_success == n or (n_guard_pair + n_guard_goal + n_deadline) == 0),
    }


def usable_states(dp_result: DPResult) -> List[TRMState]:
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


# ---------------------------------------------------------------------
# 3a -- the ceiling
# ---------------------------------------------------------------------

def measurement_3a_ceiling(dp_result: DPResult, states: List[TRMState]) -> dict:
    classes: Dict[frozenset, List[TRMState]] = defaultdict(list)
    for s in states:
        classes[frozenset(dp_result.optimal_actions[s])].append(s)
    sizes = [len(v) for v in classes.values()]
    n_singleton = sum(1 for sz in sizes if sz == 1)
    n_locations = len({s.pos for s in states})
    return {
        "n_usable_states": len(states),
        "n_locations": n_locations,
        "n_classes": len(classes),
        "class_size_mean": statistics.mean(sizes),
        "class_size_median": statistics.median(sizes),
        "class_size_max": max(sizes),
        "singleton_class_fraction": n_singleton / len(classes),
        "classes": classes,
    }


# ---------------------------------------------------------------------
# 3b -- candidate observation models
# ---------------------------------------------------------------------

def partition_stats(states: List[TRMState], dp_result: DPResult, key_fn) -> dict:
    cells: Dict[CellKey, List[TRMState]] = defaultdict(list)
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


def _guard_bits(instance: TRMGridInstance, s: TRMState) -> Tuple[int, ...]:
    """Guard-satisfiability bits for the phase's CURRENTLY-ACTIVE guard(s),
    evaluated against the state's current clock values (not accounting for
    the move about to be taken):
      phase=() [start]: no active guard -- () (vacuous, uniform key shape
        isn't required since phase is already part of the outer key).
      phase=(i,) [have_i]: one bit PER POSSIBLE NEXT SUB-GOAL j != i --
        is t_i currently inside [lo_ij, hi_ij]? (a vector, like
        gridtimed's gate_passable_bits, since here there are multiple
        live choices, not one predetermined next transition.)
      phase=(i,j) [have_i,j]: one bit -- is t_j currently <= cap_j?
    """
    if s.phase == ():
        return ()
    if len(s.phase) == 1:
        i = s.phase[0]
        bits = []
        for j in range(instance.k):
            if j == i:
                continue
            lo, hi = instance.windows[(i, j)]
            bits.append(1 if lo <= s.clocks[i] <= hi else 0)
        return tuple(bits)
    j = s.phase[1]
    return (1 if s.clocks[j] <= instance.caps[j] else 0,)


def build_candidates(instance: TRMGridInstance, dp_result: DPResult, states: List[TRMState]):
    _dp2, zctx, zones = build_state_zones(instance, dp_result)

    candidates = {
        "a_concrete_state": lambda s: s,
        "b_location_only": lambda s: (s.pos, s.phase),
        "c_zone_construction": lambda s: (s.pos, s.phase, str(zones[s])),
        "d_guard_satisfiability_bits": lambda s: (s.pos, s.phase, _guard_bits(instance, s)),
        "e_discretized_clocks_b1": lambda s: (s.pos, s.phase, tuple(c // 1 for c in s.clocks), s.g // 1),
        "e_discretized_clocks_b2": lambda s: (s.pos, s.phase, tuple(c // 2 for c in s.clocks), s.g // 2),
        "e_discretized_clocks_b4": lambda s: (s.pos, s.phase, tuple(c // 4 for c in s.clocks), s.g // 4),
    }
    return candidates, zctx, zones


def measurement_3b_candidates(instance: TRMGridInstance, dp_result: DPResult, states: List[TRMState]):
    candidates, zctx, zones = build_candidates(instance, dp_result, states)
    results = {}
    for name, key_fn in candidates.items():
        results[name] = partition_stats(states, dp_result, key_fn)
    return results, zctx, zones


# ---------------------------------------------------------------------
# 3c / 3d -- decision relevance + within-zone non-uniformity
# ---------------------------------------------------------------------

def measurement_3c_3d(instance: TRMGridInstance, dp_result: DPResult, zctx, zones) -> dict:
    # build_state_zones builds zones one layer PAST the truncation boundary
    # (successor states computed via peek() during exploration but never
    # actually popped by a truncated BFS still get a zone entry, since
    # zone construction replays every transition of every POPPED state).
    # That's harmless for 3a/3b (they iterate dp_result.states, the
    # correctly-popped set) but validate_containment must be restricted to
    # the SAME popped set here too, or n_ok+n_bad silently exceeds
    # n_usable_states -- a real inconsistency caught by cross-checking
    # these two counts against each other, not assumed benign.
    popped_zones = {s: fed for s, fed in zones.items() if s in dp_result.transitions}
    n_ok, n_bad = validate_containment(instance, zctx, popped_zones)

    n_deadlock_excluded = 0
    by_loc: Dict[Tuple[int, int], Dict[str, List[TRMState]]] = defaultdict(lambda: defaultdict(list))
    for s in dp_result.states:
        if is_terminal(s):
            continue
        if not dp_result.optimal_actions.get(s):
            n_deadlock_excluded += 1
            continue
        by_loc[(s.pos, s.phase)][str(zones[s])].append(s)

    multi_zone_locations = 0
    conflicting_locations = 0
    conflict_examples = []

    n_multistate_zones = 0
    n_nonuniform_zones = 0
    within_zone_examples = []
    n_concrete_states_in_multistate_zones = 0

    for loc, groups in by_loc.items():
        group_list = list(groups.items())

        for zs, members in group_list:
            if len(members) < 2:
                continue
            n_multistate_zones += 1
            n_concrete_states_in_multistate_zones += len(members)
            action_set_to_rep: Dict[frozenset, TRMState] = {}
            for s in members:
                acts = frozenset(dp_result.optimal_actions[s])
                if acts not in action_set_to_rep:
                    action_set_to_rep[acts] = s
            distinct = list(action_set_to_rep.items())
            found = False
            for i in range(len(distinct)):
                for j in range(i + 1, len(distinct)):
                    ai, si = distinct[i]
                    aj, sj = distinct[j]
                    if not (ai & aj):
                        found = True
                        if len(within_zone_examples) < 500:
                            within_zone_examples.append(
                                {"loc": loc, "state_i": str(si), "state_j": str(sj),
                                 "actions_i": sorted(ai), "actions_j": sorted(aj)}
                            )
                        break
                if found:
                    break
            if found:
                n_nonuniform_zones += 1

        if len(groups) < 2:
            continue
        multi_zone_locations += 1

        # Bucket by UNIONED action-set per zone before comparing PAIRS OF
        # DISTINCT action-sets -- avoids the quadratic-pairwise-zone-
        # comparison blowup the watch-list flags explicitly.
        action_set_to_rep: Dict[frozenset, TRMState] = {}
        for zs, members in group_list:
            acts = frozenset().union(*(set(dp_result.optimal_actions[s]) for s in members))
            if acts not in action_set_to_rep:
                action_set_to_rep[acts] = members[0]
        distinct_action_sets = list(action_set_to_rep.items())

        conflict = False
        for i in range(len(distinct_action_sets)):
            for j in range(i + 1, len(distinct_action_sets)):
                acts_i, rep_i = distinct_action_sets[i]
                acts_j, rep_j = distinct_action_sets[j]
                if not (acts_i & acts_j):
                    conflict = True
                    if len(conflict_examples) < 500:
                        conflict_examples.append(
                            {"loc": loc, "state_i": str(rep_i), "state_j": str(rep_j),
                             "actions_i": sorted(dp_result.optimal_actions[rep_i]),
                             "actions_j": sorted(dp_result.optimal_actions[rep_j])}
                        )
        if conflict:
            conflicting_locations += 1

    n_usable = sum(1 for s in dp_result.states if not is_terminal(s) and dp_result.optimal_actions.get(s))
    n_zones_total = len({(loc, str(z)) for loc, groups in by_loc.items() for z in groups})
    mean_states_per_zone = n_usable / n_zones_total if n_zones_total else float("nan")

    return {
        "n_containment_ok": n_ok, "n_containment_bad": n_bad,
        "n_deadlock_excluded": n_deadlock_excluded,
        "multi_zone_locations": multi_zone_locations,
        "conflicting_locations": conflicting_locations,
        "conflict_fraction": (conflicting_locations / multi_zone_locations) if multi_zone_locations else float("nan"),
        "conflict_examples": conflict_examples[:10],
        "n_multistate_zones": n_multistate_zones,
        "n_nonuniform_zones": n_nonuniform_zones,
        "within_zone_nonuniform_fraction": (n_nonuniform_zones / n_multistate_zones) if n_multistate_zones else float("nan"),
        "within_zone_examples": within_zone_examples[:10],
        "n_concrete_states_in_multistate_zones": n_concrete_states_in_multistate_zones,
        "mean_concrete_states_per_zone": mean_states_per_zone,
    }


def run(instance: TRMGridInstance, state_limit: int = 300_000) -> dict:
    dp_result = explore_and_solve(instance, state_limit=state_limit)
    states = usable_states(dp_result)
    excl = exclusion_counts(dp_result)

    m3a = measurement_3a_ceiling(dp_result, states)
    m3b, zctx, zones = measurement_3b_candidates(instance, dp_result, states)
    m3cd = measurement_3c_3d(instance, dp_result, zctx, zones)
    rand_policy = random_policy_outcomes(instance)

    return {
        "instance_label": instance.label, "w": instance.w,
        "exclusions": excl,
        "measurement_3a": {k: v for k, v in m3a.items() if k != "classes"},
        "measurement_3b": {name: {k: v for k, v in r.items() if k != "cells"} for name, r in m3b.items()},
        "measurement_3c_3d": {k: v for k, v in m3cd.items()},
        "random_policy_outcomes": rand_policy,
    }


if __name__ == "__main__":
    import json

    from .generator import make_instance

    inst, info = make_instance(grid_n=5, k=3, alpha=0.25, width=4, w=0.25, seed=0, label="sweep_default")
    print(info)
    r = run(inst)
    print(json.dumps(r, indent=2, default=str))
    with open("/tmp/trmgrid_measurement3.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("\nWrote /tmp/trmgrid_measurement3.json")
