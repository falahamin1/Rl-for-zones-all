"""Step 4 -- decision relevance: cross-zone conflict fraction and
within-zone non-uniformity (must be 0 for soundness), same methodology as
gridtimed/decision_relevance.py after its Task A soundness fix, ported to
this domain. Also reports mean concrete states per zone -- gridtimed's
failure mode was zones ending up ~1:1 with concrete states (no
compression); if that recurs here it is reported plainly, not hidden.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .diagnostics import _location_key
from .dp import DPResult
from .env import ModeSelState, is_terminal
from .instance import ModeSelInstance
from .zones import build_state_zones, validate_containment


def multi_zone_conflict_analysis(instance: ModeSelInstance, dp_result: DPResult = None, precomputed=None) -> dict:
    if precomputed is not None:
        dp_result, zctx, zones = precomputed
    else:
        dp_result, zctx, zones = build_state_zones(instance, dp_result)
    n_ok, n_bad = validate_containment(instance, zctx, zones)

    n_deadlock_excluded = 0
    by_loc: Dict[Tuple, Dict[str, List[ModeSelState]]] = defaultdict(lambda: defaultdict(list))
    for s in dp_result.states:
        if is_terminal(instance, s):
            continue
        if not dp_result.optimal_actions.get(s):
            n_deadlock_excluded += 1
            continue
        by_loc[_location_key(instance, s)][str(zones[s])].append(s)

    multi_zone_locations = 0
    conflicting_locations = 0
    conflict_examples = []
    n_multistate_zones = 0
    n_nonuniform_zones = 0
    within_zone_examples = []
    total_states_in_zones = 0
    total_zone_groups = 0

    for loc, groups in by_loc.items():
        group_list = list(groups.items())
        total_zone_groups += len(group_list)

        for zs, states in group_list:
            total_states_in_zones += len(states)
            if len(states) < 2:
                continue
            n_multistate_zones += 1
            action_set_to_rep: Dict[frozenset, ModeSelState] = {}
            for s in states:
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
                                {"loc": str(loc), "state_i": str(si), "state_j": str(sj),
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

        action_set_to_rep2: Dict[frozenset, ModeSelState] = {}
        for zs, states in group_list:
            acts = frozenset().union(*(set(dp_result.optimal_actions[s]) for s in states))
            if acts not in action_set_to_rep2:
                action_set_to_rep2[acts] = states[0]
        distinct2 = list(action_set_to_rep2.items())

        conflict = False
        for i in range(len(distinct2)):
            for j in range(i + 1, len(distinct2)):
                acts_i, rep_i = distinct2[i]
                acts_j, rep_j = distinct2[j]
                if not (acts_i & acts_j):
                    conflict = True
                    own_i = dp_result.optimal_actions[rep_i]
                    own_j = dp_result.optimal_actions[rep_j]
                    if len(conflict_examples) < 500:
                        conflict_examples.append(
                            {"loc": str(loc), "state_i": str(rep_i), "state_j": str(rep_j),
                             "actions_i": sorted(own_i), "actions_j": sorted(own_j)}
                        )
        if conflict:
            conflicting_locations += 1

    return {
        "n_states": len(dp_result.states),
        "truncated": dp_result.truncated,
        "n_containment_ok": n_ok,
        "n_containment_bad": n_bad,
        "n_deadlock_excluded": n_deadlock_excluded,
        "multi_zone_locations": multi_zone_locations,
        "conflicting_locations": conflicting_locations,
        "conflict_fraction": (conflicting_locations / multi_zone_locations) if multi_zone_locations else float("nan"),
        "conflict_examples": conflict_examples,
        "n_multistate_zones": n_multistate_zones,
        "n_nonuniform_zones": n_nonuniform_zones,
        "within_zone_nonuniform_fraction": (n_nonuniform_zones / n_multistate_zones) if n_multistate_zones else float("nan"),
        "within_zone_examples": within_zone_examples,
        "mean_states_per_zone": (total_states_in_zones / total_zone_groups) if total_zone_groups else float("nan"),
        "zones_abstracting_at_all": (total_states_in_zones / total_zone_groups) > 1.0 if total_zone_groups else False,
    }
