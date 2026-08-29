"""Measurement 3 -- decision relevance.

States at the same location are grouped by (the string form of) their
canonical zone (zones.py); if two groups' optimal-action sets are disjoint,
that's a CROSS-ZONE conflict: the DP needs clock information the location
alone doesn't provide.

A second, complementary statistic is also computed: WITHIN-ZONE
non-uniformity -- does a single zone (>1 concrete member state) itself
contain disjoint optimal-action sets? Under the original (unsound) zone
construction this was 40.65% and is the reason zones.py exists; after
Task A's fix it is the PRIMARY thing to check, reported first by the
caller (geometry_sweep.py / __main__ below), and must be 0% for the
abstraction to be sound.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .dp import DPResult, explore_and_solve
from .env import GridState, is_terminal, peek, target_of
from .instance import GridInstance
from .zones import build_state_zones, validate_containment


def multi_zone_conflict_analysis(instance: GridInstance, dp_result: DPResult = None, precomputed=None) -> dict:
    """``precomputed``, if given, is ``(dp_result, zctx, zones)`` already
    built by ``build_state_zones`` -- see geometry.analyze_geometry's
    matching parameter."""
    if precomputed is not None:
        dp_result, zctx, zones = precomputed
    else:
        dp_result, zctx, zones = build_state_zones(instance, dp_result)
    n_ok, n_bad = validate_containment(instance, zctx, zones)

    # Deadlock states (dp.py: every neighbour a closed gate -- possible for
    # Task B's generated layouts, not just a theoretical edge case, see
    # /tmp/GRIDSOUND.md) have an empty optimal_actions set by construction.
    # Excluded from conflict analysis here: two deadlocked groups sharing
    # "no valid action" is not a decision-relevant conflict, it's the same
    # (non-)decision, and counting it as disjoint-therefore-conflicting
    # would be a false positive.
    n_deadlock_excluded = 0
    by_loc: Dict[Tuple[int, int], Dict[str, List[GridState]]] = defaultdict(lambda: defaultdict(list))
    for s in dp_result.states:
        if is_terminal(s):
            continue
        if not dp_result.optimal_actions.get(s):
            n_deadlock_excluded += 1
            continue
        by_loc[s.pos][str(zones[s])].append(s)

    multi_zone_locations = 0
    conflicting_locations = 0
    conflict_examples = []

    n_multistate_zones = 0
    n_nonuniform_zones = 0
    within_zone_examples = []

    for loc, groups in by_loc.items():
        group_list = list(groups.items())

        for zs, states in group_list:
            if len(states) < 2:
                continue
            n_multistate_zones += 1
            # Same distinct-action-set bucketing as the cross-zone check
            # below -- guards against the same O(k^2) blowup if a zone
            # ever ends up with many concrete member states.
            action_set_to_rep: Dict[frozenset, GridState] = {}
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
                                {
                                    "loc": loc,
                                    "state_i": str(si), "state_j": str(sj),
                                    "clocks_i": si.clocks, "y_i": si.y,
                                    "clocks_j": sj.clocks, "y_j": sj.y,
                                    "actions_i": sorted(ai), "actions_j": sorted(aj),
                                    "gate_boundary_flip": _is_gate_boundary_flip(instance, loc, si, sj, ai, aj),
                                }
                            )
                        break
                if found:
                    break
            if found:
                n_nonuniform_zones += 1

        if len(groups) < 2:
            continue
        multi_zone_locations += 1

        # A location can have thousands of distinct zones once the sound
        # construction ties each zone ~1:1 to a concrete state (see
        # /tmp/GRIDSOUND.md Task A) -- an O(k^2) pairwise comparison over
        # every zone PAIR does not scale (this was measured directly: it
        # hung for tens of minutes on a single T2 seed with ~3000 zones at
        # one location, i.e. ~4.5M pairs). Fixed by first bucketing zones
        # by their (small, bounded by the fixed action set) UNIONED
        # action-set, and only comparing PAIRS OF DISTINCT ACTION-SETS
        # (typically single digits, never more than 2^|action set|) --
        # mathematically equivalent (whether ANY pair of zones has
        # disjoint action-sets depends only on which distinct action-sets
        # occur, not on how many zones share each one), just no longer
        # quadratic in the zone count.
        action_set_to_rep: Dict[frozenset, GridState] = {}
        for zs, states in group_list:
            acts = frozenset().union(*(set(dp_result.optimal_actions[s]) for s in states))
            if acts not in action_set_to_rep:
                action_set_to_rep[acts] = states[0]
        distinct_action_sets = list(action_set_to_rep.items())

        conflict = False
        for i in range(len(distinct_action_sets)):
            for j in range(i + 1, len(distinct_action_sets)):
                acts_i, rep_i = distinct_action_sets[i]
                acts_j, rep_j = distinct_action_sets[j]
                if not (acts_i & acts_j):
                    conflict = True
                    own_i = dp_result.optimal_actions[rep_i]
                    own_j = dp_result.optimal_actions[rep_j]
                    assert not (set(own_i) & set(own_j))
                    if len(conflict_examples) < 500:
                        conflict_examples.append(
                            {
                                "loc": loc,
                                "state_i": str(rep_i), "state_j": str(rep_j),
                                "clocks_i": rep_i.clocks, "y_i": rep_i.y,
                                "clocks_j": rep_j.clocks, "y_j": rep_j.y,
                                "actions_i": sorted(own_i), "actions_j": sorted(own_j),
                                "gate_boundary_flip": _is_gate_boundary_flip(instance, loc, rep_i, rep_j, own_i, own_j),
                            }
                        )
        if conflict:
            conflicting_locations += 1

    return {
        "n_states": len(dp_result.states),
        "truncated": dp_result.truncated,
        "n_horizon_fallback_refs": dp_result.n_horizon_fallback_refs,
        "n_deadlock_states": dp_result.n_deadlock_states,
        "n_deadlock_excluded_from_conflict_analysis": n_deadlock_excluded,
        "n_containment_ok": n_ok,
        "n_containment_bad": n_bad,
        "multi_zone_locations": multi_zone_locations,
        "conflicting_locations": conflicting_locations,
        "conflict_fraction": (conflicting_locations / multi_zone_locations) if multi_zone_locations else float("nan"),
        "conflict_examples": conflict_examples,
        "n_multistate_zones": n_multistate_zones,
        "n_nonuniform_zones": n_nonuniform_zones,
        "within_zone_nonuniform_fraction": (n_nonuniform_zones / n_multistate_zones) if n_multistate_zones else float("nan"),
        "within_zone_examples": within_zone_examples,
        "dp_result": dp_result,
        "zctx": zctx,
        "zones": zones,
    }


def _is_gate_boundary_flip(instance: GridInstance, loc: Tuple[int, int], si: GridState, sj: GridState,
                            acts_i, acts_j) -> dict:
    """Test (not assume) whether the action flip between two example states
    coincides with a GATE WINDOW boundary. Location-aware: a move only
    counts as gate-relevant if its target *from this specific location* is
    actually a gate cell. "Boundary crossing" is tested as a change in
    *reachability of the gate via some allowed delay* (peek legality for
    any delay in instance.allowed_delays), not merely whether the state's
    raw clock value currently sits inside the window."""
    moves_i = {a.split("_", 1)[1] for a in acts_i}
    moves_j = {a.split("_", 1)[1] for a in acts_j}
    differing_moves = moves_i.symmetric_difference(moves_j)

    gate_relevant = {}
    explained = False
    for move in differing_moves:
        target = target_of(loc, move)
        if not instance.in_bounds(target) or not instance.is_gate(target):
            continue
        reach_i = any(peek(instance, si, f"d{d}_{move}").legal for d in instance.allowed_delays)
        reach_j = any(peek(instance, sj, f"d{d}_{move}").legal for d in instance.allowed_delays)
        crosses = reach_i != reach_j
        gate_relevant[move] = {
            "target": target,
            "reachable_i": reach_i, "reachable_j": reach_j,
            "crosses_boundary": crosses,
        }
        explained = explained or crosses

    return {
        "differing_moves": sorted(differing_moves),
        "gate_relevant_moves": gate_relevant,
        "explained_by_gate_boundary": explained,
    }


if __name__ == "__main__":
    import json

    import zoneviews  # noqa: F401
    from .instance import DEFAULT_DELAYS, RESTRICTED_DELAYS

    out = {}
    for label, delays in [("default_delays", DEFAULT_DELAYS), ("restricted_delay0", RESTRICTED_DELAYS)]:
        inst = GridInstance(allowed_delays=delays)
        r = multi_zone_conflict_analysis(inst)
        print(f"\n=== {label} (allowed_delays={delays}) ===")
        print(f"n_states={r['n_states']} truncated={r['truncated']} "
              f"containment ok/bad={r['n_containment_ok']}/{r['n_containment_bad']}")
        print(f"multi_zone_locations={r['multi_zone_locations']} conflicting={r['conflicting_locations']} "
              f"conflict_fraction={r['conflict_fraction']:.4f}")
        print(f"n_multistate_zones={r['n_multistate_zones']} n_nonuniform_zones={r['n_nonuniform_zones']} "
              f"within_zone_nonuniform_fraction={r['within_zone_nonuniform_fraction']}")
        out[label] = {k: v for k, v in r.items() if k not in ("dp_result", "zctx", "zones")}

    with open("/tmp/gridgate_results/decision_relevance.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nWrote /tmp/gridgate_results/decision_relevance.json")
