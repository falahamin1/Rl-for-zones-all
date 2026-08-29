"""Section 5: decision relevance -- do two zones at the same location ever
require different optimal actions?

Extends ``dp.py``'s concrete-state exploration with a symbolic zone built
in lockstep for every visited state (same replay technique as
``collect.py``, just driven by the DP's exhaustive branch-over-every-
duration traversal instead of sampled rollouts, so every reachable
concrete state gets a zone rather than only ones a random policy happened
to visit). States are grouped by ``loc``; within a location, states are
further grouped by their zone's string representation (mirroring
``symbolic_rl.py``'s ``(loc, str(Z))`` key from the earlier no-deadline
audits, for direct comparability) -- a "multi-zone location" is one with
2+ such distinct zone-string groups, and a "conflict" is one where the
DP's optimal-action *set* differs (no common action) between two of those
groups.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from zoneviews.zone import Zone

from .dp import DPResult, StateKey, explore_and_solve, state_key
from .instance import SCALE, DeadlineJobShopInstance, nominal_duration
from .symbolic_env import DeadlineSymbolicJobShopEnv
from .zone_ops import DeadlineZoneContext, apply_delay, apply_invariant, apply_lower_guard, reset_clocks


def _invariant_of_loc(env: DeadlineSymbolicJobShopEnv, loc: tuple) -> List[Tuple[str, int]]:
    result = []
    for tid, status in zip(env._task_ids, loc):
        if status.startswith("active_d"):
            job_id = env._task_job[tid]
            exact_d = int(status[len("active_d"):])
            result.append((f"c_{job_id}", exact_d * SCALE))
    return result


def explore_with_zones(instance: DeadlineJobShopInstance, state_limit: int = 200_000):
    """Re-run the DP exploration (same BFS shape as dp.explore_and_solve)
    while also building each state's Federation by replay. Returns
    (DPResult, {state_key: Federation})."""
    zctx = DeadlineZoneContext(instance)
    result = explore_and_solve(instance, state_limit=state_limit)

    env = DeadlineSymbolicJobShopEnv(instance, seed=0)
    zones: Dict[StateKey, "object"] = {}

    env.restore_state(result.snapshots[result.root_key])
    zones[result.root_key] = zctx.ctx.get_zero_federation()

    from collections import deque

    order = deque([result.root_key])
    seen = {result.root_key}
    while order:
        key = order.popleft()
        Z_here = zones[key]
        trans = result.transitions.get(key, {})
        for tid, branches in trans.items():
            for _prob, _elapsed, _penalty, nkey in branches:
                if nkey in zones:
                    continue
                env.restore_state(result.snapshots[key])
                job_id = env._task_job[tid]
                loc_before_dispatch = list(env._loc)
                # A task's duration branches can, in principle, land on the
                # same next state_key (e.g. two different durations both
                # immediately failing some other job identically) -- so
                # match by replaying each branch and comparing the
                # resulting state_key, not by position.
                action = next(a for a in result.actions[key] if a["action_id"] == tid)
                Z_new = None
                for br in action["branches"]:
                    env.restore_state(result.snapshots[key])
                    d = br["d"]
                    loc_after_dispatch = list(loc_before_dispatch)
                    loc_after_dispatch[env._task_to_idx[tid]] = f"active_d{d}"

                    Zc = reset_clocks(Z_here, zctx, [zctx.c(job_id)])
                    Zc = apply_invariant(Zc, zctx, _invariant_of_loc(env, tuple(loc_after_dispatch)))

                    env.restore_state(result.snapshots[key])
                    _loc2, _actions2, _done2, _info2 = env.step_deterministic(tid, d)
                    for step in env._last_advance_path:
                        inv_before = _invariant_of_loc(env, step["loc_before"])
                        Zc = apply_delay(Zc, zctx, inv_before)
                        for ctid in step["completing_tasks"]:
                            status = step["loc_before"][env._task_to_idx[ctid]]
                            exact_d = int(status[len("active_d"):])
                            cjid = env._task_job[ctid]
                            Zc = apply_lower_guard(Zc, zctx, [(zctx.c(cjid), exact_d * SCALE)])
                        if Zc is None or Zc.is_empty():
                            raise RuntimeError("decision_relevance: zone unexpectedly empty during DP replay")

                    if state_key(env) == nkey:
                        Z_new = Zc
                        break

                if Z_new is None:
                    raise RuntimeError(f"decision_relevance: could not match branch for {tid} at {key}")

                zones[nkey] = Z_new
                if nkey not in seen:
                    seen.add(nkey)
                    order.append(nkey)

    return result, zones, zctx


def multi_zone_conflict_analysis(
    instance: DeadlineJobShopInstance, m_bound_scale: float = 1.0, state_limit: int = 200_000
) -> dict:
    result, zones, zctx = explore_with_zones(instance, state_limit=state_limit)
    m_bound = int(instance.worst_case_makespan_upper_bound() * SCALE * m_bound_scale) + SCALE
    k_vec = tuple([m_bound] * (2 * len(instance.jobs)))

    by_loc: Dict[tuple, Dict[str, List[StateKey]]] = defaultdict(lambda: defaultdict(list))
    for key, Z in zones.items():
        loc = key[0]
        if key not in result.optimal_actions:
            continue  # terminal states have no action to compare
        zone = Zone(Z, zctx.clock_names, k_vec).clip_to_k_box()
        zone_str = str(zone.federation)
        by_loc[loc][zone_str].append(key)

    multi_zone_locations = 0
    conflicting_locations = 0
    conflict_examples = []

    for loc, zone_groups in by_loc.items():
        if len(zone_groups) < 2:
            continue
        multi_zone_locations += 1

        group_action_sets = []
        for zone_str, keys in zone_groups.items():
            action_sets = [frozenset(result.optimal_actions[k]) for k in keys]
            group_action_sets.append((zone_str, keys, action_sets))

        conflict = False
        for i in range(len(group_action_sets)):
            for j in range(i + 1, len(group_action_sets)):
                zs_i, keys_i, actsets_i = group_action_sets[i]
                zs_j, keys_j, actsets_j = group_action_sets[j]
                # conflict iff EVERY pair of (state in i, state in j) has
                # disjoint optimal-action sets (mirrors earlier audits'
                # "two zones require different optimal actions").
                any_overlap = any(a & b for a in actsets_i for b in actsets_j)
                if not any_overlap:
                    conflict = True
                    conflict_examples.append(
                        {
                            "loc": loc,
                            "zone_i": zs_i,
                            "zone_j": zs_j,
                            "state_i": keys_i[0],
                            "state_j": keys_j[0],
                            "actions_i": sorted(result.optimal_actions[keys_i[0]]),
                            "actions_j": sorted(result.optimal_actions[keys_j[0]]),
                        }
                    )
        if conflict:
            conflicting_locations += 1

    return {
        "n_states": len(result.snapshots),
        "truncated": result.truncated,
        "multi_zone_locations": multi_zone_locations,
        "conflicting_locations": conflicting_locations,
        "conflict_fraction": (conflicting_locations / multi_zone_locations) if multi_zone_locations else float("nan"),
        "conflict_examples": conflict_examples,
        "dp_result": result,
        "zones": zones,
        "zctx": zctx,
    }


def slack(instance: DeadlineJobShopInstance, job_id: str, g_value: float, remaining_task_ids: List[str]) -> float:
    """slack_J = (D_J - g_J) - W_J, W_J = nominal work of the REMAINING
    tasks (from wherever the job currently is), per the task's formula."""
    w = sum(nominal_duration(instance.task_by_id(t)) for t in remaining_task_ids)
    return (instance.deadline_of(job_id) - g_value) - w
