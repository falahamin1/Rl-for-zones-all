"""Measurement 2: decision relevance -- do two zones at the same location
ever require different optimal actions? gate_track version: single global
g, interval-based zones (see collect.py), interval-aware slack formula.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Tuple

from zoneviews.zone import Zone

from .dp import DPResult, StateKey, explore_and_solve, state_key
from .instance import SCALE, GateJobShopInstance, interval, nominal_duration
from .symbolic_env import GateSymbolicJobShopEnv
from .zone_ops import GateZoneContext, apply_delay, apply_invariant, apply_lower_guard, reset_clocks


def _invariant_of_loc(env: GateSymbolicJobShopEnv, loc: tuple) -> List[Tuple[str, int]]:
    result = []
    for tid, status in zip(env._task_ids, loc):
        if status.startswith("active_d"):
            job_id = env._task_job[tid]
            nominal_d = int(status[len("active_d"):])
            _lo, hi = interval(nominal_d, env.w)
            result.append((f"c_{job_id}", int(round(hi * SCALE))))
    return result


def explore_with_zones(instance: GateJobShopInstance, state_limit: int = 200_000):
    zctx = GateZoneContext(instance)
    result = explore_and_solve(instance, state_limit=state_limit)

    env = GateSymbolicJobShopEnv(instance, seed=0)
    zones: Dict[StateKey, "object"] = {}

    env.restore_state(result.snapshots[result.root_key])
    zones[result.root_key] = zctx.ctx.get_zero_federation()

    order = deque([result.root_key])
    seen = {result.root_key}
    while order:
        key = order.popleft()
        Z_here = zones[key]
        trans = result.transitions.get(key, {})
        for tid, branches in trans.items():
            action = next(a for a in result.actions[key] if a["action_id"] == tid)
            for _prob, _elapsed, _penalty, nkey in branches:
                if nkey in zones:
                    continue
                env.restore_state(result.snapshots[key])
                job_id = env._task_job[tid]
                loc_before_dispatch = list(env._loc)

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
                            nominal_d = int(status[len("active_d"):])
                            lo, _hi = interval(nominal_d, env.w)
                            cjid = env._task_job[ctid]
                            Zc = apply_lower_guard(Zc, zctx, [(zctx.c(cjid), int(round(lo * SCALE)))])
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


def multi_zone_conflict_analysis(instance: GateJobShopInstance, state_limit: int = 200_000) -> dict:
    result, zones, zctx = explore_with_zones(instance, state_limit=state_limit)
    m_bound = int(instance.worst_case_makespan_upper_bound() * SCALE) + SCALE
    k_vec = tuple([m_bound] * (len(instance.jobs) + 1))

    by_loc: Dict[tuple, Dict[str, List[StateKey]]] = defaultdict(lambda: defaultdict(list))
    for key, Z in zones.items():
        loc = key[0]
        if key not in result.optimal_actions:
            continue
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


def slack(instance: GateJobShopInstance, job_id: str, g_value: float, c_value: float,
          active_nominal_d: "int | None", remaining_task_ids: List[str]) -> float:
    """slack_J = (D_J - g) - W_J(c_J), W_J = interval-aware remaining work:
    (hi - c_J) for the currently-active task (hi from its nominal duration's
    interval) plus nominal duration for every other remaining
    (not-yet-dispatched) task. If there is no active task, W_J is just the
    sum of nominal durations of the remaining tasks (deadline_track's
    original formula)."""
    remaining_ids = list(remaining_task_ids)
    w_j = 0.0
    if active_nominal_d is not None:
        _lo, hi = interval(active_nominal_d, instance.width)
        w_j += hi - c_value
        # the active task is remaining_ids[0] in this convention -- drop it
        # from the "nominal for the rest" sum.
        remaining_ids = remaining_ids[1:]
    w_j += sum(nominal_duration(instance.task_by_id(t)) for t in remaining_ids)
    return (instance.deadline_of(job_id) - g_value) - w_j
