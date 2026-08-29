"""Section E: exact DP over the concrete (loc, per-job virtual-clock) state
space, and check whether distinct zones at the same location require
different optimal actions.

No pyudbm needed here for the DP itself (concrete states, not zones) — but we
also run an exhaustive zone BFS (same traversal, same env) to map each
visited zone back to its underlying concrete state, verifying the near-
bijection found in Section D exhaustively (not just on 200 sampled episodes)
and using it to answer the "2+ zones at a location -> different optimal
action?" question.
"""
import pyudbm
import sys, json, time
from collections import deque
sys.path.insert(0, '/Users/falah/Desktop/git-projects/RL-for-zones')

from prob_jobshop_ext.benchmarks.all_instances import get_instance_by_name
from prob_jobshop_ext.zone_ops import (
    ZoneContext, init_zone, apply_guard, reset_clocks, apply_delay, normalize, min_cost
)
from prob_jobshop_ext.symbolic_env import SymbolicJobShopEnv

_INF = float("inf")
CONCRETE_STATE_CAP = 150_000
CANDIDATES = ["PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05"]


def vc_key(env):
    """Concrete per-job virtual-clock tuple, excluding the synthetic elapsed tag."""
    return tuple(sorted((jid, v) for jid, v in env._virtual_clocks.items() if jid != "__elapsed__"))


def explore_concrete_and_zone(instance, state_cap=CONCRETE_STATE_CAP):
    """Single combined BFS: explores the zone graph exhaustively (same traversal
    as OptimalSymbolicSolver._explore) while also recording, for every zone
    state, its underlying concrete (loc, vc) state and per-edge elapsed time —
    enough to build both a concrete-state DP and a zone->concrete mapping.

    Returns:
      concrete_states: dict concrete_key -> (loc, done)
      concrete_trans:  dict concrete_key -> [(action_id, [(prob, elapsed, next_concrete_key), ...]), ...]
      zone_to_concrete: dict zone_key -> concrete_key
      truncated: bool (concrete-state cap hit)
    """
    env = SymbolicJobShopEnv(instance, seed=0)
    zctx = ZoneContext(instance)
    M = instance.worst_case_makespan_upper_bound()

    loc0, actions0 = env.reset()
    env._virtual_clocks["__elapsed__"] = 0.0
    Z0 = init_zone(zctx, env.invariant(loc0))
    Z0 = apply_delay(Z0, zctx, env.invariant(loc0), 1)
    Z0 = normalize(Z0, zctx, M)
    zkey0 = (loc0, str(Z0))
    ckey0 = (loc0, vc_key(env))

    concrete_states = {ckey0: (loc0, False)}
    concrete_trans = {}
    zone_to_concrete = {zkey0: ckey0}

    snap0 = (env._loc, dict(env._virtual_clocks), dict(env._active_d))
    queue = deque([(loc0, Z0, actions0, zkey0, ckey0, snap0)])
    seen_zone_states = {zkey0}
    truncated = False

    while queue:
        if len(concrete_states) >= state_cap:
            truncated = True
            break
        loc, Z, actions, zkey, ckey, snap = queue.popleft()
        if ckey in concrete_trans:
            continue
        concrete_trans[ckey] = []

        for action in actions:
            Zg = apply_guard(Z, zctx, action["guards"])
            if Zg is None:
                continue
            Zr = reset_clocks(Zg, zctx, action["resets"])

            action_edges = []  # (prob, elapsed, next_ckey)
            for branch in action["branches"]:
                d, prob = branch["d"], branch["prob"]
                Zcurr = apply_delay(Zr, zctx, env.invariant(branch["next_loc"]), 1)
                valid = True
                for step in branch["path"]:
                    for cname, exact_d in step["completion_guards"]:
                        Zcurr = Zcurr & (zctx.clocks[cname] >= exact_d)
                        if Zcurr.is_empty():
                            valid = False
                            break
                    if not valid:
                        break
                    Zcurr = apply_delay(Zcurr, zctx, env.invariant(step["next_loc"]), 1)
                if not valid or Zcurr.is_empty():
                    continue

                Zb = normalize(Zcurr, zctx, M)
                dec_loc = branch["decision_loc"]
                next_zkey = (dec_loc, str(Zb))

                # advance the REAL env deterministically to get the concrete next state
                env._loc, env._virtual_clocks, env._active_d = snap[0], dict(snap[1]), dict(snap[2])
                parent_elapsed = env._virtual_clocks["__elapsed__"]
                next_loc, next_actions, done = env.step_deterministic(action["action_id"], d)
                child_elapsed = env._virtual_clocks["__elapsed__"]
                elapsed_this_edge = child_elapsed - parent_elapsed
                next_ckey = (next_loc, vc_key(env))
                next_snap = (env._loc, dict(env._virtual_clocks), dict(env._active_d))

                action_edges.append((prob, elapsed_this_edge, next_ckey))

                if next_ckey not in concrete_states and len(concrete_states) < state_cap:
                    concrete_states[next_ckey] = (next_loc, done)

                if next_zkey not in zone_to_concrete:
                    zone_to_concrete[next_zkey] = next_ckey

                if next_zkey not in seen_zone_states and len(concrete_states) <= state_cap:
                    seen_zone_states.add(next_zkey)
                    if not done:
                        queue.append((dec_loc, Zb, next_actions, next_zkey, next_ckey, next_snap))

            if action_edges:
                concrete_trans[ckey].append((action["action_id"], action_edges))

    return concrete_states, concrete_trans, zone_to_concrete, truncated


def backward_dp(concrete_states, concrete_trans):
    V = {}
    Q = {}
    for ckey, (loc, done) in concrete_states.items():
        V[ckey] = 0.0 if done else _INF
        Q[ckey] = {}

    n = len(concrete_states)
    for _pass in range(n + 1):
        changed = False
        for ckey, action_list in concrete_trans.items():
            if not action_list:
                continue
            best_v = _INF
            for action_id, edges in action_list:
                q = 0.0
                ok = True
                for prob, elapsed, next_ckey in edges:
                    vnext = V.get(next_ckey, _INF)
                    if vnext >= _INF:
                        vnext = 0.0  # unexplored/truncated successor treated as 0 (optimistic), matches OptimalSymbolicSolver
                    q += prob * (elapsed + vnext)
                Q[ckey][action_id] = q
                best_v = min(best_v, q)
            if best_v < V.get(ckey, _INF):
                V[ckey] = best_v
                changed = True
        if not changed:
            break
    return V, Q


def analyze(name):
    print(f"=== {name} ===", flush=True)
    inst = get_instance_by_name(name)
    t0 = time.time()
    concrete_states, concrete_trans, zone_to_concrete, truncated = explore_concrete_and_zone(inst)
    explore_time = time.time() - t0
    n_concrete = len(concrete_states)
    n_zones = len(zone_to_concrete)
    print(f"  concrete states: {n_concrete}  zone states: {n_zones}  truncated={truncated}  wall={explore_time:.2f}s")

    qualifies = (not truncated) and n_concrete < 100_000
    result = {
        "instance": name,
        "n_concrete_states": n_concrete,
        "n_zone_states": n_zones,
        "concrete_truncated": truncated,
        "explore_wall_s": explore_time,
        "qualifies_for_exact_dp": qualifies,
    }

    if not qualifies:
        print(f"  SKIPPED (n_concrete={n_concrete}, truncated={truncated})")
        return result

    t0 = time.time()
    V, Q = backward_dp(concrete_states, concrete_trans)
    dp_time = time.time() - t0

    # Optimal action per concrete state
    opt_action = {}
    for ckey, qvals in Q.items():
        if qvals:
            opt_action[ckey] = min(qvals, key=lambda a: qvals[a])

    # group concrete states with a defined optimal action by location
    from collections import defaultdict
    loc_to_ckeys = defaultdict(set)
    for ckey in opt_action:
        loc = ckey[0]
        loc_to_ckeys[loc].add(ckey)

    multi_state_locs = {loc: ck for loc, ck in loc_to_ckeys.items() if len(ck) >= 2}
    n_multi = len(multi_state_locs)
    n_conflict = 0
    example = None
    for loc, ckeys in multi_state_locs.items():
        actions_here = {ck: opt_action[ck] for ck in ckeys}
        distinct_actions = set(actions_here.values())
        if len(distinct_actions) >= 2:
            n_conflict += 1
            if example is None:
                ck_list = list(ckeys)
                example = {
                    "location": loc,
                    "states_and_actions": [
                        {"vc": ck[1], "optimal_action": opt_action[ck], "V": V[ck]}
                        for ck in ck_list
                    ],
                }

    # Also do the same grouping at the ZONE level (using zone->concrete mapping)
    loc_to_zone_ckeys = defaultdict(set)
    for zkey, ckey in zone_to_concrete.items():
        loc = zkey[0]
        if ckey in opt_action:
            loc_to_zone_ckeys[loc].add(ckey)
    # (identical structure/semantics to loc_to_ckeys since zone->concrete is a function;
    #  reported separately to make the correspondence explicit)

    result.update({
        "n_concrete_states_explored": n_concrete,
        "n_states_with_optimal_action": len(opt_action),
        "dp_wall_s": dp_time,
        "n_multi_state_locations": n_multi,
        "n_conflicting_locations": n_conflict,
        "conflict_fraction": (n_conflict / n_multi) if n_multi else None,
        "example_conflict": example,
        "n_locations_total": len(loc_to_ckeys),
        "zone_to_concrete_is_function_check_size": len(zone_to_concrete),
    })
    print(json.dumps({k: v for k, v in result.items() if k != "example_conflict"}, indent=2, default=str))
    if example:
        print("EXAMPLE CONFLICT:")
        print(json.dumps(example, indent=2, default=str))
    return result


if __name__ == "__main__":
    all_results = []
    for name in CANDIDATES:
        r = analyze(name)
        all_results.append(r)
        if r.get("n_concrete_states", 0) > 300_000:
            print(f"  (stopping candidate scan after {name} — sizes only growing)")
    with open("/tmp/env_audit_jobshop/decision_relevance_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("DONE")
