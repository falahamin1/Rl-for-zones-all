"""Section D: aggregation ratio experiment.

For each instance, run 200 seeded episodes under a uniform-random policy over
enabled actions. At each decision point record:
  concrete_key = (loc, exact virtual-clock valuation tuple)
  zone_key     = (loc, str(Z))   [same key form as symbolic_rl.py]
Report distinct concrete keys, distinct zone keys, aggregation ratio,
location-only control ratio, and fraction of zones containing exactly one
concrete state.
"""
import pyudbm
import sys, random, json
sys.path.insert(0, '/Users/falah/Desktop/git-projects/RL-for-zones')

from prob_jobshop_ext.benchmarks.all_instances import get_instance_by_name
from prob_jobshop_ext.zone_ops import (
    ZoneContext, init_zone, apply_guard, reset_clocks, apply_delay, normalize, min_cost
)
from prob_jobshop_ext.symbolic_env import SymbolicJobShopEnv

INSTANCES = ["PJS_01", "PJS_03", "PJS_06"]
N_EPISODES = 200
SEED = 777


def propagate_zone_branch(Z, zctx, env, action, sampled_d, M):
    """Propagate Z through the exact sampled branch of `action` (mirrors
    SymbolicRLAgent._propagate_zone but only for the matching branch)."""
    Zg = apply_guard(Z, zctx, action["guards"])
    if Zg is None:
        return None
    Zr = reset_clocks(Zg, zctx, action["resets"])
    branch = next((b for b in action["branches"] if b["d"] == sampled_d), None)
    if branch is None:
        return None
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
        return None
    return normalize(Zcurr, zctx, M)


def run_instance(name):
    inst = get_instance_by_name(name)
    env = SymbolicJobShopEnv(inst, seed=SEED)
    zctx = ZoneContext(inst)
    M = inst.worst_case_makespan_upper_bound()
    rng = random.Random(SEED)

    concrete_keys = set()
    zone_keys = set()
    locs_visited = set()
    zone_to_concrete = {}  # zone_key -> set of concrete_keys observed at it

    for ep in range(N_EPISODES):
        loc, actions = env.reset()
        Z = init_zone(zctx, env.invariant(loc))
        Z = apply_delay(Z, zctx, env.invariant(loc), 1)
        Z = normalize(Z, zctx, M)

        while True:
            # concrete key: (loc, exact virtual-clock valuation tuple)
            vc_tuple = tuple(sorted((jid, env._virtual_clocks.get(jid, 0.0))
                                     for jid in env._virtual_clocks))
            concrete_key = (loc, vc_tuple)
            zone_key = (loc, str(Z))

            concrete_keys.add(concrete_key)
            zone_keys.add(zone_key)
            locs_visited.add(loc)
            zone_to_concrete.setdefault(zone_key, set()).add(concrete_key)

            if not actions:
                break
            chosen = rng.choice(actions)
            loc_new, actions_new, done, info = env.step(chosen["action_id"])
            sampled_d = info["sampled_d"]

            Z_next = propagate_zone_branch(Z, zctx, env, chosen, sampled_d, M)
            if Z_next is None:
                # shouldn't happen since we executed the exact sampled branch
                break

            loc, actions, Z = loc_new, actions_new, Z_next
            if done:
                # record terminal state too
                vc_tuple = tuple(sorted((jid, env._virtual_clocks.get(jid, 0.0))
                                         for jid in env._virtual_clocks))
                concrete_key = (loc, vc_tuple)
                zone_key = (loc, str(Z))
                concrete_keys.add(concrete_key)
                zone_keys.add(zone_key)
                locs_visited.add(loc)
                zone_to_concrete.setdefault(zone_key, set()).add(concrete_key)
                break

    n_concrete = len(concrete_keys)
    n_zone = len(zone_keys)
    n_locs = len(locs_visited)
    agg_ratio = n_concrete / n_zone if n_zone else None
    loc_ratio = n_concrete / n_locs if n_locs else None
    n_singleton_zones = sum(1 for s in zone_to_concrete.values() if len(s) == 1)
    frac_singleton = n_singleton_zones / n_zone if n_zone else None

    result = {
        "instance": name,
        "n_episodes": N_EPISODES,
        "distinct_concrete_keys": n_concrete,
        "distinct_zone_keys": n_zone,
        "distinct_locations": n_locs,
        "aggregation_ratio_concrete_over_zone": agg_ratio,
        "location_only_control_ratio": loc_ratio,
        "frac_zones_with_exactly_one_concrete_state": frac_singleton,
        "zone_ratio_exceeds_location_control": (agg_ratio > loc_ratio) if (agg_ratio and loc_ratio) else None,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    all_results = [run_instance(n) for n in INSTANCES]
    with open("/tmp/env_audit_jobshop/aggregation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("DONE")
