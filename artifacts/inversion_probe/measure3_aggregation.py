"""Measurement 3: AGGREGATION RATIO + location-only control, same protocol
as Spike 3 (200 seeded random-policy episodes, concrete_key vs zone_key),
with zone_key computed on the CLIPPED zone (clip_to_k_box), as instructed.
THROWAWAY, /tmp only.
"""
import pyudbm as _warmup_pyudbm  # noqa: F401
import sys
import statistics
sys.path.append("/tmp/inversion_probe")
sys.path.append("/tmp")
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")

import numpy as np
import prob_jobshop_spike.zone_ops as zone_ops
from zone_common import make_propagate_one
from model import MicroEnv, MicroZoneContext, _scaled, M_REAL

N_EPISODES = 200
propagate_one = make_propagate_one(zone_ops)


def run_random_episode(deadline, seed):
    env = MicroEnv(deadline=deadline, seed=seed)
    zctx = MicroZoneContext()
    M = _scaled(M_REAL)
    bounds = {cn: M for cn in zctx.clock_names}
    policy_rng = np.random.default_rng(seed * 1_000_003 + 7)

    loc, actions = env.reset_concrete(seed=seed)
    Z = zone_ops.init_zone(zctx, env.invariant(loc))
    Z = zone_ops.apply_delay(Z, zctx, env.invariant(loc), env.rate(loc))
    Z = zone_ops.normalize(Z, zctx, M)

    records = []
    propagate_failure = False

    for _ in range(10):
        Z_clip = zone_ops.clip_to_k_box(Z, zctx, bounds)
        concrete_key = (loc, round(env._c, 9), round(env._g, 9))
        zone_key = (loc, str(Z_clip))
        records.append((concrete_key, zone_key, loc))

        if not actions:
            break

        action = actions[policy_rng.integers(len(actions))]
        action_id = action["action_id"]
        loc_new, actions_new, done, info = env.step(action_id)
        bucket_d = info["sampled_bucket"]

        res = propagate_one(env, zctx, Z, M, action, bucket_d)
        if res is None:
            propagate_failure = True
            break
        Z, dec_loc = res

        loc, actions = loc_new, actions_new
        if done:
            Z_clip = zone_ops.clip_to_k_box(Z, zctx, bounds)
            records.append(((loc, round(env._c, 9), round(env._g, 9)), (loc, str(Z_clip)), loc))
            break

    return records, propagate_failure


def run(deadline=12.0, n_episodes=N_EPISODES):
    all_records = []
    any_failure = False
    for ep in range(n_episodes):
        recs, fail = run_random_episode(deadline, seed=ep)
        all_records.extend(recs)
        any_failure = any_failure or fail

    concrete_keys = set()
    zone_keys = set()
    locs = set()
    cluster: dict = {}
    for ck, zk, loc in all_records:
        concrete_keys.add(ck)
        zone_keys.add(zk)
        locs.add(loc)
        cluster.setdefault(zk, set()).add(ck)

    sizes = [len(v) for v in cluster.values()]
    n_c = len(concrete_keys)
    n_z = len(zone_keys)
    n_l = len(locs)

    by_loc_concrete: dict = {}
    by_loc_zone: dict = {}
    for ck, zk, loc in all_records:
        by_loc_concrete.setdefault(loc, set()).add(ck)
        by_loc_zone.setdefault(loc, set()).add(zk)
    per_location = {
        loc: {
            "distinct_concrete": len(by_loc_concrete[loc]),
            "distinct_zone": len(by_loc_zone[loc]),
            "ratio": len(by_loc_concrete[loc]) / len(by_loc_zone[loc]),
        }
        for loc in by_loc_concrete
    }

    return {
        "deadline": deadline,
        "n_records": len(all_records),
        "propagate_failure": any_failure,
        "distinct_concrete": n_c,
        "distinct_zone": n_z,
        "distinct_loc": n_l,
        "aggregation_ratio_zone": n_c / n_z,
        "aggregation_ratio_loc_control": n_c / n_l,
        "cluster_size_mean": float(np.mean(sizes)),
        "cluster_size_median": float(statistics.median(sizes)),
        "cluster_size_max": int(max(sizes)),
        "frac_cluster_size_1": float(np.mean([s == 1 for s in sizes])),
        "per_location": per_location,
    }


if __name__ == "__main__":
    r = run(deadline=12.0)
    print(r)
    print()
    print(f"aggregation_ratio_zone = {r['aggregation_ratio_zone']:.3f}")
    print(f"aggregation_ratio_loc_control = {r['aggregation_ratio_loc_control']:.3f}")
    if r["aggregation_ratio_zone"] > r["aggregation_ratio_loc_control"]:
        print("CONFIRMED: zone ratio exceeds location-only control, as expected "
              "(location alone cannot separate tau<=4 from tau>4).")
    else:
        print("!!! ANOMALY: zone ratio does NOT exceed the location-only control -- "
              "unexpected given the design, flagging per instructions.")
