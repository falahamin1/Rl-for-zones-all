"""SPIKE3: Task B grid — w in {0.1,0.25} x t in {0.0,0.3,None}, across
PJS_01-05, LJS_01. Real-valued interval durations + deadlines.
THROWAWAY, /tmp only.
"""
from __future__ import annotations
import sys
import json
import statistics

import pyudbm as _warmup_pyudbm  # noqa: F401
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")
sys.path.append("/tmp")

import numpy as np

from zone_common import make_propagate_one
from spike3_common import get_base_instance, make_deadline_variant

import prob_jobshop_spike.symbolic_env as sym_env
import prob_jobshop_spike.zone_ops as zone_ops

INSTANCES = ["PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05", "LJS_01"]
WIDTHS = [0.1, 0.25]
TIGHTNESS = [0.0, 0.3, None]
N_EPISODES = 200
MAX_STEPS = 5000

propagate_one = make_propagate_one(zone_ops)


def run_random_episode(instance, w, seed):
    sym_env.INTERVAL_WIDTH = w
    env = sym_env.SymbolicJobShopEnv(instance, seed=seed)
    zctx = zone_ops.ZoneContext(instance)
    M = int(round(instance.worst_case_makespan_upper_bound() * (1 + w) * sym_env.SCALE))
    job_ids = [j.job_id for j in instance.jobs]
    policy_rng = np.random.default_rng(seed * 1_000_003 + 7)

    loc, actions = env.reset()
    Z = zone_ops.init_zone(zctx, env.invariant(loc))
    Z = zone_ops.apply_delay(Z, zctx, env.invariant(loc), env.rate(loc))
    Z = zone_ops.normalize(Z, zctx, M)

    records = []
    propagate_failure = False

    for _ in range(MAX_STEPS):
        concrete_key = (loc, tuple(round(env._virtual_clocks[jid], 9) for jid in job_ids))
        zone_key = (loc, str(Z))
        records.append((concrete_key, zone_key, loc))

        if not actions:
            break

        idx = policy_rng.integers(len(actions))
        action = actions[idx]
        task_id = action["action_id"]

        loc_new, actions_new, done, info = env.step(task_id)
        sampled_d = info["sampled_d"]

        result = propagate_one(env, zctx, Z, M, action, sampled_d)
        if result is None:
            propagate_failure = True
            break
        Z, dec_loc = result

        loc = loc_new
        actions = actions_new
        if done:
            break

    return records, propagate_failure, env._last_deadline_violated


def run_one(name, w, t):
    base = get_base_instance(name)
    inst = make_deadline_variant(base, t)

    all_records = []
    any_failure = False
    n_deadline_violated = 0
    for ep in range(N_EPISODES):
        recs, fail, dv = run_random_episode(inst, w, seed=ep)
        all_records.extend(recs)
        any_failure = any_failure or fail
        if dv:
            n_deadline_violated += 1

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
    n_distinct_concrete = len(concrete_keys)
    n_distinct_zone = len(zone_keys)
    n_distinct_loc = len(locs)

    frac_deadline_violated = n_deadline_violated / N_EPISODES

    row = {
        "instance": name,
        "w": w,
        "t": t,
        "n_records": len(all_records),
        "propagate_failure": any_failure,
        "distinct_concrete": n_distinct_concrete,
        "distinct_zone": n_distinct_zone,
        "distinct_loc": n_distinct_loc,
        "aggregation_ratio_zone": n_distinct_concrete / n_distinct_zone,
        "aggregation_ratio_loc_control": n_distinct_concrete / n_distinct_loc,
        "cluster_size_mean": float(np.mean(sizes)),
        "cluster_size_median": float(statistics.median(sizes)),
        "cluster_size_max": int(max(sizes)),
        "frac_cluster_size_1": float(np.mean([s == 1 for s in sizes])),
        "frac_episodes_deadline_violated": frac_deadline_violated,
        "deadline_knob_flag": (
            "ALWAYS-VIOLATED (100%)" if frac_deadline_violated == 1.0 else
            "NEVER-VIOLATED (0%)" if frac_deadline_violated == 0.0 else
            None
        ) if t is not None else "N/A (no deadline)",
    }
    return row


if __name__ == "__main__":
    all_rows = []
    for name in INSTANCES:
        for w in WIDTHS:
            for t in TIGHTNESS:
                row = run_one(name, w, t)
                all_rows.append(row)
                print(
                    f"{name:8s} w={w:.2f} t={str(t):5s} records={row['n_records']:6d} "
                    f"fail={row['propagate_failure']!s:5s} "
                    f"distinct(concrete/zone/loc)={row['distinct_concrete']}/{row['distinct_zone']}/{row['distinct_loc']} "
                    f"agg_ratio_zone={row['aggregation_ratio_zone']:.3f} "
                    f"agg_ratio_loc_control={row['aggregation_ratio_loc_control']:.3f} "
                    f"cluster(mean/median/max)={row['cluster_size_mean']:.3f}/{row['cluster_size_median']:.1f}/{row['cluster_size_max']} "
                    f"frac_size1={row['frac_cluster_size_1']:.3f} "
                    f"deadline_viol_frac={row['frac_episodes_deadline_violated']:.3f} "
                    f"flag={row['deadline_knob_flag']}",
                    flush=True,
                )

    with open("/tmp/spike3_taskB_results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print("Saved /tmp/spike3_taskB_results.json")
