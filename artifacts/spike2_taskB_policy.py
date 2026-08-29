"""SPIKE2 Task B (continued): policy-consistency check for PJS_01-03.

Builds a FRESH exact exhaustive DP over CONCRETE (loc, exact-clock-tuple)
states (not the zone graph) via SymbolicJobShopEnv.step_deterministic(),
branching over every possible sampled duration at each dispatch — same
overall shape as OptimalSymbolicSolver's zone-graph DP, but keyed by exact
concrete states, and using the corrected (elapsed-based, non-cumulative)
Bellman-Ford-style relaxation (V initialised to +inf, only ever decreases
when a genuinely better path is found — this is standard-correct and
deliberately avoids the "treat unresolved child as 0" bug found in
OptimalSymbolicSolver during the earlier audit).

Then reuses Task B's own (loc, exact-clock-tuple) -> zone_key clustering
(same 200-episode random rollout, same seeds, deterministic — reproduces
identical clusters) to check: for every zone cluster with >=2 distinct
concrete states, do the concrete states' sets of tied-optimal actions have a
non-empty intersection?

THROWAWAY, /tmp only.
"""
from __future__ import annotations
import sys
import json
from collections import deque

import pyudbm as _warmup_pyudbm  # noqa: F401
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")
sys.path.append("/tmp")

import prob_jobshop_spike.symbolic_env as sym_env
import prob_jobshop_spike.zone_ops as zone_ops
from prob_jobshop_ext.benchmarks.all_instances import get_instance_by_name

from spike2_taskB import run_random_episode, N_EPISODES  # reuse exact same rollout code/seeds

_INF = float("inf")
_TOL = 1e-6

INSTANCES = ["PJS_01", "PJS_02", "PJS_03"]
STATE_CAP = 200_000


def concrete_key_of(env, job_ids):
    return tuple(int(round(env._virtual_clocks[jid])) for jid in job_ids)


def build_concrete_dp(instance, state_cap=STATE_CAP):
    job_ids = [j.job_id for j in instance.jobs]
    env = sym_env.SymbolicJobShopEnv(instance)

    loc0, actions0 = env.reset()
    key0 = (loc0, concrete_key_of(env, job_ids))
    snap0 = env.save_state()

    states = {key0: (loc0, snap0, actions0, False)}
    transitions: dict = {}
    queue = deque([key0])
    truncated = False

    while queue:
        key = queue.popleft()
        if key in transitions:
            continue
        loc, snap, actions, done = states[key]
        transitions[key] = []
        if done:
            continue

        for action in actions:
            task_id = action["action_id"]
            branch_trans = []
            for branch in action["branches"]:
                d, prob = branch["d"], branch["prob"]
                env.restore_state(snap)
                next_loc, next_actions, next_done = env.step_deterministic(task_id, d)
                elapsed = env._last_elapsed
                next_key = (next_loc, concrete_key_of(env, job_ids))
                next_snap = env.save_state()
                branch_trans.append((d, prob, next_key, elapsed))

                if next_key not in states:
                    if len(states) >= state_cap:
                        truncated = True
                    else:
                        states[next_key] = (next_loc, next_snap, next_actions, next_done)
                        queue.append(next_key)
            transitions[key].append((task_id, branch_trans))

    return states, transitions, truncated


def backward_dp(states, transitions):
    V = {key: (0.0 if done else _INF) for key, (loc, snap, actions, done) in states.items()}
    Q = {key: {} for key in states}

    for _pass in range(len(states) + 1):
        changed = False
        for key, action_list in transitions.items():
            if not action_list:
                continue
            best = _INF
            for task_id, trans in action_list:
                q = 0.0
                for d, prob, nk, elapsed in trans:
                    vnext = V.get(nk, _INF)
                    q += prob * (elapsed + vnext)
                Q[key][task_id] = q
                if q < best:
                    best = q
            if best < V[key] - 1e-12:
                V[key] = best
                changed = True
        if not changed:
            break

    return V, Q


def optimal_action_set(Q_key: dict) -> set:
    if not Q_key:
        return set()
    best = min(Q_key.values())
    return {a for a, q in Q_key.items() if q <= best + _TOL}


def run_one(name):
    inst = get_instance_by_name(name)
    job_ids = [j.job_id for j in inst.jobs]

    states, transitions, truncated = build_concrete_dp(inst)
    V, Q = backward_dp(states, transitions)
    unresolved = sum(1 for v in V.values() if v == _INF)

    # Rebuild the SAME clusters Task B found (identical seeds -> identical rollout)
    cluster: dict = {}
    for ep in range(N_EPISODES):
        recs, fail = run_random_episode(inst, seed=ep)
        for ck, zk, loc in recs:
            cluster.setdefault(zk, set()).add(ck)

    multi_clusters = {zk: ck_set for zk, ck_set in cluster.items() if len(ck_set) >= 2}
    n_multi = len(multi_clusters)
    n_consistent = 0
    n_missing_dp = 0
    inconsistent_examples = []

    for zk, ck_set in multi_clusters.items():
        opt_sets = []
        missing = False
        for ck in ck_set:
            if ck not in Q:
                missing = True
                continue
            opt_sets.append(optimal_action_set(Q[ck]))
        if missing:
            n_missing_dp += 1
            continue
        common = set.intersection(*opt_sets) if opt_sets else set()
        if common:
            n_consistent += 1
        else:
            inconsistent_examples.append((zk, list(ck_set)[:4], [list(s) for s in opt_sets]))

    row = {
        "instance": name,
        "n_concrete_states": len(states),
        "concrete_dp_truncated": truncated,
        "n_unresolved_V": unresolved,
        "n_multi_state_clusters": n_multi,
        "n_missing_from_dp": n_missing_dp,
        "n_policy_consistent": n_consistent,
        "n_policy_inconsistent": n_multi - n_missing_dp - n_consistent,
        "frac_policy_consistent_of_checkable": (
            n_consistent / (n_multi - n_missing_dp) if (n_multi - n_missing_dp) > 0 else None
        ),
        "inconsistent_examples": inconsistent_examples[:5],
    }
    return row


if __name__ == "__main__":
    all_rows = []
    for name in INSTANCES:
        row = run_one(name)
        all_rows.append(row)
        print(
            f"{name:8s} n_concrete_states={row['n_concrete_states']:6d} trunc={row['concrete_dp_truncated']!s:5s} "
            f"unresolved_V={row['n_unresolved_V']} multi_clusters={row['n_multi_state_clusters']} "
            f"missing_from_dp={row['n_missing_from_dp']} consistent={row['n_policy_consistent']} "
            f"inconsistent={row['n_policy_inconsistent']} frac_consistent={row['frac_policy_consistent_of_checkable']}",
            flush=True,
        )
        if row["inconsistent_examples"]:
            print("  INCONSISTENT EXAMPLES:", row["inconsistent_examples"])

    with open("/tmp/spike2_taskB_policy_results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print("Saved /tmp/spike2_taskB_policy_results.json")
