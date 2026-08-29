"""SPIKE3: policy consistency + decision relevance, PJS_01-03 @ w=0.25, t=0.3.

Builds an EXHAUSTIVE bucket-driven exact concrete-state DP (branching over
every possible sampled duration bucket at each dispatch, via
step_deterministic — same overall approach as Spike 2's policy-consistency
solver), this time under Spike 3's real-valued-interval + deadline zone
semantics, tracking each concrete state's zone in parallel (via
propagate_one, matched by bucket d — mirrors symbolic_rl.py's own
branch-matching pattern). This gives a MUCH more complete state/cluster
population than the 200-episode random rollout in spike3_taskB.py (that
script's clusters come from whatever a finite random sample happened to
visit; this one enumerates every reachable bucket-outcome combination).

Methodology note (also stated in /tmp/SPIKE3.md): "concrete state" here is
BUCKET-driven (nominal durations), not real-valued — an exhaustive DP over
a genuinely continuous state space is not tractable/well-defined the way a
finite branching-factor DP is. elapsed/V/Q are therefore computed using
step_deterministic()'s bucket-driven timing (self._active_real_dur stays
unset in this whole flow, so _advance_to_decision falls back to the bucket
value — see symbolic_env.py). This is a DIFFERENT, narrower operationalization
than spike3_taskB.py's real-valued rollout-based aggregation-ratio grid.

Branches whose ZONE propagation is symbolically infeasible (e.g. a deadline
the branch cannot satisfy) are dropped from the DP entirely — consistent
with how the zone graph already treats an empty zone as "prune this
branch"; the concrete step_deterministic() call itself does NOT enforce
deadlines (nothing stops a real system from missing one), so this is a
deliberate modeling choice to compute the optimal policy AS IF deadline-
violating outcomes are simply unavailable, matching the zone abstraction's
own semantics. Flagged explicitly, not silently assumed.

THROWAWAY, /tmp only.
"""
from __future__ import annotations
import sys
import json
from collections import deque

import pyudbm as _warmup_pyudbm  # noqa: F401
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")
sys.path.append("/tmp")

from zone_common import make_propagate_one
from spike3_common import get_base_instance, make_deadline_variant

import prob_jobshop_spike.symbolic_env as sym_env
import prob_jobshop_spike.zone_ops as zone_ops

_INF = float("inf")
_TOL = 1e-6

INSTANCES = ["PJS_01", "PJS_02", "PJS_03"]
W = 0.25
T = 0.3
STATE_CAP = 200_000

propagate_one = make_propagate_one(zone_ops)


def concrete_key_of(env, job_ids):
    return tuple(int(round(env._virtual_clocks[jid])) for jid in job_ids)


def build_concrete_dp_with_zones(instance, state_cap=STATE_CAP):
    sym_env.INTERVAL_WIDTH = W
    job_ids = [j.job_id for j in instance.jobs]
    env = sym_env.SymbolicJobShopEnv(instance)
    zctx = zone_ops.ZoneContext(instance)
    M = int(round(instance.worst_case_makespan_upper_bound() * (1 + W) * sym_env.SCALE))

    loc0, actions0 = env.reset()
    Z0 = zone_ops.init_zone(zctx, env.invariant(loc0))
    Z0 = zone_ops.apply_delay(Z0, zctx, env.invariant(loc0), env.rate(loc0))
    Z0 = zone_ops.normalize(Z0, zctx, M)
    key0 = (loc0, concrete_key_of(env, job_ids))
    snap0 = env.save_state()

    states = {key0: (loc0, snap0, actions0, False, Z0)}
    transitions: dict = {}
    queue = deque([key0])
    truncated = False
    n_deadline_pruned_branches = 0

    while queue:
        key = queue.popleft()
        if key in transitions:
            continue
        loc, snap, actions, done, Z = states[key]
        transitions[key] = []
        if done:
            continue

        for action in actions:
            task_id = action["action_id"]
            branch_trans = []
            for branch in action["branches"]:
                d, prob = branch["d"], branch["prob"]

                pres = propagate_one(env, zctx, Z, M, action, d)
                if pres is None:
                    n_deadline_pruned_branches += 1
                    continue
                Z_next, _dec_loc_symbolic = pres

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
                        states[next_key] = (next_loc, next_snap, next_actions, next_done, Z_next)
                        queue.append(next_key)
            if branch_trans:
                transitions[key].append((task_id, branch_trans))

    return states, transitions, truncated, n_deadline_pruned_branches


def backward_dp(states, transitions):
    V = {key: (0.0 if done else _INF) for key, (loc, snap, actions, done, Z) in states.items()}
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
    inst = make_deadline_variant(get_base_instance(name), T)
    states, transitions, truncated, n_pruned = build_concrete_dp_with_zones(inst)
    V, Q = backward_dp(states, transitions)
    unresolved = sum(1 for v in V.values() if v == _INF)

    cluster: dict = {}       # zone_key -> set of concrete_keys
    loc_zones: dict = {}     # loc -> set of zone_keys
    for key, (loc, snap, actions, done, Z) in states.items():
        zk = (loc, str(Z))
        cluster.setdefault(zk, set()).add(key)
        loc_zones.setdefault(loc, set()).add(zk)

    multi_clusters = {zk: ck_set for zk, ck_set in cluster.items() if len(ck_set) >= 2}
    n_consistent = 0
    n_missing_dp = 0
    inconsistent_examples = []
    zone_action_set: dict = {}   # zk -> well-defined optimal action set (for decision relevance)

    for zk, ck_set in cluster.items():
        opt_sets = []
        missing = False
        for ck in ck_set:
            if ck not in Q or not Q[ck]:
                missing = True
                continue
            opt_sets.append(optimal_action_set(Q[ck]))
        if missing or not opt_sets:
            if len(ck_set) >= 2:
                n_missing_dp += 1
            continue
        common = set.intersection(*opt_sets) if len(opt_sets) > 1 else opt_sets[0]
        if len(ck_set) >= 2:
            if common:
                n_consistent += 1
            else:
                inconsistent_examples.append((zk, list(ck_set)[:4], [list(s) for s in opt_sets]))
        if common:
            zone_action_set[zk] = common
        # if inconsistent (common empty) with 2+ states, leave zk out of
        # zone_action_set — no single well-defined action for decision relevance

    n_multi = len(multi_clusters)

    # Decision relevance: for each location with >=2 well-defined zones,
    # do any two of those zones have DISJOINT optimal-action-sets?
    n_multi_zone_locs = 0
    n_relevant_locs = 0
    relevant_examples = []
    for loc, zks in loc_zones.items():
        defined = [(zk, zone_action_set[zk]) for zk in zks if zk in zone_action_set]
        if len(defined) < 2:
            continue
        n_multi_zone_locs += 1
        found = False
        for i in range(len(defined)):
            for j in range(i + 1, len(defined)):
                if not (defined[i][1] & defined[j][1]):
                    found = True
                    if len(relevant_examples) < 5:
                        relevant_examples.append((loc, defined[i], defined[j]))
                    break
            if found:
                break
        if found:
            n_relevant_locs += 1

    row = {
        "instance": name,
        "n_concrete_states": len(states),
        "concrete_dp_truncated": truncated,
        "n_deadline_pruned_branches": n_pruned,
        "n_unresolved_V": unresolved,
        "n_multi_state_clusters": n_multi,
        "n_missing_from_dp": n_missing_dp,
        "n_policy_consistent": n_consistent,
        "n_policy_inconsistent": n_multi - n_missing_dp - n_consistent,
        "frac_policy_consistent_of_checkable": (
            n_consistent / (n_multi - n_missing_dp) if (n_multi - n_missing_dp) > 0 else None
        ),
        "inconsistent_examples": inconsistent_examples[:5],
        "n_multi_zone_locations": n_multi_zone_locs,
        "n_decision_relevant_locations": n_relevant_locs,
        "frac_decision_relevant": (n_relevant_locs / n_multi_zone_locs) if n_multi_zone_locs else None,
        "relevant_examples": relevant_examples,
    }
    return row


if __name__ == "__main__":
    all_rows = []
    for name in INSTANCES:
        row = run_one(name)
        all_rows.append(row)
        print(
            f"{name:8s} n_concrete_states={row['n_concrete_states']:6d} trunc={row['concrete_dp_truncated']!s:5s} "
            f"pruned_branches={row['n_deadline_pruned_branches']} unresolved_V={row['n_unresolved_V']} "
            f"multi_clusters={row['n_multi_state_clusters']} missing_from_dp={row['n_missing_from_dp']} "
            f"consistent={row['n_policy_consistent']} inconsistent={row['n_policy_inconsistent']} "
            f"frac_consistent={row['frac_policy_consistent_of_checkable']} | "
            f"multi_zone_locs={row['n_multi_zone_locations']} decision_relevant={row['n_decision_relevant_locations']} "
            f"frac_relevant={row['frac_decision_relevant']}",
            flush=True,
        )
        if row["inconsistent_examples"]:
            print("  INCONSISTENT:", row["inconsistent_examples"])

    with open("/tmp/spike3_policy_results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print("Saved /tmp/spike3_policy_results.json")
