"""Zone-graph BFS + exact backward-induction DP for the micro-automaton.
Reuses zone_common.make_propagate_one (fully generic -- only needs
env.invariant()) and zone_ops's primitives directly, exactly as Spike 3's
own exact-DP scripts did. THROWAWAY, /tmp only.
"""
from __future__ import annotations
import pyudbm as _warmup_pyudbm  # noqa: F401 -- MUST import before repo root hits sys.path
import sys
import time
from collections import deque

sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")
sys.path.append("/tmp")

from zone_common import make_propagate_one
import prob_jobshop_spike.zone_ops as zone_ops

from model import MicroEnv, MicroZoneContext, _scaled, M_REAL

_INF = float("inf")

propagate_one = make_propagate_one(zone_ops)


def explore_zone_graph(deadline, state_cap=10_000):
    """BFS over (loc, str(Z)) states. Returns dict with:
      states:      key -> (loc, Z, actions)
      transitions: key -> [(action_id, price, [(d, prob, next_key), ...]), ...]
    """
    env = MicroEnv(deadline=deadline)
    zctx = MicroZoneContext()
    M = _scaled(M_REAL)

    loc0, actions0 = env.reset()
    Z0 = zone_ops.init_zone(zctx, env.invariant(loc0))
    Z0 = zone_ops.apply_delay(Z0, zctx, env.invariant(loc0), env.rate(loc0))
    Z0 = zone_ops.normalize(Z0, zctx, M)
    key0 = (loc0, str(Z0))

    states = {key0: (loc0, Z0, actions0)}
    transitions: dict = {}
    queue = deque([key0])
    truncated = False
    t0 = time.time()

    while queue:
        if len(states) >= state_cap:
            truncated = True
            break
        key = queue.popleft()
        if key in transitions:
            continue
        loc, Z, actions = states[key]
        transitions[key] = []

        for action in actions:
            trans = []
            for branch in action["branches"]:
                res = propagate_one(env, zctx, Z, M, action, branch["d"])
                if res is None:
                    continue  # branch's zone is empty -- infeasible, edge unavailable
                Z_b, dec_loc = res
                next_key = (dec_loc, str(Z_b))
                trans.append((branch["d"], branch["prob"], next_key))
                if next_key not in states:
                    states[next_key] = (dec_loc, Z_b, env._compute_actions(dec_loc))
                    queue.append(next_key)
            if trans:
                transitions[key].append((action["action_id"], action["price"], trans))

    wall = time.time() - t0
    return {
        "states": states,
        "transitions": transitions,
        "truncated": truncated,
        "wall_s": wall,
        "zctx": zctx,
        "M": M,
    }


def backward_dp(states, transitions):
    """V[terminal] = 0 (a state with no outgoing transitions is terminal in
    this model). Q[key][action_id] = price + sum_d prob*V[next]. Standard
    Bellman-Ford-style relaxation (V starts at +inf, only ever decreases) --
    same corrected style used throughout Spikes 2/3 (deliberately NOT the
    "treat unresolved child as 0" pattern found buggy in the original
    OptimalSymbolicSolver during the initial audit).
    """
    V = {}
    Q = {key: {} for key in states}
    for key, action_list in transitions.items():
        V[key] = 0.0 if not action_list else _INF

    for _pass in range(len(states) + 1):
        changed = False
        for key, action_list in transitions.items():
            if not action_list:
                continue
            best = _INF
            for action_id, price, trans in action_list:
                q = price
                for d, prob, nk in trans:
                    vnext = V.get(nk, _INF)
                    q += prob * vnext
                Q[key][action_id] = q
                if q < best:
                    best = q
            if best < V[key] - 1e-12:
                V[key] = best
                changed = True
        if not changed:
            break

    return V, Q


def optimal_action_set(Q_key: dict, tol=1e-9) -> set:
    if not Q_key:
        return set()
    best = min(Q_key.values())
    return {a for a, q in Q_key.items() if q <= best + tol}
