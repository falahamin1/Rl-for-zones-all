"""Exact DP over the concrete state space, gate_track version (single
global g instead of per-job g_J's; see symbolic_env.py's module docstring
for the nominal-value-as-realized-duration convention used here)."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from .instance import GateJobShopInstance
from .symbolic_env import GateSymbolicJobShopEnv

StateKey = Tuple


def state_key(env: GateSymbolicJobShopEnv) -> StateKey:
    loc, vc, g, ad, rd, fj = env.save_state()
    vc_k = tuple(sorted((k, int(round(v))) for k, v in vc.items()))
    ad_k = tuple(sorted(ad.items()))
    fj_k = tuple(sorted(fj))
    return (loc, vc_k, int(round(g)), ad_k, fj_k)


class DPResult:
    def __init__(self):
        self.snapshots: Dict[StateKey, tuple] = {}
        self.actions: Dict[StateKey, List[dict]] = {}
        self.transitions: Dict[StateKey, Dict[str, List[tuple]]] = {}
        self.V: Dict[StateKey, float] = {}
        self.optimal_actions: Dict[StateKey, List[str]] = {}
        self.root_key: Optional[StateKey] = None
        self.root_initial_penalty: float = 0.0
        self.truncated = False


def explore_and_solve(instance: GateJobShopInstance, state_limit: int = 200_000) -> DPResult:
    result = DPResult()
    env = GateSymbolicJobShopEnv(instance, seed=0)
    _loc0, actions0, info0 = env.reset()
    root_key = state_key(env)
    result.root_key = root_key
    result.root_initial_penalty = len(info0["newly_failed"]) * instance.penalty

    result.snapshots[root_key] = env.save_state()
    result.actions[root_key] = actions0

    queue = deque([root_key])
    visited = {root_key}

    while queue:
        if len(result.snapshots) > state_limit:
            result.truncated = True
            break
        key = queue.popleft()
        env.restore_state(result.snapshots[key])
        loc = env._loc
        actions = result.actions[key]

        is_terminal = all(s in ("done", "failed") for s in loc)
        if is_terminal or not actions:
            result.transitions[key] = {}
            continue

        trans: Dict[str, List[tuple]] = {}
        for a in actions:
            tid = a["action_id"]
            branch_list = []
            for br in a["branches"]:
                d, prob = br["d"], br["prob"]
                env.restore_state(result.snapshots[key])
                g_before = env._g
                _loc2, actions2, done2, info2 = env.step_deterministic(tid, d)
                g_after = env._g
                elapsed = g_after - g_before
                penalty = len(info2["newly_failed"]) * instance.penalty
                nkey = state_key(env)
                if nkey not in result.snapshots:
                    result.snapshots[nkey] = env.save_state()
                    result.actions[nkey] = actions2
                    if nkey not in visited:
                        visited.add(nkey)
                        queue.append(nkey)
                branch_list.append((prob, elapsed, penalty, nkey))
            trans[tid] = branch_list
        result.transitions[key] = trans

    memo: Dict[StateKey, float] = {}
    import sys

    sys.setrecursionlimit(10000)

    def V(key: StateKey) -> float:
        if key in memo:
            return memo[key]
        trans = result.transitions.get(key, {})
        if not trans:
            memo[key] = 0.0
            return 0.0
        best_cost = float("inf")
        best_actions = []
        for tid, branches in trans.items():
            q = sum(prob * (elapsed + penalty + V(nkey)) for prob, elapsed, penalty, nkey in branches)
            if q < best_cost - 1e-9:
                best_cost = q
                best_actions = [tid]
            elif abs(q - best_cost) <= 1e-9:
                best_actions.append(tid)
        memo[key] = best_cost
        result.optimal_actions[key] = best_actions
        return best_cost

    for key in list(result.snapshots.keys()):
        V(key)
    result.V = memo
    return result
