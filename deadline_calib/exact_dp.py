"""Exact DP (exhaustive BFS + backward value iteration) over the concrete
state space of a ``DeadlineJobShopEnv``.

The transition graph is a genuine DAG: every non-terminal transition either
strictly increases ``g`` (time elapsing) or strictly grows the set of
done/failed tasks at zero elapsed time (an immediate job failure) -- never
both zero elapsed time AND an unchanged location, so no state can ever be
revisited. This means a single backward pass over states in *reverse BFS
discovery order* is already a valid reverse topological order; no
iterative fixed point is needed (unlike the earlier no-deadline audits'
zone-graph DP, which had to special-case a self-referential "clamped at the
cap" bucket -- that doesn't arise here since ``g`` is never clamped).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from prob_jobshop_deadline.deadline_env import DONE_LIKE, DeadlineJobShopEnv
from prob_jobshop_deadline.instance import ProbJobShopInstance

StateKey = Tuple


def _key(state: tuple) -> StateKey:
    loc, g, vc_items, Z, done_job_ids, late_checked = state
    return (loc, g, vc_items, done_job_ids, late_checked)


@dataclass
class DPResult:
    instance: ProbJobShopInstance
    V: Dict[StateKey, float] = field(default_factory=dict)
    policy: Dict[StateKey, Optional[str]] = field(default_factory=dict)
    # action_id -> list of (prob, next_key, reward) per state key
    transitions: Dict[StateKey, Dict[str, List[Tuple[float, StateKey, float]]]] = field(default_factory=dict)
    state_by_key: Dict[StateKey, tuple] = field(default_factory=dict)
    order: List[StateKey] = field(default_factory=list)  # BFS discovery order, root first
    truncated: bool = False
    n_states: int = 0
    stuck_states: List[StateKey] = field(default_factory=list)
    zone_mismatches: List[StateKey] = field(default_factory=list)


def _zone_matrix(Z) -> Optional[list]:
    dbms = Z.to_dbm_list()
    if len(dbms) != 1:
        return None  # non-convex federation -- shouldn't happen here, flagged separately if it does
    return dbms[0].to_matrix()


def solve(
    instance: ProbJobShopInstance,
    fail_penalty: Optional[float] = None,
    state_limit: int = 200_000,
) -> DPResult:
    env = DeadlineJobShopEnv(instance, seed=0, fail_penalty=fail_penalty)
    loc0, actions0 = env.reset()
    root_state = env.get_state()
    root_key = _key(root_state)

    result = DPResult(instance=instance)
    result.state_by_key[root_key] = root_state
    zone_by_key: Dict[StateKey, list] = {root_key: _zone_matrix(root_state[3])}

    visited = {root_key}
    order = [root_key]
    queue = deque([root_key])

    while queue:
        if len(visited) > state_limit:
            result.truncated = True
            break
        key = queue.popleft()
        state = result.state_by_key[key]
        env.set_state(state)
        loc = state[0]

        if all(s in DONE_LIKE for s in loc):
            result.transitions[key] = {}
            continue

        actions = env._settle_and_compute_actions()
        if env.stuck_this_episode:
            result.stuck_states.append(key)

        action_map: Dict[str, List[Tuple[float, StateKey, float]]] = {}
        for action in actions:
            branch_results = []
            for branch in action["branches"]:
                env.set_state(state)
                env._settle_and_compute_actions()
                _, _, _, reward, _ = env.step_deterministic(action["action_id"], branch["d"])
                next_state = env.get_state()
                next_key = _key(next_state)

                mat = _zone_matrix(next_state[3])
                if next_key in zone_by_key:
                    if zone_by_key[next_key] != mat:
                        result.zone_mismatches.append(next_key)
                else:
                    zone_by_key[next_key] = mat

                if next_key not in visited:
                    visited.add(next_key)
                    result.state_by_key[next_key] = next_state
                    order.append(next_key)
                    queue.append(next_key)

                branch_results.append((branch["prob"], next_key, reward))
            action_map[action["action_id"]] = branch_results

        result.transitions[key] = action_map

    result.order = order
    result.n_states = len(order)

    # Backward value iteration in reverse discovery order (valid reverse
    # topological order for this DAG -- see module docstring).
    for key in reversed(order):
        state = result.state_by_key[key]
        loc = state[0]
        if all(s in DONE_LIKE for s in loc):
            result.V[key] = 0.0
            result.policy[key] = None
            continue

        action_map = result.transitions.get(key, {})
        if not action_map:
            result.V[key] = float("nan")
            result.policy[key] = None
            continue

        best_action, best_q = None, -float("inf")
        for action_id, branch_list in action_map.items():
            q = sum(prob * (reward + result.V[next_key]) for prob, next_key, reward in branch_list)
            if q > best_q:
                best_q = q
                best_action = action_id
        result.V[key] = best_q
        result.policy[key] = best_action

    result.zone_by_key = zone_by_key
    return result


def zone_key(result: DPResult, key: StateKey) -> Tuple[tuple, str]:
    """``(loc, canonical-zone-string)`` -- same convention as the earlier
    no-deadline audits' zone graphs, for direct comparability."""
    state = result.state_by_key[key]
    loc, g, vc_items, Z, done_job_ids, late_checked = state
    return (loc, str(Z))
