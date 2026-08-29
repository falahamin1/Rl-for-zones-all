"""Exact DP over the concrete state space.

Unlike gridtimed's grid-navigation domain (genuine revisit cycles via
switches), this domain is ACYCLIC by construction: every dispatch either
advances some job's cursor (irreversible -- cursors only increase) or
fails a job (irreversible -- failed only goes False->True). "Total
progress" (sum of cursors, plus failed jobs counted as fully advanced) is
strictly monotonic and bounded by n_tasks_total + n_jobs, so plain
memoized recursion (no topological-sort workaround needed) terminates
safely.

For a truncated exploration (state_limit reached), a successor state that
was enqueued but never actually processed (no transitions entry) is
treated as reaching a conservative fallback cost -- "assume every job in
the instance fails" -- rather than the wrong default of implicitly-zero
(which would make unexplored territory look free). Counted and reported,
same convention as gridtimed's ``n_horizon_fallback_refs``.
"""
from __future__ import annotations

import sys
from collections import deque
from typing import Dict, List

from .env import ModeSelState, StepResult, all_actions, initial_state, is_legal, is_terminal, step_deterministic
from .instance import ModeSelInstance, parse_action_id

_TIE_EPS = 1e-9


class DPResult:
    def __init__(self, instance: ModeSelInstance):
        self.root: ModeSelState = initial_state(instance)
        self.states: List[ModeSelState] = []
        self.transitions: Dict[ModeSelState, Dict[str, StepResult]] = {}
        self.V: Dict[ModeSelState, float] = {}
        self.optimal_actions: Dict[ModeSelState, List[str]] = {}
        self.truncated = False
        self.n_horizon_fallback_refs = 0


def explore_and_solve(instance: ModeSelInstance, state_limit: int = 200_000) -> DPResult:
    result = DPResult(instance)
    root = result.root

    queue = deque([root])
    seen = {root}
    order: List[ModeSelState] = []

    while queue:
        if len(seen) > state_limit:
            result.truncated = True
            break
        s = queue.popleft()
        order.append(s)
        if is_terminal(instance, s):
            result.transitions[s] = {}
            continue
        trans: Dict[str, StepResult] = {}
        for aid in all_actions(instance):
            task_id, mode = parse_action_id(aid)
            if not is_legal(instance, s, task_id, mode):
                continue
            res = step_deterministic(instance, s, task_id, mode)
            trans[aid] = res
            if res.next_state not in seen:
                seen.add(res.next_state)
                queue.append(res.next_state)
        result.transitions[s] = trans

    result.states = order

    fallback_cost = sum(instance.failure_penalty(j.job_id) for j in instance.jobs)
    memo: Dict[ModeSelState, float] = {}
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

    def V(key: ModeSelState) -> float:
        if key in memo:
            return memo[key]
        if key not in result.transitions:
            result.n_horizon_fallback_refs += 1
            memo[key] = fallback_cost
            return fallback_cost
        trans = result.transitions[key]
        if not trans:
            memo[key] = 0.0
            return 0.0
        best_cost = float("inf")
        best_actions: List[str] = []
        for aid, res in trans.items():
            q = res.cost + V(res.next_state)
            if q < best_cost - _TIE_EPS:
                best_cost = q
                best_actions = [aid]
            elif abs(q - best_cost) <= _TIE_EPS:
                best_actions.append(aid)
        memo[key] = best_cost
        result.optimal_actions[key] = best_actions
        return best_cost

    for key in order:
        V(key)
    result.V = memo
    return result
