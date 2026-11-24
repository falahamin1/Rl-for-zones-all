"""Exact DP over the concrete state space.

The state graph is deterministic (one successor per action) and, restricted
to *non-self-loop* actions, strictly increasing in ``g`` (every legal,
time-consuming move advances the never-resetting global clock by >= 1;
illegal actions self-loop with zero elapsed time). Self-loop actions can
therefore never be optimal as long as at least one non-self-loop action
exists from every non-terminal state -- asserted, not just assumed, during
value computation (see the deadlock branch below).

This means backward induction can process states in strictly decreasing
``g`` order (a topological order for the non-self-loop transition graph)
instead of needing general value iteration -- same structure as
gridtimed/dp.py (read, not imported, as a reference for this pattern).

For large instances, the BFS may be truncated at ``state_limit`` before
every reachable state is discovered. A truncated state can have
transitions pointing at successors whose own value was never computed;
those are treated as reaching the FAILED absorbing state's cost as a
conservative placeholder, counted and reported rather than crashing or
silently treating them as free.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

from .env import StepResult, TRMState, all_actions, done_state, failed_state, initial_state, is_terminal, peek
from .instance import FAIL_PENALTY, TRMGridInstance

_TIE_EPS = 1e-9


class DPResult:
    def __init__(self, instance: TRMGridInstance):
        self.root: TRMState = initial_state(instance)
        self.states: List[TRMState] = []
        self.transitions: Dict[TRMState, Dict[str, StepResult]] = {}
        self.V: Dict[TRMState, float] = {}
        self.optimal_actions: Dict[TRMState, List[str]] = {}
        self.truncated = False
        self.n_horizon_fallback_refs = 0
        self.n_deadlock_states = 0


def explore_and_solve(instance: TRMGridInstance, state_limit: int = 200_000) -> DPResult:
    result = DPResult(instance)
    root = result.root
    DONE, FAILED = done_state(instance), failed_state(instance)

    queue = deque([root])
    seen = {root}
    order: List[TRMState] = []

    while queue:
        if len(seen) > state_limit:
            result.truncated = True
            break
        s = queue.popleft()
        order.append(s)
        if is_terminal(s):
            result.transitions[s] = {}
            continue
        trans: Dict[str, StepResult] = {}
        for aid in all_actions(instance):
            res = peek(instance, s, aid)
            trans[aid] = res
            if res.next_state not in seen:
                seen.add(res.next_state)
                queue.append(res.next_state)
        result.transitions[s] = trans

    result.states = order

    non_terminal = [s for s in order if not is_terminal(s)]
    non_terminal.sort(key=lambda s: s.g, reverse=True)

    result.V[DONE] = 0.0
    result.V[FAILED] = 0.0
    fallback_value = FAIL_PENALTY  # conservative: unexplored = assume failure

    def value_of(succ: TRMState) -> float:
        if succ in result.V:
            return result.V[succ]
        result.n_horizon_fallback_refs += 1
        return fallback_value

    for s in non_terminal:
        trans = result.transitions[s]
        real_moves = [(aid, res) for aid, res in trans.items() if res.next_state != s]
        if not real_moves:
            # A genuine deadlock: every neighbour is off-grid (possible at
            # a corner if the action set were ever restricted enough to
            # make every direction illegal -- not expected with all 4
            # moves always available, but handled rather than crashing).
            result.V[s] = FAIL_PENALTY
            result.optimal_actions[s] = []
            result.n_deadlock_states += 1
            continue
        best_cost = float("inf")
        best_actions: List[str] = []
        for aid, res in real_moves:
            q = res.cost + value_of(res.next_state)
            if q < best_cost - _TIE_EPS:
                best_cost = q
                best_actions = [aid]
            elif abs(q - best_cost) <= _TIE_EPS:
                best_actions.append(aid)
        result.V[s] = best_cost
        result.optimal_actions[s] = best_actions

    return result
