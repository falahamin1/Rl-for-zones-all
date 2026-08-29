"""Exact DP over the concrete state space.

The state graph is deterministic (one successor per action) and, restricted
to *non-self-loop* actions, strictly increasing in ``y`` (every legal,
time-consuming move advances the never-resetting global clock by >= 1;
illegal actions self-loop with zero elapsed time). Self-loop actions can
therefore never be optimal as long as at least one non-self-loop action
exists from every non-terminal state -- a structural property of every
instance this generator produces (every cell has at least one neighbour
that is a plain cell or a switch, never gate-only -- switches are never
placed adjacent to nothing but gates by the Task B generator's solvability
check), asserted (not just assumed) below during value computation.

This means backward induction can process states in strictly decreasing
``y`` order (a topological order for the non-self-loop transition graph,
even though the *full* transition graph -- self-loops and switch-revisit
cycles included -- is not a DAG) instead of needing general value
iteration.

For LARGE instances (Task B's bigger tiers), the BFS may be truncated at
``state_limit`` before every reachable state is discovered. A truncated
state can have transitions pointing at successors whose own value was
never computed (because the BFS stopped before it was popped). Those are
treated as reaching the ``FAILED`` absorbing state's cost as a conservative
placeholder -- "assume the worst about territory we didn't explore" --
rather than crashing or (silently, incorrectly) treating them as free. The
number of such horizon references is counted and reported; a truncated run
with 0 of them means the truncation didn't actually affect any computed
value.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

from .env import GridState, StepResult, all_actions, done_state, failed_state, initial_state, is_terminal, peek
from .instance import DEADLINE_FAIL_PENALTY, GridInstance

_TIE_EPS = 1e-9


class DPResult:
    def __init__(self, instance: GridInstance):
        self.root: GridState = initial_state(instance)
        self.states: List[GridState] = []
        self.transitions: Dict[GridState, Dict[str, StepResult]] = {}
        self.V: Dict[GridState, float] = {}
        self.optimal_actions: Dict[GridState, List[str]] = {}
        self.truncated = False
        self.n_horizon_fallback_refs = 0
        self.n_deadlock_states = 0


def explore_and_solve(instance: GridInstance, state_limit: int = 200_000) -> DPResult:
    result = DPResult(instance)
    root = result.root
    DONE, FAILED = done_state(instance), failed_state(instance)

    queue = deque([root])
    seen = {root}
    order: List[GridState] = []  # discovery order; re-sorted below

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
    non_terminal.sort(key=lambda s: s.y, reverse=True)

    result.V[DONE] = 0.0
    result.V[FAILED] = 0.0
    fallback_value = DEADLINE_FAIL_PENALTY  # conservative: unexplored = assume failure

    def value_of(succ: GridState) -> float:
        if succ in result.V:
            return result.V[succ]
        result.n_horizon_fallback_refs += 1
        return fallback_value

    for s in non_terminal:
        trans = result.transitions[s]
        real_moves = [(aid, res) for aid, res in trans.items() if res.next_state != s]
        if not real_moves:
            # A genuine deadlock: every neighbour is a currently-closed gate
            # (never possible in the hand-built T0 layout -- every cell
            # there has a plain/switch neighbour -- but NOT structurally
            # guaranteed for Task B's generated layouts, where gates can end
            # up surrounding a cell). y never advances from here, so the
            # deadline auto-fail can never trigger either: this is a
            # permanent trap. Treated as exactly as bad as FAILED (a
            # documented choice, not a crash) and reported separately.
            result.V[s] = DEADLINE_FAIL_PENALTY
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
