"""Concrete, deterministic transition function for a trmgrid v2 (branching)
instance. Pure-functional by design, same discipline as v1: a ``TRMState``
is a small hashable dataclass, so the exact DP (``dp.py``) and the
hand-verified paper gate (``measurement1.py``) can both call
:func:`step`/:func:`peek` directly.

Phase transitions and guards (``instance.py``'s module docstring has the
full spec):
  - phase=() [start] + target is ANY sub-goal i -> phase=(i,), reset t_i.
    No guard (first visit is always unconstrained, same as v1's start->A).
  - phase=(i,) [have_i] + target is sub-goal j != i -> guard: t_i (after
    this move's elapse, before reset) must lie in [lo_ij, hi_ij]. Pass:
    phase=(i,j), reset t_j. Fail: TERMINAL FAILURE.
  - phase=(i,j) [have_i,j] + target == GOAL -> guard: t_j <= cap_j. Pass
    (and g <= D): TERMINAL SUCCESS. Fail: TERMINAL FAILURE.
  - Any OTHER touch of a sub-goal (e.g. re-touching i while already
    "have_i" or "have_i,j", or touching the third, unused sub-goal while
    "have_i,j") resets that sub-goal's clock unconditionally (same
    physical-arrival rule as v1) but does not advance the phase or check
    any guard -- it is simply not the RM's currently active transition.

Guard violations and deadline expiry are TERMINAL FAILURES (move happens,
time elapses, THEN the episode ends with a penalty) -- not a blocked/
illegal move, same departure from gridtimed's gate-window convention as
v1, carried over unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from .instance import (
    DELTA,
    MOVES,
    REWARD_SUCCESS,
    TIME_COST,
    ILLEGAL_PENALTY,
    FAIL_PENALTY,
    Phase,
    Position,
    TRMGridInstance,
)

Loc = Union[Position, str]  # a grid (row, col), or "DONE" / "FAILED"


@dataclass(frozen=True)
class TRMState:
    phase: Phase
    pos: Loc
    clocks: Tuple[int, ...]
    g: int


def done_state(instance: TRMGridInstance) -> TRMState:
    return TRMState(phase=(), pos="DONE", clocks=(0,) * instance.k, g=0)


def failed_state(instance: TRMGridInstance) -> TRMState:
    return TRMState(phase=(), pos="FAILED", clocks=(0,) * instance.k, g=0)


def initial_state(instance: TRMGridInstance) -> TRMState:
    return TRMState(phase=(), pos=instance.start, clocks=(0,) * instance.k, g=0)


def is_terminal(state: TRMState) -> bool:
    return state.pos in ("DONE", "FAILED")


def all_actions(instance: TRMGridInstance) -> List[str]:
    return [instance.action_id(d, m) for d in instance.allowed_delays for m in MOVES]


def _parse(aid: str) -> Tuple[int, str]:
    delay_part, move = aid.split("_", 1)
    return int(delay_part[1:]), move


def target_of(pos: Position, move: str) -> Position:
    dr, dc = DELTA[move]
    return (pos[0] + dr, pos[1] + dc)


@dataclass(frozen=True)
class StepResult:
    next_state: TRMState
    cost: float
    elapsed: int
    legal: bool
    newly_done: bool
    newly_failed: bool
    fail_reason: str = ""  # "" | "guard_pair" | "guard_goal" | "deadline"


def peek(instance: TRMGridInstance, state: TRMState, aid: str) -> StepResult:
    assert not is_terminal(state), "peek() called on an absorbing state"
    delay, move = _parse(aid)
    target = target_of(state.pos, move)

    if not instance.in_bounds(target):
        return StepResult(state, ILLEGAL_PENALTY, 0, legal=False, newly_done=False, newly_failed=False)

    elapsed = delay + 1
    new_clocks = list(c + elapsed for c in state.clocks)
    new_g = state.g + elapsed
    phase = state.phase

    is_sg = instance.is_sub_goal(target)
    tgt_idx = instance.sub_goal_index(target) if is_sg else None

    if phase == () and is_sg:
        i = tgt_idx
        new_phase: Phase = (i,)
        new_clocks[i] = 0
    elif len(phase) == 1 and is_sg and tgt_idx != phase[0]:
        i = phase[0]
        j = tgt_idx
        lo, hi = instance.windows[(i, j)]
        if not (lo <= new_clocks[i] <= hi):
            cost = TIME_COST * elapsed + FAIL_PENALTY
            return StepResult(failed_state(instance), cost, elapsed, legal=True, newly_done=False,
                               newly_failed=True, fail_reason="guard_pair")
        new_phase = (i, j)
        new_clocks[j] = 0
    elif len(phase) == 2 and target == instance.goal_pos:
        j = phase[1]
        cap_j = instance.caps[j]
        if new_clocks[j] > cap_j:
            cost = TIME_COST * elapsed + FAIL_PENALTY
            return StepResult(failed_state(instance), cost, elapsed, legal=True, newly_done=False,
                               newly_failed=True, fail_reason="guard_goal")
        if new_g > instance.deadline:
            cost = TIME_COST * elapsed + FAIL_PENALTY
            return StepResult(failed_state(instance), cost, elapsed, legal=True, newly_done=False,
                               newly_failed=True, fail_reason="deadline")
        cost = TIME_COST * elapsed - REWARD_SUCCESS
        return StepResult(done_state(instance), cost, elapsed, legal=True, newly_done=True, newly_failed=False)
    else:
        new_phase = phase
        if is_sg:
            new_clocks[tgt_idx] = 0  # incidental touch -- unconditional reset, no phase change

    if new_g > instance.deadline:
        cost = TIME_COST * elapsed + FAIL_PENALTY
        return StepResult(failed_state(instance), cost, elapsed, legal=True, newly_done=False,
                           newly_failed=True, fail_reason="deadline")

    next_state = TRMState(phase=new_phase, pos=target, clocks=tuple(new_clocks), g=new_g)
    return StepResult(next_state, TIME_COST * elapsed, elapsed, legal=True, newly_done=False, newly_failed=False)


def action_mask(instance: TRMGridInstance, state: TRMState) -> Dict[str, bool]:
    return {aid: peek(instance, state, aid).legal for aid in all_actions(instance)}


def step(instance: TRMGridInstance, state: TRMState, aid: str) -> StepResult:
    assert aid in all_actions(instance), f"{aid} not in the fixed action set for this instance"
    return peek(instance, state, aid)
