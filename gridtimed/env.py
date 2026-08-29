"""Concrete, deterministic transition function for a grid instance.

Pure-functional by design (no mutable env object): a ``GridState`` is a
small hashable dataclass, so the exact DP (``dp.py``) and the hand-verified
sanity gate (``sanity_gate.py``) can both call :func:`step` directly on
whatever state they're inspecting without any save/restore machinery.

``GridState.clocks`` is a tuple of per-switch clock values, in the same
order as ``instance.switches`` -- generalizes the original fixed
``s1``/``s2`` fields to an arbitrary switch count for Task B's tiered
instances.
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
    DEADLINE_FAIL_PENALTY,
    GridInstance,
    Position,
)

Loc = Union[Position, str]  # a grid (row, col), or "DONE" / "FAILED"


@dataclass(frozen=True)
class GridState:
    pos: Loc
    clocks: Tuple[int, ...]
    y: int


def done_state(instance: GridInstance) -> GridState:
    return GridState(pos="DONE", clocks=(0,) * instance.n_switches, y=0)


def failed_state(instance: GridInstance) -> GridState:
    return GridState(pos="FAILED", clocks=(0,) * instance.n_switches, y=0)


def initial_state(instance: GridInstance) -> GridState:
    return GridState(pos=instance.start, clocks=(0,) * instance.n_switches, y=0)


def is_terminal(state: GridState) -> bool:
    return state.pos in ("DONE", "FAILED")


def all_actions(instance: GridInstance) -> List[str]:
    return [instance.action_id(d, m) for d in instance.allowed_delays for m in MOVES]


def _parse(aid: str) -> Tuple[int, str]:
    delay_part, move = aid.split("_", 1)
    return int(delay_part[1:]), move


def target_of(pos: Position, move: str) -> Position:
    dr, dc = DELTA[move]
    return (pos[0] + dr, pos[1] + dc)


@dataclass(frozen=True)
class StepResult:
    next_state: GridState
    cost: float
    elapsed: int
    legal: bool
    newly_done: bool
    newly_failed: bool


def peek(instance: GridInstance, state: GridState, aid: str) -> StepResult:
    """Compute the result of taking ``aid`` from ``state`` without mutating
    anything -- used identically by :func:`step`, the action-mask builder,
    and the DP's exhaustive BFS."""
    assert not is_terminal(state), "peek() called on an absorbing state"
    delay, move = _parse(aid)
    target = target_of(state.pos, move)
    DONE, FAILED = done_state(instance), failed_state(instance)

    if not instance.in_bounds(target):
        return StepResult(state, ILLEGAL_PENALTY, 0, legal=False, newly_done=False, newly_failed=False)

    elapsed = delay + 1
    new_clocks = list(c + elapsed for c in state.clocks)
    new_y = state.y + elapsed

    if instance.is_gate(target):
        lo, hi = instance.gate_window(target)
        idx = instance.gate_index(target)
        val = new_clocks[idx]
        if not (lo <= val <= hi):
            return StepResult(state, ILLEGAL_PENALTY, 0, legal=False, newly_done=False, newly_failed=False)
        # entering a gate does not reset anything
    elif instance.is_switch(target):
        new_clocks[instance.switch_index(target)] = 0

    if target == instance.goal and new_y <= instance.deadline:
        cost = TIME_COST * elapsed - REWARD_SUCCESS
        return StepResult(DONE, cost, elapsed, legal=True, newly_done=True, newly_failed=False)
    if new_y > instance.deadline:
        cost = TIME_COST * elapsed + DEADLINE_FAIL_PENALTY
        return StepResult(FAILED, cost, elapsed, legal=True, newly_done=False, newly_failed=True)

    next_state = GridState(pos=target, clocks=tuple(new_clocks), y=new_y)
    return StepResult(next_state, TIME_COST * elapsed, elapsed, legal=True, newly_done=False, newly_failed=False)


def action_mask(instance: GridInstance, state: GridState) -> Dict[str, bool]:
    return {aid: peek(instance, state, aid).legal for aid in all_actions(instance)}


def step(instance: GridInstance, state: GridState, aid: str) -> StepResult:
    assert aid in all_actions(instance), f"{aid} not in the fixed action set for this instance"
    return peek(instance, state, aid)
