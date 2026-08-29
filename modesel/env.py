"""Concrete simulator: dispatch (task_id, mode), then auto-advance time to
the next decision point.

Feasibility is checked ON THE EDGE, in exactly two places, never as a
blanket invariant (a blanket invariant -- e.g. forbidding g from ever
exceeding some bound at every instant -- would deadlock the time-advance
step the moment any job's remaining work can't fit, even transiently):

1. At DISPATCH time: a (task, mode) action is only ever offered/legal if
   dispatching it keeps that job's own best-case completion (this task's
   full committed duration + FAST for whatever's left after it) within its
   deadline. A dispatched task's duration is fixed thereafter (no more
   mode choice for it), so once legal, it stays feasible forever -- an
   already-dispatched task can never later become the reason a job fails.
2. During ADVANCE (time passing while other jobs run): every WAITING,
   non-done, non-failed job is re-checked after each time step -- if even
   its best possible remaining path (FAST for everything left, machine
   contention ignored -- a documented simplification matching
   deadline_track/gate_track's precedent) can no longer fit, it fails
   right there, once, with its own penalty, and is dropped from further
   consideration ("the invariant is dropped there").

``JobState.active_duration`` holds the ACTIVE task's realized duration
directly (not just its mode) so the advance loop is identical whether that
duration came from the deterministic nominal value (``step_deterministic``,
used by the exact DP) or a real uniform sample (``step``, used for
calibration) -- see instance.py's module docstring for why the DP stays on
nominal values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .instance import FAST, MODES, ModeSelInstance, parse_action_id

TIME_COST = 1.0
COST_EPS = 1e-9


@dataclass(frozen=True)
class JobState:
    cursor: int
    active_mode: Optional[str]
    active_duration: Optional[float]
    c: float
    failed: bool


@dataclass(frozen=True)
class ModeSelState:
    jobs: Tuple[JobState, ...]
    g: float


def initial_state(instance: ModeSelInstance) -> ModeSelState:
    return ModeSelState(
        jobs=tuple(JobState(cursor=0, active_mode=None, active_duration=None, c=0, failed=False) for _ in instance.jobs),
        g=0,
    )


def is_terminal(instance: ModeSelInstance, state: ModeSelState) -> bool:
    return all(
        js.failed or js.cursor == instance.jobs[i].n_tasks
        for i, js in enumerate(state.jobs)
    )


def all_actions(instance: ModeSelInstance) -> List[str]:
    return [instance.action_id(t.task_id, m) for t in instance.all_tasks for m in MODES]


def dispatch_is_feasible(instance: ModeSelInstance, job_idx: int, state: ModeSelState, mode: str) -> bool:
    job = instance.jobs[job_idx]
    js = state.jobs[job_idx]
    task = job.tasks[js.cursor]
    remaining = task.duration(mode) + sum(t.d_fast for t in job.tasks[js.cursor + 1:])
    return state.g + remaining <= job.deadline + COST_EPS


def waiting_job_is_feasible(instance: ModeSelInstance, job_idx: int, state: ModeSelState) -> bool:
    """Best-case (all-FAST from here, machine contention ignored --
    documented simplification, see module docstring) feasibility check for
    a currently-waiting job."""
    job = instance.jobs[job_idx]
    js = state.jobs[job_idx]
    remaining = sum(t.d_fast for t in job.tasks[js.cursor:])
    return state.g + remaining <= job.deadline + COST_EPS


def is_legal(instance: ModeSelInstance, state: ModeSelState, task_id: str, mode: str) -> bool:
    task = instance.task_by_id(task_id)
    job_idx = instance.job_index(task.job_id)
    js = state.jobs[job_idx]
    if js.failed or js.cursor != task.index or js.active_mode is not None:
        return False
    for k, other_js in enumerate(state.jobs):
        if k == job_idx or other_js.failed or other_js.active_mode is None:
            continue
        other_job = instance.jobs[k]
        if other_js.cursor < other_job.n_tasks and other_job.tasks[other_js.cursor].machine == task.machine:
            return False
    return dispatch_is_feasible(instance, job_idx, state, mode)


def action_mask(instance: ModeSelInstance, state: ModeSelState) -> Dict[str, bool]:
    if is_terminal(instance, state):
        return {aid: False for aid in all_actions(instance)}
    mask = {}
    for t in instance.all_tasks:
        for m in MODES:
            mask[instance.action_id(t.task_id, m)] = is_legal(instance, state, t.task_id, m)
    return mask


def _has_legal_action(instance: ModeSelInstance, state: ModeSelState) -> bool:
    for t in instance.all_tasks:
        for m in MODES:
            if is_legal(instance, state, t.task_id, m):
                return True
    return False


def _dispatch(instance: ModeSelInstance, state: ModeSelState, job_idx: int, mode: str, duration) -> ModeSelState:
    jobs = list(state.jobs)
    js = jobs[job_idx]
    jobs[job_idx] = JobState(cursor=js.cursor, active_mode=mode, active_duration=duration, c=0, failed=False)
    return ModeSelState(jobs=tuple(jobs), g=state.g)


def advance_and_resolve(instance: ModeSelInstance, state: ModeSelState, max_iters: Optional[int] = None):
    """Repeatedly resolve automatic time passage until a decision point
    (a legal action exists) or the episode is terminal. Returns
    (state, cost, newly_failed_job_ids). Raises RuntimeError on a genuine
    deadlock (no active task AND no legal action, but not terminal -- a
    real bug, never expected here since every WAITING job's own task
    always frees its own machine once done, but asserted rather than
    assumed) or on exceeding the iteration cap (a real bug, not a modeling
    choice)."""
    cap = max_iters or (instance.n_tasks_total + instance.n_jobs + 2)
    total_cost = 0.0
    newly_failed: List[str] = []
    trace: List[AdvanceMicroStep] = []

    for _ in range(cap):
        if is_terminal(instance, state) or _has_legal_action(instance, state):
            return state, total_cost, newly_failed, trace

        candidates = [
            (js.active_duration - js.c, i)
            for i, js in enumerate(state.jobs)
            if not js.failed and js.active_mode is not None
        ]
        if not candidates:
            raise RuntimeError(
                f"advance_and_resolve: no active task and no legal action, but not terminal -- "
                f"a genuine deadlock, state={state}"
            )
        delta = min(r for r, _ in candidates)
        assert delta > COST_EPS, f"advance step must make positive progress, got delta={delta}"

        new_jobs = []
        completing_jobs = []
        for i, js in enumerate(state.jobs):
            if js.failed or js.active_mode is None:
                new_jobs.append(js)
                continue
            new_c = js.c + delta
            if new_c >= js.active_duration - COST_EPS:
                job = instance.jobs[i]
                new_jobs.append(JobState(cursor=js.cursor + 1, active_mode=None, active_duration=None, c=0, failed=False))
                completing_jobs.append(i)
            else:
                new_jobs.append(JobState(cursor=js.cursor, active_mode=js.active_mode, active_duration=js.active_duration, c=new_c, failed=False))
        state = ModeSelState(jobs=tuple(new_jobs), g=state.g + delta)
        total_cost += delta * TIME_COST

        # auto-fail check, on the edge (g just advanced) -- waiting jobs only
        new_jobs2 = list(state.jobs)
        newly_failed_this_step = []
        for i, js in enumerate(new_jobs2):
            job = instance.jobs[i]
            if js.failed or js.active_mode is not None or js.cursor == job.n_tasks:
                continue
            if not waiting_job_is_feasible(instance, i, state):
                new_jobs2[i] = JobState(cursor=js.cursor, active_mode=None, active_duration=None, c=0, failed=True)
                newly_failed.append(job.job_id)
                newly_failed_this_step.append(i)
                total_cost += instance.failure_penalty(job.job_id)
        state = ModeSelState(jobs=tuple(new_jobs2), g=state.g)
        trace.append(AdvanceMicroStep(delta=delta, completing_jobs=tuple(completing_jobs),
                                       newly_failed_jobs=tuple(newly_failed_this_step)))

    raise RuntimeError(f"advance_and_resolve: exceeded cap {cap} iterations without reaching a decision point")


@dataclass(frozen=True)
class AdvanceMicroStep:
    delta: float
    completing_jobs: Tuple[int, ...]
    newly_failed_jobs: Tuple[int, ...]


@dataclass(frozen=True)
class StepResult:
    next_state: ModeSelState
    cost: float
    done: bool
    newly_failed: Tuple[str, ...]
    dispatched_job_idx: int
    dispatched_mode: str
    trace: Tuple[AdvanceMicroStep, ...]


def step_deterministic(instance: ModeSelInstance, state: ModeSelState, task_id: str, mode: str) -> StepResult:
    """Dispatch using the NOMINAL (exact integer) duration for ``mode`` --
    the DP's own transition function."""
    if not is_legal(instance, state, task_id, mode):
        raise ValueError(f"{task_id}_{mode} is not legal in state {state}")
    task = instance.task_by_id(task_id)
    job_idx = instance.job_index(task.job_id)
    dispatch_cost = instance.cost_penalty if mode == FAST else 0.0
    s2 = _dispatch(instance, state, job_idx, mode, task.duration(mode))
    s3, advance_cost, newly_failed, trace = advance_and_resolve(instance, s2)
    return StepResult(s3, dispatch_cost + advance_cost, is_terminal(instance, s3), tuple(newly_failed),
                       job_idx, mode, tuple(trace))


def interval(d: float, w: float) -> Tuple[float, float]:
    return d * (1 - w), d * (1 + w)


def step(instance: ModeSelInstance, state: ModeSelState, task_id: str, mode: str, rng) -> StepResult:
    """Dispatch using a REAL duration sampled uniform on the mode's true
    interval ``[d*(1-w), d*(1+w)]`` -- used for calibration, NOT the exact
    DP (see module docstring). Integer/nominal-only sampling would
    silently reproduce the degenerate single-point-zone case."""
    if not is_legal(instance, state, task_id, mode):
        raise ValueError(f"{task_id}_{mode} is not legal in state {state}")
    task = instance.task_by_id(task_id)
    job_idx = instance.job_index(task.job_id)
    dispatch_cost = instance.cost_penalty if mode == FAST else 0.0
    lo, hi = interval(task.duration(mode), instance.w)
    realized = float(rng.uniform(lo, hi))
    s2 = _dispatch(instance, state, job_idx, mode, realized)
    s3, advance_cost, newly_failed, trace = advance_and_resolve(instance, s2)
    return StepResult(s3, dispatch_cost + advance_cost, is_terminal(instance, s3), tuple(newly_failed),
                       job_idx, mode, tuple(trace))
