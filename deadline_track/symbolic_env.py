"""Deadline-augmented job-shop symbolic environment.

Fork of ``prob_jobshop_ext.symbolic_env.SymbolicJobShopEnv``. Adds:

* A ``failed`` per-task status (set on every not-yet-dispatched task of a
  job the moment that job's deadline becomes provably unreachable). A job
  can only be discovered infeasible when checking its next ready task for
  dispatch (a job has at most one *active* task at a time -- precedence is
  strictly sequential within a job -- so there is never an active task
  belonging to an already-failed job; only "waiting" tasks need relabeling).
* Per-job deadline clocks (``g_J``, real elapsed time since episode start,
  concrete side -- the zone-side counterpart lives in
  ``deadline_track.zone_ops``), advanced by the same ``dt`` as
  ``_virtual_clocks`` at every micro-step, never reset.
* Feasibility is checked *on the edge*: only when a task becomes an
  otherwise-legal dispatch candidate (precedence satisfied, machine free).
  It is never imposed as an unconditional per-location invariant (that
  would block time-elapse for every other, unrelated job the moment one
  deadline is exceeded and deadlock the whole automaton). Infeasible jobs
  are auto-failed -- this is not an agent-chosen action, it happens eagerly
  while building the action list, precisely so the environment can never
  reach a decision point with zero available actions purely because of an
  expired deadline (see ``calibration.py``'s "no action available" check,
  which exists to catch exactly this failure mode if this eager-fail logic
  is broken).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .instance import SCALE, DeadlineJobShopInstance


class DeadlineSymbolicJobShopEnv:
    def __init__(self, instance: DeadlineJobShopInstance, seed: Optional[int] = None):
        self.instance = instance
        self._task_ids: List[str] = [t.task_id for j in instance.jobs for t in j.tasks]
        self._task_to_idx = {tid: i for i, tid in enumerate(self._task_ids)}
        self._task_job: Dict[str, str] = {t.task_id: j.job_id for j in instance.jobs for t in j.tasks}
        self._job_task_ids: Dict[str, List[str]] = {j.job_id: [t.task_id for t in j.tasks] for j in instance.jobs}
        self._min_dur: Dict[str, int] = {
            t.task_id: t.distribution.min_duration() for j in instance.jobs for t in j.tasks
        }

        self._rng = np.random.default_rng(seed)

        self._loc: Optional[tuple] = None
        self._virtual_clocks: Dict[str, float] = {}  # c_J, job_id -> elapsed since last task start
        self._g_values: Dict[str, float] = {}  # g_J, job_id -> elapsed since episode start, never reset
        self._active_d: Dict[str, int] = {}
        self._last_advance_path: List[dict] = []
        self._failed_jobs: Set[str] = set()

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[tuple, List[dict], dict]:
        self._loc = tuple("waiting" for _ in self._task_ids)
        self._virtual_clocks = {j.job_id: 0.0 for j in self.instance.jobs}
        self._g_values = {j.job_id: 0.0 for j in self.instance.jobs}
        self._active_d = {}
        self._failed_jobs = set()
        loc, actions, newly_failed = self._advance_and_resolve(list(self._loc))
        self._loc = loc
        return self._loc, actions, {"newly_failed": newly_failed}

    def step(self, task_id: str) -> Tuple[tuple, List[dict], bool, dict]:
        loc = list(self._loc)
        idx = self._task_to_idx[task_id]
        assert loc[idx] == "waiting", f"Task {task_id} is not waiting"

        dist = self.instance.task_by_id(task_id).distribution
        d = int(self._rng.choice(dist.durations, p=dist.probabilities))

        loc[idx] = f"active_d{d}"
        job_id = self._task_job[task_id]
        self._virtual_clocks[job_id] = 0.0
        self._active_d[job_id] = d

        loc, actions, newly_failed = self._advance_and_resolve(loc)
        self._loc = loc

        done = all(s in ("done", "failed") for s in self._loc)
        return self._loc, actions, done, {"sampled_d": d, "newly_failed": newly_failed}

    def step_deterministic(self, task_id: str, d: int) -> Tuple[tuple, List[dict], bool, dict]:
        """Dispatch task_id with an exact, caller-chosen duration d (no
        sampling) -- used by the exact DP solver to branch over every
        possible duration outcome rather than following one random rollout."""
        loc = list(self._loc)
        idx = self._task_to_idx[task_id]
        assert loc[idx] == "waiting", f"Task {task_id} is not waiting"

        loc[idx] = f"active_d{d}"
        job_id = self._task_job[task_id]
        self._virtual_clocks[job_id] = 0.0
        self._active_d[job_id] = d

        loc, actions, newly_failed = self._advance_and_resolve(loc)
        self._loc = loc

        done = all(s in ("done", "failed") for s in self._loc)
        return self._loc, actions, done, {"newly_failed": newly_failed}

    def save_state(self) -> tuple:
        return (
            self._loc,
            dict(self._virtual_clocks),
            dict(self._g_values),
            dict(self._active_d),
            set(self._failed_jobs),
        )

    def restore_state(self, snapshot: tuple) -> None:
        self._loc, vc, gv, ad, fj = snapshot
        self._virtual_clocks = dict(vc)
        self._g_values = dict(gv)
        self._active_d = dict(ad)
        self._failed_jobs = set(fj)

    def invariant(self, loc: tuple) -> List[Tuple[str, int]]:
        """Task-completion invariants only (scaled) -- deliberately no
        blanket g_J <= D_J invariant here, see module docstring."""
        result = []
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active_d"):
                job_id = self._task_job[tid]
                exact_d = int(status[len("active_d"):])
                result.append((f"c_{job_id}", exact_d * SCALE))
        return result

    def rate(self, loc: tuple) -> int:
        return 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _has_dispatch_pure(self, loc) -> bool:
        active_machines = set()
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active"):
                active_machines.add(self.instance.task_by_id(tid).machine)
        for tid, status in zip(self._task_ids, loc):
            if status != "waiting":
                continue
            pred_id = self.instance.predecessor_of(tid)
            if pred_id is not None and loc[self._task_to_idx[pred_id]] != "done":
                continue
            if self.instance.task_by_id(tid).machine not in active_machines:
                return True
        return False

    def _advance_and_resolve(self, loc: List[str]) -> Tuple[tuple, List[dict], List[str]]:
        """Loop advance-time <-> resolve-actions until either a real
        dispatch action exists or the episode is terminal.

        This loop exists because ``_advance_to_decision`` only checks
        precedence/machine readiness (``_has_dispatch_pure``), not deadline
        feasibility -- it can correctly stop with some task "ready" that
        ``_resolve_actions`` then discovers is infeasible and auto-fails.
        If *every* currently-ready task gets auto-failed this pass while
        other machines still have tasks actively running, the environment
        is not stuck, it just needs to advance time further for those to
        finish -- a single advance+resolve pass would incorrectly report a
        deadlock (zero available actions, not terminal) in that case. Capped
        at ``len(self._task_ids) + 1`` iterations (an upper bound on how
        many times a well-formed instance's task set can be fully
        consumed); exceeding it is asserted as a genuine bug, not silently
        handled, per the task's "flag it as a bug, do not work around it."
        """
        all_newly_failed: List[str] = []
        for _ in range(len(self._task_ids) + 1):
            loc = self._advance_to_decision(loc)
            loc_t, actions, newly_failed = self._resolve_actions(loc)
            loc = list(loc_t)
            all_newly_failed.extend(newly_failed)
            if actions:
                return tuple(loc), actions, all_newly_failed
            if all(s in ("done", "failed") for s in loc):
                return tuple(loc), actions, all_newly_failed
            if not newly_failed:
                raise RuntimeError(
                    "DEADLINE_TRACK BUG: _advance_and_resolve made no progress "
                    f"(no new actions, no new failures, not terminal) at loc={tuple(loc)}"
                )
        raise RuntimeError("DEADLINE_TRACK BUG: _advance_and_resolve exceeded iteration cap")

    def _check_feasible(self, job_id: str, ready_task_id: str) -> bool:
        """Arithmetic-shortcut feasibility (operational, used live by the
        env): is there any duration assignment for job_id's remaining tasks
        (starting at ready_task_id) that keeps g_{job_id} <= D_{job_id}?
        Since g_J is never reset and always advances at rate 1 regardless
        of which job's task is running (module docstring / zone_ops.py),
        this reduces to pure best-case-duration arithmetic and is provably
        equivalent to the real zone-based check
        ``zone_ops.feasible_best_case`` -- see ``sanity_gate.py`` and
        ``calibration.py`` for where that equivalence is verified on
        sampled states rather than assumed. Using the arithmetic form live
        (instead of rebuilding a Federation on every dispatch-candidate
        check) keeps a 500-episode x 6-instance x 5-tightness calibration
        sweep fast.
        """
        job_tasks = self._job_task_ids[job_id]
        start = job_tasks.index(ready_task_id)
        best_case = sum(self._min_dur[tid] for tid in job_tasks[start:])
        g_now = self._g_values[job_id]
        lhs_scaled = int(round((g_now + best_case) * SCALE))
        rhs_scaled = self.instance.deadline_scaled(job_id)
        return lhs_scaled <= rhs_scaled

    def _resolve_actions(self, loc: List[str]) -> Tuple[tuple, List[dict], List[str]]:
        """Single pass: for every otherwise-ready task, check its job's
        feasibility; infeasible -> fail the whole job (mark its remaining
        "waiting" tasks failed, no action offered); feasible -> real
        dispatch action, branched over the task's duration distribution
        exactly as the base package does."""
        active_machines = set()
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active"):
                active_machines.add(self.instance.task_by_id(tid).machine)

        newly_failed: List[str] = []
        actions: List[dict] = []

        for tid, status in zip(self._task_ids, loc):
            if status != "waiting":
                continue
            pred_id = self.instance.predecessor_of(tid)
            if pred_id is not None and loc[self._task_to_idx[pred_id]] != "done":
                continue
            task = self.instance.task_by_id(tid)
            if task.machine in active_machines:
                continue

            job_id = self._task_job[tid]
            if job_id in self._failed_jobs:
                continue  # already failed earlier this same resolve pass -- shouldn't happen, defensive

            if not self._check_feasible(job_id, tid):
                for jt in self._job_task_ids[job_id]:
                    jt_idx = self._task_to_idx[jt]
                    if loc[jt_idx] == "waiting":
                        loc[jt_idx] = "failed"
                self._failed_jobs.add(job_id)
                newly_failed.append(job_id)
                continue

            dist = task.distribution
            branches = []
            for d, prob in zip(dist.durations, dist.probabilities):
                next_loc_imm = list(loc)
                next_loc_imm[self._task_to_idx[tid]] = f"active_d{d}"
                branches.append({"d": d, "prob": prob, "next_loc": tuple(next_loc_imm)})
            actions.append({"action_id": tid, "resets": [f"c_{job_id}"], "branches": branches})

        return tuple(loc), actions, newly_failed

    def _advance_to_decision(self, loc: list) -> list:
        #: Micro-step path recorded for this call, consumed by
        #: ``deadline_track.collect`` to replay the exact same completion
        #: sequence into a symbolic Federation. Each entry:
        #: {"completing_tasks": [task_id, ...], "loc_before": tuple, "loc_after": tuple}
        #: -- deliberately no "dt" field: dt is not needed to build the
        #: zone (invariant/guard bounds are exact durations, not dt), and
        #: dt is float-accumulated internally here in a way that isn't
        #: guaranteed to exactly equal the SCALE-integer arithmetic
        #: zone_ops.py uses -- recomputing bounds from exact_d avoids ever
        #: needing the two to agree bit-for-bit.
        self._last_advance_path: List[dict] = []
        while True:
            if self._has_dispatch_pure(loc):
                break
            if all(s in ("done", "failed") for s in loc):
                break

            remaining = {}
            for tid, status in zip(self._task_ids, loc):
                if status.startswith("active_d"):
                    job_id = self._task_job[tid]
                    exact_d = int(status[len("active_d"):])
                    rem = exact_d - self._virtual_clocks.get(job_id, 0.0)
                    remaining[tid] = max(0.0, rem)

            if not remaining:
                break

            dt = min(remaining.values())
            for job_id in self._virtual_clocks:
                self._virtual_clocks[job_id] += dt
            for job_id in self._g_values:
                self._g_values[job_id] += dt

            loc_before = tuple(loc)
            completing_tasks = []
            for tid, rem in remaining.items():
                if abs(rem - dt) < 1e-9:
                    idx = self._task_to_idx[tid]
                    loc[idx] = "done"
                    job_id = self._task_job[tid]
                    self._active_d.pop(job_id, None)
                    completing_tasks.append(tid)

            self._last_advance_path.append(
                {"completing_tasks": completing_tasks, "loc_before": loc_before, "loc_after": tuple(loc)}
            )

        return loc
