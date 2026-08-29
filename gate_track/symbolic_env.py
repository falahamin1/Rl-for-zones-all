"""Concrete environment: interval-valued durations + single global deadline
clock g.

Two duration-sampling paths, deliberately different (stated explicitly,
not left implicit):

* ``step()`` (used by the random-policy calibration/violation-rate sweeps
  and by zone sampling) draws a REAL number uniformly on
  ``[lo, hi] = [d*(1-w), d*(1+w)]`` for the chosen nominal branch ``d`` --
  per the task, integer/nominal-only sampling here would silently
  reproduce the old degenerate (single-point-zone) case.
* ``step_deterministic(task_id, d)`` (used by the exact DP, Measurement 2)
  treats the branch's nominal value ``d`` itself as the realized duration
  for transition/ordering/elapsed-time purposes. This is not an
  approximation of convenience: ``[d*(1-w), d*(1+w)]`` is symmetric around
  ``d``, so ``E[Uniform(lo,hi)] == d`` exactly, for any ``w``. The DP's
  *cost* accounting is therefore exact under this choice. What it does NOT
  capture is order uncertainty among concurrently-active tasks whose
  intervals overlap (which of two tasks with nominal 3 and 4 actually
  finishes first is only certain up to interval overlap) -- the DP
  resolves ordering deterministically by nominal value, same tie-breaking
  as the non-interval model. This is a modeling choice, stated here and in
  the report, not hidden in the DP's state-space definition.

Zone geometry (built by ``collect.py``, not here) uses the true
``[lo, hi]`` interval for every completed task's guard/invariant
regardless of which sampling path produced the transition -- that's what
makes Measurement 1's zones non-degenerate even though DP transitions are
nominal-deterministic.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .instance import SCALE, GateJobShopInstance, interval


class GateSymbolicJobShopEnv:
    def __init__(self, instance: GateJobShopInstance, seed: Optional[int] = None):
        self.instance = instance
        self.w = instance.width
        self._task_ids: List[str] = [t.task_id for j in instance.jobs for t in j.tasks]
        self._task_to_idx = {tid: i for i, tid in enumerate(self._task_ids)}
        self._task_job: Dict[str, str] = {t.task_id: j.job_id for j in instance.jobs for t in j.tasks}
        self._job_task_ids: Dict[str, List[str]] = {j.job_id: [t.task_id for t in j.tasks] for j in instance.jobs}
        self._min_nominal: Dict[str, int] = {
            t.task_id: t.distribution.min_duration() for j in instance.jobs for t in j.tasks
        }

        self._rng = np.random.default_rng(seed)

        self._loc: Optional[tuple] = None
        self._virtual_clocks: Dict[str, float] = {}  # c_J
        self._g: float = 0.0  # single global deadline clock
        self._active_d: Dict[str, int] = {}  # job_id -> NOMINAL duration of active task
        self._realized_d: Dict[str, float] = {}  # job_id -> REAL realized duration of active task (concrete only)
        self._failed_jobs: Set[str] = set()
        self._last_advance_path: List[dict] = []

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[tuple, List[dict], dict]:
        self._loc = tuple("waiting" for _ in self._task_ids)
        self._virtual_clocks = {j.job_id: 0.0 for j in self.instance.jobs}
        self._g = 0.0
        self._active_d = {}
        self._realized_d = {}
        self._failed_jobs = set()
        loc, actions, newly_failed = self._advance_and_resolve(list(self._loc))
        self._loc = loc
        return self._loc, actions, {"newly_failed": newly_failed}

    def step(self, task_id: str) -> Tuple[tuple, List[dict], bool, dict]:
        """Real random-policy step: samples nominal d (per branch
        probabilities), then a REAL realized duration uniform on
        [d*(1-w), d*(1+w)]."""
        loc = list(self._loc)
        idx = self._task_to_idx[task_id]
        assert loc[idx] == "waiting", f"Task {task_id} is not waiting"

        dist = self.instance.task_by_id(task_id).distribution
        d = int(self._rng.choice(dist.durations, p=dist.probabilities))
        lo, hi = interval(d, self.w)
        realized = float(self._rng.uniform(lo, hi))

        loc[idx] = f"active_d{d}"
        job_id = self._task_job[task_id]
        self._virtual_clocks[job_id] = 0.0
        self._active_d[job_id] = d
        self._realized_d[job_id] = realized

        loc, actions, newly_failed = self._advance_and_resolve(loc)
        self._loc = loc

        done = all(s in ("done", "failed") for s in self._loc)
        return self._loc, actions, done, {"sampled_d": d, "realized_d": realized, "newly_failed": newly_failed}

    def step_deterministic(self, task_id: str, d: int) -> Tuple[tuple, List[dict], bool, dict]:
        """DP step: nominal d IS the realized duration -- see module
        docstring for why this is exact for expected-cost purposes."""
        loc = list(self._loc)
        idx = self._task_to_idx[task_id]
        assert loc[idx] == "waiting", f"Task {task_id} is not waiting"

        loc[idx] = f"active_d{d}"
        job_id = self._task_job[task_id]
        self._virtual_clocks[job_id] = 0.0
        self._active_d[job_id] = d
        self._realized_d[job_id] = float(d)

        loc, actions, newly_failed = self._advance_and_resolve(loc)
        self._loc = loc

        done = all(s in ("done", "failed") for s in self._loc)
        return self._loc, actions, done, {"newly_failed": newly_failed}

    def save_state(self) -> tuple:
        return (
            self._loc,
            dict(self._virtual_clocks),
            self._g,
            dict(self._active_d),
            dict(self._realized_d),
            set(self._failed_jobs),
        )

    def restore_state(self, snapshot: tuple) -> None:
        self._loc, vc, g, ad, rd, fj = snapshot
        self._virtual_clocks = dict(vc)
        self._g = g
        self._active_d = dict(ad)
        self._realized_d = dict(rd)
        self._failed_jobs = set(fj)

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

    def _check_feasible(self, job_id: str, ready_task_id: str) -> bool:
        """Arithmetic-shortcut feasibility, re-derived for this model (see
        module docstring + gate_track/validate_feasibility.py): the global
        clock g is now literally shared (a single scalar, not merely
        provably-equal-by-dynamics across per-job copies), so
        g_now + best_case_remaining <= D_J is immediately the right
        condition with no separate "are the g's actually equal" argument
        needed. best_case per remaining task uses its SMALLEST nominal
        duration scaled by (1-w) -- the most optimistic realizable value
        under the interval model."""
        job_tasks = self._job_task_ids[job_id]
        start = job_tasks.index(ready_task_id)
        best_case = sum(self._min_nominal[tid] * (1.0 - self.w) for tid in job_tasks[start:])
        lhs_scaled = int(round((self._g + best_case) * SCALE))
        rhs_scaled = self.instance.deadline_scaled(job_id)
        return lhs_scaled <= rhs_scaled

    def _advance_and_resolve(self, loc: List[str]) -> Tuple[tuple, List[dict], List[str]]:
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
                    "GATE_TRACK BUG: _advance_and_resolve made no progress "
                    f"(no new actions, no new failures, not terminal) at loc={tuple(loc)}"
                )
        raise RuntimeError("GATE_TRACK BUG: _advance_and_resolve exceeded iteration cap")

    def _resolve_actions(self, loc: List[str]) -> Tuple[tuple, List[dict], List[str]]:
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
                continue

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
        self._last_advance_path = []
        while True:
            if self._has_dispatch_pure(loc):
                break
            if all(s in ("done", "failed") for s in loc):
                break

            remaining = {}
            for tid, status in zip(self._task_ids, loc):
                if status.startswith("active_d"):
                    job_id = self._task_job[tid]
                    rem = self._realized_d[job_id] - self._virtual_clocks.get(job_id, 0.0)
                    remaining[tid] = max(0.0, rem)

            if not remaining:
                break

            dt = min(remaining.values())
            for job_id in self._virtual_clocks:
                self._virtual_clocks[job_id] += dt
            self._g += dt

            loc_before = tuple(loc)
            completing_tasks = []
            for tid, rem in remaining.items():
                if abs(rem - dt) < 1e-9:
                    idx = self._task_to_idx[tid]
                    loc[idx] = "done"
                    job_id = self._task_job[tid]
                    self._active_d.pop(job_id, None)
                    self._realized_d.pop(job_id, None)
                    completing_tasks.append(tid)

            self._last_advance_path.append(
                {"completing_tasks": completing_tasks, "loc_before": loc_before, "loc_after": tuple(loc)}
            )

        return loc
