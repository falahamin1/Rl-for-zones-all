"""The mode-selection instance: N jobs, each a sequential chain of tasks,
each task needing a specific machine (machines serve one task at a time).
Dispatching a task also chooses a MODE: FAST (short, costs ``cost_penalty``)
or SLOW (long, free). A global clock ``g`` never resets; each job has its
own deadline ``D_j``.

Durations are nominal integers (``d_fast`` < ``d_slow``) for the exact DP's
own bookkeeping -- matching this repo's established convention (gate_track,
gridtimed's Task A fix): the DP transitions deterministically on the
nominal value; only the zone construction (modesel/zones.py, Step 4) uses
the true real-valued interval ``[d_m*(1-w), d_m*(1+w))]``. Concrete
simulation stays in exact integer units throughout -- SCALE=20 exists only
at the pyudbm boundary in zone_ops.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

FAST = "FAST"
SLOW = "SLOW"
MODES = (FAST, SLOW)

SCALE = 20


@dataclass(frozen=True)
class Task:
    task_id: str
    job_id: str
    index: int  # position in the job's chain, 0-based
    machine: str
    d_fast: int
    d_slow: int

    def duration(self, mode: str) -> int:
        return self.d_fast if mode == FAST else self.d_slow


@dataclass(frozen=True)
class Job:
    job_id: str
    tasks: Tuple[Task, ...]  # sequential chain, index 0..T-1
    deadline: int  # D_j

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def slow_makespan(self) -> int:
        return sum(t.d_slow for t in self.tasks)

    def fast_makespan(self) -> int:
        return sum(t.d_fast for t in self.tasks)


@dataclass(frozen=True)
class ModeSelInstance:
    jobs: Tuple[Job, ...]
    cost_penalty: float  # c_fast, applied once per FAST dispatch, any task
    w: float = 0.0
    seed: int = 0
    label: str = "M0"

    def __post_init__(self):
        for job in self.jobs:
            for i, t in enumerate(job.tasks):
                assert t.index == i
                assert t.job_id == job.job_id
                assert t.d_fast < t.d_slow, f"{t.task_id}: d_fast must be < d_slow"
        object.__setattr__(self, "_task_index", {t.task_id: t for t in self.all_tasks})
        object.__setattr__(self, "_job_by_id_map", {j.job_id: j for j in self.jobs})
        object.__setattr__(self, "_job_position", {j.job_id: i for i, j in enumerate(self.jobs)})

    @property
    def all_tasks(self) -> List[Task]:
        return [t for j in self.jobs for t in j.tasks]

    @property
    def n_jobs(self) -> int:
        return len(self.jobs)

    @property
    def n_tasks_total(self) -> int:
        return sum(j.n_tasks for j in self.jobs)

    def task_by_id(self, task_id: str) -> Task:
        return self._task_index[task_id]

    def job_by_id(self, job_id: str) -> Job:
        return self._job_by_id_map[job_id]

    def job_index(self, job_id: str) -> int:
        return self._job_position[job_id]

    def failure_penalty(self, job_id: str) -> float:
        # P_j = 3x the job's own worst-case (all-SLOW) makespan -- same
        # convention as deadline_track/gate_track's P=3x nominal makespan.
        return 3.0 * self.job_by_id(job_id).slow_makespan()

    def action_id(self, task_id: str, mode: str) -> str:
        return f"{task_id}_{mode}"


def action_id(task_id: str, mode: str) -> str:
    return f"{task_id}_{mode}"


def parse_action_id(aid: str) -> Tuple[str, str]:
    task_id, mode = aid.rsplit("_", 1)
    return task_id, mode
