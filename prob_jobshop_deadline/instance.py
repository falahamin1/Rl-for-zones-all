from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskDistribution:
    durations: List[int]
    probabilities: List[float]

    def __post_init__(self):
        if len(self.durations) != len(self.probabilities):
            raise ValueError(
                f"durations and probabilities must have the same length, "
                f"got {len(self.durations)} and {len(self.probabilities)}"
            )

    def expected_duration(self) -> float:
        return sum(d * p for d, p in zip(self.durations, self.probabilities))

    def max_duration(self) -> int:
        return max(self.durations)

    def min_duration(self) -> int:
        return min(self.durations)


@dataclass
class Task:
    task_id: str    # format: "J1_T1"
    machine: str    # format: "m1"
    distribution: TaskDistribution


@dataclass
class Job:
    job_id: str     # format: "J1"
    tasks: List[Task]
    #: Absolute deadline measured against g_J (the never-resetting per-job
    #: clock introduced for Track 2), i.e. wall-clock time since episode
    #: start -- NOT against c_J (which resets on every task start). ``None``
    #: means this job has no deadline at all (unconstrained, behaves exactly
    #: like the no-deadline model).
    deadline: Optional[int] = None
    #: Release time, also measured against g_J: job cannot be dispatched
    #: before g_J >= release. Defaults to 0 (dispatchable from the start,
    #: same as the no-deadline model).
    release: int = 0

    def nominal_work(self) -> float:
        """Sum of expected task durations in this job -- the "nominal
        work" W_J used both to derive D_J = W_J * (1 + t) and, later, to
        compute per-job slack = (D_J - g_J) - W_J_remaining."""
        return sum(t.distribution.expected_duration() for t in self.tasks)


@dataclass
class ProbJobShopInstance:
    name: str
    description: str
    machines: List[str]
    jobs: List[Job]

    # Lookup caches — built in __post_init__, not declared as dataclass fields
    def __post_init__(self):
        self._task_index: Dict[str, Task] = {}
        self._task_to_job: Dict[str, Job] = {}
        self._predecessor: Dict[str, Optional[str]] = {}
        for job in self.jobs:
            for i, task in enumerate(job.tasks):
                self._task_index[task.task_id] = task
                self._task_to_job[task.task_id] = job
                self._predecessor[task.task_id] = job.tasks[i - 1].task_id if i > 0 else None

    def all_tasks(self) -> List[Task]:
        return [task for job in self.jobs for task in job.tasks]

    def task_by_id(self, task_id: str) -> Task:
        return self._task_index[task_id]

    def job_of_task(self, task_id: str) -> Job:
        return self._task_to_job[task_id]

    def predecessor_of(self, task_id: str) -> Optional[str]:
        return self._predecessor[task_id]

    def expected_makespan_lower_bound(self) -> float:
        machine_load: Dict[str, float] = {m: 0.0 for m in self.machines}
        for task in self.all_tasks():
            machine_load[task.machine] += task.distribution.expected_duration()
        return max(machine_load.values())

    def worst_case_makespan_upper_bound(self) -> int:
        return sum(task.distribution.max_duration() for task in self.all_tasks())

    def num_tasks(self) -> int:
        return sum(len(job.tasks) for job in self.jobs)

    def mean_expected_duration(self) -> float:
        tasks = self.all_tasks()
        return sum(t.distribution.expected_duration() for t in tasks) / len(tasks)

    def remaining_nominal_work(self, job: "Job", from_task_id: Optional[str] = None) -> float:
        """Sum of expected durations for ``job``'s tasks from ``from_task_id``
        onward (inclusive); with ``from_task_id=None``, the task after
        whichever ones are already "done"/"failed" is used implicitly by the
        caller passing the right task id. Used for W_J in slack_J = (D_J -
        g_J) - W_J.
        """
        if from_task_id is None:
            return job.nominal_work()
        idx = next(i for i, t in enumerate(job.tasks) if t.task_id == from_task_id)
        return sum(t.distribution.expected_duration() for t in job.tasks[idx:])


def make_deadline_variant(
    base: ProbJobShopInstance,
    t: float,
    release_spread: Optional[List[int]] = None,
    name_suffix: Optional[str] = None,
) -> ProbJobShopInstance:
    """Programmatically derive a deadline variant of ``base``.

    Per job J: ``D_J = round(W_J * (1 + t))`` where ``W_J`` is the job's
    nominal (expected) work (see :meth:`Job.nominal_work`); deadlines are
    later checked against ``g_J``, the never-resetting clock, not ``c_J``
    -- see ``deadline_env.py``. ``t=0`` means "exactly as tight as the
    nominal work allows" (near-guaranteed violations under a bad-luck
    duration draw or any contention delay); larger ``t`` gives more slack.

    :param release_spread: optional list of release times, one per job in
        ``base.jobs`` order (cycled if shorter than the job count), applied
        as each job's ``release``. ``None`` means every job keeps the
        default ``release=0``.
    """
    import copy

    jobs = []
    for i, job in enumerate(base.jobs):
        new_job = copy.deepcopy(job)
        w_j = new_job.nominal_work()
        new_job.deadline = int(round(w_j * (1.0 + t)))
        if release_spread:
            new_job.release = release_spread[i % len(release_spread)]
        jobs.append(new_job)

    suffix = name_suffix if name_suffix is not None else f"_deadline_t{t}"
    return ProbJobShopInstance(
        name=base.name + suffix,
        description=base.description + f" (deadline variant, t={t})",
        machines=list(base.machines),
        jobs=jobs,
    )
