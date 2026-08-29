"""Deadline-augmented job-shop instance model.

Scratch fork of ``prob_jobshop_ext.instance`` -- ``prob_jobshop_ext`` itself
is never imported for mutation, only read from (its ``Task``/``Job``/
``TaskDistribution``/``ProbJobShopInstance`` classes and its
``benchmarks.all_instances.get_instance_by_name`` loader). This module adds
per-job deadlines/release-times on top without touching the original.

"Nominal duration" of a task, used both to generate ``D_J`` and later (in
``decision_relevance.py``) to compute remaining-work slack, is defined here
as ``task.distribution.expected_duration()`` -- there is no single
canonical "unperturbed" duration field on ``TaskDistribution`` itself (the
``benchmarks/_utils.py`` formulas that generated these distributions don't
retain their pre-formula base value), so the probability-weighted mean is
the most defensible directly-computable stand-in. Stated explicitly here
since it's a modeling choice, not a given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prob_jobshop_ext.instance import Job, ProbJobShopInstance, Task, TaskDistribution

#: Scale factor for encoding fractional deadlines as pyudbm-integer bounds.
#: Task durations are always integers, so concrete elapsed time (and hence
#: every reachable g_J value) is always an exact integer already; only
#: D_J = nominal_work * (1+t) can be fractional. Scaling *every* clock
#: constraint in the zone by SCALE (not just the deadline ones) keeps one
#: consistent tick unit across the whole Federation, since up() advances
#: all clocks together and can't be given a per-clock rate.
SCALE = 20


def load_base_instance(name: str) -> ProbJobShopInstance:
    """Load a base (no-deadline) instance by name from either the small
    (PJS_*) or large (LJS_*) registry."""
    from prob_jobshop_ext.benchmarks.all_instances import ALL_INSTANCE_NAMES
    from prob_jobshop_ext.benchmarks.all_instances import get_instance_by_name as _get_small
    from prob_jobshop_ext.benchmarks.all_large_instances import get_large_instance_by_name as _get_large

    if name in ALL_INSTANCE_NAMES:
        return _get_small(name)
    return _get_large(name)


def nominal_duration(task: Task) -> float:
    return task.distribution.expected_duration()


def nominal_work(job: Job, from_index: int = 0) -> float:
    """Sum of nominal durations of job.tasks[from_index:] -- i.e. remaining
    nominal work if a job is partway through, or the whole job's nominal
    work if from_index=0."""
    return sum(nominal_duration(t) for t in job.tasks[from_index:])


@dataclass
class DeadlineJobShopInstance(ProbJobShopInstance):
    #: job_id -> D_J, the *operational* (already SCALE-rounded, but reported
    #: back in real units: operational_deadline = round(D_J_raw*SCALE)/SCALE)
    #: deadline used by both the concrete simulation and the zone check.
    deadlines: Dict[str, float] = field(default_factory=dict)
    #: job_id -> r_J, release time (default 0), same rounding convention.
    release: Dict[str, float] = field(default_factory=dict)
    tightness: float = 0.0
    penalty: float = 0.0
    base_name: str = ""

    def deadline_of(self, job_id: str) -> float:
        return self.deadlines[job_id]

    def release_of(self, job_id: str) -> float:
        return self.release.get(job_id, 0.0)

    def deadline_scaled(self, job_id: str) -> int:
        return round(self.deadlines[job_id] * SCALE)

    def release_scaled(self, job_id: str) -> int:
        return round(self.release.get(job_id, 0.0) * SCALE)


def make_deadline_variant(
    base: ProbJobShopInstance,
    t: float,
    release_spread: Optional[List[int]] = None,
    penalty_mult: float = 3.0,
) -> DeadlineJobShopInstance:
    """Programmatically derive a deadline variant of ``base``.

    D_J = nominal_work(job J) * (1 + t), rounded to the operational
    SCALE-integer grid (both the raw and the rounded value are kept: the
    rounded one is what the env/zone actually enforce, the raw one is
    reported for readability).

    :param release_spread: if given, a list of integer release-time values
        cycled across jobs in job order (e.g. ``[0, 1, 2]`` on a 3-job
        instance gives r_J1=0, r_J2=1, r_J3=2). If ``None``, every job has
        r_J=0 (all released at episode start).
    :param penalty_mult: P = penalty_mult * base.expected_makespan_lower_bound()
        ("nominal makespan" = the standard critical-machine-load lower
        bound already defined on ``ProbJobShopInstance``).
    """
    deadlines: Dict[str, float] = {}
    release: Dict[str, float] = {}
    for j in base.jobs:
        w = nominal_work(j)
        d_raw = w * (1.0 + t)
        deadlines[j.job_id] = round(d_raw * SCALE) / SCALE

    if release_spread:
        for i, j in enumerate(base.jobs):
            r_raw = float(release_spread[i % len(release_spread)])
            release[j.job_id] = round(r_raw * SCALE) / SCALE
    else:
        release = {j.job_id: 0.0 for j in base.jobs}

    nominal_makespan = base.expected_makespan_lower_bound()
    penalty = penalty_mult * nominal_makespan

    suffix = f"_t{t:g}" + ("_staggered" if release_spread else "")
    return DeadlineJobShopInstance(
        name=base.name + suffix,
        description=base.description + f" [deadline variant t={t}, P={penalty:.2f}]",
        machines=list(base.machines),
        jobs=base.jobs,  # tasks/jobs themselves are shared, read-only
        deadlines=deadlines,
        release=release,
        tightness=t,
        penalty=penalty,
        base_name=base.name,
    )
