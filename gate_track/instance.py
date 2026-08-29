"""Instance model for the geometry gate: single global deadline clock +
interval-valued durations.

Fork of ``deadline_track.instance`` (which is itself a fork of
``prob_jobshop_ext.instance`` -- neither is modified). Two changes from
``deadline_track``:

* One global deadline clock ``g`` (never reset) instead of one ``g_J`` per
  job -- ``deadline_track``'s own geometry measurement proved every ``g_J``
  is *always* exactly equal to every other (100.000% mutual pinning, zero
  exceptions, both release-time variants), so per-job copies were a real
  modeling error, not a design with headroom. A single clock is the
  faithful model of what was actually being represented.
* Interval-valued task durations, reinstating Spike 3's construction:
  ``[lo, hi] = [d*(1-w), d*(1+w)]`` for the discrete branch's nominal
  duration ``d`` and a module-level-style width knob ``w`` (a per-variant
  constant, swept externally across {0.0, 0.1, 0.25} by ``geometry.py`` --
  "module constant" in the sense of being fixed for a given instance
  variant, not re-drawn per task or per episode).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prob_jobshop_ext.instance import Job, ProbJobShopInstance, Task, TaskDistribution

from deadline_track.instance import load_base_instance  # noqa: F401  (re-exported)

SCALE = 20


def nominal_duration(task: Task) -> float:
    return task.distribution.expected_duration()


def nominal_work(job: Job, from_index: int = 0) -> float:
    return sum(nominal_duration(t) for t in job.tasks[from_index:])


def interval(d: float, w: float):
    """[lo, hi] = [d*(1-w), d*(1+w)]. Note E[Uniform(lo,hi)] == d exactly
    (symmetric interval), used throughout to justify treating the nominal
    branch value as the exact expected elapsed-time contribution for DP
    purposes even though the concrete simulation samples a real value."""
    return d * (1.0 - w), d * (1.0 + w)


@dataclass
class GateJobShopInstance(ProbJobShopInstance):
    deadlines: Dict[str, float] = field(default_factory=dict)
    release: Dict[str, float] = field(default_factory=dict)
    tightness: float = 0.0
    width: float = 0.0  # w
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


def make_gate_variant(
    base: ProbJobShopInstance,
    t: float,
    w: float,
    release_spread: Optional[List[int]] = None,
    penalty_mult: float = 3.0,
) -> GateJobShopInstance:
    deadlines: Dict[str, float] = {}
    release: Dict[str, float] = {}
    for j in base.jobs:
        wk = nominal_work(j)
        d_raw = wk * (1.0 + t)
        deadlines[j.job_id] = round(d_raw * SCALE) / SCALE

    if release_spread:
        for i, j in enumerate(base.jobs):
            r_raw = float(release_spread[i % len(release_spread)])
            release[j.job_id] = round(r_raw * SCALE) / SCALE
    else:
        release = {j.job_id: 0.0 for j in base.jobs}

    nominal_makespan = base.expected_makespan_lower_bound()
    penalty = penalty_mult * nominal_makespan

    suffix = f"_t{t:g}_w{w:g}" + ("_staggered" if release_spread else "")
    return GateJobShopInstance(
        name=base.name + suffix,
        description=base.description + f" [gate variant t={t}, w={w}, P={penalty:.2f}]",
        machines=list(base.machines),
        jobs=base.jobs,
        deadlines=deadlines,
        release=release,
        tightness=t,
        width=w,
        penalty=penalty,
        base_name=base.name,
    )
