"""SPIKE3 shared helpers: deadline-variant generator + instance/env plumbing.
THROWAWAY, /tmp only.
"""
from __future__ import annotations
import sys
import copy

sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")
sys.path.append("/tmp")

import prob_jobshop_spike.symbolic_env as sym_env
import prob_jobshop_spike.zone_ops as zone_ops
from prob_jobshop_spike.instance import ProbJobShopInstance, Job  # scratch copy: has .deadline
from prob_jobshop_ext.benchmarks.all_instances import get_instance_by_name
from prob_jobshop_ext.benchmarks.all_large_instances import get_large_instance_by_name

# NOTE: prob_jobshop_ext's own Job dataclass has no `deadline` field (that
# addition lives only in the scratch copy's instance.py). We therefore build
# deadline-variant instances using the SCRATCH COPY's Job/ProbJobShopInstance
# classes, re-hydrated from the stock benchmark data (Task/TaskDistribution
# objects are identical between the two copies — plain dataclasses, no
# behavior — so reusing them directly across modules is safe).


def get_base_instance(name: str):
    return get_large_instance_by_name(name) if name.startswith("LJS") else get_instance_by_name(name)


def make_deadline_variant(base_instance, tightness):
    """Programmatically generate a deadline-augmented COPY of `base_instance`.

    tightness: float in [0,1], or None for "no deadline" (returns an
    equivalent instance with every job.deadline = None — explicit, not just
    reuse of the original object, so all three t-values in the grid go
    through the exact same code path).

    D_J = (sum of nominal durations in job J) * (1 + t), where "nominal
    duration" = each task's TaskDistribution.expected_duration() (this
    instance suite's own hand-crafted PJS_01-05/LJS_01 files don't store a
    single separate "nominal" value the way the formula-generated PJS_07-10/
    LJS_02-10 files do — E[d] is the well-defined, already-available stand-in
    used here; noted explicitly in /tmp/SPIKE3.md as an interpretation
    choice, not given verbatim by the task).
    """
    new_jobs = []
    for job in base_instance.jobs:
        if tightness is None:
            deadline = None
        else:
            nominal_sum = sum(t.distribution.expected_duration() for t in job.tasks)
            deadline = nominal_sum * (1 + tightness)
        new_jobs.append(Job(job_id=job.job_id, tasks=job.tasks, deadline=deadline))
    return ProbJobShopInstance(
        name=base_instance.name,
        description=base_instance.description,
        machines=base_instance.machines,
        jobs=new_jobs,
    )


def build_env_zctx_M(instance, w):
    sym_env.INTERVAL_WIDTH = w
    env = sym_env.SymbolicJobShopEnv(instance)
    zctx = zone_ops.ZoneContext(instance)
    # SPIKE3: M must be scaled AND widened to (1+w) — durations now range up
    # to hi = max_duration*(1+w), which can exceed the original
    # worst_case_makespan_upper_bound()'s per-task max. See /tmp/SPIKE3.md.
    M_real = instance.worst_case_makespan_upper_bound() * (1 + w)
    M_scaled = int(round(M_real * sym_env.SCALE))
    return env, zctx, M_scaled
