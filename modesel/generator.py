"""Step 2 -- a parameterized instance generator, not hand-written tiers.

Each job's task-to-machine routing is a random permutation of the machine
set (truncated/cycled if n_tasks != n_machines) -- classic job-shop
routing, every tier here uses n_tasks == n_machines so this is always a
clean permutation, never a cycle. Each task gets an independent random
base duration (used as d_fast); d_slow = d_fast * fast_slow_ratio,
rounded up to stay strictly greater than d_fast. D_j = (job's own
all-SLOW makespan) * (1 + tightness), exactly as specified.
"""
from __future__ import annotations

import random
from typing import Tuple

from .instance import Job, ModeSelInstance, Task

DEFAULT_FAST_SLOW_RATIO = 2.0
DEFAULT_COST_PENALTY = 6.0  # > max possible (d_slow - d_fast) = base_max*(ratio-1) = 5*1.0, so SLOW is
                             # always cheaper than FAST when both are feasible (matches M0's design ratio)
BASE_DURATION_RANGE = (2, 5)


def make_instance(
    n_jobs: int,
    n_tasks: int,
    n_machines: int,
    fast_slow_ratio: float,
    cost_penalty: float,
    tightness: float,
    w: float,
    seed: int,
    label: str = "",
) -> ModeSelInstance:
    rng = random.Random(seed)
    machines = [f"M{i}" for i in range(n_machines)]

    jobs = []
    for j in range(n_jobs):
        job_id = f"J{j}"
        routing = machines[:]
        rng.shuffle(routing)
        if n_tasks <= n_machines:
            routing = routing[:n_tasks]
        else:
            routing = [routing[i % n_machines] for i in range(n_tasks)]

        tasks = []
        for t_idx in range(n_tasks):
            d_fast = rng.randint(*BASE_DURATION_RANGE)
            d_slow = max(d_fast + 1, round(d_fast * fast_slow_ratio))
            tasks.append(
                Task(task_id=f"{job_id}_T{t_idx}", job_id=job_id, index=t_idx,
                     machine=routing[t_idx], d_fast=d_fast, d_slow=d_slow)
            )
        slow_makespan = sum(t.d_slow for t in tasks)
        deadline = int(round(slow_makespan * (1 + tightness)))
        jobs.append(Job(job_id=job_id, tasks=tuple(tasks), deadline=deadline))

    return ModeSelInstance(jobs=tuple(jobs), cost_penalty=cost_penalty, w=w, seed=seed, label=label or f"gen_{n_jobs}x{n_tasks}_{n_machines}m")
