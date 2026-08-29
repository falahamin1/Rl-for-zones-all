"""Zone operations for the mode-selection domain.

Clocks: ``c0..c{n_jobs-1}`` (one per job, elapsed-since-dispatch of that
job's currently active task, reset at each dispatch AND again once that
task completes -- a completed task's clock carries no further information,
same reasoning as gridtimed's ``e`` clock) and ``g`` (global, never
resets). No extra bookkeeping clock is needed here the way gridtimed
needed ``e``: MODE is the exact, agent-chosen decision (analogous to
gridtimed's delay after its soundness fix), and the per-job clock IS the
task-specific clock the duration interval naturally bounds -- exactly
gate_track's ``c_J`` pattern, not a new mechanism.

SCALE=20 at the pyudbm boundary; concrete simulation (env.py/dp.py) stays
in exact units throughout.
"""
from __future__ import annotations

from typing import List, Tuple

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from pyudbm import Context

from .instance import ModeSelInstance, SCALE

GLOBAL_CLOCK = "g"


def _scale(x) -> int:
    return int(round(x * SCALE))


class ModeSelZoneContext:
    def __init__(self, instance: ModeSelInstance):
        self.job_clock_names = [f"c{i}" for i in range(instance.n_jobs)]
        self._clock_names = self.job_clock_names + [GLOBAL_CLOCK]
        self.ctx = Context(self._clock_names)
        self.clocks = {name: getattr(self.ctx, name) for name in self._clock_names}

    @property
    def clock_names(self) -> List[str]:
        return list(self._clock_names)

    def c(self, job_idx: int) -> str:
        return f"c{job_idx}"


def reset_clocks(Z, zctx: ModeSelZoneContext, names: List[str]):
    for name in names:
        assert name != GLOBAL_CLOCK, f"{name} is the global clock and must never be reset"
    Z2 = Z
    for name in names:
        Z2 = Z2.reset_value(zctx.clocks[name])
    return Z2


def apply_upper_guard(Z, zctx: ModeSelZoneContext, pairs: List[Tuple[str, float]]):
    Z2 = Z
    for name, bound in pairs:
        Z2 = Z2 & (zctx.clocks[name] <= _scale(bound))
    return Z2


def apply_lower_guard(Z, zctx: ModeSelZoneContext, pairs: List[Tuple[str, float]]):
    Z2 = Z
    for name, bound in pairs:
        Z2 = Z2 & (zctx.clocks[name] >= _scale(bound))
    return Z2


def normalize(Z, zctx: ModeSelZoneContext, k_bound_real: float):
    bounds = {zctx.clocks[name]: _scale(k_bound_real) for name in zctx.clock_names}
    return Z.extrapolate_max_bounds(bounds)
