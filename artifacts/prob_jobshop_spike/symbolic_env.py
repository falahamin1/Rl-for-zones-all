from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .instance import ProbJobShopInstance


# ---------------------------------------------------------------------------
# SPIKE 3 MODIFICATION — real-valued interval durations + deadlines.
#
# INTERVAL_WIDTH (w): each duration bucket d becomes a REAL interval
# [lo, hi] = [d*(1-w), d*(1+w)] (no floor/ceil — genuinely real-valued,
# unlike Spike 1's integer-clamped version). Set by the driver script.
#
# SCALE: pyudbm's Clock comparison operators REJECT float bounds outright
#   (TypeError: '<=' not supported between instances of 'Clock' and
#   'float' — confirmed empirically; see /tmp/SPIKE3.md). Since lo/hi are
#   generally non-integer, the SYMBOLIC side represents them as SCALED
#   INTEGERS (scaled_bound = round(real_bound * SCALE)) — all pyudbm-facing
#   constraints live in "SCALE-ths of a time unit". SCALE=20 is exact (no
#   rounding at all) for both tested widths w in {0.1, 0.25}, since
#   d*0.9*20, d*1.1*20, d*0.75*20, d*1.25*20 are all exact integers for
#   integer d. The CONCRETE side never touches pyudbm and stays in real
#   (unscaled) float units throughout, per the task's explicit requirement.
# ---------------------------------------------------------------------------
INTERVAL_WIDTH: float = 0.25
SCALE: int = 20


def _interval_bounds_real(d: int) -> Tuple[float, float]:
    """[lo, hi] = [d*(1-w), d*(1+w)] — real-valued, no rounding."""
    w = INTERVAL_WIDTH
    return d * (1 - w), d * (1 + w)


def _scaled(x: float) -> int:
    return int(round(x * SCALE))


class SymbolicJobShopEnv:
    """UPTA interface for the job-shop scheduling problem.

    Location: tuple of task statuses in flat job-order.
      "waiting"    — task not yet dispatched
      "active_d{d}" — task dispatched with exact duration d (distinct state per d)
      "done"       — task completed

    action_space item:
        {
          "action_id": task_id,          # e.g. "J1_T1"
          "guards":    [],               # no timing guards for dispatch
          "resets":    ["c_J1"],         # job clock to reset
          "price":     0,
          "branches":  [                 # one per possible duration
            {
              "d":           int,        # exact sampled duration
              "prob":        float,      # P(duration = d)
              "next_loc":    tuple,      # immediate location after dispatch (active_d{d})
              "path":        [...],      # completion steps to next decision point
              "decision_loc": tuple,     # final location where agent decides next
            }, ...
          ],
        }

    path item:
        {
          "completion_guards": [(clock_name, exact_d), ...],  # exact c_J = exact_d
          "next_loc":          loc_tuple,
        }

    invariant(loc) returns [(clock_name, exact_d)] for each active_d{X} task —
    the exact upper bound equals the sampled duration, so combined with the
    completion guard c_J >= exact_d we get c_J = exact_d in the zone.

    step() samples a concrete duration from the task's distribution and encodes
    it in the location.  The returned info dict contains {"sampled_d": d} so
    the symbolic RL agent can select the matching zone branch.
    """

    def __init__(self, instance: "ProbJobShopInstance", seed: Optional[int] = None):
        self.instance = instance
        self._task_ids: List[str] = [
            t.task_id for j in instance.jobs for t in j.tasks
        ]
        self._task_to_idx = {tid: i for i, tid in enumerate(self._task_ids)}

        self._task_job: dict = {
            t.task_id: j.job_id
            for j in instance.jobs for t in j.tasks
        }
        self._job_by_id: dict = {j.job_id: j for j in instance.jobs}  # SPIKE3: for .deadline lookup

        self._max_dur: dict = {
            t.task_id: t.distribution.max_duration()
            for j in instance.jobs for t in j.tasks
        }

        self._rng = np.random.default_rng(seed)

        self._loc: Optional[tuple] = None
        self._virtual_clocks: dict = {}   # job_id -> elapsed since last task start
        self._active_d: dict = {}         # job_id -> duration bucket of current task
        self._last_elapsed: float = 0.0   # SPIKE2 (Task A): set by _advance_to_decision
        self._active_real_dur: dict = {}  # SPIKE3: job_id -> REAL sampled duration in [lo,hi]
        self._last_deadline_violated: bool = False  # SPIKE3: set by step() when a job's
                                                      # own realized duration exceeds its deadline

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[tuple, List[dict]]:
        self._loc = tuple("waiting" for _ in self._task_ids)
        self._virtual_clocks = {j.job_id: 0.0 for j in self.instance.jobs}
        self._active_d = {}
        self._active_real_dur = {}
        return self._loc, self._compute_actions(self._loc)

    def step(self, task_id: str) -> Tuple[tuple, List[dict], bool, dict]:
        """Execute one dispatch action, sampling a concrete duration.

        SPIKE3 MODIFICATION: samples the duration bucket d as before (same
        probabilities), then samples a REAL duration uniformly from that
        bucket's [lo,hi] interval (a true float — CRITICAL per spike
        instructions, not rounded/clamped to an integer). The real value
        (not the bucket) drives concrete decision-point timing via
        _advance_to_decision. If the job carries a deadline and the real
        sampled duration alone exceeds it, records a deadline violation
        (self._last_deadline_violated) — purely a post-hoc measurement flag;
        does not alter dispatch/enablement logic.
        """
        loc = list(self._loc)
        idx = self._task_to_idx[task_id]
        assert loc[idx] == "waiting", f"Task {task_id} is not waiting"

        dist = self.instance.task_by_id(task_id).distribution
        d = int(self._rng.choice(dist.durations, p=dist.probabilities))
        lo, hi = _interval_bounds_real(d)
        real_dur = float(self._rng.uniform(lo, hi)) if hi > lo else float(lo)

        loc[idx] = f"active_d{d}"
        job_id = self._task_job[task_id]
        self._virtual_clocks[job_id] = 0.0
        self._active_d[job_id] = d
        self._active_real_dur[job_id] = real_dur

        job = self._job_by_id[job_id]
        if job.deadline is not None and real_dur > job.deadline:
            self._last_deadline_violated = True

        loc = self._advance_to_decision(loc)
        self._loc = tuple(loc)

        done = all(s == "done" for s in self._loc)
        actions = [] if done else self._compute_actions(self._loc)
        return self._loc, actions, done, {
            "sampled_d": d,
            "elapsed": self._last_elapsed,
            "sampled_real_dur": real_dur,
        }

    def step_deterministic(self, task_id: str, d: int) -> Tuple[tuple, List[dict], bool]:
        """Dispatch task_id with exact duration d (no sampling).  Used by the optimal solver."""
        loc = list(self._loc)
        idx = self._task_to_idx[task_id]
        assert loc[idx] == "waiting", f"Task {task_id} is not waiting"
        loc[idx] = f"active_d{d}"
        job_id = self._task_job[task_id]
        self._virtual_clocks[job_id] = 0.0
        self._active_d[job_id] = d
        loc = self._advance_to_decision(loc)
        self._loc = tuple(loc)
        done = all(s == "done" for s in self._loc)
        actions = [] if done else self._compute_actions(self._loc)
        return self._loc, actions, done

    def save_state(self) -> tuple:
        """Snapshot current env state for later restore."""
        return (self._loc, dict(self._virtual_clocks), dict(self._active_d))

    def restore_state(self, snapshot: tuple) -> None:
        """Restore env to a previously saved snapshot."""
        self._loc, vc, ad = snapshot
        self._virtual_clocks = dict(vc)
        self._active_d = dict(ad)

    def invariant(self, loc: tuple) -> List[Tuple[str, int]]:
        """Upper-bound constraints (SCALED integers) for active tasks:
        [(clock_name, scaled_hi)], PLUS [(clock_name, scaled_deadline)] for
        any active task whose job carries a deadline.

        SPIKE3 MODIFICATION: hi = interval(d).hi (real-valued, then scaled to
        an integer — pyudbm rejects float bounds, see module docstring), not
        the exact bucket value d. Combined with the completion guard
        c_J >= scaled_lo (see _compute_path), the zone keeps genuine
        [lo, hi] room. A job deadline is an ADDITIONAL entry for the same
        clock — apply_invariant() intersects all given pairs, so
        c_J <= hi AND c_J <= D_J is exactly c_J <= min(hi, D_J); if D_J is
        tighter than what the completion guard (c_J >= lo) needs, the
        resulting zone is empty and that branch is pruned by the existing
        emptiness check in _compute_path/propagate — no special-casing
        needed for "missing a deadline".
        """
        result = []
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active_d"):
                job_id = self._task_job[tid]
                d = int(status[len("active_d"):])
                _, hi = _interval_bounds_real(d)
                result.append((f"c_{job_id}", _scaled(hi)))
                job = self._job_by_id[job_id]
                if job.deadline is not None:
                    result.append((f"c_{job_id}", _scaled(job.deadline)))
        return result

    def rate(self, loc: tuple) -> int:
        return 1

    def clock_names(self) -> List[str]:
        return [f"c_{j.job_id}" for j in self.instance.jobs]

    def M(self) -> int:
        return max(t.distribution.max_duration() for t in self.instance.all_tasks())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _has_dispatch_pure(self, loc) -> bool:
        """Return True if any task can be dispatched (no path computation)."""
        active_machines = set()
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active"):
                active_machines.add(self.instance.task_by_id(tid).machine)

        for tid, status in zip(self._task_ids, loc):
            if status != "waiting":
                continue
            pred_id = self.instance.predecessor_of(tid)
            if pred_id is not None:
                if loc[self._task_to_idx[pred_id]] != "done":
                    continue
            if self.instance.task_by_id(tid).machine not in active_machines:
                return True
        return False

    def _compute_path(self, loc: tuple, tid: str, d: int):
        """Compute the completion-guard path for dispatching tid with duration bucket d.

        Uses COPIES of the current virtual clock state so the env is not mutated.

        SPIKE3 MODIFICATION: dt = min(remaining) STRUCTURAL/topological event
        ordering still uses the bucket value d "as before" (per spike
        instructions) — this decides which task's completion triggers which
        path step, unchanged from the exact-duration model. What changes is
        only the CONSTRAINT emitted at each completion: c_J >= scaled_lo
        (interval(d).lo, scaled) instead of c_J >= exact_d — "the symbolic
        side does not pin clocks". invariant() supplies the matching
        c_J <= scaled_hi (and c_J <= scaled_deadline, if any) upper bound.

        Returns
        -------
        path : list of {"completion_guards": [(clock_name, scaled_lo), ...], "next_loc": tuple}
        decision_loc : tuple — final decision point location
        """
        job_id = self._task_job[tid]
        sim_loc = list(loc)
        sim_loc[self._task_to_idx[tid]] = f"active_d{d}"

        virtual_clocks = dict(self._virtual_clocks)
        virtual_clocks[job_id] = 0.0

        path: List[dict] = []

        while not self._has_dispatch_pure(sim_loc) and not all(
            s == "done" for s in sim_loc
        ):
            remaining = {}
            for t_id, status in zip(self._task_ids, sim_loc):
                if status.startswith("active_d"):
                    jid = self._task_job[t_id]
                    exact_d = int(status[len("active_d"):])
                    rem = exact_d - virtual_clocks.get(jid, 0.0)
                    remaining[t_id] = max(0.0, rem)

            if not remaining:
                break

            dt = min(remaining.values())
            for jid in virtual_clocks:
                virtual_clocks[jid] += dt

            completing_guards: List[Tuple[str, int]] = []
            for t_id, rem in remaining.items():
                if abs(rem - dt) < 1e-9:
                    cjid = self._task_job[t_id]
                    exact_d = int(sim_loc[self._task_to_idx[t_id]][len("active_d"):])
                    lo, _ = _interval_bounds_real(exact_d)   # SPIKE3: >= lo, not == exact_d
                    completing_guards.append((f"c_{cjid}", _scaled(lo)))
                    sim_loc[self._task_to_idx[t_id]] = "done"

            path.append({
                "completion_guards": completing_guards,
                "next_loc":          tuple(sim_loc),
            })

        return path, tuple(sim_loc)

    def _compute_actions(self, loc) -> List[dict]:
        """Return enabled dispatch actions, each with a list of duration branches."""
        active_machines = set()
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active"):
                active_machines.add(self.instance.task_by_id(tid).machine)

        actions = []
        for tid, status in zip(self._task_ids, loc):
            if status != "waiting":
                continue
            pred_id = self.instance.predecessor_of(tid)
            if pred_id is not None:
                pred_idx = self._task_to_idx[pred_id]
                if loc[pred_idx] != "done":
                    continue
            task = self.instance.task_by_id(tid)
            if task.machine in active_machines:
                continue

            job_id = self._task_job[tid]
            dist = task.distribution

            branches = []
            for d, prob in zip(dist.durations, dist.probabilities):
                next_loc_imm = list(loc)
                next_loc_imm[self._task_to_idx[tid]] = f"active_d{d}"
                path, decision_loc = self._compute_path(loc, tid, d)
                branches.append({
                    "d":            d,
                    "prob":         prob,
                    "next_loc":     tuple(next_loc_imm),
                    "path":         path,
                    "decision_loc": decision_loc,
                })

            actions.append({
                "action_id": tid,
                "guards":    [],
                "resets":    [f"c_{job_id}"],
                "price":     0,
                "branches":  branches,
            })
        return actions

    def _advance_to_decision(self, loc: list) -> list:
        """Advance loc until a dispatch is available.

        SPIKE2 MODIFICATION (Task A): also accumulates elapsed wall-clock
        time into self._last_elapsed (a side channel, not a return-signature
        change — step_deterministic()'s 3-tuple callers stay unaffected).

        SPIKE3 MODIFICATION: uses the REAL sampled duration
        (self._active_real_dur[job_id], set by .step()) when available,
        falling back to the duration bucket for jobs where no real value was
        recorded (step_deterministic(), used by the exact concrete-state DP
        — see /tmp/SPIKE3.md's policy-consistency methodology note on why
        that DP deliberately stays bucket-driven rather than real-valued).
        """
        elapsed = 0.0
        while True:
            if self._has_dispatch_pure(loc):
                break
            if all(s == "done" for s in loc):
                break

            remaining = {}
            for tid, status in zip(self._task_ids, loc):
                if status.startswith("active_d"):
                    job_id = self._task_job[tid]
                    bucket_d = int(status[len("active_d"):])
                    real_d = self._active_real_dur.get(job_id, bucket_d)
                    rem = real_d - self._virtual_clocks.get(job_id, 0.0)
                    remaining[tid] = max(0.0, rem)

            if not remaining:
                break

            dt = min(remaining.values())
            elapsed += dt
            for job_id in self._virtual_clocks:
                self._virtual_clocks[job_id] += dt

            for tid, rem in remaining.items():
                if abs(rem - dt) < 1e-9:
                    idx = self._task_to_idx[tid]
                    loc[idx] = "done"
                    job_id = self._task_job[tid]
                    self._active_d.pop(job_id, None)
                    self._active_real_dur.pop(job_id, None)

        self._last_elapsed = elapsed
        return loc
