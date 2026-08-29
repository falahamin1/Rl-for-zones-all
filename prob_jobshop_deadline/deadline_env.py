"""Combined concrete+symbolic job-shop environment with per-job deadlines.

Unlike the no-deadline model (where ``SymbolicJobShopEnv`` and the zone
propagation in ``symbolic_rl.py`` are two independently-usable views of the
same dynamics), here they cannot be separated: whether a dispatch edge is
even OFFERED depends on a zone-emptiness check against ``g_J <= D_J``, so
action computation itself requires a live pyudbm Federation. This class
therefore owns both the concrete simulation state (location, per-job
virtual clocks, sampled active durations) and the symbolic zone in lockstep.

Deadline semantics (see also the module-level design note at the bottom):

* A dispatch edge for a task of job J is offered only if AT LEAST ONE of
  its duration branches keeps the resulting zone non-empty under
  ``g_J <= D_J`` (checked by intersecting that branch's fully-propagated
  zone with the deadline constraint, mirroring exactly how completion
  guards are already checked elsewhere in this codebase). This is a
  per-edge check, not a blanket invariant -- see ``zone_ops.apply_delay``'s
  docstring for why an unconditional invariant would deadlock the
  automaton.
* If a job's next-eligible task (precedence satisfied, machine free) is
  attempted and has ZERO surviving branches, the job transitions to a
  FAILED status for all its remaining tasks, in one step, incurring
  ``fail_penalty``. No time elapses for this transition (the task was
  never actually dispatched). "Failed" is treated exactly like "done" for
  precedence/termination purposes elsewhere in the automaton.
* Because the per-edge check only requires ONE surviving branch (not all),
  a dispatched task's REALIZED (concretely sampled) duration can still be
  one of the branches that would have been infeasible had it been the only
  option. If that happens on a job's LAST task, the job completes but
  g_J > D_J at that point -- a genuine deadline miss that the pre-dispatch
  check could not have caught (by design: it only guarantees at least one
  feasible outcome existed, not that the realized one would be it). This
  module treats that identically to an explicit FAILED transition for
  reward purposes (``-fail_penalty``) and counts it as a "deadline
  violation" for calibration, but does NOT retroactively mark the task
  "failed" (it did complete) -- see ``_check_late_completions``.

Reward convention: ``-elapsed`` from the concrete side (same sign
convention as ``region_env.py:86-111`` / the delta-free Track 1 modules),
plus ``-fail_penalty`` per violated job (explicit failure or late
completion, each counted at most once per job per episode).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .instance import Job, ProbJobShopInstance
from .zone_ops import ZoneContext, apply_delay, apply_guard, apply_invariant, reset_clocks

DONE_LIKE = ("done", "failed")


class DeadlineJobShopEnv:
    def __init__(self, instance: ProbJobShopInstance, seed: Optional[int] = None, fail_penalty: Optional[float] = None):
        self.instance = instance
        self._task_ids: List[str] = [t.task_id for j in instance.jobs for t in j.tasks]
        self._task_to_idx: Dict[str, int] = {tid: i for i, tid in enumerate(self._task_ids)}
        self._task_job: Dict[str, Job] = {t.task_id: j for j in instance.jobs for t in j.tasks}
        self._job_by_id: Dict[str, Job] = {j.job_id: j for j in instance.jobs}

        self.zctx = ZoneContext(instance)
        self._rng = np.random.default_rng(seed)

        if fail_penalty is None:
            nominal_makespan = instance.expected_makespan_lower_bound()
            fail_penalty = 3.0 * max(nominal_makespan, 1.0)
        self.fail_penalty = float(fail_penalty)

        self._loc: Optional[tuple] = None
        self._virtual_clocks: Dict[str, float] = {}
        self._g: float = 0.0
        self._Z = None
        self._done_job_ids: set = set()
        self._late_checked: set = set()
        self._current_actions: List[dict] = []
        self._pending_reward: float = 0.0

        # Diagnostics accumulated per episode, read by the calibration harness.
        self.violations_this_episode: List[str] = []
        self.stuck_this_episode = False

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[tuple, List[dict]]:
        self._loc = tuple("waiting" for _ in self._task_ids)
        self._virtual_clocks = {j.job_id: 0.0 for j in self.instance.jobs}
        self._g = 0.0
        self._done_job_ids = set()
        self._late_checked = set()
        self._pending_reward = 0.0
        self.violations_this_episode = []
        self.stuck_this_episode = False

        self._Z = self.zctx.ctx.get_zero_federation()
        actions = self._settle_and_compute_actions()
        return self._loc, actions

    # ------------------------------------------------------------------
    # State get/set -- used by the exact-DP BFS (deadline_calib/exact_dp.py),
    # which needs to pin this env to an arbitrary already-computed
    # (loc, g, virtual_clocks, Z) node and recompute its action set, without
    # replaying a whole episode from reset(). No Federation copying is
    # needed: Z is threaded through by reference from whichever branch
    # produced it (Federations are immutable from the outside -- every
    # zone_ops op returns a new one -- so aliasing a reference is safe).
    # ------------------------------------------------------------------

    def get_state(self) -> tuple:
        return (
            self._loc,
            self._g,
            tuple(sorted(self._virtual_clocks.items())),
            self._Z,
            frozenset(self._done_job_ids),
            frozenset(self._late_checked),
        )

    def set_state(self, state: tuple) -> None:
        loc, g, vc_items, Z, done_job_ids, late_checked = state
        self._loc = loc
        self._g = g
        self._virtual_clocks = dict(vc_items)
        self._Z = Z
        self._done_job_ids = set(done_job_ids)
        self._late_checked = set(late_checked)
        self._pending_reward = 0.0
        self.violations_this_episode = []

    def step(self, task_id: str) -> Tuple[tuple, List[dict], bool, float, dict]:
        """Dispatch ``task_id`` (must be in the most recently returned
        action list), sample a concrete duration among its surviving
        branches, advance to the next decision point (or terminal),
        applying deadline failures/late-completion checks along the way.

        :return: ``(loc, actions, done, reward, info)`` -- reward covers
            everything since the *previous* ``step()``/``reset()`` call,
            including any penalty incurred by a job failing during this
            call's settling phase.
        """
        action = next(a for a in self._current_actions if a["action_id"] == task_id)
        durations = [b["d"] for b in action["branches"]]
        probs = np.array([b["prob"] for b in action["branches"]], dtype=np.float64)
        probs = probs / probs.sum()  # renormalize over surviving (deadline-feasible) branches only
        d = int(self._rng.choice(durations, p=probs))
        return self._dispatch(task_id, d)

    def step_deterministic(self, task_id: str, d: int) -> Tuple[tuple, List[dict], bool, float, dict]:
        """Like :meth:`step`, but dispatches an EXACT chosen duration ``d``
        (which must be one of ``task_id``'s surviving branches) instead of
        sampling. Used by the exact-DP BFS to enumerate every branch of
        every action rather than following one random trajectory.
        """
        return self._dispatch(task_id, d)

    def _dispatch(self, task_id: str, d: int) -> Tuple[tuple, List[dict], bool, float, dict]:
        action = next(a for a in self._current_actions if a["action_id"] == task_id)
        job = self._task_job[task_id]
        chosen_branch = next(b for b in action["branches"] if b["d"] == d)

        loc = list(self._loc)
        idx = self._task_to_idx[task_id]
        loc[idx] = f"active_d{d}"
        self._virtual_clocks[job.job_id] = 0.0
        self._loc = tuple(loc)
        self._Z = chosen_branch["zone_after_dispatch"]

        self._advance_concrete_to_decision()
        self._check_late_completions()
        actions = self._settle_and_compute_actions()

        done = all(s in DONE_LIKE for s in self._loc)
        reward = self._pending_reward
        self._pending_reward = 0.0
        return self._loc, actions, done, reward, {"violations": list(self.violations_this_episode)}

    # ------------------------------------------------------------------
    # Concrete-side helpers (mirror symbolic_env.py's _advance_to_decision,
    # extended to treat "failed" like "done" and to track global time g).
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
            if pred_id is not None and loc[self._task_to_idx[pred_id]] not in DONE_LIKE:
                continue
            if self.instance.task_by_id(tid).machine not in active_machines:
                return True
        return False

    def _advance_concrete_to_decision(self) -> None:
        loc = list(self._loc)
        while True:
            if self._has_dispatch_pure(loc):
                break
            if all(s in DONE_LIKE for s in loc):
                break

            remaining = {}
            for tid, status in zip(self._task_ids, loc):
                if status.startswith("active_d"):
                    job_id = self._task_job[tid].job_id
                    exact_d = int(status[len("active_d"):])
                    rem = exact_d - self._virtual_clocks.get(job_id, 0.0)
                    remaining[tid] = max(0.0, rem)

            if not remaining:
                # Nothing active and nothing dispatchable, but not all
                # done/failed either: a genuine deadlock. Flagged, not
                # papered over -- see module docstring / DEADLINE.md.
                self.stuck_this_episode = True
                break

            dt = min(remaining.values())
            self._g += dt
            self._pending_reward -= dt
            for job_id in self._virtual_clocks:
                self._virtual_clocks[job_id] += dt

            for tid, rem in remaining.items():
                if abs(rem - dt) < 1e-9:
                    idx = self._task_to_idx[tid]
                    job = self._task_job[tid]
                    loc[idx] = "done"
                    if self._is_last_task(job, tid):
                        self._done_job_ids.add(job.job_id)

        self._loc = tuple(loc)

    @staticmethod
    def _is_last_task(job: Job, task_id: str) -> bool:
        return job.tasks[-1].task_id == task_id

    def _check_late_completions(self) -> None:
        """A job whose LAST task just became 'done' might still have missed
        its deadline (see module docstring): the pre-dispatch edge check
        only guarantees SOME branch was feasible, not that the realized one
        would be."""
        for job_id in self._done_job_ids - self._late_checked:
            job = self._job_by_id[job_id]
            self._late_checked.add(job_id)
            if job.deadline is None:
                continue
            if self._g > job.deadline + 1e-9:
                self.violations_this_episode.append(job_id)
                self._pending_reward -= self.fail_penalty

    # ------------------------------------------------------------------
    # Symbolic action computation (needs the live Federation).
    # ------------------------------------------------------------------

    def _settle_and_compute_actions(self) -> List[dict]:
        """Recompute available actions from the current (loc, Z), applying
        deadline-driven job failures before returning. Failing a job can
        never unblock another (a failed task was never dispatched, so it
        never held a machine), so a single detect-then-apply pass is
        sufficient; the follow-up recompute below is a defensive check that
        this invariant actually holds (asserts if it doesn't, rather than
        silently looping).
        """
        actions, failed_job_ids = self._compute_actions_and_failures(self._loc, self._Z)
        if failed_job_ids:
            loc = list(self._loc)
            for job_id in failed_job_ids:
                job = self._job_by_id[job_id]
                for t in job.tasks:
                    i = self._task_to_idx[t.task_id]
                    if loc[i] not in DONE_LIKE:
                        loc[i] = "failed"
                self.violations_this_episode.append(job_id)
                self._pending_reward -= self.fail_penalty
            self._loc = tuple(loc)

            actions, more_failed = self._compute_actions_and_failures(self._loc, self._Z)
            assert not more_failed, f"unexpected cascading failure after applying {failed_job_ids}: {more_failed}"

        if not actions and not all(s in DONE_LIKE for s in self._loc):
            self.stuck_this_episode = True

        self._current_actions = actions
        return actions

    def _compute_actions_and_failures(self, loc: tuple, Z) -> Tuple[List[dict], List[str]]:
        active_machines = set()
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active"):
                active_machines.add(self.instance.task_by_id(tid).machine)

        actions: List[dict] = []
        attempted_job_ids: set = set()

        for tid, status in zip(self._task_ids, loc):
            if status != "waiting":
                continue
            pred_id = self.instance.predecessor_of(tid)
            if pred_id is not None and loc[self._task_to_idx[pred_id]] not in DONE_LIKE:
                continue
            task = self.instance.task_by_id(tid)
            if task.machine in active_machines:
                continue

            job = self._task_job[tid]
            if self._g + 1e-9 < job.release:
                continue  # release-time gate: not eligible yet, not a failure

            attempted_job_ids.add(job.job_id)

            branches = []
            for d, prob in zip(task.distribution.durations, task.distribution.probabilities):
                path, decision_loc = self._compute_path(loc, tid, d)
                Zb = self._propagate_branch(Z, tid, d, path, job)
                if Zb is None:
                    continue
                branches.append({"d": d, "prob": prob, "decision_loc": decision_loc, "zone_after_dispatch": Zb})

            if branches:
                actions.append({"action_id": tid, "job_id": job.job_id, "branches": branches})

        failed = [
            jid
            for jid in attempted_job_ids
            if self._job_by_id[jid].deadline is not None and not any(a["job_id"] == jid for a in actions)
        ]
        return actions, failed

    def _propagate_branch(self, Z, tid: str, d: int, path: List[dict], job: Job) -> Optional[object]:
        job_clock = f"c_{job.job_id}"
        Z_r = reset_clocks(Z, self.zctx, [job_clock])
        Z_curr = apply_delay(Z_r, self.zctx, self._invariant_for_dispatch(tid, d), 1)

        for step in path:
            for cname, exact_d in step["completion_guards"]:
                Z_curr = Z_curr & (self.zctx.clocks[cname] >= exact_d)
                if Z_curr.is_empty():
                    return None
            Z_curr = apply_delay(Z_curr, self.zctx, self._invariant_for_loc(step["next_loc"]), 1)

        if job.deadline is not None:
            Z_curr = Z_curr & (self.zctx.clocks[f"g_{job.job_id}"] <= job.deadline)
            if Z_curr.is_empty():
                return None

        return Z_curr

    def _invariant_for_dispatch(self, tid: str, d: int) -> List[Tuple[str, int]]:
        loc = list(self._loc)
        loc[self._task_to_idx[tid]] = f"active_d{d}"
        return self._invariant_for_loc(tuple(loc))

    def _invariant_for_loc(self, loc: tuple) -> List[Tuple[str, int]]:
        result = []
        for tid, status in zip(self._task_ids, loc):
            if status.startswith("active_d"):
                job_id = self._task_job[tid].job_id
                exact_d = int(status[len("active_d"):])
                result.append((f"c_{job_id}", exact_d))
        return result

    def _compute_path(self, loc: tuple, tid: str, d: int):
        """Ported from symbolic_env.py's ``_compute_path``, extended to
        treat "failed" the same as "done" (via ``DONE_LIKE``)."""
        job = self._task_job[tid]
        sim_loc = list(loc)
        sim_loc[self._task_to_idx[tid]] = f"active_d{d}"

        virtual_clocks = dict(self._virtual_clocks)
        virtual_clocks[job.job_id] = 0.0

        path: List[dict] = []
        while not self._has_dispatch_pure(sim_loc) and not all(s in DONE_LIKE for s in sim_loc):
            remaining = {}
            for t_id, status in zip(self._task_ids, sim_loc):
                if status.startswith("active_d"):
                    jid = self._task_job[t_id].job_id
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
                    cjid = self._task_job[t_id].job_id
                    exact_d = int(sim_loc[self._task_to_idx[t_id]][len("active_d"):])
                    completing_guards.append((f"c_{cjid}", exact_d))
                    sim_loc[self._task_to_idx[t_id]] = "done"

            path.append({"completion_guards": completing_guards, "next_loc": tuple(sim_loc)})

        return path, tuple(sim_loc)
