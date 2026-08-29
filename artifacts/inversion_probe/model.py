"""inversion_probe: a hand-verifiable micro-automaton, positive-control test
of the Spike 2/3 measurement pipeline. THROWAWAY, /tmp only. Does not touch
prob_jobshop_ext or prob_jobshop_spike (only imports generic, already-written
zone algebra from prob_jobshop_spike.zone_ops and /tmp/zone_common.py).

---------------------------------------------------------------------------
MODEL, exactly as specified, plus one necessary design decision (flagged):
---------------------------------------------------------------------------
Two clocks: c (task-local, resets at every task start), g (job clock, NEVER
resets — kept locally in THIS model; Spike 2's package-wide `delta` removal
is a separate, unrelated fact about zone_ops.py's generic ZoneContext, which
this probe reuses unmodified — nothing about that removal prevents a model
from defining its own never-resetting clock as part of ITS OWN automaton).

Task 1: invariant c<=9, completion guard c>=1. Concrete .step() samples a
REAL duration uniform on [1,9].

DESIGN DECISION (necessary, not specified verbatim by the task — flagged
here and again in PROBE.md): the task's spec for Task 1 gives exactly ONE
flat interval (c<=9, c>=1), which by itself produces exactly ONE symbolic
zone at the post-task-1 decision point (g ranging over the whole [1,9], not
split at 4). That zone alone cannot answer "for each [zone], the optimal
action" (plural) or let "decision relevance" compare sibling zones, since
there would only be one. To make Step 1 and Step 2 answerable at all
(rather than trivially degenerate), Task 1 is represented symbolically as
NINE DISCRETE INTEGER BUCKETS tau in {1,...,9} (mirroring exactly how every
prior spike's job-shop model represents a probabilistic/interval duration:
discrete buckets driving symbolic branch/location structure, a real value
driving concrete execution) — each bucket PINS c (and hence g, which ticks
in lockstep with c during task 1) to that exact integer via
invariant c<=tau AND completion guard c>=tau (both bounds equal — this is
the ONE place in this probe where the symbolic side does pin a clock,
deliberately, precisely BECAUSE the task's own hand-verifiable ground truth
is stated in terms of "tau, the realized completion time of task 1" as a
single quantity, not a range). Each bucket is given probability 1/9 (a
discrete approximation of Uniform(1,9) — stated explicitly, not implied).
The CONCRETE side samples a genuinely continuous real_tau ~ Uniform(1,9)
independently of this, and maps it to the nearest integer bucket in [1,9]
(clipped at the boundaries) purely to select which symbolic branch to
propagate — exactly the same "bucket drives symbolic structure, real value
drives concrete timing" split used throughout Spikes 1 and 3.

On task 1 completion (location ("done","waiting")) the agent chooses FAST
or SLOW for task 2:
  FAST: invariant c<=2, guard c>=2, cost +5 (deterministic, no bucketing needed)
  SLOW: invariant c<=8, guard c>=8, cost 0
Global deadline: invariant g<=D in EVERY location (task 1 active, and while
task 2 is active) — D is a module parameter (default 12), swept in the
sensitivity analysis. A branch whose zone becomes empty under this
invariant is infeasible and unavailable — enforced BOTH symbolically (the
existing emptiness-check machinery in zone_ops, no new logic) AND
concretely (the concrete env's _compute_actions excludes SLOW when
real_tau + 8 > D — a deliberate choice, discussed in PROBE.md, to make the
"unavailable edge" language literal on the concrete side too, unlike Spike
3's job-shop deadlines which were checked post-hoc only).

Objective: minimize total cost (mode penalty only — no elapsed-time/
makespan cost anywhere in this model, unlike every prior spike).

Ground truth (asserted in test_ground_truth.py): SLOW iff tau<=4 (i.e. D=12
minus SLOW's duration 8), FAST otherwise. At tau<=4 SLOW is feasible
(tau+8<=12) and strictly cheaper (0<5) so it's chosen; at tau>4 SLOW is
infeasible so FAST is the only (hence optimal-by-default) option.
---------------------------------------------------------------------------

Kept from Spike 2/3, reused not reimplemented: delta-free ZoneContext
pattern, zone_ops.clip_to_k_box / is_bounded / normalize / apply_guard /
reset_clocks / apply_delay / apply_invariant / init_zone (imported directly
from prob_jobshop_spike.zone_ops — those functions are fully generic, they
only need a zctx with .clocks/.clock_names, never referenced job-shop
specifics), zone_common.affine_dim / exact_vertices_fraction / exact_vertices_and_facets
(imported from /tmp/zone_common.py), and the SCALE=20 integer encoding
(pyudbm rejects float bounds — see AUDIT/SPIKE3 for the original discovery;
re-confirmed not to matter here since every bound in this model happens to
already be an integer, but the mechanism is kept for consistency, per
instructions, and is a harmless no-op multiplication in that case).
"""
from __future__ import annotations
import pyudbm as _warmup_pyudbm  # noqa: F401 -- MUST import before repo root hits sys.path
import sys
from typing import Dict, List, Optional, Tuple

sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")
sys.path.append("/tmp")

import numpy as np
from pyudbm import Context
import prob_jobshop_spike.zone_ops as zone_ops  # generic zone algebra, reused as-is

SCALE = 20
DEADLINE_DEFAULT = 12.0
M_REAL = 24.0  # extrapolation/clip bound in real units — generous headroom above
               # the largest deadline swept (24) and the largest reachable g (D+2 at most)


def _scaled(x: float) -> int:
    return int(round(x * SCALE))


class MicroZoneContext:
    """Same delta-free pattern as Spike 2's zone_ops.ZoneContext, instantiated
    directly for this model's two clocks (c, g) rather than derived from a
    ProbJobShopInstance's job list — ZoneContext's own __init__ is hardcoded
    to `instance.jobs`, not reusable as-is for a non-job-shop model, so this
    is a minimal parallel class following the exact same shape/pattern.
    """

    def __init__(self):
        self._clock_names = ["c", "g"]
        self.ctx = Context(self._clock_names)
        self.clocks = {name: getattr(self.ctx, name) for name in self._clock_names}

    @property
    def clock_names(self) -> List[str]:
        return self._clock_names


class MicroEnv:
    """Symbolic action/branch structure + concrete real-valued stepping for
    the micro-automaton. Mirrors SymbolicJobShopEnv's action/branch dict
    shape (action_id/guards/resets/price/branches, branch d/prob/next_loc/
    path/decision_loc) closely enough to reuse zone_ops primitives directly,
    but is written bespoke (not forced through zone_common.explore(), which
    is hard-coded to a class literally named SymbolicJobShopEnv) since this
    model is small enough that a self-contained explorer is clearer.

    Location: (task1_status, task2_status)
      task1_status: "waiting" | "active_tau{k}" (k in 1..9) | "done"
      task2_status: "waiting" | "active_FAST" | "active_SLOW" | "done"
    """

    def __init__(self, deadline: float = DEADLINE_DEFAULT, seed: Optional[int] = None):
        self.deadline = deadline
        self._rng = np.random.default_rng(seed)
        self._loc: Optional[Tuple[str, str]] = None
        self._c: float = 0.0
        self._g: float = 0.0
        self._last_elapsed: float = 0.0   # real time advanced this step (for concrete_key tracking only)
        self._last_cost: float = 0.0      # mode penalty incurred this step (the actual objective)
        self._last_sampled_bucket: Optional[int] = None
        self._last_real_tau: Optional[float] = None

    # ------------------------------------------------------------------
    def reset(self):
        self._loc = ("waiting", "waiting")
        self._c = 0.0
        self._g = 0.0
        return self._loc, self._compute_actions(self._loc)

    def invariant(self, loc: Tuple[str, str]) -> List[Tuple[str, int]]:
        """[(clock_name, scaled_bound)] — task-local bound(s) for whichever
        sub-task is active, PLUS the global deadline g<=D in EVERY location
        (per spec: "invariant g <= 12 in every location"), including while
        task 1 is active (redundant there since tau<=9<D always for every
        deadline tested, but included structurally, not assumed).
        """
        t1, t2 = loc
        result: List[Tuple[str, int]] = []
        if t1.startswith("active_tau"):
            tau = int(t1[len("active_tau"):])
            result.append(("c", _scaled(tau)))
        if t2 == "active_FAST":
            result.append(("c", _scaled(2)))
        elif t2 == "active_SLOW":
            result.append(("c", _scaled(8)))
        result.append(("g", _scaled(self.deadline)))
        return result

    def rate(self, loc) -> int:
        return 1

    def clock_names(self) -> List[str]:
        return ["c", "g"]

    # ------------------------------------------------------------------
    def _compute_actions(self, loc: Tuple[str, str]) -> List[dict]:
        t1, t2 = loc
        actions: List[dict] = []

        if t1 == "waiting":
            branches = []
            for tau in range(1, 10):
                branches.append({
                    "d": tau,
                    "prob": 1.0 / 9.0,
                    "next_loc": (f"active_tau{tau}", t2),
                    "path": [{
                        "completion_guards": [("c", _scaled(tau))],
                        "next_loc": ("done", t2),
                    }],
                    "decision_loc": ("done", t2),
                })
            actions.append({
                "action_id": "dispatch_task1",
                "guards": [],
                "resets": ["c"],
                "price": 0.0,
                "branches": branches,
            })

        elif t1 == "done" and t2 == "waiting":
            for mode, dur, cost in [("FAST", 2, 5.0), ("SLOW", 8, 0.0)]:
                actions.append({
                    "action_id": f"dispatch_{mode}",
                    "guards": [],
                    "resets": ["c"],
                    "price": cost,
                    "branches": [{
                        "d": dur,
                        "prob": 1.0,
                        "next_loc": (t1, f"active_{mode}"),
                        "path": [{
                            "completion_guards": [("c", _scaled(dur))],
                            "next_loc": (t1, "done"),
                        }],
                        "decision_loc": (t1, "done"),
                    }],
                })

        return actions

    # ------------------------------------------------------------------
    # Concrete (real-valued) side
    # ------------------------------------------------------------------
    def _enabled_actions_concrete(self) -> List[dict]:
        """Like _compute_actions, but for the CONCRETE side: excludes SLOW
        when it is not actually achievable given the deadline and the
        REAL (not bucket) elapsed time so far — makes 'that edge is
        unavailable' literal on the concrete side too, not just symbolic.
        See PROBE.md for why this differs from Spike 3's job-shop deadlines
        (which were checked post-hoc only, never gating action availability).
        """
        actions = self._compute_actions(self._loc)
        t1, t2 = self._loc
        if t1 == "done" and t2 == "waiting":
            filtered = []
            for a in actions:
                if a["action_id"] == "dispatch_SLOW" and self._g + 8 > self.deadline:
                    continue
                filtered.append(a)
            return filtered
        return actions

    def reset_concrete(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._loc = ("waiting", "waiting")
        self._c = 0.0
        self._g = 0.0
        return self._loc, self._enabled_actions_concrete()

    def step(self, action_id: str):
        """Execute one action on the concrete side. Returns
        (loc, actions, done, info). info contains 'elapsed', 'cost',
        'sampled_bucket' (matches the branch to propagate symbolically) and,
        for the task-1 dispatch, 'real_tau'.
        """
        t1, t2 = self._loc

        if action_id == "dispatch_task1":
            real_tau = float(self._rng.uniform(1.0, 9.0))
            bucket = int(np.clip(round(real_tau), 1, 9))
            self._c = 0.0
            elapsed = real_tau
            self._c += elapsed
            self._g += elapsed
            self._loc = ("done", t2)
            self._last_elapsed = elapsed
            self._last_cost = 0.0
            self._last_sampled_bucket = bucket
            self._last_real_tau = real_tau
            done = False
            return self._loc, self._enabled_actions_concrete(), done, {
                "elapsed": elapsed, "cost": 0.0, "sampled_bucket": bucket, "real_tau": real_tau,
            }

        elif action_id in ("dispatch_FAST", "dispatch_SLOW"):
            mode = "FAST" if action_id == "dispatch_FAST" else "SLOW"
            dur = 2.0 if mode == "FAST" else 8.0
            cost = 5.0 if mode == "FAST" else 0.0
            self._c = 0.0
            self._c += dur
            self._g += dur
            self._loc = (t1, "done")
            self._last_elapsed = dur
            self._last_cost = cost
            self._last_sampled_bucket = int(dur)
            done = True
            return self._loc, [], done, {
                "elapsed": dur, "cost": cost, "sampled_bucket": int(dur), "real_tau": None,
            }

        raise ValueError(f"unknown action_id {action_id!r}")
