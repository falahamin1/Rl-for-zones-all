# Spike 2: bounded zones (Task A) and the aggregation ratio (Task B)

**Status: throwaway spike, not merged, package not refactored.** All code
lives in `/tmp/prob_jobshop_spike/` (the same scratch copy as Spike 1) plus
`/tmp/zone_common.py`, `/tmp/spike2_taskA.py`, `/tmp/spike2_taskB.py`,
`/tmp/spike2_taskB_policy.py` (throwaway measurement scripts).
`prob_jobshop_ext/` at the repo root was never touched.

**Spike 1's interval-duration change was discarded first**, per instructions:
`symbolic_env.py` was copied back byte-for-byte from the untouched original
(`diff` confirmed identical) before any Spike 2 work started. Spike 2's
changes are layered on top of that reverted, exact-duration baseline.

**Files touched in this spike, and exactly why:**
- `zone_ops.py` — `delta` removed from `ZoneContext`; `normalize()` updated;
  `min_cost()` deleted; `clip_to_k_box()` and `is_bounded()` added. (Task A,
  as instructed.)
- `symbolic_env.py` — `_advance_to_decision()` now also accumulates elapsed
  time into a new `self._last_elapsed` side-channel attribute; `step()`'s
  returned info dict gained an `"elapsed"` key. **This was necessary, not
  optional**: the task's Task A instructions say to take elapsed time "from
  the concrete side, exactly as region_env.py's step() already does" —
  `SymbolicJobShopEnv.step()` did not expose elapsed time at all before this
  change, so there was nothing to take. Scoped to be a pure addition: no
  existing return signature changed (`step_deterministic()`'s 3-tuple return
  is untouched, so `symbolic_optimal.py`'s call site still unpacks correctly
  — though that module is broken anyway once `min_cost` is gone, see below).
- `symbolic_rl.py` — the `min_cost` import removed (module-level, required:
  the name no longer exists in `zone_ops.py`, so the import would otherwise
  raise `ImportError`), and **four** lines in `_run_episode` changed, not
  literally one: `prev_delta = min_cost(Z, ...)` removed, the reward line
  changed to `r = info["elapsed"]`, and the two `return min_cost(Z_nx, ...)`
  statements (episode-end cost reporting, used by `train()`'s
  `episode_costs`/checkpoint tracking) changed to `return total_elapsed`.
  **Read "that one reward line" as covering the full, tightly-coupled set of
  `min_cost` usages needed to keep `_run_episode` functional** — a literal
  single-line edit would leave the method calling a name that no longer
  exists. `evaluate_real()` was not touched (it never called `min_cost`; it
  only used `str(Z)` as a dict key, unaffected by delta's removal).
- **Not touched, as instructed**: `region_env.py`, `pta_env.py`,
  `simulator.py`, `region_rl.py`, `ddpg_*.py`. **Broken as an unavoidable
  side effect, not touched/fixed**: `symbolic_optimal.py` imports and
  extensively uses `zone_ops.min_cost`, which no longer exists — that module
  is no longer importable after this spike's `zone_ops.py` change. Not used
  by any measurement in this report (Task B's policy-consistency check uses
  a freshly-built exact concrete-state DP instead, described below, not
  `OptimalSymbolicSolver`).

---

## Task A — removing `delta`, bounding zones

### Boundedness (measured directly on the post-removal zones, before clipping)

| Instance | n_states | truncated@20k | bounded_frac | n_terminal | terminal_bounded_frac |
|---|---:|:---:|---:|---:|---:|
| PJS_01 | 60 | No | **0.0000** | 16 | 0.0000 |
| PJS_02 | 133 | No | 0.2331 | 37 | 0.0000 |
| PJS_03 | 1,799 | No | 0.0217 | 467 | 0.0000 |
| PJS_04 | 4,373 | No | 0.1319 | 917 | 0.0000 |
| PJS_05 | 20,000 | **Yes** | 0.1489 | 0 (cap hit first) | — |
| LJS_01 | 20,000 | **Yes** | **0.0000** | 0 (cap hit first) | — |

**Do NOT assume 100% — confirmed, this is nowhere near 100%.** It is not
even uniformly nonzero: PJS_01 and LJS_01 measured **exactly 0.0000** across
every single reachable state. Where it is nonzero (PJS_02-05), it is still
small (2–23%). **Every terminal (all-tasks-done) state checked, across every
instance, is unbounded** (0/16, 0/37, 0/467, 0/917 — 0% in all four cases
where terminal states were reached before the cap).

**Why, mechanistically** (verified directly against the actual DBM output,
not inferred): removing `delta` did not fix the unboundedness — it turns out
**every job clock exhibits the same behavior `delta` had**. A job clock only
ever gets an upper-bound invariant while that job's task is *currently
active* (`invariant()` only emits an entry for `active_d{X}` statuses).
Before a job's first task dispatches, and *after* one of its tasks completes
but before the next one dispatches, that job's clock has no invariant
applied and drifts unboundedly upward along with real time, exactly like
`delta` used to. Confirmed by direct inspection of the very first
post-`reset()` zone for PJS_01: `(c_J1==c_J2)` — no upper bound on either
clock, unbounded, before a single task has even been dispatched. A **fully
bounded** zone requires *every* job to have a currently-active task
simultaneously (no job waiting, none freshly completed) — rare, and requires
enough machines relative to jobs (PJS_02's 3 machines / 2 jobs gets to
23%; LJS_01's 4 machines / 6 jobs never manages it in 20,000 states).
Terminal states are unbounded unconditionally, because at termination every
job's *last* task has completed and none has a next task to re-bound it.

### `clip_to_k_box()` — verification

Applied `clip_to_k_box(Z, zctx, {cn: M for cn in zctx.clock_names})` (M =
`worst_case_makespan_upper_bound()`, the same bound already used by
`normalize()`) to **every** reachable zone in every instance:

| Instance | n zones checked | n_clip_empty | n_clip_unbounded (after clip) |
|---|---:|---:|---:|
| PJS_01 | 60 | 0 | 0 |
| PJS_02 | 133 | 0 | 0 |
| PJS_03 | 1,799 | 0 | 0 |
| PJS_04 | 4,373 | 0 | 0 |
| PJS_05 | 20,000 | 0 | 0 |
| LJS_01 | 20,000 | 0 | 0 |

**Every single clipped zone, across every instance and every reachable
state, is non-empty and bounded — zero exceptions, zero pyudbm rejections.**
`clip_to_k_box` does exactly what it should: turns the (routinely unbounded)
extrapolated zone into a guaranteed-bounded one via exact intersection, with
no observed edge case where the box clipped a zone down to emptiness.

### Collapse check

Compared, over the exact same reached-state population from one BFS run
(extrapolated-zone keying, matching how the model already keys states
everywhere else in the codebase — clipping was *not* used to drive
reachability/exploration itself, only applied after the fact to the fixed
set of already-reached states):

| Instance | n_states (baseline) | distinct extrapolated `str(Z)` | distinct clipped `str(Z)` | distinct `(loc, clipped str(Z))` | merge groups (loc-keyed) | merge groups (pure-str) |
|---|---:|---:|---:|---:|---:|---:|
| PJS_01 | 60 | 53 | 53 | 60 | 0 (max 0) | 0 (max 0) |
| PJS_02 | 133 | 127 | 127 | 133 | 0 (max 0) | 0 (max 0) |
| PJS_03 | 1,799 | 1,697 | 1,697 | 1,799 | 0 (max 0) | 0 (max 0) |
| PJS_04 | 4,373 | 3,793 | 3,793 | 4,373 | 0 (max 0) | 0 (max 0) |
| PJS_05 | 20,000 | 18,126 | 18,126 | 20,000 | 0 (max 0) | 0 (max 0) |
| LJS_01 | 20,000 | 12,961 | 12,961 | 20,000 | 0 (max 0) | 0 (max 0) |

**Zero merge groups, in every instance, under either comparison.**
Clipping never once caused two previously-distinct zones to become
string-identical — `distinct clipped str(Z)` equals `distinct extrapolated
str(Z)` exactly, digit for digit, in all six rows. **The clipped view loses
no state distinctions the unclipped one makes.** (Separately, note
`distinct (loc, str(Z))` always equals the full `n_states` baseline — i.e.
the codebase's actual keying convention, `(loc, str(Z))`, never collapses
two BFS-distinct states together even before clipping, since `loc` alone
already disambiguates most of what would otherwise coincide; the
*pure*-`str(Z)`-only count, ignoring `loc`, is what's lower — reported as
requested, "compare on `str(Z)`.")

### affine_dim / vertices / facets on clipped zones

Vertex/facet sample: up to 1,000 zones/instance (all of them for PJS_01/02).
`exact` = finite-extreme-point combinatorial method (reduced-dimension,
capped at `m<=18, k<=4` for tractability — see Spike 1's note on why this
cap exists). `qhull` = `HalfspaceIntersection`+`ConvexHull` on the same
reduced system — **legitimately applicable here**, unlike in Spike 1, since
`clip_to_k_box` zones are bounded by construction (verified above), and
Spike 1's specific failure mode (qhull on an *unbounded* polyhedron) cannot
occur.

| Instance | dim mean/max | exact vert mean | exact unk_frac | exact facet mean | qhull vert mean | qhull unk_frac | qhull facet mean | disagreements |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PJS_01 | 1.92/2 | 3.83 | 0.000 | 3.83 | 4.00 | 0.083 | 4.00 | 0 |
| PJS_02 | 1.95/2 | 3.71 | 0.000 | 3.71 | 3.81 | 0.053 | 3.81 | 0 |
| PJS_03 | 2.90/3 | 6.49 | 0.000 | 9.05 | 6.54 | 0.012 | 9.14 | 0 |
| PJS_04 | 2.80/3 | 6.44 | 0.000 | 8.98 | 6.54 | 0.023 | 9.15 | 0 |
| PJS_05 | 3.62/4 | 6.02 | 0.716 | 8.17 | 11.23 | 0.006 | 33.20 | **5** |
| LJS_01 | 5.00/6 | 6.53 | 0.943 | 9.05 | 22.04 | 0.000 | 264.17 | 0 (but see below) |

**Cross-check disagreement, flagged as instructed.** 5 disagreements on
PJS_05 (all "exact=6, qhull=8"), 0 flagged on LJS_01 *only because* the
exact method's `unk_frac` there is 94.3% (almost nothing to compare against
— not because the two methods agreed on the overlap). **I hand-verified one
"exact=6, qhull=8" case exactly** (a different, LJS_01 zone hit during
development, same 6-vs-8 signature): solving the reduced 3-variable system
by hand (`x∈[0,2], y∈[max(7,x+7),9], z∈[max(7,x+7,y),420]`) gives exactly 6
vertices — `(0,7,7), (0,7,420), (0,9,9), (0,9,420), (2,9,9), (2,9,420)` —
matching the exact-method output digit for digit. **Qhull's answer of 8 is
wrong on this bounded polytope.** Given all 6 observed disagreements
(1 hand-verified + 5 on PJS_05) share the identical "+2 spurious vertices"
signature, this looks systematic (plausibly from qhull's handling of the
several redundant/non-facet-defining constraints these reduced systems
contain), not random noise — but the root cause inside qhull was not
further diagnosed (out of scope). **Practical consequence for reading this
table**: the `exact` columns are ground truth where available; the `qhull`
columns should be treated as approximate/occasionally-overcounting,
especially useful for coverage (near-0% unknown) where `exact` cannot reach
(LJS_01, PJS_05's high-dimension tail) but not as a trusted precise value
there.

---

## Task B — the aggregation ratio

200 seeded uniform-random-policy episodes per instance (policy RNG seeded
independently per episode, separate from the env's own duration-sampling
RNG), run against the CONCRETE `SymbolicJobShopEnv.step()` (post-Task-A
code). At every decision point, recorded `concrete_key = (loc, exact
integer virtual-clock tuple)` and `zone_key = (loc, str(Z))`, with `Z`
propagated alongside the concrete rollout through the exact sampled branch
at each step (mirroring `symbolic_rl.py`'s own branch-matching pattern).
Zero `propagate` failures (the symbolic branch always matched the concrete
sampled outcome, in every one of 1,200 episodes across 6 instances) — no
model/pyudbm inconsistency between the concrete and symbolic sides.

| Instance | episodes | records | distinct concrete | distinct zone | **agg. ratio (zone)** | control ratio (loc-only) | cluster mean/median/max | frac cluster size=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PJS_01 | 200 | 800 | 27 | 41 | **0.659** | 1.929 | 1.171 / 1.0 / 3 | 0.854 |
| PJS_02 | 200 | 1,000 | 79 | 106 | **0.745** | 3.160 | 1.198 / 1.0 / 4 | 0.849 |
| PJS_03 | 200 | 1,200 | 288 | 444 | **0.649** | 4.364 | 1.023 / 1.0 / 3 | 0.980 |
| PJS_04 | 200 | 1,800 | 537 | 875 | **0.614** | 3.196 | 1.034 / 1.0 / 3 | 0.967 |
| PJS_05 | 200 | 2,400 | 1,474 | 1,737 | **0.849** | 2.094 | 1.003 / 1.0 / 2 | 0.997 |
| LJS_01 | 200 | 4,800 | 4,095 | 4,304 | **0.951** | 1.130 | 1.000 / 1.0 / 1 | 1.000 |

**The decisive number, reported exactly as it came out: the aggregation
ratio is below 1.0 for every single instance.** Zones do not merely fail to
compress the concrete state space beyond what the discrete location already
does — they produce *more* distinct symbolic identities than there are
distinct concrete states actually visited (0.61–0.95×, never above 1). The
control ratio (concrete states per distinct *location alone*, ignoring
clocks) is comfortably above 1 in every instance (1.13–4.36×) — i.e.
location-based aggregation is real and substantial, while zone-based
aggregation (on top of location) is net negative. Cluster sizes confirm
this directly: **85–100% of zones (by instance) contain exactly one
distinct concrete state** (`frac cluster size=1`), and the largest cluster
found anywhere is 4 (PJS_02). The trend across instances (ratio climbing
toward LJS_01's 0.951, cluster fractions climbing toward 1.000) shows the
effect getting *more* pronounced, not less, as instances scale up — larger
instances have progressively less real zone-level aggregation, not more.

**Mechanistic note** (offered as a plausible explanation of the *direction*
of the effect — the numbers above are the actual finding, this paragraph is
interpretation, clearly marked as such): in this exact-duration model, the
sampled duration outcome is baked directly into `loc` itself
(`"active_dX"`), so two rollouts landing on the same `(loc, zone_key)` must
already have taken an identical sequence of discrete duration outcomes up
to that point for every currently-relevant task — leaving very little room
for the *zone geometry* to still be usefully merging genuinely different
concrete histories. Meanwhile the zone string can differ between rollouts
that happen to coincide on the exact clock-value tuple (e.g. differing
constraints among clocks belonging to *other*, not-yet-relevant jobs, which
don't show up in the raw clock-value tuple at all) — which is the direction
that pushes the ratio below 1.

### Policy consistency (PJS_01–03)

Built a **fresh, independent exact DP over concrete states** (not
`OptimalSymbolicSolver`, which is broken after Task A's `zone_ops.py`
change — see above): BFS over `(loc, exact-clock-tuple)` via
`step_deterministic`, branching over every possible sampled duration at
each dispatch, backward relaxation `V[s] = min_a Σ_d P(d)(elapsed_d +
V[s'_d])` initialised at `V=+inf` (not `V=0`, deliberately — this is the
standard correct Bellman-Ford-style relaxation and specifically avoids the
"treat an unresolved child as contributing 0 extra cost" bug found in
`OptimalSymbolicSolver` during the earlier audit of this codebase). Then,
for every zone cluster (from Task B's own 200-episode rollout, same seeds —
reproduces identical clusters deterministically) containing 2+ distinct
concrete states, checked whether the concrete states' sets of tied-optimal
actions (within `1e-6` tolerance) share at least one action in common.

| Instance | n_concrete_states | truncated | multi-state clusters | missing from DP | **policy-consistent** | **policy-inconsistent** |
|---|---:|:---:|---:|---:|---:|---:|
| PJS_01 | 31 | No | 6 | 0 | **6 / 6 (100%)** | 0 |
| PJS_02 | 105 | No | 16 | 0 | **16 / 16 (100%)** | 0 |
| PJS_03 | 667 | No | 9 | 0 | **9 / 9 (100%)** | 0 |

**No policy-inconsistent cluster was found — not a show-stopper, by this
check.** All 31 multi-state clusters checked across PJS_01–03 (every
multi-state cluster the 200-episode rollout produced, matched 100% against
the exact concrete DP, no missing lookups) shared at least one common
optimal action among their member concrete states. Given how small these
clusters are (Task B's own numbers: max cluster size 3–4, 85–98% of
clusters are singletons anyway), this check has limited statistical power —
it rules out unsoundness *in the specific clusters this random-policy
rollout happened to produce*, not in the full zone graph exhaustively.

---

## Summary

Both Task A and Task B produced clean, unambiguous, mostly negative
answers about the value of the zone abstraction as currently constructed.
**Task A**: removing `delta` did not bound the zones — the exact same
unbounded-clock pathology `delta` had turns out to be a structural property
of every job clock in this model (unconstrained whenever its task is not
*currently* active), so bounded-zone fraction is 0% for two of six
instances tested and never exceeds 23% for the rest; terminal states are
unconditionally unbounded in every instance. `clip_to_k_box` reliably fixes
this (100% non-empty, 100% bounded, zero pyudbm rejections across ~46,000
zones checked) and, measured directly, causes **zero** loss of state
distinctions relative to the unclipped zones (zero merge groups in every
instance, both loc-keyed and pure-zone-string-keyed). The one genuine
methodological finding here is that `qhull`'s `HalfspaceIntersection` can
still silently produce **wrong** (not just inapplicable) vertex counts even
on now-*bounded* zones — a different failure mode from Spike 1's
unbounded-polyhedron issue, confirmed by hand-verification and consistent
across all 6 observed disagreements. **Task B**: the aggregation ratio —
the number the task called decisive — is **below 1.0 for every one of the
six tested instances** (0.61–0.95×), meaning the zone abstraction, as
currently implemented in this exact-integer-duration model, does not
compress the concrete state space at all; it fragments it more finely than
raw concrete-state identity does, while location-alone aggregation
(1.13–4.36×) is real. Within the (small, size ≤4) clusters that do form, no
policy inconsistency was found across all 31 checked in PJS_01–03 against a
freshly-built, independently-verified exact concrete-state DP — so no
soundness violation surfaced, but the sample of multi-state clusters
available to check this on was itself small, a direct consequence of the
Task B finding.
