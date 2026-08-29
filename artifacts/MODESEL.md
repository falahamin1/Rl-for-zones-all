# MODESEL: Mode-Selection Job-Shop, Diagnostic-First

New package `modesel/`, built from scratch -- does not fork or import from `prob_jobshop_ext`, `deadline_track`, `gate_track`, `gridtimed`, or `supplementary/` (verified: no such imports anywhere in `modesel/*.py`, grep-checked). Reuses only `zoneviews/` (not directly -- see below) and `pyudbm` directly. None of those five directories were modified (verified: 0 files in them newer than the start of this work). Diagnostic-first as instructed: environment + exact DP were built and gated (Step 1) before any zone-construction code existed at all; zones were built later, only for Step 4, once Steps 1-3 had already produced results.

**Total wall-clock: calibration sweep + full 5-tier x 3-seed run (DP + Step 3 diagnostic + Step 4 geometry/decision-relevance at both w values) = under 90s.** All 15 tier-seed combinations' exact DP completed WITHOUT truncation -- a stronger result than gridtimed's equivalent tiers, discussed below.

## The model

N jobs, each a sequential chain of tasks; task `(j,t)` needs machine `m(j,t)` (machines serve one task at a time). One global clock `g`, never resets (asserted in `zone_ops.reset_clocks`). Each job has its own deadline `D_j`. Dispatching a task also picks a MODE: FAST (short, costs `cost_penalty`) or SLOW (long, free). Feasibility is checked ON THE EDGE in exactly two places (never as a blanket invariant, which would deadlock time-advance the moment any job's remaining work transiently can't fit):

1. **At dispatch**: a `(task, mode)` action is only offered if that job's own best-case completion (this task's full committed duration + FAST for whatever's left) stays within `D_j`. A dispatched task's duration is fixed thereafter, so a legal dispatch can never later become the reason its job fails.
2. **During time-advance**: every WAITING (not dispatched, not done, not failed) job is re-checked after each completion event; if even its best possible remaining path (all-FAST, machine contention ignored -- a documented simplification matching `deadline_track`/`gate_track`'s own precedent) can no longer fit, it fails right there, once, and is dropped from further consideration.

Durations are nominal integers for the exact DP (`step_deterministic`); the true per-move duration is `Uniform(d_m(1-w), d_m(1+w))`, sampled for real by `step()` (used only for calibration) -- integer-only sampling would silently reproduce the degenerate point-zone case, so `step()` genuinely samples real values. `SCALE=20` at the pyudbm boundary; concrete simulation stays in exact integer units throughout.

**A structural difference from gridtimed worth flagging up front**: this domain is provably acyclic (every dispatch either advances a job's cursor or fails it, both irreversible), so the exact DP is plain memoized recursion, not gridtimed's topological-sort-by-clock workaround -- and deadlock is structurally impossible here too (every job either has a machine-blocked task, in which case *something* is occupying that machine and will eventually complete, or is itself dispatchable). Confirmed empirically: 0 deadlocked states across all 15 tier-seed runs (below), not just argued.

## Step 1 -- THE PAPER GATE (blocking)

M0: 2 jobs, 1 task each, sharing one machine. `d_fast=3, d_slow=5, cost_penalty=3, D=9` (both jobs, symmetric). Only one task can run at a time, so the second job's dispatch decision happens at a `g` that depends on which mode the first job used -- the same decision, two different concrete `g` values:

- **Zone I** (first job dispatched FAST, arrives at g=3): second job SLOW -> cost 5 (feasible, 3+5=8<=9); FAST -> cost 6 (feasible, 3+3=6<=9). **SLOW optimal, V=5.**
- **Zone J** (first job dispatched SLOW, arrives at g=5): second job SLOW would complete at 5+5=10>9 -- **not even offered** (feasibility checked on the dispatch edge, not after the fact); FAST -> cost 6 (feasible, 5+3=8<=9), the only legal option. **FAST optimal (forced), V=6.**

```
zone_I (g=3): expected J1_T0_SLOW (V=5.0), got ['J1_T0_SLOW'] (V=5.0) -- PASS
zone_J (g=5): expected J1_T0_FAST (V=6.0), got ['J1_T0_FAST'] (V=6.0) -- PASS
Step 1 (paper gate): PASSED.
```

**PASSED** -- proceeding to Steps 2-4.

## Step 2 -- generator and calibration

`generator.make_instance(n_jobs, n_tasks, n_machines, fast_slow_ratio, cost_penalty, tightness, w, seed)`: each job's task-to-machine routing is a random permutation of the machine set (every tier here has `n_tasks == n_machines`, so this is always a clean permutation); each task gets an independent random base duration (`d_fast`, range 2-5), `d_slow = round(d_fast * fast_slow_ratio)`. `D_j = job's own all-SLOW makespan * (1 + tightness)`, exactly as specified. `fast_slow_ratio=2.0`, `cost_penalty=6.0` (chosen to safely exceed the maximum possible `d_slow - d_fast = 5`, so SLOW is always cheaper than FAST when both are feasible, matching M0's design ratio) used uniformly across all generated tiers.

**Calibration sweep** (500 episodes/tightness/tier, uniform-random policy over legal actions, real duration sampling via `step()`, `w=0.25`):

| Tier | t=0.0 | t=0.1 | t=0.25 | t=0.5 | t=1.0 | chosen t | in [5%,50%] band? |
|---|---:|---:|---:|---:|---:|---:|---|
| M0 | 74.0% | 67.8% | 49.2% | 50.2% | 0.0% | 0.25 | yes |
| M1 | 51.8% | 0.0% | 24.8% | 4.2% | 0.0% | 0.25 | yes |
| M2 | 80.6% | 68.0% | 75.2% | 20.2% | 8.4% | 0.5 | yes |
| M3 | 82.4% | 51.4% | 4.6% | 2.0% | 1.2% | 0.25 | **no** |
| M4 | 98.8% | 85.6% | 82.8% | 31.8% | 1.4% | 0.5 | yes |

**M3 has no tightness value in band at the swept grid** -- reported plainly rather than hidden: the sweep has a gap between t=0.1 (51.4%, just over) and t=0.25 (4.6%, just under 5%). t=0.25 (closest, just under the floor) was used for M3's tier run below rather than searching a finer grid.

**Fraction of episodes with no action available at a non-terminal state: 0.0000 everywhere, all 25 sweep cells (5 tiers x 5 tightness values), 12,500 episodes total.** This is the expected result, not a coincidence -- since infeasible dispatches are filtered at the action-offering stage and any WAITING job's own blocking machine is always occupied by *something* that will eventually complete, this fallback-mechanism check was never expected to trip; confirmed rather than assumed.

## Tier tractability

| Tier | n_jobs x n_tasks, machines | tightness | mean n_states (3 seeds) | exact (no truncation)? |
|---|---|---:|---:|---|
| M0 | 2x1, 1m | 0.25 | 6.7 | yes |
| M1 | 2x2, 2m | 0.25 | 30.7 | yes |
| M2 | 3x2, 2m | 0.5 | 125.0 | yes |
| M3 | 3x3, 3m | 0.25 | 466.0 | yes |
| **M4** | 4x3, 3m | 0.5 | **5857.3** | **yes** |

**All 5 tiers, all 3 seeds (15/15), completed EXACTLY -- no truncation anywhere, `state_limit=200,000` never approached (max was M4 seed 0 at 8,588 states).** M4 comfortably clears the ">= 2000 states" power requirement on every seed (8588, 3485, 5499). Unlike gridtimed (where T2-T4 needed a `state_limit`-truncated, sampled treatment), this domain's combinatorial structure (discrete mode choices and completion orderings, not gridtimed's fine-grained per-cell clock geometry) keeps even the largest tier here tractable -- worth flagging as a genuine cross-domain structural difference, not a weaker test.

## Step 3 -- THE PARTITION DIAGNOSTIC

Exclusions applied throughout (0 deadlocked states confirmed at every tier-seed, matching the structural argument above):

| Tier (seed 0) | total states | terminal excluded | deadlock excluded | usable |
|---|---:|---:|---:|---:|
| M0 | 6 | 3 | 0 | 3 |
| M1 | 32 | 7 | 0 | 25 |
| M2 | 147 | 24 | 0 | 123 |
| M3 | 569 | 26 | 0 | 543 |
| M4 | 8588 | 107 | 0 | 8481 |

### Measurement 1 -- the ceiling

| Tier (seed 0) | usable states | n_locations | **ceiling (classes)** | class mean | class median | class max | singleton frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 3 | 3 | 3 | 1.00 | 1 | 1 | 100.0% |
| M1 | 25 | 13 | 13 | 1.92 | 1 | 5 | 53.8% |
| M2 | 123 | 62 | 29 | 4.24 | 2 | 20 | 44.8% |
| M3 | 543 | 161 | 51 | 10.65 | 3 | 84 | 31.4% |
| **M4** | **8481** | 1261 | **243** | 34.90 | 3 | 1004 | 30.5% |

At M4 (the highest-power tier): 8481 usable states compress to a ceiling of 243 classes (~35x) -- notably coarser than gridtimed's ~54x at a much smaller scale (3436 states -> 64 classes), and coarser than the 1261 raw locations too (243 < 1261, unlike gridtimed where the ceiling was *finer* than location count). This is a first hint, confirmed below, that mode-selection's optimal-action structure is less atomized than grid-navigation's.

### Measurement 2 -- candidate observation models (M4 seed 0, sorted by cell count)

| Candidate | cells | soundness | mean cell size | max cell size | singleton frac |
|---|---:|---:|---:|---:|---:|
| (b) location only | 1261 | 60.51% | 6.73 | 79 | 12.29% |
| (d) slack ordering | 1480 | 64.86% | 5.73 | 79 | 17.36% |
| (e) discretized g, b=4 | 3000 | 93.60% | 2.83 | 32 | 44.13% |
| (e) discretized g, b=2 | 4227 | 96.92% | 2.01 | 19 | 54.98% |
| (c) slack-bucket, b=4 | 4981 | 97.77% | 1.70 | 13 | 67.64% |
| (e) discretized g, b=1 | 5201 | 98.31% | 1.63 | 10 | 63.62% |
| (c) slack-bucket, b=2 | 7031 | 99.36% | 1.21 | 4 | 84.87% |
| (a) concrete state [control] | 8481 | 100.00% | 1.00 | 1 | 100.00% |
| (c) slack-bucket, b=1 | 8481 | 100.00% | 1.00 | 1 | 100.00% |

**No candidate is both SOUND (>= 95%) and COARSENING (mean cell size >= 3), at any tier tested (M2, M3, M4 all checked explicitly, same pattern every time).** The closest is (e) discretized-g at b=4: 93.60% sound, mean 2.83 -- close on both axes but short of both bars simultaneously. (b) location-only and (d) slack-ordering achieve real compression (mean 5.7-6.7x) but only ~60-65% soundness. This is the same qualitative result gridtimed found (no candidate cleared both bars there either), now confirmed in a second, structurally different domain.

### Measurement 3 -- is the ceiling zone-expressible? (THE DECISIVE ONE)

M4 seed 0, top 10 classes, contamination against 8481 usable states:

| class size | action | contamination | % of other states | clean? |
|---:|---|---:|---:|---|
| 1004 | `J2_T2_SLOW` | 5483 / 7477 | 73.3% | no |
| 856 | `J1_T2_SLOW` | 4819 / 7625 | 63.2% | no |
| 757 | `J0_T2_SLOW` | 4980 / 7724 | 64.5% | no |
| 685 | `J3_T2_SLOW` | 5687 / 7796 | 72.9% | no |
| 415 | `J2_T1_SLOW` | 4430 / 8066 | 54.9% | no |
| 407 | `J0_T1_FAST` | 5554 / 8074 | 68.8% | no |
| 338 | `J1_T1_SLOW` | 5549 / 8143 | 68.1% | no |
| 287 | `J3_T1_SLOW` | 3506 / 8194 | 42.8% | no |
| 281 | `J3_T1_FAST` | 5481 / 8200 | 66.8% | no |
| 272 | `J1_T1_FAST` | 5861 / 8209 | 71.4% | no |

**0/10 clean at M4** (self-check passed first: every class's own defining points were confirmed contained in its own hull before trusting any contamination count). **0/10 at M2 and M3 too** (M1: 0-1/10 across seeds; M0: 1-4/10, but M0's classes are trivially tiny, size 1, at only 3-4 usable states total -- not a meaningful test of the hypothesis).

**Against the grid-navigation baseline (0/10 clean, contamination up to 89%) directly, as asked: also 0/10 clean, contamination 42.8%-73.3%** -- somewhat lower peak contamination than grid-navigation, but the qualitative result is identical: **the ceiling is not zone-expressible here either.**

**HYPOTHESIS TEST RESULT: refuted.** The hypothesis under test was that mode-selection optimal actions are thresholds on remaining slack, and a threshold on a linear function of clocks is a half-space, so these classes should be convex where grid-navigation's were not. They are not convex. Slack *is* linear in the clocks (`slack_j = (D_j - g) - remaining_SLOW_work_j`, and both terms are linear in `g` and the per-job elapsed clocks), and a single-job threshold on it genuinely would be a half-space -- but the *optimal action* is not simply "the job with minimum slack" (candidate (d), slack-ordering, is only 64.86% sound at M4, confirming the ranking itself doesn't determine the action either). The true optimal-action boundary depends jointly on multiple jobs' slacks *and* machine occupancy in a way that a single linear threshold does not capture, so the resulting classes are not half-spaces, and their DBM hulls sweep in the majority of all other reachable states. This is the most valuable outcome this run could produce, and it points against, not for, the zone-expressibility of mode-selection's decision structure -- reported plainly rather than reframed.

## Step 4 -- decision relevance

Geometry and decision-relevance computed at `w in {0.0, 0.25}` for every tier-seed, exhaustively (no sampling anywhere; even M4's 8481-zone case ran in well under a minute).

| Tier (seed 0) | n_clocks | dim distribution (w=0.25) | dim>=2 frac | facets mean | vertices mean | bounded | strict | cross-zone conflict (w=0.0 / w=0.25) | within-zone nonuniform | mean states/zone |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| M0 | 3 | {0:1, 1:2} | 0.0% | 6.00 | 1.67 | 100% | 0% | nan / nan (no multi-zone locations this seed) | 0% (n/a) | 1.00 |
| M1 | 3 | {0:1, 1:24} | 0.0% | 6.00 | 1.96 | 100% | 0% | 0.364 / 0.364 | 0% (n/a) | 1.00 |
| M2 | 4 | {0:1, 1:110, 2:12} | 9.8% | 8.10 | 2.28 | 100% | 0% | 0.348 / 0.348 | 0% (n/a) | 1.00 |
| M3 | 4 | {0:7, 1:203, 2:333} | 61.3% | 8.96 | 4.18 | 100% | 0% | 0.350 / 0.350 | 0% (n/a) | 1.00 |
| **M4** | 5 | {0:9, 1:2373, 2:5419, 3:680} | 71.9% | 11.02 | 4.97 | 100% | 0% | 0.450 / 0.450 | **0% (n/a)** | **1.00** |

**Within-zone non-uniformity is 0% at every tier and every seed (15/15) -- and, exactly as gridtimed found after its own Task A fix, `n_multistate_zones=0` everywhere: the check is satisfied structurally, not narrowly.** The sound per-state zone construction (delay -> here, MODE -- exact per dispatch, real width only from genuine duration uncertainty) generalizes cleanly to this domain.

**`w=0.0` and `w=0.25` give IDENTICAL cross-zone conflict fractions at every tier, to 3 decimal places.** Same structural reason as gridtimed: **mean states per zone = 1.00 everywhere** -- zones here are, again, ~1:1 with concrete states, so grouping by zone reduces to grouping by concrete state, independent of `w`. **This is gridtimed's exact failure mode, reported plainly as instructed, not hidden**: the current sound zone construction achieves essentially zero compression in this domain too. Cross-zone conflict fractions themselves (35-45% typically, up to 45% at M4) are, however, dramatically above every job-shop-family baseline reported earlier in this project (0-2.6%), and above the ceiling-vs-location comparison in Measurement 1 already suggested this: decision-relevant information genuinely is NOT capturable by location alone, here or in grid-navigation -- it's specifically that *zones*, as currently constructed (1:1 with concrete states), don't compress it into anything smaller than the raw state itself.

### Worked example (generated tier, in addition to Step 1's hand-verified one)

M4 seed 0, location `(J0 waiting T0, J1 active T1 FAST, J2 waiting T1, J3 waiting T0)`:

```
g=5:  optimal J3_T0_SLOW
g=10: optimal J3_T0_FAST
```

Same task (`J3_T0`), same location, optimal mode flips from SLOW to FAST purely as a function of `g` -- an independently-discovered instance of exactly the same mechanism Step 1's hand-built M0 example was designed to exercise, confirming the paper-gate mechanism recurs in generated, non-hand-built instances, not merely in the constructed example.

## Watch-list items (explicitly checked against, per the task)

- **Clip bound scaled per instance, not global**: `geometry.k_bound_for(instance)` computes `max(D_j) + margin` fresh per instance (`_K_MARGIN=4`), exactly avoiding the class of bug that hit gridtimed's `K_BOUND` twice (T1-T4's larger deadlines silently emptying zones tuned for T0's `D=16`). No empty-zone crash occurred anywhere in this run.
- **Quadratic pairwise conflict detection**: `decision_relevance.py` buckets by distinct optimal-action set *before* the pairwise comparison (same fix gridtimed needed after a real 15-minute hang on a T2 seed) -- built in from the start here, not retrofitted after a hang. M4's 8481-zone run completed in well under a minute, consistent with the fix being effective, not merely absent of a stress case (M4 alone has locations with up to 79 concrete states, i.e. up to 79 candidate zones per location before bucketing -- large enough to have hit the old O(k^2) bug had it been present).
- **Blanket deadline invariant deadlocking time-elapse**: avoided by construction (feasibility checked only at the two edges described in "The model" above); confirmed 0 deadlocks across all 15 runs, and 0 `RuntimeError`s from `advance_and_resolve`'s cap-exceeded/no-progress guards.
- **Integer duration sampling degenerating zones**: `w=0.0` was run explicitly as a control at every tier, both the hand-built Step-1 M0 instance and every generated tier. Result, checked directly: **100% dim-0 at every single tier** (M0: 3/3, M1: 25/25, M2: 123/123, M3: 543/543, M4: 8481/8481 zones all at affine dimension 0) -- the fully-degenerate control, exactly as expected, consistent with gridtimed's and gate_track's own `w=0` controls, and a clean confirmation that the width seen at `w=0.25` (up to 71.9% dim>=2 at M4) is genuinely coming from the duration-jitter mechanism and nothing else. `w=0.25` uses genuine real-valued interval sampling in `step()` and genuine real-valued interval bounds in the zone construction (`zones.py::interval`), never integer-only.

## Anomalies

One real bug was found and fixed during Step 4, caught by running `validate_containment` (not by inspection):

1. **A per-job clock's "not yet dispatched" semantics didn't match the zone construction's implicit clock semantics.** In the concrete simulator, a WAITING job's elapsed-since-dispatch clock `c_i` simply stays at 0 until that job is actually dispatched -- it is not a continuously-ticking clock the way gridtimed's switch clocks were (which genuinely track `y` until first touched, by explicit design). But pyudbm's `up()` advances *every* declared clock uniformly with no "selective freeze" -- so the first version of `zones.py` let every waiting job's clock silently drift with `g` through every `up()` call, while the concrete state it was meant to match stayed at exactly 0. First full validation run: 4 of 5 M0 states failed `validate_containment` (only the root, trivially at all-zeros, passed). **Fixed** by resetting every clock *not* currently active back to 0 once per transition (after the full micro-step trace, not after every micro-step -- provably equivalent and cheaper, since nothing during the trace ever reads or constrains a non-active clock's value, and `reset_value` discards a clock's entire prior relationship to other clocks regardless of how it drifted). Re-validated: 100% containment (5/5 on M0, 8481/8481 on M4) after the fix.

No `pyudbm` rejection, no unexpectedly-empty zone (beyond the one already-fixed containment bug above, which was a semantic mismatch, not an emptiness crash), no deadlock, and no vertex-counter disagreement occurred anywhere in the final (post-fix) runs reported above.

## Files

`modesel/{instance,env,dp,sanity_gate,generator,calibration,diagnostics,zone_ops,zones,geometry,decision_relevance,tiers}.py`. Entry points: `python -m modesel.sanity_gate`, `python -m modesel.calibration`, `python -m modesel.tiers`. Result JSON: `/tmp/modesel_results/{calibration,tiers}.json` (also `/tmp/modesel_results_calibration.json`, `calibration.py`'s own direct output path). `zoneviews/`, `pyudbm/`, `prob_jobshop_ext/`, `deadline_track/`, `gate_track/`, `gridtimed/` were not modified (verified: 0 files in those directories newer than the start of this work). `modesel/` reuses `zoneviews.zone.Zone` directly for `geometry.py`'s affine-dimension/facet/vertex measurements (as instructed); Measurement 3's smallest-DBM-hull construction is new, from-scratch machinery built directly on `pyudbm` (a different geometric operation -- containment of a point set -- than anything `Zone` exposes), not a modification to `zoneviews`.
