# Spike 3: real-valued interval durations + deadlines, on top of Spike 2

**Status: throwaway spike, not merged, package not refactored.** All code
lives in `/tmp/prob_jobshop_spike/` (the running scratch copy) plus
`/tmp/zone_common.py`, `/tmp/spike3_common.py`, `/tmp/spike3_taskB.py`,
`/tmp/spike3_policy.py` (throwaway measurement scripts). `prob_jobshop_ext/`
at the repo root was never touched. **Spike 2's delta removal and
`clip_to_k_box()` are kept** (per instructions) — this spike layers on top
of them, not instead of them.

**Files touched, cumulative diff vs. `prob_jobshop_ext/`:**
`instance.py` (new, this spike: `Job.deadline` field), `symbolic_env.py`
(this spike: interval/deadline semantics, on top of Spike 2's elapsed-time
addition), `symbolic_rl.py` (Spike 2 only, unchanged this spike),
`zone_ops.py` (Spike 2 only, unchanged this spike — `clip_to_k_box`/
`is_bounded`/no-`delta` all carried forward as-is). `region_env.py`,
`pta_env.py`, `simulator.py`, `region_rl.py`, `ddpg_*.py` untouched, same as
every prior spike. `symbolic_optimal.py` remains broken (Spike 2's `delta`
removal deleted `min_cost`, which it depends on) — still not fixed, out of
scope, same as before.

---

## Change 1 — real-valued interval durations

**pyudbm rejection, encountered and worked around — flagged as instructed:**
```
>>> zctx.clocks['c_J1'] <= 5.25
TypeError: '<=' not supported between instances of 'Clock' and 'float'
```
pyudbm's `Clock` comparison operators require an **integer** bound — they do
not attempt any float-to-int coercion, they raise immediately. Since
`[lo, hi] = [d*(1-w), d*(1+w)]` is generally non-integer, the symbolic side
represents bounds as **scaled integers**: `scaled_bound = round(real_bound *
SCALE)`, with `SCALE = 20` — chosen because it is *exact* (zero rounding
error) for both tested widths: `d*0.9*20`, `d*1.1*20`, `d*0.75*20`,
`d*1.25*20` are all exact integers for integer `d`. All pyudbm-facing
constraints (invariant, completion guard, `M`, deadlines) live in these
scaled units; **the concrete side never touches pyudbm and stays in real,
unscaled float units throughout**, per the task's explicit requirement.
Consequence worth noting for anyone reading raw zone strings from this
spike: a printed bound like `c_J1<=10` is **10 scaled units = 0.5 real time
units**, not 10 real time units.

Implementation (in `symbolic_env.py`, on top of Spike 2's version):
- `step()`: samples the duration **bucket** `d` exactly as before (same
  categorical probabilities, unchanged — bucket identity still drives
  `loc` encoding, machine/precedence enablement, and branch topology, same
  scope decision as Spike 1), then samples a **real** duration
  `real_dur = rng.uniform(lo, hi)` — a true float, **not** rounded or
  clamped to an integer. Confirmed empirically: recorded elapsed times like
  `5.205`, `1.933`, `3.227` (genuinely fractional, not just occasionally
  integer-valued by luck).
- `invariant()`: `c_J <= scaled(hi)` (was `c_J <= scaled_d`).
- `_compute_path()`'s completion guards: `c_J >= scaled(lo)` (was
  `c_J >= scaled_d`) — "the symbolic side does not pin clocks."
- `_compute_path()`'s `dt = min(remaining)` **still uses the bucket value
  for event-ordering/topology**, per the task's own instruction ("uses the
  concrete sampled durations as before") — only the *constraint value*
  emitted at each completion changed, not which task's completion triggers
  which path step.
- `_advance_to_decision()` (concrete side): prefers the real sampled
  duration (`self._active_real_dur[job_id]`) over the bucket when
  available, falling back to the bucket for `step_deterministic()` callers
  (the exact concrete DP — see Task B methodology note below).

**Anomaly found and flagged: strict inequalities appear for the first time
across this entire multi-spike investigation.** Every prior spike (audit,
Spike 1, Spike 2) never once observed `dbm.is_strict(i,j) == True` on any
zone. Under interval durations, they appear **frequently**:

| Instance | t | n zones | zones with ≥1 strict constraint | fraction |
|---|---|---:|---:|---:|
| PJS_01 | none | 18 | 11 | 0.611 |
| PJS_01 | 0.3 | 18 | 11 | 0.611 |
| PJS_02 | none | 51 | 38 | 0.745 |
| PJS_02 | 0.3 | 51 | 38 | 0.745 |

The fraction is **identical with and without a deadline** — this rules out
deadlines as the cause; it's the interval-width mechanism itself. Example
observed directly: zone `(16<c_J2 & c_J2<=c_J1)` — a genuine `c_J2 > 16`
(scaled units). Root cause not fully traced into pyudbm's C++ internals
(out of scope for the time available), but the most plausible mechanism,
stated as a hypothesis and clearly labeled as such: `normalize()`'s
`extrapolate_max_bounds` docstring says low lower bounds get "clamped to the
negated maximal constant" during k-normalization — this clamping plausibly
introduces the open/strict boundary artifact, and would naturally happen
more often once interval widening (and the ×20 scale factor) pushes more
values close to the `M` extrapolation boundary than the tighter
exact-duration case ever did. **Practical consequence, checked**: both
vertex-counting methods below (which both treat all constraints as
non-strict for solving, a standard and mathematically sound simplification
for *locating* extreme points — see next section) agreed on every zone
checked despite this, so it did not produce an observable vertex-count
error in practice, but it is a genuine, newly-appearing property of this
model variant worth carrying forward.

---

## Change 2 — deadlines

`Job.deadline: Optional[float] = None` added to `instance.py` (scratch
copy). `invariant()` adds `(c_J, scaled(D_J))` as an **additional** entry
alongside `(c_J, scaled(hi))` whenever the active task's job has a set
deadline — `apply_invariant()` already intersects every given pair, so this
is exactly `c_J <= min(hi, D_J)` with no special-casing. A branch whose
completion guard (`c_J >= lo`) cannot be satisfied under the tightened
invariant produces an empty zone, which the **existing** emptiness check in
`_compute_path`/`propagate` already prunes — "missing a deadline makes the
branch infeasible" required zero new pruning logic.

**Deadline-variant generator** (`/tmp/spike3_common.py:make_deadline_variant`,
not added to the package): `D_J = (Σ task.distribution.expected_duration()
for task in J.tasks) * (1+t)`. **Interpretation choice, stated explicitly**:
"nominal duration" isn't a single stored value for PJS_01-05/LJS_01 (unlike
the formula-generated PJS_07-10/LJS_02-10, which do store one) — `E[d]` per
task is the well-defined stand-in used here.

**Positive-control verification**: forced an absurdly tight deadline
(`D_J = 0.5`, real time units) on PJS_01 and confirmed the pruning mechanism
actually fires — exploration stalled at 7 states (root + 6 immediate
dispatch options, zero further progress), because every branch's own `lo`
already exceeds `0.5`. No pyudbm rejection, no unexpected empty root zone —
exactly the designed "edge simply unavailable" behavior.

**Deadline-tightness knob is miscalibrated for most instances — flagged
prominently, exactly as the task asked.** Measured fraction of the 200
random-policy episodes with ≥1 deadline violation, per instance:

| Instance | t=0.0 (w=0.10) | t=0.0 (w=0.25) | t=0.3 (both w) |
|---|---:|---:|---:|
| PJS_01 | **0.000** | **0.000** | 0.000 |
| PJS_02 | **0.000** | **0.000** | 0.000 |
| PJS_03 | 0.070 | 0.365 | 0.000 |
| PJS_04 | **0.000** | **0.000** | 0.000 |
| PJS_05 | **0.000** | **0.000** | 0.000 |
| LJS_01 | **0.000** | **0.000** | 0.000 |

**Every instance except PJS_03 shows 0% violation even at the "tight"
t=0.0 setting.** Mechanism, confirmed by construction, not guessed: the
violation check compares a **single task's** real sampled duration against
`D_J`, but `D_J` sums nominal duration **over every task in the job**. For
any job with 2+ tasks and low-to-moderate duration variance, no single
task's realization comes close to the *sum* of all its siblings' nominal
durations, so the deadline (as literally specified by the given formula) is
essentially unenforceable per-task. **PJS_03 is the one exception, and it is
directly explained by the instance's own documented character**: PJS_03 is
described in this suite's README as "high variance, wide spread" (durations
like 1 vs. 8 on the same task) — occasional outlier draws, especially
combined with the `×1.25` interval widening at `w=0.25`, are large enough to
individually exceed the whole-job nominal sum, producing a genuinely
non-degenerate 7–36.5% violation rate. This is a property of the formula
**as given** (sum-over-job compared per-task), not a bug in how it was
wired up — reported factually rather than silently worked around.

---

## Task B grid — aggregation ratio, w × t, all 6 instances

200 seeded random-policy episodes per cell, `concrete_key` now built from
**real-valued** clock tuples (rounded to 9 decimals purely for hashing
robustness, not semantically altered). Zero `propagate_failure`s across all
36 grid cells.

| Instance | w | t | records | distinct concrete | distinct zone | **agg. ratio** | loc-control ratio | cluster mean/median/max | frac size=1 | deadline viol. frac |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PJS_01 | 0.10 | any | 800 | 407 | 42 | **9.690** | 29.071 | 9.69/4.0/59 | 0.310 | 0.000 |
| PJS_01 | 0.25 | any | 800 | 407 | 42 | **9.690** | 29.071 | 9.69/4.0/59 | 0.310 | 0.000 |
| PJS_02 | 0.10 | any | 1,000 | 607 | 100 | **6.070** | 24.280 | 6.07/5.0/25 | 0.140 | 0.000 |
| PJS_02 | 0.25 | any | 1,000 | 607 | 142 | **4.275** | 24.280 | 4.28/3.0/22 | 0.331 | 0.000 |
| PJS_03 | 0.10 | 0.0 | 1,200 | 810 | 450 | **1.800** | 12.656 | 1.80/1.0/13 | 0.649 | 0.070 |
| PJS_03 | 0.10 | 0.3/none | 1,200 | 810 | 450 | **1.800** | 12.656 | 1.80/1.0/13 | 0.649 | 0.000 |
| PJS_03 | 0.25 | 0.0 | 1,200 | 810 | 466 | **1.738** | 12.656 | 1.74/1.0/13 | 0.665 | 0.365 |
| PJS_03 | 0.25 | 0.3/none | 1,200 | 810 | 466 | **1.738** | 12.656 | 1.74/1.0/13 | 0.665 | 0.000 |
| PJS_04 | 0.10 | any | 1,800 | 1,237 | 881 | **1.404** | 8.138 | 1.43/1.0/8 | 0.715 | 0.000 |
| PJS_04 | 0.25 | any | 1,800 | 1,237 | 938 | **1.319** | 7.731 | 1.35/1.0/6 | 0.756 | 0.000 |
| PJS_05 | 0.10 | any | 2,400 | 1,854 | 1,758 | **1.055** | 2.813 | 1.07/1.0/5 | 0.953 | 0.000 |
| PJS_05 | 0.25 | any | 2,400 | 1,854 | 1,765 | **1.050** | 2.809 | 1.07/1.0/5 | 0.956 | 0.000 |
| LJS_01 | 0.10 | any | 4,800 | 4,241 | 4,320 | **0.982** | 1.179 | 1.001/1.0/2 | 0.999 | 0.000 |
| LJS_01 | 0.25 | any | 4,800 | 4,241 | 4,323 | **0.981** | 1.163 | 1.000/1.0/2 | 0.999 | 0.000 |

(Rows collapsed where `t` doesn't change the number shown — the deadline
never prunes any branch actually visited by the random rollout except for
PJS_03 at `t=0.0`, exactly matching the violation-fraction finding above:
if the deadline is never binding, it cannot change which zones get
visited either, so `t=0.0` and `t=0.3`/`none` produce byte-identical
aggregation numbers everywhere except PJS_03. Full 36-row table in
`/tmp/spike3_taskB_results.json`.)

**The reversal from Spike 2, exactly as one would predict from "integer
sampling reproduces the degenerate case we just measured."** Spike 2's
ratio was below 1.0 for every instance (0.61–0.95×). Real-valued sampling
flips this dramatically for small instances: **PJS_01 now aggregates
9.69×**, PJS_02 4.3–6.1×, PJS_03 ~1.7–1.8×. Mechanism, directly observable
in the numbers: `distinct_concrete` is now only ~51–68% of `n_records`
(PJS_01: 407/800) instead of Spike 2's near-100% (real-valued floats
essentially never collide by luck, so most of that gap comes from genuinely
revisiting the *same* zone-and-location combination on different episodes,
not from duplicate concrete keys).

**But the effect shrinks monotonically with instance size, and reverses
again by the largest instance tested.** Ratio trend: 9.69 (PJS_01) → 4.3–6.1
(PJS_02) → 1.7–1.8 (PJS_03) → 1.3–1.4 (PJS_04) → 1.05 (PJS_05) → **0.98
(LJS_01) — back below 1.0**, matching Spike 2's finding almost exactly for
the one large instance tested. `frac_cluster_size_1` tracks this precisely:
0.31 (PJS_01) climbing to 0.999 (LJS_01) — i.e. by LJS_01, essentially every
zone contains exactly one concrete state again, same qualitative picture as
Spike 2, real-valued sampling notwithstanding. The location-only control
ratio also shrinks with size (29.1 → 1.16) but stays **above** the zone
ratio at every single grid point checked — zones never out-aggregate raw
location identity in this grid, at any instance size or width.

`w` (0.10 vs 0.25) has a real but secondary effect, and only where it
matters: identical for PJS_01 (9.690 both), a real difference for PJS_02
(6.070 → 4.275) and PJS_04 (1.404 → 1.319), negligible beyond that.

---

## Policy consistency & decision relevance — PJS_01–03 @ w=0.25, t=0.3

**Methodology note, stated explicitly (this is a genuine departure from the
literal Task-B protocol, necessitated by real-valued durations):** an
exhaustive DP over a continuous concrete-state space is not
tractable/well-defined the way Spike 2's finite-integer-branching DP was.
This check therefore falls back to the **same bucket-driven exact DP
methodology as Spike 2** (`step_deterministic`, `_active_real_dur` unset →
`_advance_to_decision` falls back to the bucket value), built **exhaustively**
(branching over every possible duration bucket, not sampled by 200 random
episodes) — this is a materially larger and more complete population than
Spike 2's rollout-derived clusters, tracking each concrete state's **zone**
in parallel via `propagate_one` (bucket-matched). Branches whose zone
propagation is symbolically infeasible (deadline miss) are dropped from the
DP — a deliberate choice, computing the optimal policy *as if*
deadline-violating outcomes are unavailable, consistent with how the zone
graph already treats them; flagged here rather than silently assumed.

| Instance | n_concrete_states | truncated | deadline-pruned branches | multi-state clusters | missing-from-DP | **policy-consistent** | **inconsistent** |
|---|---:|:---:|---:|---:|---:|---:|---:|
| PJS_01 | 31 | No | 0 | **3** | 0 | 3/3 (100%) | 0 |
| PJS_02 | 105 | No | 0 | **13** | 6 | 7/7 (100%) | 0 |
| PJS_03 | 667 | No | 0 | **70** | 14 | 56/56 (100%) | 0 |

**"If there are now thousands rather than 31, say so prominently"** — they
are **not** thousands, but the count did change, non-uniformly: PJS_01 has
*fewer* multi-state clusters under this exhaustive-exploration methodology
(3, vs. Spike 2's rollout-derived 6, on a different DP setup so not
directly comparable in origin, but worth noting the number moved rather
than trivially matching), while PJS_03 grew substantially (70, vs. Spike 2's
9). No policy-inconsistent cluster was found in any of the 66 checkable
multi-state clusters across all three instances (0 inconsistent) — same
clean result as Spike 2, now on a larger and more exhaustive population, so
somewhat more evidence against unsoundness, though still not a full
guarantee (missing-from-DP — states where every action was deadline-pruned
— excludes 6 and 14 clusters from the PJS_02/03 check respectively; these
are reported, not silently dropped).

**Decision relevance — "the number that says timing matters":**

| Instance | multi-zone locations | decision-relevant locations | **fraction** |
|---|---:|---:|---:|
| PJS_01 | 7 | 0 | **0.000** |
| PJS_02 | 17 | 0 | **0.000** |
| PJS_03 | 54 | 0 | **0.000** |

**Zero, across all three instances and all 78 multi-zone locations
checked.** Every location that had 2+ zones with well-defined optimal
actions had those zones' optimal-action-sets **overlap** — at least one
action was optimal regardless of which specific zone (which specific timing
history) you were actually in. By this measurement, at `w=0.25, t=0.3`,
timing information carried by the zone beyond location identity **never**
changed which action was optimal, for any of the 78 multi-zone locations
tested across PJS_01–03.

---

## Summary

**Change 1 (real-valued intervals)**: required scaled-integer bounds
because pyudbm rejects float bounds outright (confirmed, worked around,
documented) — and, unexpectedly, produces genuine strict inequalities for
the first time in this whole investigation (61–75% of PJS_01/02 zones),
traced to the interval-width mechanism itself (present with or without a
deadline), not fully root-caused into pyudbm internals but empirically
verified not to break vertex counting. **Change 2 (deadlines)**: the
pruning mechanism itself works correctly (verified via a positive-control
absurdly-tight-deadline test), but the specified formula — job-total
nominal duration compared against a single task's realized duration — makes
the tightness knob essentially inert (0% violations even at t=0.0) for
every instance except the one instance whose own documented character
(PJS_03, "high variance, wide spread") is enough to occasionally produce a
single outlier draw that exceeds the whole job's nominal budget. **Task B
grid**: real-valued sampling reverses Spike 2's below-1.0 finding
dramatically for small instances (PJS_01: 9.69×) but the effect shrinks
monotonically with instance size and crosses back below 1.0 by LJS_01
(0.98×) — location-alone aggregation is higher than zone aggregation at
every single point in the grid, regardless of size or width. **Policy
consistency**: no inconsistency found across 66 checkable multi-state
clusters (exhaustively enumerated, not just randomly sampled) spanning
PJS_01–03. **Decision relevance**: zero of 78 multi-zone locations across
PJS_01–03 showed genuinely conflicting optimal actions between zones
sharing a location — by this test, timing did not matter for any location
checked at `w=0.25, t=0.3`.
