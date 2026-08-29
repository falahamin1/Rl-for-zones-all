# Geometry Gate: Interval Durations + Single Global Deadline Clock

Scratch fork of `deadline_track` in a new `gate_track/` package (`prob_jobshop_ext/` and `deadline_track/` were not modified — verified: 0 files newer than the start of this work). One shared-library fix was made in `zoneviews/zone.py` (`v_rep()`, item 5 of the task, explicitly in scope) — validated against the pre-fix brute-force algorithm on 270 real zones (0 mismatches) and against `zoneviews`'s own 39-test suite (all still pass) before being relied on for anything below.

**Total wall-clock, Measurement 1 (the gate): 721.1s (~12.0 min).** Measurement 2 + recalibration together added well under 5 more minutes. No PPO, no encoders, no training anywhere in this track.

## Model changes made

1. **Per-job `g_J` deleted, replaced by one global clock `g`.** `deadline_track`'s own geometry report proved every `g_J` is *always* exactly equal to every other (100.000% mutual pinning, zero exceptions, both release-time variants) — a single clock is the faithful model of what was actually there, not a simplification. Enforced non-reset via an actual assertion in `reset_clocks` (`assert cname != GLOBAL_CLOCK`), same pattern as `delta` in the original package.
2. **Interval-valued durations reinstated** (Spike 3's construction): invariant `c_J <= hi`, completion guard `c_J >= lo`, `[lo,hi] = [d*(1-w), d*(1+w)]`, same branch probabilities as before. `.step()` (used for calibration/violation-rate sweeps and for zone sampling) samples a **real** number uniform on `[lo,hi]`, not an integer — confirmed this matters: an early smoke test using integer-only sampling silently reproduced the degenerate single-point-zone case, which is exactly the failure mode the task warned about.
3. **Deadline mechanism preserved exactly**: edge-only feasibility checking, eager auto-fail to an absorbing `"failed"` status, the `deadline_track` advance↔resolve loop fix (re-verified: 0 deadlocks across every stress test run in this track, several hundred episodes each), the uniform worst-case-makespan clip bound (also re-verified necessary here: a waiting job's `c_J` is still not bounded by a single task's duration).
4. **Feasibility-check equivalence re-derived, not reused.** The arithmetic shortcut is arguably more directly justified now (`g` is one literal shared scalar, not `n` copies proven equal by a separate dynamics argument) — but the *best-case remaining work* term changed (now `min_nominal_duration * (1-w)` per remaining task). Cross-validated directly against `zone_ops.feasible_best_case` (a real DBM reset/invariant/delay/guard chain): **first run found 2/3682 disagreements** — traced to a real bug in the *validation script* (rounding `g` to the nearest integer before scaling instead of after, discarding up to 0.5 real units of precision). Fixed; re-ran: **3682/3682 agree (100.0000%)**, across PJS_04 (w=0.1, w=0.25), LJS_01 (w=0.25), PJS_05 (w=0.25).
5. **`v_rep()` fixed**: computes `affine_dim()` first via a weighted union-find that also tracks each clock's exact offset to its pinning-class representative (`zoneviews/zone.py::_pinned_offsets`), short-circuits to the single point directly at `dim=0` (no enumeration), and otherwise rewrites every `h_rep()` half-space into the reduced `affine_dim`-dimensional coordinate system before enumerating `C(facets, affine_dim)` subsets — not `C(facets, full_clock_dim)`. Validated against the old brute-force algorithm on 270 real sampled zones spanning affine dims 1-3: **0 mismatches**.

### Design decision flagged explicitly (not hidden in the DP's state-space definition)

The exact DP (Measurement 2) branches over the same finite discrete nominal-duration outcomes as before, treating each branch's nominal value `d` as its realized duration for transition/ordering/elapsed-time purposes. This is exact for *cost* accounting — `[d(1-w), d(1+w)]` is symmetric around `d`, so `E[Uniform(lo,hi)] = d` exactly, for any `w` — but it does not capture order uncertainty among concurrently-active tasks whose intervals overlap (which of two tasks with nominal 3 and 4 actually finishes first is only certain up to interval overlap in the true continuous model). The DP resolves ordering deterministically by nominal value, same tie-breaking as `deadline_track`. The **zone geometry** built for each DP-reachable state (used for Measurement 1 and for the multi-zone-location grouping in Measurement 2) uses the *true* `[lo,hi]` interval regardless — this is what makes the zones non-degenerate even though DP transitions are nominal-deterministic. Random-policy simulation (calibration, zone sampling) uses genuine real-valued sampling throughout, per item 2.

## Sanity gate (blocking, run before trusting anything above)

Same hand-built 2-job instance as `deadline_track`'s gate (durations 3 and 5, shared machine, minimum-slack-first computable on paper), checked at **w=0.0 as a regression control** (must exactly reproduce `deadline_track`'s result) and again at **w=0.25**:

| | w | Expected order | DP result |
|---|---:|---|---|
| A tight (D_A=3, D_B=8) | 0.0 | A first | `['A_T1']` ✓, V(root)=8.00 |
| B tight (D_A=8, D_B=5) | 0.0 | B first | `['B_T1']` ✓, V(root)=8.00 |
| A tight | 0.25 | A first | `['A_T1']` ✓, V(root)=8.00 |
| B tight | 0.25 | B first | `['B_T1']` ✓, V(root)=8.00 |

**PASSED**, all four, including the regression check.

## Measurement 1 — Geometry (THE GATE)

**Control (w=0.0)**: affine dim = 0 for every single sampled zone, on all 6 instances, with zero exceptions — exactly reproduces `deadline_track`'s degenerate result. Confirmed before reading anything else.

**Fraction of zones with affine dim ≥ 2, w=0.25** (the literal gate condition — "must exceed roughly 20% on at least PJS_04 and PJS_05"):

| Instance | w=0.0 (control) | w=0.1 | w=0.25 |
|---|---:|---:|---:|
| PJS_01 | 0.0% | 0.0% | 0.0% |
| PJS_02 | 0.0% | 25.2% | 26.2% |
| PJS_03 | 0.0% | 37.3% | 37.4% |
| **PJS_04** | 0.0% | 54.9% | **55.1%** |
| **PJS_05** | 0.0% | 63.6% | **62.5%** |
| LJS_01 | 0.0% | 79.0% | 78.8% |

**GATE PASSED.** PJS_04 (55.1%) and PJS_05 (62.5%) both clear the ~20% threshold by roughly 3×, not marginally.

Full geometry table at w=0.25 (2000 zones sampled where available; vertices sampled from the first 200):

| Instance | dim-0 frac | dim distribution | strict frac | facets (mean) | bounded frac | vertices (mean, n sampled, errors) |
|---|---:|---|---:|---:|---:|---|
| PJS_01 | 55.6% | {0:500, 1:399} | 0.0% | 6.00 | 100% | 1.45 (200, 0) |
| PJS_02 | 42.6% | {0:500,1:366,2:218,3:90} | 0.0% | 6.13 | 100% | 2.32 (200, 0) |
| PJS_03 | 34.5% | {0:500,1:408,2:319,3:181,4:43} | 0.0% | 8.14 | 100% | 4.12 (200, 0) |
| PJS_04 | 33.8% | {0:675,1:223,2:452,3:517,4:133} | 0.0% | 8.43 | 100% | 5.10 (200, 0) |
| PJS_05 | 26.3% | {0:526,1:224,2:314,3:363,4:435,5:138} | 0.0% | 10.81 | 100% | 10.48 (200, 0) |
| LJS_01 | 16.1% | {0:321,1:102,2:96,3:140,4:216,5:398,6:559,7:168} | 0.0% | 18.27 | 100% | 86.34 (200, **13 `VertexEnumerationLimit`**) |

**Strict-inequality fraction is exactly 0.0% everywhere** — both the interval invariant (`c_J <= hi`) and completion guard (`c_J >= lo`) are non-strict by construction (matching Spike 3's stated construction), and `w` doesn't introduce strictness anywhere else in the model, so this is architecturally expected, not a finding to interpret further. **Bounded fraction is 100% everywhere** (the clip-to-k-box bound is always sufficient, no empty-zone crashes — see the two real bugs already fixed under "Model changes" above, both caught before this final run). LJS_01's 13/200 vertex-enumeration failures are all `VertexEnumerationLimit` (genuinely large `C(facets, affine_dim)` at its higher clock count and dim up to 7 — e.g. dim=7 zones with ~18 facets can still exceed 200,000 subsets), not `UnboundedZone` or any other error — a real combinatorial-cost observation, not a bug.

## Re-calibration at w=0.25 (before Measurement 2, as required)

500 episodes/instance, paired seeds across t, real-valued interval sampling. All 6 instances still land in the requested [5%,50%] band; 5 of 6 instances' calibrated t are unchanged from `deadline_track`'s exact-duration calibration, one changed:

| Instance | New t (w=0.25) | Violation fraction | Old t (exact durations) | Changed? |
|---|---:|---:|---:|:---:|
| PJS_01 | 0.1 | 15.6% | 0.0 | **yes** |
| PJS_02 | 0.0 | 29.4% | 0.0 | no |
| PJS_03 | 1.0 | 17.2% | 1.0 | no |
| PJS_04 | 0.25 | 13.0% | 0.25 | no |
| PJS_05 | 1.0 | 25.0% | 1.0 | no |
| LJS_01 | 1.0 | 36.8% | 1.0 | no |

`deadlock_fraction = 0.0000` for every one of these 30 cells (15,000 episodes).

## Measurement 2 — Decision relevance (exact DP, w=0.25, recalibrated t, PJS_01-04)

| Instance | t | States | Truncated | Multi-zone locations | Conflicting locations | Conflict fraction | No-deadline baseline | Deadline-no-interval baseline |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| PJS_01 | 0.1 | 31 | No | 7 | 0 | 0.0% | 0.0% | 0.0% |
| PJS_02 | 0.0 | 151 | No | 17 | 0 | 0.0% | 0.0% | 0.0% |
| PJS_03 | 1.0 | 1,047 | No | 54 | 1 | **1.85%** | 0.0% | **1.85%** |
| PJS_04 | 0.25 | 2,286 | No | 144 | 0 | 0.0% | **1.4%** | 0.0% |

PJS_03's one conflicting location is the *same* location found in `deadline_track`'s report (J1's second task active with nominal duration 8, J2_T2/J3_T1 both otherwise-ready) — same instance, same t, so this is expected continuity, not independent confirmation of anything new about *which* locations conflict. State counts shifted slightly from `deadline_track`'s (e.g. PJS_04: 2,242→2,286) even where t is unchanged: the interval-aware feasibility check's best-case bound (`min_nominal*(1-w)`) is strictly more optimistic than the old exact-duration bound, so slightly fewer jobs fail early, growing the reachable state space a little — a real, expected consequence of the model change, not noise.

### Worked examples (two, both at PJS_03's one conflicting location — the task asked for "at least two conflicts"; this location's 3 zone groups produced exactly 2 pairwise conflicting comparisons, both reported below rather than only one)

**Location** (both examples): `J1_T1=done, J1_T2=active(nominal=8), J2_T1=done, J2_T2=waiting, J3_T1=waiting, J3_T2=waiting`. J1's second task is mid-flight (nominal duration 8, interval [6.0,10.0] at w=0.25); J2_T2 and J3_T1 are the two actually-dispatchable candidates.

Slack formula used throughout (per the task): `slack_J = (D_J - g) - W_J`, `W_J = (hi - c_J)` for J's *active* task (`hi` from its nominal duration's interval) plus nominal duration for every other remaining task — richer than `deadline_track`'s all-nominal formula, and it changes the numbers: J1 (which has the long-running active task) now shows *much lower* slack than before, because `hi=10.0` (the pessimistic bound for the realized-8 task) replaces the old formula's plain nominal 5.0.

**Example 0** (g=1 vs g=8):

| | Zone I (g=1) | Zone J (g=8) |
|---|---|---|
| **DP-optimal action** | `J3_T1` | `J2_T2` |
| J1 (active, hi=10.0): D, W, **slack** | 18.0, 10.0, **7.0** | 18.0, 10.0, **0.0** |
| J2: D, W, **slack** | 16.6, 5.0, **10.6** | 16.6, 5.0, **3.6** |
| J3: D, W, **slack** | 16.2, 8.1, **7.1** | 16.2, 8.1, **0.1** |

**Example 1** (g=4 vs g=8, same location, a different pair of zone groups):

| | Zone I (g=4) | Zone J (g=8) |
|---|---|---|
| **DP-optimal action** | `J3_T1` | `J2_T2` |
| J1 (active, hi=10.0): D, W, **slack** | 18.0, 10.0, **4.0** | 18.0, 10.0, **0.0** |
| J2: D, W, **slack** | 16.6, 5.0, **7.6** | 16.6, 5.0, **3.6** |
| J3: D, W, **slack** | 16.2, 8.1, **4.1** | 16.2, 8.1, **0.1** |

**Slack-ordering inversion, tested (not assumed) in both examples: it does not invert, again.** In every zone of both examples, ranked by slack: `J1 < J3 < J2` (J1's freshly-refined slack is now the *global* minimum, but J1 isn't a dispatch candidate — it's already active). Restricted to the two actual candidates, `J3 < J2` (J3 more urgent) holds in **every** zone of **both** examples — the ranking among the jobs that can actually be dispatched never flips — yet the DP's optimal action still flips from `J3_T1` to `J2_T2` as `g` increases. This reproduces `deadline_track`'s finding under a materially different (and more refined) slack formula, now with two independent zone-pair comparisons agreeing rather than one: the minimum-slack-first heuristic is not what the DP is actually optimizing, even when the slack computation is sharpened to account for the active task's realized-outcome interval rather than its a-priori nominal value.

## Anomalies

- Two real bugs, both caught by running the code and fixed before any number above was trusted: (1) the feasibility validation script's premature `g`-rounding (2/3682 spurious disagreements → 0/3682 after fix); (2) none new to the environment/DP itself — `deadline_track`'s two known bugs (deadlock loop, clip-bound sizing) were re-verified fixed, not re-broken, under the new interval model (0 deadlocks, 100% bounded fraction throughout).
- `v_rep()`'s exact enumeration still hits `VertexEnumerationLimit` on 13/200 sampled LJS_01 zones at w=0.25 (up from a hard 200/200 failure in `deadline_track`'s report, before the affine-dim-first fix) — a genuine combinatorial cost at LJS_01's higher clock count and now-real dimensionality (up to 7), not a bug; every failure is `VertexEnumerationLimit`, never `UnboundedZone` or anything else.
- No `pyudbm` rejection, no unexpectedly-empty zone, and no deadlock occurred anywhere in the final (post-fix) runs reported above.

## Files

`gate_track/{instance,zone_ops,symbolic_env,dp,collect,decision_relevance}.py`, `gate_track/{sanity_gate,validate_feasibility,geometry,calibration,run_measurement2}.py` (entry points). `zoneviews/zone.py` modified (`v_rep()` fix, `_pinned_offsets` helper added) — all 39 pre-existing `zoneviews` tests still pass. Result JSON: `/tmp/gate_results/{geometry,calibration,section5}.json`. `prob_jobshop_ext/` and `deadline_track/` were not modified (verified).
