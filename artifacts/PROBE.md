# Probe: positive-control test of the Spike 2/3 measurement pipeline

**Status: new module, scratch, `/tmp/inversion_probe/`. `prob_jobshop_ext/`
was never touched** (verified: `grep -rn "prob_jobshop_ext" *.py` finds only
one docstring mention, zero imports). This is **not a benchmark** — it is a
tiny, hand-verifiable synthetic automaton built specifically to test whether
the measurement machinery from Spikes 2/3 can detect a threshold effect we
already know the answer to.

**Kept from Spike 2/3, reused (not reimplemented)**: the delta-free
`ZoneContext` pattern (a minimal parallel class, `MicroZoneContext`, same
shape, instantiated for this model's own two clocks), `zone_ops.clip_to_k_box`
/ `is_bounded` / `normalize` / `apply_guard` / `reset_clocks` / `apply_delay`
/ `apply_invariant` / `init_zone` (imported directly from
`prob_jobshop_spike.zone_ops` — unmodified, fully generic functions),
`zone_common.exact_vertices_fraction` / `exact_vertices_and_facets` /
`affine_dim` / `make_propagate_one`, and the `SCALE=20` integer encoding.
**No scipy.spatial/qhull import anywhere in this module** (grep-verified).

---

## Necessary design decision (flagged, as instructed)

The task's literal spec for Task 1 (`invariant c<=9, completion guard c>=1`)
is a single flat interval — by itself it produces exactly **one** symbolic
zone at the post-task-1 decision point (g ranging over the whole [1,9]),
which cannot answer "for each zone, the optimal action" (there'd only be
one) or let decision relevance compare sibling zones (needs 2+). To make
Steps 1–2 answerable at all, Task 1 is represented with **nine discrete
integer buckets** `tau in {1,...,9}`, each *pinning* `c` (and hence `g`,
which ticks in lockstep with `c`) to that exact integer via
`invariant c<=tau AND guard c>=tau` — mirroring exactly the "discrete
bucket drives symbolic structure, real value drives concrete execution"
split used throughout Spikes 1 and 3, with bucket probability `1/9` each (a
discrete approximation of `Uniform(1,9)`, stated explicitly). The concrete
side samples a genuinely continuous `real_tau ~ Uniform(1,9)` and maps it to
the nearest integer bucket (clipped to [1,9]) purely to select which
symbolic branch to propagate. Full rationale in `model.py`'s module
docstring.

---

## Sanity / regression checks

- `test_ground_truth.py`: **PASS** — all 9 buckets' DP-computed optimal
  action matches `SLOW iff tau<=4, FAST otherwise` exactly, at `D=12`.
- Zero pyudbm rejections anywhere across the entire probe (all measurements
  below, 4 deadlines × 3 measurement types + cross-check = dozens of
  zone-graph builds). The one process-level pyudbm issue encountered was
  the **already-known** import-shadowing bug (a literal `pyudbm/` directory
  at the repo root shadows the installed package unless `import pyudbm`
  happens before the repo root hits `sys.path`) — same root cause
  documented in AUDIT.md §1, not a new finding; every driver script here
  imports `pyudbm` as its first line specifically to avoid it.
- `clip_to_k_box`: checked on every explored zone across all 4 deadlines
  (98 zones total) — **zero empty results, zero unbounded results**, same
  as Spike 2's finding on the real benchmark.

---

## Measurement 1 — threshold recovery (D=12)

| tau | g interval (real units) | optimal action | zone string |
|---:|---|---|---|
| 1 | [1.00, 12.00] | SLOW | `20<=c & c<=g & g<=240` |
| 2 | [2.00, 12.00] | SLOW | `40<=c & c<=g & g<=240` |
| 3 | [3.00, 12.00] | SLOW | `60<=c & c<=g & g<=240` |
| 4 | [4.00, 12.00] | SLOW | `80<=c & c<=g & g<=240` |
| 5 | [5.00, 12.00] | FAST | `100<=c & c<=g & g<=240` |
| 6 | [6.00, 12.00] | FAST | `120<=c & c<=g & g<=240` |
| 7 | [7.00, 12.00] | FAST | `140<=c & c<=g & g<=240` |
| 8 | [8.00, 12.00] | FAST | `160<=c & c<=g & g<=240` |
| 9 | [9.00, 12.00] | FAST | `180<=c & c<=g & g<=240` |

**9 zones found (expected 9), all distinct — the zone graph DOES
distinguish `g<=4` from `g>4` at this location.** Split confirmed exactly
at tau=4/5, matching the hand-derived ground truth.

**Note on the reported g-interval, worth being precise about**: each zone's
*lower* bound on g is exact and tight (`g >= tau`, matching the bucket).
The *upper* bound is **not** tau — it's the global deadline (12) for every
bucket. This is a direct, mechanistic consequence of reusing the same
zone-propagation pattern as every prior spike's `_compute_path`/`propagate`:
after the completion guard pins `c==tau` exactly, the code applies one more
`apply_delay` (an `up()` call) to reach the decision location — this is the
*same* structural step that, in the job-shop model, re-establishes bounds
for other still-active clocks; here, since the decision location's own
invariant no longer mentions `c` at all, that `up()` call relaxes the just
established `c<=tau` pin back toward the box bound before intersecting with
the (much weaker) decision-location invariant. **This did not break the
threshold measurement** — the DP's emptiness check for the SLOW branch
still correctly uses the zone's *minimum* reachable g (which remains
exactly `tau`, via the surviving `c<=g` relation from the untouched
reset-to-0 tie), so feasibility is determined correctly regardless. But it
does mean the zone's *reported interval* is looser than "pinned to a
point," which is worth knowing before reading zone strings from this whole
spike lineage as if bucket zones are always single points.

---

## Measurement 2 — decision relevance (Spike 3 exact metric)

| Deadline | multi-zone locations | decision-relevant | **fraction** |
|---:|---:|---:|---:|
| 10 | 1 | 1 | **1.0** |
| 12 | 1 | 1 | **1.0** |
| 16 | 1 | 1 | **1.0** |
| 24 | 1 | **0** | **0.0** |

**Strictly positive at D=10/12/16, and `('done','waiting')` — the
post-task-1 location — is the (only) relevant location in every positive
case, exactly as expected.** The metric is not broken. At **D=24 (negative
control)**: SLOW is feasible for every bucket (`tau+8<=24` for all
`tau<=9`) and, being strictly cheaper (0<5), is optimal for all 9 buckets —
all zones at the one multi-zone location agree, so decision relevance
correctly falls to exactly **0**. Confirmed by direct inspection: at D=24,
`slow_buckets=[1..9]`, `fast_buckets=[]`.

Threshold location tracks the deadline exactly, as it should:

| Deadline | SLOW buckets | FAST buckets | threshold tau |
|---:|---|---|---:|
| 10 | 1,2 | 3,4,5,6,7,8,9 | 2 |
| 12 | 1,2,3,4 | 5,6,7,8,9 | 4 |
| 16 | 1,2,...,8 | 9 | 8 |
| 24 | 1,2,...,9 | (none) | 9 |

(`threshold_tau = D - 8` exactly, as derived by hand — `D=10→2`, `D=12→4`,
`D=16→8`, `D=24→9` capped at the max bucket 9 since `D-8=16>9`.)

---

## Measurement 3 — aggregation ratio (D=12) — **anomaly found, flagged loudly**

| Deadline | records | distinct concrete | distinct zone | distinct loc | **agg. ratio (zone)** | **loc-control ratio** |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 600 | 384 | 20 | 3 | 19.200 | **128.000** |
| 12 | 600 | 401 | 23 | 3 | 17.435 | **133.667** |
| 16 | 600 | 401 | 27 | 3 | 14.852 | **133.667** |
| 24 | 600 | 401 | 28 | 3 | 14.321 | **133.667** |

**`!!! ANOMALY`: the zone ratio does NOT exceed the location-only control
at any deadline tested — the opposite of the task's stated expectation
("the zone ratio is expected to exceed the control").** Reported exactly as
found, per instructions, with the mechanism traced rather than papered
over:

**Per-location breakdown (D=12) reveals why**, and shows the zone
abstraction is *not* actually failing — the pooled ratio is a misleading
statistic in this specific tiny model:

| Location | distinct concrete | distinct zone | **per-location ratio** |
|---|---:|---:|---:|
| `('waiting','waiting')` | 1 | 1 | 1.0 (trivial — always the same start state) |
| `('done','waiting')` | 200 | 9 | **22.2** |
| `('done','done')` | 200 | 13 | **15.4** |

**Within every non-trivial location, the zone abstraction aggregates
substantially — 22× and 15× respectively — exactly as intended.** The
pooled/global ratio inverts this because this toy model has only **3**
possible location values total (by construction — a strictly sequential
2-task model has no room for the many distinct intermediate locations a
real job-shop instance has), so the location-only control ratio
(`distinct_concrete / distinct_loc`) gets an artificially tiny denominator
"for free," independent of how well location actually separates anything
*within* itself. The zone ratio's denominator (20–28 distinct zones) is
larger than 3, so even though zones out-aggregate concrete states far more
effectively *per location*, the **pooled** zone ratio ends up smaller than
the **pooled** location ratio. This is a structural property of comparing
pooled ratios in a model with very few distinct locations, not a failure of
the zone graph to capture the threshold (Measurements 1 and 2 already
independently confirm it does). Also note: `distinct_concrete=401` (not
600) even under fully continuous real-valued sampling — traced to the
`('waiting','waiting')` initial state being byte-identical across all 200
episodes (zero randomness before the first dispatch), collapsing what would
otherwise be 200 distinct rows into 1 and accounting for almost the entire
597→401 gap on its own.

**Read together with Measurement 2, the overall picture is**: the zone
graph *can* represent this threshold (Measurement 1), *does* make it show
up as a genuine location-level decision conflict (Measurement 2, including
the D=24 negative control behaving exactly as predicted), and *does*
aggregate substantially wherever it has room to (per-location breakdown of
Measurement 3) — but the specific **pooled** aggregation-ratio statistic,
exactly as literally specified, is not a reliable readout of that in a
model this structurally small, because it is dominated by how few distinct
locations exist at all. This is reported as a property of the **statistic**
under these conditions, not a recommendation to change it.

---

## Vertex-counter cross-check (exact Fraction vs. Spike 2 float method)

Every explored zone at every tested deadline (20+23+27+28 = 98 zones
total), clipped via `clip_to_k_box`, checked with both
`zone_common.exact_vertices_fraction` (this spike's rational-arithmetic
method, no qhull) and `zone_common.exact_vertices_and_facets` (Spike 2's
float+tolerance combinatorial method):

| Deadline | zones checked | disagreements | zones with a strict constraint |
|---:|---:|---:|---:|
| 10 | 20 | **0** | 0 |
| 12 | 23 | **0** | 0 |
| 16 | 27 | **0** | 0 |
| 24 | 28 | **0** | 0 |

**Zero disagreements, and — unlike Spike 3's job-shop interval model
(61–75% of zones) — zero strict inequalities observed anywhere in this
probe.** Consistent with the model design: Task 1's buckets *pin* `c`
exactly (no `[lo,hi]` spread the way Spike 3's interval-duration widening
introduced), so there is no analogous mechanism here to produce the
open/strict boundary artifact traced (as a hypothesis) to k-normalization
clamping in Spike 3.

---

## Summary

All three requested measurements ran; none were skipped. The positive
control **passes** for the properties it was explicitly built to test:
Measurement 1 confirms the zone graph represents the tau<=4/tau>4 threshold
exactly (9 distinct zones, split precisely between bucket 4 and 5, matching
hand-derived ground truth `threshold_tau = D-8`), and Measurement 2
confirms decision relevance is strictly positive at every deadline where a
threshold genuinely exists (D=10,12,16) with `('done','waiting')` correctly
identified as the relevant location, and correctly falls to exactly 0 at
the D=24 negative control where SLOW is unconditionally optimal — the
metric is not broken. Measurement 3's **pooled** aggregation ratio came
back below the location-only control at every deadline tested, contrary to
the task's stated expectation; traced to a specific, reported mechanism
(this toy model has only 3 possible locations total, giving the
location-only control an artificially favorable denominator that a
per-location breakdown shows is not representative of within-location
behavior, where the zone ratio is 15–22×). One design decision was
necessary and not specified verbatim by the task (discrete bucketing of
Task 1's otherwise-flat interval) — stated explicitly, with the alternative
(a single unsplit zone) noted as the direct cause of Steps 1–2 otherwise
being unanswerable. Zero pyudbm rejections, zero empty/unbounded clipped
zones, and zero vertex-counter disagreements were found anywhere in this
probe.
