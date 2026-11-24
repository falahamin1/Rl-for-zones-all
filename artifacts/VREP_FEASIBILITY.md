# VREP_FEASIBILITY: Step 0 Gate for the v_rep / GNN Encoder Addition

**Both checks pass. Proceeding to Steps 1-3.**

Methodology: 500 zones per instance (M3, M4, T2, T3), sampled via the same
uniform-random-policy rollout used throughout this project's diagnostics
(`difference_probe.collect_dataset`). A new content-keyed LRU cache for
`Zone.v_rep()` (`zoneviews.raster.cached_v_rep`) was added, reusing the
exact same `_zone_content_key` fingerprint the `PRESUBMIT` fix introduced
for `projection_stack` — not `str(zone)`, which was the address-based bug
that caused a real crash there. This is a genuine new piece of
infrastructure (v_rep had no cross-call cache before this), not a
reuse-in-name-only.

## 0a. v_rep() cost

| Instance | mean | median | p95 | max | limit-hit rate | cache hit rate (4-epoch replay) | projected 300k-step hours, uncached | ...with caching |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| modesel/M3 | 0.66ms | 0.60ms | 1.08ms | 1.19ms | 0.0% | 92.9% | 0.22h | 0.016h |
| modesel/M4 | 1.31ms | 1.12ms | 1.58ms | 19.2ms | 0.0% | 87.5% | 0.44h | 0.055h |
| gridtimed/T2 | 1.33ms | 1.11ms | 1.33ms | 39.2ms | 0.0% | 85.3% | 0.44h | 0.065h |
| gridtimed/T3 | 1.88ms | 1.74ms | 2.10ms | 23.5ms | 0.0% | 85.6% | 0.63h | 0.091h |

**Zero `VertexEnumerationLimit` hits across all 2000 sampled zones** (500 ×
4 instances). Worst case is 0.63 wall-clock hours per 300k-step run,
*uncached* — three orders of magnitude under the ~20-hour stop threshold.
With caching (measured hit rate 85-93% under a realistic 4-epoch replay of
the same 500-zone sample, matching `PPOConfig.n_epochs`), projected cost
drops to under 6 minutes per run on every instance. `deepset_v` is not a
cost concern on any of the four target instances.

Projection method, stated plainly: `total_touches = 300,000 steps ×
4 epochs`, each touch charged the *measured mean successful-call* cost
(cache misses only, uncached column) or that cost scaled by `(1 -
hit_rate)` (cached column). This slightly undercounts real touches (rollout
collection itself adds one more pass not multiplied by epochs), so the
true uncached number is closer to ~25% higher — still nowhere near the
20-hour threshold even doubled.

## 0b. Zone degeneracy

| Instance | dim 0 | dim 1 | dim 2 | dim 3 | dim 4 | dim-0 fraction | mean affine dim |
|---|---:|---:|---:|---:|---:|---:|---:|
| modesel/M3 | 126 | 217 | 157 | - | - | **25.2%** | 1.06 |
| modesel/M4 | 94 | 166 | 222 | 18 | - | 18.8% | 1.33 |
| gridtimed/T2 | 51 | 224 | 213 | 10 | 2 | 10.2% | 1.38 |
| gridtimed/T3 | 45 | 325 | 108 | 22 | - | 9.0% | 1.21 |

**No instance is majority dim-0** — the highest is modesel/M3 at 25.2%,
well under half. Worth flagging plainly rather than burying: a quarter of
M3's zones are still single-point/degenerate (v_rep encodes one vertex,
carries almost nothing there), so `deepset_v`'s M3 result should be read
knowing that ~1 in 4 of its training decision points give the encoder
essentially no shape information regardless of what the network does with
it — this is a property of the instance's own zone geometry, not a
network capacity or implementation issue. The other three instances are
substantially less degenerate (9-19% dim-0), with gridtimed's tiers the
least degenerate of the four (9-10%).

## Decision

Both gates pass with wide margin on all four instances. No array split is
needed — `deepset_v` and `gnn` both run on the full four-instance set,
same as the original four encoders. Proceeding to Steps 1-3.
