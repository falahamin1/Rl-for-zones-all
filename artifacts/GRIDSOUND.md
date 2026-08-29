# GRIDSOUND: Fixing gridtimed's Zone Soundness, Then Scaling Up

Two tasks, in order: Task A (blocking — fix the within-zone non-uniformity
bug flagged in `/tmp/GRIDGATE.md`'s Anomalies §5) had to pass before Task B
(a parameterized generator, 5 tiers) was attempted. It passed. `gridtimed/`
was extended in place (`prob_jobshop_ext/`, `deadline_track/`, `gate_track/`,
`zoneviews/` all remain unmodified — verified: 0 files in those directories
newer than the start of this work). No encoders, no PPO, no training.

**Total wall-clock: Task A's w-sweep 68.5s + Task B's 5-tier x 3-seed run
272.5s ≈ 341s (~5.7 min).**

## Task A — the soundness bug, and the fix

### What was wrong

The previous zone construction built one symbolic zone per *sequence of
locations visited*, with each edge's delay **widened** to the full
`[min_delay+1, max_delay+1]` range regardless of which specific delay the
agent actually chose. Delay is a decision variable the agent picks and
knows it picked — not an unobserved random quantity — so this merged
concrete trajectories that took genuinely different actions into one zone.
Measured consequence: 40.65% of multi-point zones had internally disjoint
optimal-action sets. No zone-level policy could be optimal.

### The fix

1. **Delay is now exact per edge**, not widened: elapsed time for an edge
   is `delay + duration`, where `delay` is the specific value the action
   chose. The internal edge clock `e` is bounded via `reset -> up() ->
   invariant(e <= delay+1+w) -> guard(e >= delay+1)` — pinned to a
   constant whenever `w=0`.
2. **Genuine uncertainty now comes only from `w`**: a real, physically
   unobserved per-move duration `~ Uniform(1, 1+w)`, added on top of the
   exact delay. `w` is a `GridInstance` field (not a bare module constant,
   so it can be swept/generated), defaulting to `0.0`.
3. **SCALE=20** at the pyudbm boundary (gate windows, deadline, and the
   edge invariant/guard are all real-valued once `w>0`; concrete
   simulation — `env.py`/`dp.py` — stays in real/exact units throughout,
   unchanged by this fix).
4. **Architectural simplification, not just a parameter change**: zones
   are now attached 1:1 to concrete DP-reachable states by replaying each
   state's own discovery path (`zones.py::build_state_zones`), riding on
   the DP's own (already-finite, since the deadline bounds it) transition
   graph. The old `zone_graph.py` — a separate location-only BFS needing
   k-normalization + subsumption to terminate against location-graph
   cycles — is superseded and no longer called by geometry.py or
   decision_relevance.py (left in the repo, unmodified, as the artifact
   behind `/tmp/GRIDGATE.md`'s numbers). No new fixed-point machinery was
   needed to replace it.
5. **`env.py`/`dp.py` (the concrete simulator) were NOT touched.**
   `sanity_gate.py`'s Measurement 1 hand computation and its DP match are
   byte-identical to before — reconfirmed as a regression check (below)
   before trusting anything downstream.

### Regression check (env/dp untouched)

```
zone_I: expected d0_RIGHT (V=-98.0), got ['d0_RIGHT'] (V=-98.0) -- PASS
zone_J: expected d1_DOWN (V=-97.0), got ['d1_DOWN'] (V=-97.0) -- PASS
Measurement 1 (paper gate): PASSED.
```

### Within-zone non-uniformity — reported first and prominently, as required

| w | within-zone non-uniform fraction | n multi-state zones | n zones | n states |
|---:|---:|---:|---:|---:|
| **0.0** | **0% (n/a — 0 multi-state zones)** | 0 | 3436 | 3438 |
| **0.25** | **0% (n/a — 0 multi-state zones)** | 0 | 3436 | 3438 |
| **0.5** | **0% (n/a — 0 multi-state zones)** | 0 | 3436 | 3438 |

**0% at every `w` tested — the abstraction is sound. Proceeding to Task B.**

Not just empirically 0%: `n_multistate_zones=0` at every `w`, meaning the
check wasn't merely satisfied, it was never even put to the test on a zone
with more than one concrete member. This has a structural explanation,
checked directly (not merely assumed): under the fixed construction, a
zone's exact numeric position is determined by the same exact-delay
history that determines the concrete state producing it; for two genuinely
different concrete trajectories to land on the identical zone in a domain
this reset-heavy (switch resets break the naive "same total delay, any
order" symmetry that could otherwise cause coincidental overlaps), no
counterexample occurred across 3436 zones at any of the three `w` values.
Task B's larger, more topologically complex tiers (below) provide a
second, independent test of this at up to 7 clocks and ~85k states — still
0 multi-state zones anywhere.

### w=0.0 control

```
affine_dim_distribution: {0: 3436}   (100% dim-0, as expected)
pinned(s1,s2): 1.0
facets: 8.0 (exactly 2 x 4 clocks -- a minimal spanning-tree pin)
vertices: 1.0 (every zone is a single point)
```

Exactly reproduces the fully-degenerate control expected when there is no
duration jitter at all — matches the delay-restricted-to-`{0}` result from
`/tmp/GRIDGATE.md`'s Measurement 4, as it should (both represent "zero
timing freedom").

### Full geometry / decision-relevance table

| | w=0.0 | w=0.25 | w=0.5 |
|---|---:|---:|---:|
| affine dim distribution | {0:3436} | {0:536,1:137,2:823,3:1940} | {0:536,1:137,2:823,3:1940} |
| dim >= 2 fraction | 0.00% | 80.41% | 80.41% |
| pinned(s1,s2) | 100.00% | 21.62% | 21.62% |
| pinned(y,s1) / pinned(y,s2) | 100% / 100% | 24.10% / 23.95% | 24.10% / 23.95% |
| strict-inequality fraction | 0.00% | 0.00% | 0.00% |
| facets (mean / min / max) | 8.00 / 8 / 8 | 9.09 / 8 / 10 | 8.85 / 7 / 10 |
| vertices (mean / min / max, 0 errors) | 1.00 / 1 / 1 | 5.77 / 1 / 10 | 5.53 / 1 / 10 |
| bounded fraction | 100% | 100% | 100% |
| cross-zone conflict fraction | 50.00% | 50.00% | 50.00% |
| multi-zone locations | 8 | 8 | 8 |
| containment (validated) | 3436/3436 | 3436/3436 | 3436/3436 |

**w=0.25 and w=0.5 have identical affine-dimension distributions.**
Checked, not glossed over: dimension is a purely combinatorial fact about
*which* clock pairs end up exactly pinned, determined by the path topology
(which edges' widths accumulated onto which clocks), not by the numeric
value of `w` itself as long as `w>0` — so the same pinning *pattern*
recurs at both widths, only the zones' *size* differs (reflected in the
vertex/facet numbers, not the dimension histogram). **Cross-zone conflict
fraction is exactly 50.00% at every `w`, including w=0.0** — a direct
structural consequence of zones now being ~1:1 with concrete states (see
above): grouping by zone is, in this domain, equivalent to grouping by
concrete state, which doesn't depend on `w` at all (`w` only widens each
zone's *shape*, not which concrete states get grouped together).

### Strict inequalities (tested, not assumed)

**Still 0% even with real-valued jitter.** Both the invariant (`e <=
delay+1+w`) and the guard (`e >= delay+1`) were implemented non-strict
(`<=`/`>=`), matching the gate windows and deadline elsewhere in this
domain, and matching `[1, 1+w]`'s own closed-interval notation. This was a
deliberate choice, not an oversight — but it is a live design choice, not
a forced one: switching the invariant to strict (`e < delay+1+w`, on the
grounds that a continuous uniform draw lands exactly on its boundary with
probability zero) would be a one-line change in `zone_ops.edge_delay`, and
would start exercising the H-rep encoder's strict-facet path, which
nothing in this domain currently touches. Flagged for whoever picks this
up next rather than silently decided either way.

## Task B — scaling up

### The generator

`generator.make_instance(grid_n, n_switches, n_gates, window_width, D, w,
seed)`: places `start=(0,0)`, `goal=(grid_n-1,grid_n-1)` fixed, then
switches and their paired gates (gate `i` uses switch `i`'s clock) at
pseudo-random free cells, with each gate's window `[a, a+window_width]`
also pseudo-random. A cheap concrete-BFS solvability check (GOAL reachable
within `D`, ignoring cost/optimality entirely, budget-capped) rejects and
resamples on failure, distinguishing **confirmed unsolvable** from
**inconclusive** (budget exhausted before either GOAL or the full reachable
set was found) — a real, reported distinction, not a single bucket.

T0 here (`grid_n=3`, 2 switches, 2 gates) is **structurally analogous to**,
not bit-identical to, the hand-built canonical instance `instance.py`
defaults to — that hand-built instance remains the one `sanity_gate.py`
verifies against and is untouched by the generator.

Rejection rates (3 seeds, summed):

| Tier | attempts | rejected unsolvable | rejected inconclusive |
|---|---:|---:|---:|
| T0 | 5 | 2 | 0 |
| T1 | 3 | 0 | 0 |
| T2 | 3 | 0 | 0 |
| T3 | 3 | 0 | 0 |
| **T4** | **21** | 0 | **18** |

The check is doing real work, not degenerate (T0 rejects ~40% of naive
placements as genuinely unsolvable; T4's 8x8/5-switch layouts mostly hit
the reachability-BFS budget before confirming either way — an honest
"inconclusive" outcome from a bounded check at that scale, not a false
"unsolvable").

### Tiers (3 seeds each, w=0.25, uniform state_limit=150,000)

| Tier | grid | switches/gates | clocks | mean states | DP tractable | multi-zone locs | cross-zone conflict | within-zone nonuniform | dim>=2 frac | pinned(switch pair) mean | facets | vertices | mean wall |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 3x3 | 2/2 | 4 | 2190 | **yes** | 8.0 | 12.50% | **0%** | 73.06% | 27.42% | 8.70 | 4.37 | 1.58s |
| T1 | 4x4 | 2/2 | 4 | 11613 | **yes** | 15.0 | 2.22% | **0%** | 78.88% | 20.85% | 8.93 | 5.58 | 3.90s |
| T2 | 5x5 | 3/3 | 5 | 80819 | no (truncated) | 24.0 | 88.89% | **0%** | 93.85% | 8.97% | 12.02 | 10.11 | 19.35s |
| T3 | 6x6 | 4/4 | 6 | 69575 | no (truncated) | 35.0 | 77.14% | **0%** | 98.02% | 23.61% | 13.49 | 7.89 | 18.74s |
| **T4** | 8x8 | 5/5 | 7 | 66004 | no (truncated) | **63.0** | 78.31% | **0%** | 96.20% | 39.57% | 15.30 | 7.33 | 23.89s |

**Within-zone non-uniformity is 0% at EVERY tier and every seed** (15/15
tier-seed runs) — the soundness fix holds up to 7 clocks and ~85k states,
not just at T0's scale.

**Which tiers meet which requirement:**
- **>= 50 multi-zone locations (statistical power for conflict fraction):
  only T4** (63, 63, 61 across the 3 seeds). T3 (35) is the next closest
  and does not clear it.
- **Too large for exact DP but still trainable: T2, T3, T4.** All three
  truncate at `state_limit=150,000` (T2: ~81k states discovered but still
  truncated because far more remain in the frontier; T3/T4 similar).
  "Trainable" doesn't require exact DP — the concrete env steps
  identically regardless of DP tractability — but the DP-derived
  conflict-fraction numbers for these three tiers are **approximate**,
  computed over a BFS-frontier-limited subset with a substantial fraction
  of transitions hitting the conservative horizon-fallback value (244k-328k
  fallback references against 59k-85k actually-explored states per
  tier-seed — i.e. many explored states' successors point beyond the
  truncation boundary). T0 and T1 are both exact (well under the budget:
  ~2.2k and ~11.6k states respectively) and trainable.

**Facets and vertices grow with clock count** (T0: 4 clocks, 8.7 facets →
T4: 7 clocks, 15.3 facets), as expected; **vertex-enumeration never hit
`VertexEnumerationLimit` or `UnboundedZone`** at any tier (0 errors across
all 3600 vertex-computation attempts, 400/tier-seed, sampled since larger
tiers exceed the 4000-zone geometry cap).

**pinned(switch pair) mean is non-monotonic across tiers** (T0: 27% → T2:
9% → T4: 40%) — noted, not glossed over. A plausible (not fully verified)
mechanism: more switches means more pairs overall, and a larger fraction
of those pairs involve two switches that have *neither* been touched yet
at some point in the reachable set (both still trivially tracking `y`
identically) — inflating the mean pinning for higher switch counts via
more "never both touched" pairs, on top of whatever genuine structural
pinning exists. This was not independently confirmed by isolating the
never-touched-pair subpopulation; flagged as the most likely explanation,
not asserted as settled.

### gymnasium.Env readiness check (T0-scale instance, w=0.25, actually exercised)

```json
{
  "has_gymnasium": true,
  "single_env": {
    "fixed_discrete_n": 12,
    "mask_shape_consistent_every_step": true,
    "point_shape_consistent_every_step": true,
    "zone_object_usable_every_step": true,
    "observation_space_excludes_zone": true
  },
  "same_process_vector_env": {
    "n_vector_envs": 3,
    "works": true,
    "each_subenv_has_independent_pyudbm_context": true,
    "note": "NOT gymnasium.vector.SyncVectorEnv (Federation objects aren't picklable and have no gym.Space representation) -- a hand-rolled same-process analog, same as train/env.py's."
  }
}
```

Fixed `Discrete(N)` action space (`N = len(allowed_delays)*4`, unchanged
shape every step — illegal actions stay in the fixed action set as
documented self-loops rather than shrinking the space). Observation dict
has `point` (`(K+1,)` float32: switch clocks then `y`), `zone` (a live
`zoneviews.zone.Zone`, deliberately excluded from `observation_space` —
same reason `train/env.py` excludes it: no `gym.Space` represents a live
pyudbm object), and `mask` (`(N,)` bool). Reset + 25 steps run cleanly;
mask/point shapes stay fixed every step; the zone stays usable
(`.affine_dim()` callable) every step. A same-process 3-way vector env
(mirroring `train/env.py`'s hand-rolled `SyncVectorEnv`, not
`gymnasium.vector.SyncVectorEnv`, for the identical reason train/env.py
gives) was built and stepped 25 times; each sub-env's pyudbm `Context` was
confirmed independent (`id()`-distinct, not a shared singleton) — Zone
objects DO live cleanly in a same-process vector env, tested directly, not
assumed from train/env.py's precedent alone.

## Anomalies

Four real issues were found and fixed during this work, all caught by
running the code (not by inspection), consistent with this repo's
established "cross-validate before trusting" discipline:

1. **(Task A's core finding, already described above)** the pre-existing
   delay-widening zone construction caused 40.65% within-zone
   non-uniformity. Root-caused and fixed as Task A's main deliverable.
2. **Genuine deadlock states exist in generated (non-hand-built) layouts.**
   The hand-built T0 instance structurally guarantees every cell has a
   plain/switch (never gate-only) neighbour, so `dp.py`'s original code
   asserted this and crashed the first time a generated T0-tier seed (seed
   2) produced a cell surrounded only by currently-closed gates — 28 such
   deadlock states in that one run. **Fixed**: `dp.py` now treats a
   zero-legal-action state as a permanent trap (V = DEADLINE_FAIL_PENALTY,
   empty optimal-action set, counted in `n_deadlock_states`) instead of
   crashing; `decision_relevance.py` excludes deadlocked states from
   conflict analysis (two "both stuck" groups sharing no valid action is
   not a decision-relevant conflict). Worth flagging structurally: a
   permanent deadlock never triggers the deadline auto-fail either, since
   `y` cannot advance from a state with zero legal moves — a genuine trap
   the original design's safety net doesn't catch.
3. **The clip-bound (`K_BOUND`) bug recurred, exactly the class flagged in
   `/tmp/GRIDGATE.md` and `deadline_track` before it.** `K_BOUND=20` was
   tuned for T0's `D=16` and silently emptied zones (`ValueError: ...
   empty federation`) for generated tiers with larger `D` (T1: 21, up to
   T4: 43). **Fixed**: `geometry.py` and `gym_readiness.py` now compute
   `k_bound_real = max(K_BOUND, instance.deadline + 4)`, scaled per
   instance rather than a fixed global constant. This is the third time
   this exact bug class has appeared across this project's tracks
   (deadline_track's `m_c` note, gate_track's uniform-bound fix, now
   here) — worth naming as a recurring failure mode: any bound derived
   from *one* instance's parameters and then reused as a global default
   will eventually meet a differently-scaled instance and silently break.
4. **A real O(k²) performance bug**, not a correctness bug, but severe
   enough to report: the cross-zone (and within-zone) conflict-detection
   loop compared every PAIR of zones at a location. Harmless at T0's scale
   (8 zones/location) but combinatorially explosive once zones are ~1:1
   with concrete states (Task A's fix) at Task B's scale — a single T2
   seed with ~3000 zones at one location produced ~4.5M pairwise
   comparisons and the run never completed within a 15-minute wait budget.
   Diagnosed by isolating each pipeline stage with direct timing (DP: 6s,
   zone construction: 3s, containment validation: 1s — all fast;
   conflict analysis: the hang) rather than guessing. **Fixed**: bucket
   zones by their (small, bounded by the fixed action set) UNIONED
   optimal-action-set first, and only compare pairs of DISTINCT
   action-sets — mathematically equivalent (whether any pair of zones has
   disjoint action-sets depends only on which distinct action-sets occur),
   not quadratic in zone count. Re-verified against the pre-fix numbers on
   T0 (unchanged) before trusting the fix; full 5-tier x 3-seed run then
   completed in 272.5s.

No `pyudbm` rejection, no other unexpectedly-empty zone, and no vertex-counter
disagreement occurred anywhere in the final (post-fix) runs reported above.

## Files

`gridtimed/{instance,env,dp,sanity_gate,zone_ops,zones,geometry,decision_relevance,generator,tiers,gym_readiness}.py`.
`zone_graph.py` retained unmodified but no longer called (superseded, see
Task A). Entry points: `python -m gridtimed.sanity_gate`, `python -m
gridtimed.geometry`, `python -m gridtimed.decision_relevance`, `python -m
gridtimed.tiers`, `python -m gridtimed.gym_readiness`. Result JSON:
`/tmp/gridsound_results/{task_a_sweep,tiers,gym_readiness}.json`.
`prob_jobshop_ext/`, `deadline_track/`, `gate_track/`, `zoneviews/` were not
modified (verified: 0 files in those directories newer than the start of
this work).
