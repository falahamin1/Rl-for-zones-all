# GRIDGATE: A Fresh Grid-Navigation Timed-Automata Gate

New package `gridtimed/`, built from scratch -- does not fork or import from `supplementary/frozen_lake`, `supplementary/taxi`, or `Timed-RM` (verified: no such imports anywhere in `gridtimed/*.py`, grep-checked). Reuses only `zoneviews/` (`Zone`, the affine-dim-first `v_rep()` fix, `HalfSpace`/`UnboundedZone`/`VertexEnumerationLimit`) and `pyudbm` directly. `prob_jobshop_ext/`, `deadline_track/`, and `gate_track/` were not modified (verified: 0 files in those three directories newer than the start of this work). No encoders, no PPO, no training anywhere in this track.

**Total wall-clock for the full pipeline (sanity gate + both geometry runs + both decision-relevance runs, default and delay-restricted): 1.7s.** This is a genuinely tiny instance by design (9 grid cells, 4 clocks, <3500 concrete states) -- exhaustive analysis throughout, no sampling anywhere.

## The instance

3x3 grid, agent at `(0,0)`, goal at `(2,2)`:

```
(0,0)START (0,1)SWITCH1 (0,2)
(1,0)SWITCH2 (1,1)      (1,2)GATE1
(2,0)        (2,1)GATE2 (2,2)GOAL
```

Clocks: `y` (global, never resets -- enforced by an assertion in `zone_ops.reset_clocks`, same pattern as `delta`/`g` elsewhere in this repo), `s1`/`s2` (reset to 0 on entering switch 1 / switch 2; before their first activation they advance from 0 like any other clock, i.e. all three clocks are identical at episode start and only diverge once a switch is first touched -- this is why the pinning fractions below are population statistics over the *whole* episode, not just late-episode states).

Gate windows (chosen to produce a hand-verifiable inversion, see Measurement 1): **gate1 passable iff `s1` in `[4, 6]`**, **gate2 passable iff `s2` in `[5, 7]`**. Deadline `D = 16` (`y <= D` required at the instant of reaching goal). Actions: `delay in {0,1,2}` (or `{0}` for Measurement 4) x `move in {UP,DOWN,LEFT,RIGHT}` -- a fixed 12-action (or 4-action) space; illegal actions (walls, or a gate entered outside its window) are still valid to select: they self-loop, cost a small penalty (2.0), and elapse no time. Legal moves cost `1.0` per elapsed time unit; reaching goal with `y<=D` gives `+100` (net); exceeding `D` before reaching goal gives a `-50` terminal penalty and absorbs into a `FAILED` state (structurally required for the zone graph to terminate at all -- see "finite without extrapolation?" below). No slip; movement is fully deterministic.

The concrete environment (`env.py`) is intentionally pure-functional (`GridState` is a small frozen, hashable dataclass; `peek(state, action_id)` is a pure transition function) rather than a stateful object with save/restore -- this domain is small and deterministic enough that the extra machinery `gate_track`/`deadline_track` need for that isn't warranted here.

### A necessary internal clock: `e`

Representing "this edge's elapsed time is in `[min_delay+1, max_delay+1]`" symbolically requires a clock that resets on *every* edge, not just at task-dispatch (`c_J`'s role in `gate_track`) or never (`y`). `zone_ops.py` adds one, `e`, reset immediately at both the start and the end of every edge (start: to begin measuring; end: to close the book once the invariant/guard have been applied, so it doesn't leak into whatever the next edge measures). Because `e` is always reset before a zone is *read out*, it is always pinned to a constant in every reported zone -- contributes exactly 0 to affine dimension (pinning doesn't require the offset to be zero, only that it's exact) -- confirmed empirically: for the fully-degenerate delay-restricted case every zone has exactly 8 facets, which is precisely `2 x 4 clocks` (a minimal spanning-tree equality representation), not `2 x 5` -- i.e. `h_rep()`'s redundancy elimination is correctly absorbing `e`'s trivial pin via transitivity through the other clocks, not leaving spurious extra facets. This was checked, not assumed, before trusting `facet_count()` anywhere below.

## Measurement 1 -- THE PAPER GATE (blocking)

Two hand-picked states at the center cell `(1,1)`, both reached via the fixed 4-move path `(0,0) -RIGHT-> switch1 -DOWN-> (1,1) -LEFT-> switch2 -RIGHT-> (1,1)`, using the same delay at all 4 steps (all-0 for zone I, all-2 for zone J):

- **Zone I** `(s1=3, s2=1, y=4)`: RIGHT (gate1 attempt) is legal for every delay in `{0,1,2}` (arrival `s1 in {4,5,6}`, all inside `[4,6]`); DOWN (gate2 attempt) is illegal for every delay (arrival `s2 in {2,3,4}`, all outside `[5,7]`, and delay is capped at 2 so it cannot be waited into range from here). Hand-computed optimal: **RIGHT** (delay 0), reaching goal at `y=6`, `V=-98`.
- **Zone J** `(s1=9, s2=3, y=12)`: RIGHT is illegal for every delay (arrival `s1 in {10,11,12}`, already past `[4,6]` -- `s1` only increases, so no amount of *this* action's delay recovers it); DOWN is legal for delay in `{1,2}` (arrival `s2 in {5,6}`, inside `[5,7]`). Hand-computed optimal: **DOWN** (delay 1), reaching goal at `y=15`, `V=-97`.

```
zone_I: expected d0_RIGHT (V=-98.0), got ['d0_RIGHT'] (V=-98.0) -- PASS
zone_J: expected d1_DOWN (V=-97.0), got ['d1_DOWN'] (V=-97.0) -- PASS
Measurement 1 (paper gate): PASSED.
```

Both states are genuinely DP-reachable (not merely hypothetical) and the exact DP's optimal action and value match the hand computation exactly at both. **PASSED** -- proceeding to Measurements 2-4.

## Measurement 2 -- pinning and zone geometry

Zones are built via a location-graph BFS (`zone_graph.py`): from each of the 9 grid cells, an edge per direction, guarded by the target gate's window when entering a gate, reset when entering a switch, and (necessarily -- see below) capped by the same `y<=D` deadline guard the concrete env uses. Because the location graph has real cycles (a switch can be revisited indefinitely) unlike `gate_track`/`deadline_track` (acyclic once tasks complete), termination requires k-normalization (`extrapolate_max_bounds`, `k=20`) plus a subsumption check (`Federation.__le__`) against a per-location passed list. Analyzed **exhaustively** (every zone in the final passed lists, not sampled) -- the full zone graph is only 244 (default delays) or 447 (delay-restricted) zones.

### (a) Mutual pinning of s1 and s2 -- reported first, as required

| | default delays `{0,1,2}` | delay restricted to `{0}` |
|---|---:|---:|
| **`pinned(s1, s2)` fraction** | **0.41%** | **100.00%** |

Under the default action set this is **not** near 1.0 -- the design does not have the same failure as `deadline_track`'s `g_J` clocks (there, every job's deadline clock was *provably* always exactly equal because a single-machine flow forces simultaneous dispatch instants; here `s1` and `s2` are reset at genuinely independent times, chosen by the agent, in whatever order it likes). Under the delay-restricted variant, pinning jumps to exactly 1.0 -- expected and explained below, not itself evidence of a design flaw (it's a different, deliberately degenerate regime, see Measurement 4).

### (b) y-s1 / y-s2 pinning

| | default delays | delay restricted to `{0}` |
|---|---:|---:|
| `pinned(y, s1)` | 20.49% | 100.00% |
| `pinned(y, s2)` | 13.93% | 100.00% |

Non-trivial under default delays (reflects the population of zones from episode segments before the *relevant* switch has ever been touched, where that clock still equals `y` identically) -- expected, not fatal.

### (c) Affine dimension distribution (over clipped zones)

| dim | default delays | delay restricted to `{0}` |
|---:|---:|---:|
| 0 | 1 (0.41%) | 447 (100.00%) |
| 1 | 10 (4.10%) | 0 |
| 2 | 97 (39.75%) | 0 |
| 3 | 136 (55.74%) | 0 |
| **dim >= 2** | **95.49%** | **0.00%** |

Neither of the two "fatal" conditions the task warned about holds under default delays: pinning is nowhere near 1.0, and dimension is overwhelmingly >= 2, not concentrated at 0 or 1. The design survives Measurement 2.

### (d)/(e) Facets, vertices, boundedness, strictness

| | default delays | delay restricted to `{0}` |
|---|---:|---:|
| strict-inequality fraction | 0.00% | 0.00% |
| facets (mean / min / max) | 8.57 / 7 / 10 | 8.00 / 8 / 8 |
| vertices (mean / min / max, exact, 0 errors) | 6.13 / 1 / 12 | 1.00 / 1 / 1 |
| bounded fraction | 100.00% | 100.00% |

Strict fraction is exactly 0% everywhere -- every guard/invariant in this model (gate windows, the deadline, the edge-elapsed bound) is non-strict `<=`/`>=` by construction, so this is architecturally expected. Bounded fraction is 100% throughout (see the clip-bound note below for why this required a real fix, not just the `K_BOUND=20` choice alone).

### Finite without extrapolation, or was k-normalization needed?

**Measured, not assumed: k-normalization was necessary.** A version of the same BFS with `extrapolate_max_bounds` removed (subsumption checking alone, everything else identical) did not terminate within a 200,000-edge budget (>120s). With k-normalization (`k=20`) restored, the full BFS terminates in 17-29ms and explores only 672-1228 edges (359-341 pruned as subsumed, 70-441 pruned as empty). This is genuinely load-bearing machinery, unlike `gate_track`/`deadline_track` where the task-completion structure makes the state space acyclic on its own.

## Measurement 3 -- decision relevance

For every DP-reachable concrete state, the matching *symbolic* zone is built by replaying the same sequence of grid locations visited (not the specific delay chosen -- the zone widens delay to the full `[min_delay+1, max_delay+1]` range, exactly the same "concrete uses one value, the zone uses the whole allowed range" asymmetry `gate_track` uses for task durations). Two flavors of decision relevance are reported, because the design surfaced a stronger-than-expected phenomenon (see below):

**Cross-zone conflict fraction** (the statistic directly comparable to the job-shop baselines): a location has >1 distinct zone group, and two of those groups have disjoint DP-optimal-action sets.

| | multi-zone locations | conflicting | **conflict fraction** | job-shop baselines (all variants) |
|---|---:|---:|---:|---|
| default delays `{0,1,2}` | 8 | 3 | **37.50%** | 0% - 2.6% |
| delay restricted to `{0}` | 8 | 4 | **50.00%** | 0% - 2.6% |

Both figures are an order of magnitude above every job-shop-family baseline measured in this repo (`ENV_AUDIT.md`, `DEADLINE.md`, `GATE.md`: 0%-2.6% throughout). Container-validated: every zone was checked to actually `contain()` its own concrete point (`IntValuation`) before being trusted for grouping -- **3436/3436 (default) and 408/408 (restricted) valid**, 0 disagreements, after fixing a real bug (below).

**Within-zone non-uniformity** (a *stronger*, additional finding, not part of the task's named statistic but surfaced directly by the hand-verified Measurement 1 example -- see "Anomalies"): a single symbolic zone (one location-history, delay widened to the full range) can itself contain concrete points with disjoint optimal-action sets, meaning even "which zone you're in" is an insufficient summary for optimal control, not just "which location."

| | zones with >=2 concrete member states | non-uniform | **within-zone non-uniform fraction** |
|---|---:|---:|---:|
| default delays `{0,1,2}` | 155 | 63 | **40.65%** |
| delay restricted to `{0}` | 0 | 0 | n/a (every zone is a single point, dim=0 everywhere -- see Measurement 2) |

### Worked example 1 (within-zone, formally verified in Measurement 1)

`ZONE_I = (s1=3, s2=1, y=4)` and `ZONE_J = (s1=9, s2=3, y=12)`, both at location `(1,1)`, are contained in the *literal same* symbolic zone (`str(zone_I) == str(zone_J)`, verified directly): both reach `(1,1)` via the identical 4-move location sequence, differing only in per-step delay. Optimal actions: `d0_RIGHT` (I) vs. `d1_DOWN` (J) -- disjoint. Tested (not assumed) whether this coincides with a gate-window boundary, using actual reachability-via-some-allowed-delay rather than raw current-value membership (see "Anomalies" for why that distinction mattered):

```
RIGHT (-> gate1, s1, window [4,6]): reachable from I = True,  reachable from J = False  -- crosses
DOWN  (-> gate2, s2, window [5,7]): reachable from I = False, reachable from J = True   -- crosses
explained_by_gate_boundary: True
```

Both differing moves flip gate-reachability between the two states. This is the mechanism the instance was designed to exercise, confirmed by direct test.

### Worked example 2 (cross-zone, location `(1,1)`, a second, independently-discovered conflict)

`I = (s1=2, s2=1, y=2)` (reached having touched **only** switch2 so far, before switch1's first activation -- `s1` here is still tracking `y` directly, not yet a "switch1 elapsed" clock) vs. `J = (s1=9, s2=8, y=9)`, in two different zone groups at the same location:

```
I optimal: d1_RIGHT   (-> gate1 attempt; s1=2, arrival via delay 1 = 4, inside [4,6])
J optimal: d0_UP       (-> revisit switch1, resetting s1; neither gate reachable from J at all)
gate1 reachability: reachable(I)=True, reachable(J)=False -- crosses
gate2 reachability: reachable(I)=False, reachable(J)=False -- does not cross (irrelevant to this pair)
explained_by_gate_boundary: True
```

### A conflict NOT explained by a gate boundary (tested, not assumed)

Two of the three cross-zone conflicts at `(0,0)` -- e.g. `I=(s1=0,s2=0,y=0)` (episode start) optimal `DOWN` (visit switch2 first) vs. `J=(s1=5,s2=1,y=5)` optimal `RIGHT` (visit switch1 -- again) -- have `gate_boundary_flip=False`: neither differing move (`DOWN`->switch2, `RIGHT`->switch1) targets a gate at all from `(0,0)`, so this flip is about switch-visit-order/timing preference, not any gate window directly. Reported honestly rather than force-fit to the gate-window narrative; no further causal claim is made about *why* switch order flips (that would need its own dedicated investigation, out of scope here).

## Measurement 4 -- delay-action sensitivity (delay restricted to `{0}`)

Both geometry and decision relevance were re-run (not assumed) with `allowed_delays=(0,)`. Side-by-side:

| | default delays `{0,1,2}` | delay restricted to `{0}` |
|---|---:|---:|
| `pinned(s1,s2)` | 0.41% | **100.00%** |
| affine dim >= 2 fraction | 95.49% | **0.00%** |
| cross-zone conflict fraction | 37.50% | 50.00% |
| within-zone non-uniform fraction | 40.65% | n/a (0 multi-state zones) |

Restricting delay to `{0}` collapses the geometry completely (every zone becomes a single point -- `e` is pinned to exactly 1 rather than ranging over `[1,3]`, so no edge introduces any real freedom at all, and two states differing only in *how many times* they looped are literally different, non-subsumed, single-point zones rather than points within one wider zone). This directly confirms delay is the source of geometric width in this model, not merely a convenience: with it removed, "which zone" and "which concrete point" become the same question (within-zone non-uniformity is structurally impossible, 0 multi-state zones), while cross-zone conflict fraction *rises slightly* (37.5% -> 50.0%) -- removing waiting flexibility does not make the location-alone abstraction more sufficient, it makes it slightly less so, because the agent can no longer time a *single* action to land inside a window and instead must reach the right (s1, s2, y) combination via extra whole moves, producing more, thinner zones per location with less overlap in their reachable actions.

## Also reported

- **Total reachable concrete states**: 3438 (default delays), 410 (delay-restricted). **Total reachable zones**: 244 (default delays), 447 (delay-restricted). Neither BFS was truncated (`state_limit=200,000` never approached).
- **Wall-clock**: full pipeline (Measurement 1 + both geometry runs + both decision-relevance runs) = **1.7s** total.
- **Finite without extrapolation?** No -- see Measurement 2's dedicated note. k-normalization (`k=20`) is necessary and was verified necessary by directly removing it and observing non-termination within a 200k-edge budget.

## Anomalies

Two real bugs were found and fixed during this work, both caught by cross-validating a "shortcut" against an independent check before trusting it (the established discipline in this repo) rather than by inspection:

1. **The internal edge-elapsed clock `e` was not reset after use.** First version of `zone_graph.py`/`decision_relevance.py`'s edge construction correctly reset `e` to 0 at the *start* of each edge (to begin measuring) but left it sitting in `[min_delay+1, max_delay+1]` afterward instead of resetting it again immediately. Consequence: (a) `Federation.contains()` validation of concrete points against their own replayed zone failed almost universally (3435/3436 disagreements) since the concrete point's implicit `e=0` didn't match the zone's un-reset `e` range; (b) `geometry.py`'s affine-dimension counts were inflated by a spurious extra dimension in most zones (pre-fix: `{0:1,1:8,2:40,3:140,4:75}`, dim>=2 = 96.6%; post-fix: `{0:1,1:10,2:97,3:136}`, dim>=2 = 95.5% -- the *qualitative* conclusion doesn't change, but the numbers were wrong until fixed). **Fixed** by resetting `e` a second time immediately after each edge's guard/invariant/reset steps, in both `zone_graph.py` and `decision_relevance.py`. Re-validated: containment 3436/3436 (default), 408/408 (restricted), 100% agreement.
2. **The zone-graph BFS was originally deadline-unconstrained.** Without intersecting each edge with `y<=D`, some zones' *minimum* reachable `y` (from enough switch-revisit loops) exceeded `K_BOUND=20` even though the BFS terminated (via subsumption) -- these zones then silently went empty under `clip_to_k_box()`, crashing `Zone.__init__` (`ValueError: ... empty federation`) the first time geometry.py was run to completion. Same class of bug as `deadline_track`'s clip-bound note (a too-small effective bound silently emptying zones), here caused by the zone graph modeling an unbounded game instead of the actual (deadline-bounded) one. **Fixed** by intersecting every edge with the same `y<=D` guard the concrete env uses, tying "reachable" to the same meaning in both Measurement 2 and Measurement 3. Re-verified: 0 clip-to-k-box failures across both delay settings.
3. **The first version of the "gate boundary flip" test was too coarse in two ways**, both caught before trusting the reported `explained_by_gate_boundary` values: (a) it treated any differing move name (e.g. `RIGHT`) as gate-relevant regardless of *which location* it was taken from, incorrectly attributing a switch-visit-order flip at `(0,0)` to "gate1's window" purely because `s1`'s numeric value happened to fall inside `[4,6]` -- fixed to check the move's actual target *from that specific location*; (b) it tested raw current-value window membership rather than reachability via any allowed delay, producing false negatives for states that could still legally enter a gate by choosing a larger delay -- fixed to use `peek()`-based reachability across `instance.allowed_delays` directly, the same test the DP itself effectively performs.
4. No `pyudbm` rejection, no other unexpectedly-empty zone, and no deadlock occurred anywhere in the final (post-fix) runs. Deadlock is additionally structurally impossible by construction here (every non-goal cell has at least one neighbour that is a plain cell or a switch, never gate-only), asserted at runtime in `dp.py`'s value computation, not just claimed.
5. The task's Measurement 1 hand-designed inversion example turned out to be a **within-zone** (not cross-zone) phenomenon once actually built (see Measurement 3) -- both flagged and reported as its own, additional statistic rather than silently reclassified or ignored, since it is arguably the stronger finding, not a weaker or off-target one.

## Files

`gridtimed/{instance,env,dp,sanity_gate,zone_ops,zone_graph,geometry,decision_relevance}.py`. Entry points: `python -m gridtimed.sanity_gate`, `python -m gridtimed.geometry`, `python -m gridtimed.decision_relevance`. Result JSON: `/tmp/gridgate_results/{sanity_gate,geometry,decision_relevance}.json`. `zoneviews/`, `pyudbm/`, `prob_jobshop_ext/`, `deadline_track/`, `gate_track/` were not modified (verified: 0 files in those directories newer than the start of this work; `gridtimed/` reuses `zoneviews.zone.Zone` and `pyudbm` directly, and does not import from `supplementary/` or `Timed-RM/`).
