# TRMGRID: A Fresh Timed-Reward-Machine Grid Benchmark — Design Gate + Partition Diagnostic

New package `trmgrid/`, built fresh: reuses only `zoneviews/` and `pyudbm`.
Does not import, fork, or modify `gridtimed`, `modesel`,
`supplementary/frozen_lake`, `supplementary/taxi`, or `Timed-RM` — those
were **read**, not imported, purely as a reference for already-proven DBM-
operation patterns (the edge-clock uncertain-elapse technique, the
deterministic-concrete-DP-plus-symbolic-zone split, backward induction by
global-clock descending order). No encoders, no PPO, no training were
built — only environment, exact DP, and zone construction, per the
explicit scope. **No direction recommended.**

---

## The model

An NxN grid, three sub-goal cells A, B, C. TRM: `start —[visit A, no
constraint]→ gotA —[visit B, requires t_A ≥ t_min]→ gotB —[visit C,
requires t_B ≤ t_max]→ done`, plus a global invariant `g ≤ D` throughout.

**Clocks**: `t_A`, `t_B`, `t_C` each reset to 0 the instant the agent is
*physically at* that sub-goal's cell — unconditionally, regardless of
current RM phase (touching B while still in phase `start`, say, resets
`t_B` as a harmless side effect but doesn't advance the RM or check any
guard). `g` is global and never resets, asserted in `reset_clocks` the
same way `gridtimed`/`modesel` assert their own non-resetting clocks.
Because the agent chooses *when* to reach each sub-goal, these three
clocks reset at genuinely independent times — this is the property
Measurement 2 exists to check, not assume.

**Actions**: `(delay, move)`, `delay ∈ {0,1,2}`, `move ∈
{UP,DOWN,LEFT,RIGHT}` — fixed Discrete space, explicit boolean mask
(illegal only for stepping off the grid; guard violations are NOT
blocked/illegal — see below). A move's true duration is uniform on
`[1, 1+w]`; the concrete exact-DP transitions deterministically on
`elapsed = delay + 1` (the interval's minimum — mirrors `gridtimed`'s own
documented convention, read as a reference), and `w` enters *only* through
the zone construction's uncertain-elapse mechanism (`zone_ops.edge_delay`)
— this is deliberate, not an oversight: the DP needs a well-defined
deterministic transition function to solve exactly, while `w > 0` is what
gives the *symbolic* zones genuine, non-degenerate geometric width.
`SCALE = 20` at the pyudbm boundary.

**A deliberate departure from `gridtimed`'s own convention, on the task's
explicit instruction**: `gridtimed`'s gate-window violations are *illegal*
moves (blocked, self-loop, small penalty, no time elapses). Here, a TRM
guard violation (`t_A < t_min` at the A→B attempt, `t_B > t_max` at the
B→C attempt) is a **terminal failure with a penalty** — the move happens,
time elapses, and the automaton independently judges the task failed. This
is closer to real RM semantics (a labeling function + automaton
transition, not a physically-blocked action) and is what makes the t_min
lower bound actually capable of *inverting* the optimal action rather than
merely pruning it.

**Reward**: `TIME_COST=1.0` per elapsed unit, `REWARD_SUCCESS=100.0` on
completing C within all constraints, `FAIL_PENALTY=50.0` on guard
violation or deadline expiry, `ILLEGAL_PENALTY=2.0` for stepping off the
grid (no time elapses).

---

## Measurement 1 — PAPER GATE: **PASSED**

Hand-built instance: 4×4 grid, `START=(0,0)`, `A=(0,2)`, `B=(3,2)`,
`C=(3,3)`, `t_min=4`, `t_max=30` (generous — not this gate's focus),
`D=20` (generous). Two states at the **identical location** `(2,2)`,
phase `gotA`, reached via the same fixed 4-move path
(`RIGHT,RIGHT[→A, t_A:=0],DOWN,DOWN`) with two different uniform delay
profiles on the 2 post-A moves:

- **Zone P** (both post-A delays = 0): `t_A=2` at `(2,2)`. Direct entry to
  B with `delay=0` gives arrival `t_A=3 < t_min=4` — **guard violation**
  (cost `1+50=51`). `delay=1` gives arrival `t_A=4 ≥ 4` — succeeds, cost
  `2 + (1·1-100) = -97`. `delay=2` also succeeds but costs more (`-96`).
  **By hand: optimal = `d1_DOWN`, V = -97.**
- **Zone Q** (both post-A delays = 2): `t_A=6` at `(2,2)`. `delay=0` gives
  arrival `t_A=7 ≥ 4` — already safe, cheapest. **By hand: optimal =
  `d0_DOWN`, V = -98.**

Written into `trmgrid/measurement1.py` *before* running the DP. Result:

```
zone_P: expected d1_DOWN (V=-97.0), got ['d1_DOWN'] (V=-97.0) -- PASS
zone_Q: expected d0_DOWN (V=-98.0), got ['d0_DOWN'] (V=-98.0) -- PASS
Measurement 1 (paper gate): PASSED.
```

Independent cross-check (not part of the formal gate, but reassuring):
the DP's own root value (`V=-93.0`) matches a by-hand-derived globally
optimal path (`RIGHT,RIGHT→A, DOWN,DOWN,DOWN(d1)→B at t_A=4 exactly,
RIGHT→C`; total elapsed 7, cost `7-100=-93`) computed independently of
the P/Q states above.

The optimal *delay* flips (1 → 0) purely because of how much `t_A` has
already accumulated relative to `t_min`, at the identical grid location —
a genuine inversion driven by the lower bound, not a pruning effect.

---

## Measurement 2 — CLOCK INDEPENDENCE: **GATE PASSED**

Run on both the hand-built 4×4 instance (exhaustive BFS, 66,935 states,
66,933 zones, no truncation) and a generated 6×6 instance (`t_min=6,
t_max=18, D=24`, seed 0, solvable on the first placement attempt; 90,864
states, 90,862 zones, no truncation; Measurement 2's zone sample capped at
4,000 of 90,862 — reported as sampled, not silently exhaustive).

### w=0 control (both instances)

| Instance | affine_dim distribution | pinned (all subgoal pairs) |
|---|---|---|
| hand-built 4×4 | `{0: 4000}` — **100% dim 0** | 1.0 |
| generated 6×6 | `{0: 4000}` — **100% dim 0** | (not separately re-tabulated; dim-0 collapse alone is the control's pass condition) |

Collapses to dim 0 everywhere, as required — nothing besides `w` is
producing geometric width.

### w=0.25 (the real operating point)

| Metric | hand-built 4×4 (exhaustive) | generated 6×6 (capped sample, n=4000/90862) |
|---|---:|---:|
| pinned(t_A,t_B) | 27.1% | 31.3% |
| pinned(t_A,t_C) | 27.6% | 28.7% |
| pinned(t_B,t_C) | 28.8% | 28.0% |
| pinned(g,t_A) | 38.8% | — |
| pinned(g,t_B) | 37.2% | — |
| pinned(g,t_C) | 44.4% | — |
| affine_dim distribution | `{0:1028, 1:14, 2:254, 3:1301, 4:1403}` | `{0:996, 1:32, 2:457, 3:1536, 4:979}` |
| affine dim ≥ 2 fraction | **73.95%** | **74.3%** |
| affine dim 0-or-1 fraction | 26.05% | 25.7% |
| facets mean (min–max) | 11.17 | 10.99 (9–14) |
| vertices mean (min–max), n computed | 8.69, n=400, 0 errors | 7.64, n=4000 cap→400 sampled for vertices, min1–max21, 0 errors |
| bounded fraction | 1.0 | 1.0 |
| strict fraction | 0.0 | 0.0 |

**GATE**: `{"passed": true, "reasons": []}` on both instances. Sub-goal
clock pairs are mutually pinned in only ~27-31% of zones (well under the
"most zones" failure threshold), and affine dimension is dominated by dim
3-4 (67.6% combined on the small instance, well above the 0-1
concentration failure threshold). **This is the opposite of the Taxi TRM
failure mode** — the design does not collapse to line segments.

One genuine, understood (not concerning) oddity: pinning fractions and the
full affine-dimension histogram came out **numerically identical** between
`w=0.25` and `w=0.5` on the hand-built instance (facet/vertex means differ
slightly, dimension/pinning do not). This is mathematically expected, not
a bug: pinning and affine dimension are *combinatorial* properties (which
pairs of clocks happen to have reset at different times along a given
discovered path) that don't depend on the *magnitude* of `w` as long as
`w > 0` — only the w=0 vs w>0 boundary is a genuine phase transition.
Facet/vertex counts, which depend on exact bound magnitudes, do shift
slightly with `w`.

No pyudbm rejections, no empty zones (`Z2.is_empty()` assertion in
`zones.py` never triggered), no vertex-counter disagreements
(`n_vertex_errors=0` throughout), 100% zone-containment validity on both
instances (`validate_containment`: 66,933/66,933 and 90,862/90,862 OK).

---

## Measurement 3 — PARTITION DIAGNOSTIC

Gate passed, so this is meaningful. Run on the generated 6×6 instance
(90,864 states; 2 terminal excluded, 0 deadlocked excluded — every cell
always has at least one in-bounds neighbor with 4 full-range moves, so
deadlock is structurally impossible here, confirmed empirically, not just
assumed: `n_deadlock_states=0`).

### 3a — the ceiling

90,862 usable states, 36 locations, **41 distinct optimal-action-SET
classes**. Class size mean=2216.1, median=1156, max=17,043,
**singleton fraction = 0.0%** — no optimal-action-set class has only one
member; every class is shared across many states.

### 3b — candidate observation models

| Candidate | n_cells | soundness | mean cell size | singleton frac |
|---|---:|---:|---:|---:|
| (a) concrete state [control] | 90,862 | 100.0% | 1.000 | 100.0% |
| (b) **location only** (pos + RM phase) | 105 | **94.29%** | 865.35 | 0.0% |
| (c) zone construction | 90,862 | 100.0% | 1.000 | 100.0% |
| (d) location + guard-satisfiability bit | 128 | 91.41% | 709.86 | 0.0% |
| (e) discretized clocks, b=1 | 90,862 | 100.0% | 1.000 | 100.0% |
| (e) discretized clocks, b=2 | 23,290 | **99.61%** | **3.90** | 14.73% |
| (e) discretized clocks, b=4 | 6,778 | **99.04%** | **13.41** | 7.91% |

**Sound (≥95%) AND coarsening (mean cell size ≥3)**: discretized clocks at
`b=2` and `b=4`, both cleanly. **Zone construction is sound but does not
coarsen at all** (mean cell size 1.000 — see 3d). **Location-only just
misses the 95% soundness bar (94.29%)** despite being massively
coarsening (mean cell size 865) — a large fraction of its mass is sound,
but not enough to call it a reliable abstraction on its own.

**The number the task asked to lead with**: location-only soundness here
is **94.29%**, dramatically higher than modesel's reported 60.5% (the
reason symbolic encodings gave no advantage there) — trmgrid's location
alone comes much closer to determining the optimal action than modesel's
did. This is a cautionary signal in the same direction as modesel's
finding, not the opposite: location-only is *closer to sufficient* here
than it was on gridtimed, even though it technically falls short of the
95% bar. Reported plainly, not softened.

### 3c — decision relevance

**105 multi-zone locations** (matches candidate (b)'s cell count, as
expected — same location/phase grouping), **6 conflicting** (disjoint
optimal-action sets across zones at the same location) = **conflict
fraction 5.71%**.

| Domain | conflict fraction |
|---|---:|
| gridtimed T2 | 37.5% |
| modesel M4 | 45% |
| job-shop family | 0–2.6% |
| **trmgrid (this benchmark)** | **5.71%** (6/105) |

Much closer to the job-shop family's near-zero end than to
gridtimed/modesel's high-conflict end, though not as low as job-shop's
ceiling. **Statistical power note, as requested**: 105 multi-zone
locations is a real but modest sample — 6 conflicts is a small numerator,
so this fraction should be read with the denominator in mind, not just
the percentage.

### 3d — within-zone non-uniformity

**`n_multistate_zones = 0`** — with `mean_concrete_states_per_zone = 1.0`,
there are, exactly as observed in Measurement 2's zone-count numbers
(90,862 zones for 90,862 usable states), **zero zones with more than one
concrete member state to even check**. `within_zone_nonuniform_fraction`
is therefore `0/0` (reported as NaN, not a misleading `0.0`) — this is
**not** a confirmed-sound result, it's a vacuous one: the sound zone
construction here ties zones 1:1 to states by design (replaying each
state's own exact discovery path, per `zones.py`'s docstring), so it
isn't abstracting anything at all yet — the identical pattern
`gridtimed`/`modesel` both show (read, not imported, as the point of
comparison). This is exactly why candidate (c) in 3b shows `mean_cell_size
= 1.000`: the per-state zone construction, as built, is a relabeling, not
a coarsening. Any actual coarsening in this domain comes from the
discretized-clock candidates in 3b, not from the zone construction itself.

Containment: 90,862/90,862 OK, 0 bad.

---

## Watch-list, checked explicitly

- **Clip bound scaled per instance**: `k_bound_real = max(K_BOUND,
  instance.deadline + 4)` in `measurement2.py`, never a bare global
  constant — verified by construction (same pattern `gridtimed/geometry.py`
  uses, read as reference) and by the fact neither instance (D=20 vs D=24)
  produced an empty-zone assertion.
- **Quadratic pairwise conflict detection**: avoided — `measurement3.py`
  buckets by distinct optimal-action SET before any pairwise comparison
  (same fix `gridtimed/decision_relevance.py` already applied), not
  zone-pair-by-zone-pair.
- **Blanket guard invariant deadlocking time elapse**: not applicable to
  this design — guards are checked ON THE EDGE (at the exact moment of the
  A→B or B→C transition attempt), never as a standing state invariant
  that could block time from elapsing before an edge happens. Checked, not
  found.
- **Integer duration sampling silently degenerating zones**: explicitly
  avoided and directly measured — `w=0` (the degenerate case) collapses to
  100% dim-0 as its OWN control, confirming the w=0.25 width (73.95%/74.3%
  dim≥2) is genuinely coming from the real-valued `[1,1+w]` elapse, not
  something else.
- **Generated layouts producing states with zero legal actions**: handled
  (`dp.py`'s deadlock branch, `n_deadlock_states` counted, never crashes)
  — measured `n_deadlock_states = 0` on both instances tested (every cell
  on these grids always has ≥1 in-bounds neighbor across 4 full-range
  moves; TRM guard failures never block a move, they terminate it, so
  there is no "gate-surrounded" deadlock analogue here at all).

**Reachability, finiteness, wall-clock**:

| | hand-built 4×4 | generated 6×6 |
|---|---:|---:|
| total reachable states | 66,935 | 90,864 |
| total zones (= non-terminal states) | 66,933 | 90,862 |
| DP truncated? | No | No |
| DP+zone-construction wall-clock | 7.37s | 3.69s |

The zone graph is **finite without any extrapolation or k-normalization**
— bounded structurally by the deadline `g ≤ D` (every clock either resets
or is capped by `D`, same finiteness argument `gridtimed`'s own DP relies
on, read as reference). `zone_ops.normalize`/`extrapolate_max_bounds` is
implemented (mirroring the reference pattern) but **never actually called**
in any measurement here — it exists for future use (e.g. an
approximate/truncated-tier DP on a much larger instance), not because
termination required it. The only "k" in use is `k_bound_real = max(20,
deadline+4)`, purely for `zoneviews.zone.Zone`'s own box-clip parameter
(an encoder-facing convenience for `affine_dim`/`v_rep`/`h_rep`), unrelated
to state-space termination.

No pyudbm rejections, no empty zones, no deadlocks, no vertex-counter
disagreements anywhere in this run.

---

## Files delivered

`trmgrid/`: `__init__.py`, `instance.py`, `env.py`, `dp.py`, `zone_ops.py`,
`zones.py`, `generator.py`, `measurement1.py`, `measurement2.py`,
`measurement3.py`. No `gym_env.py`, no encoders, no PPO — out of scope by
explicit instruction. `/tmp/TRMGRID.md`, this report, plus raw JSON dumps
at `/tmp/trmgrid_measurement2.json`, `/tmp/trmgrid_measurement2_large.json`,
`/tmp/trmgrid_measurement3.json` — copied to `artifacts/`.

**No direction recommended.** Both gates passed and the partition
diagnostic ran cleanly, but whether that's grounds to build encoders here
is exactly the judgment call left open.
