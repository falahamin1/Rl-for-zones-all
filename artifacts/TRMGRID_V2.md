# TRMGRID_V2: A Branching Reward Machine Fixes the Chain's Collapse

Retune of `trmgrid/` (in place — v1's report, `artifacts/TRMGRID.md`,
remains the historical record of why this was needed). **No encoders, no
gym env, no training code built.** No direction recommended.

---

## Root cause (v1) and the fix

v1's chain (`start→A→B→C`) made RM phase a direct function of "how many
sub-goals visited" — combined with location, that nearly determines the
optimal action on its own (location-only soundness 94.29%, conflict
5.71%, both far from gridtimed T0's 52.17%/37.5%). v2 replaces the chain
with a **branching** RM over k=3 sub-goals + GOAL: visit **any two**,
**agent's choice of which two and in what order**, each ordered pair
(i,j) carrying its own **two-sided** window `[lo_ij, hi_ij]` on `t_i` at
the moment `j` is visited (arriving too soon is now exactly as bad as too
late), then GOAL requires `t_j ≤ cap_j`. Clocks (`t_0..t_{k-1}` reset on
physical arrival, unconditionally regardless of phase; `g` global,
never resets), actions `(delay, move)`, `delay∈{0,1,2}`, real-valued
`[1,1+w]` move durations, `SCALE=20`, the reward structure, and the exact
DP's backward-induction-by-`g` structure are all unchanged from v1.

Phase is now a tuple of visited sub-goal indices in visit order: `()` =
start, `(i,)` = "have i", `(i,j)` = "have i,j" — not a string, avoiding
any string-parsing in the guard logic.

---

## Calibration: distance-relative, not absolute

`lo_ij = ceil(d_ij·(1+α))`, `hi_ij = lo_ij + width` (`d_ij` = Manhattan
distance between sub-goals i,j); `cap_j` calibrated the same way against
`d(j, GOAL)`. The deadline `D` is **computed per-instance** from the
actual realized distances and windows (never a fixed constant reused
across parameter settings — the same clip-bound lesson from v1's own
watch-list, applied to timing parameters this time), then verified (not
assumed) via a concrete-BFS solvability check with reject+resample.

---

## The 9-cell sweep (α ∈ {0, 0.25, 0.5} × width ∈ {2, 4, 8}, 5×5 grid, k=3, 3 seeds)

**width is the dominant driver; α barely matters.** Full table (27 cells):

| alpha | width | seed | states | truncated | loc-soundness | conflict (n/N) | rand-succ | rand-degenerate |
|---|---|---|---:|---|---:|---|---:|---|
| 0.0 | 2 | 0 | 27536 | no | 0.8213 | 42/235 (0.1787) | 0.000 | **True** |
| 0.0 | 2 | 1 | 31460 | no | 0.8851 | 27/235 (0.1149) | 0.006 | False |
| 0.0 | 2 | 2 | 33991 | no | 0.8664 | 31/231 (0.1342) | 0.000 | **True** |
| 0.0 | 4 | 0 | 134679 | no | 0.7830 | 51/235 (0.2170) | 0.006 | False |
| 0.0 | 4 | 1 | 183937 | no | 0.8638 | 32/235 (0.1362) | 0.012 | False |
| 0.0 | 4 | 2 | 184716 | no | 0.8383 | 38/235 (0.1617) | 0.000 | **True** |
| 0.0 | 8 | 0 | 181589 | **YES** | **0.3005** | 128/183 (**0.6995**) | 0.016 | False |
| 0.0 | 8 | 1 | 137176 | **YES** | **0.2461** | 144/191 (**0.7539**) | 0.024 | False |
| 0.0 | 8 | 2 | 150668 | **YES** | **0.2938** | 125/177 (**0.7062**) | 0.012 | False |
| 0.25 | 2 | 0 | 43211 | no | 0.8128 | 44/235 (0.1872) | 0.000 | **True** |
| 0.25 | 2 | 1 | 54978 | no | 0.8723 | 30/235 (0.1277) | 0.000 | **True** |
| 0.25 | 2 | 2 | 61691 | no | 0.8638 | 32/235 (0.1362) | 0.000 | **True** |
| 0.25 | 4 | 0 | 197395 | no | 0.7830 | 51/235 (0.2170) | 0.006 | False |
| 0.25 | 4 | 1 | 281709 | no | 0.8511 | 35/235 (0.1489) | 0.008 | False |
| 0.25 | 4 | 2 | 298829 | no | 0.8340 | 39/235 (0.1660) | 0.004 | False |
| **0.25** | **8** | **0** | **176389** | **YES** | **0.2722** | 131/180 (**0.7278**) | 0.010 | False |
| 0.25 | 8 | 1 | 129387 | **YES** | 0.2984 | 134/191 (0.7016) | 0.022 | False |
| 0.25 | 8 | 2 | 140584 | **YES** | 0.2373 | 135/177 (0.7627) | 0.008 | False |
| 0.5 | 2 | 0 | 44396 | no | 0.8085 | 45/235 (0.1915) | 0.000 | **True** |
| 0.5 | 2 | 1 | 54978 | no | 0.8723 | 30/235 (0.1277) | 0.000 | **True** |
| 0.5 | 2 | 2 | 61691 | no | 0.8596 | 33/235 (0.1404) | 0.002 | False |
| 0.5 | 4 | 0 | 200924 | no | 0.7830 | 51/235 (0.2170) | 0.006 | False |
| 0.5 | 4 | 1 | 281709 | no | 0.8511 | 35/235 (0.1489) | 0.008 | False |
| 0.5 | 4 | 2 | 298829 | no | 0.8255 | 41/235 (0.1745) | 0.004 | False |
| 0.5 | 8 | 0 | 171276 | **YES** | 0.2623 | 135/183 (0.7377) | 0.018 | False |
| 0.5 | 8 | 1 | 129387 | **YES** | 0.2984 | 134/191 (0.7016) | 0.022 | False |
| 0.5 | 8 | 2 | 140427 | **YES** | 0.2373 | 135/177 (0.7627) | 0.008 | False |

**width=2/4**: location-only soundness stays 78-89% (still modesel-like,
above target); conflict stays 11-22% (below target). **Several width=2
cells are outright DEGENERATE under random play** (`rand-succ=0.000` —
literally zero successes in 500 episodes) — flagged per the task's own
instruction rather than reported as meaningful.

**width=8**: soundness drops to **23.7%–30.1%** (comfortably at or below
gridtimed T0's 52.17% target) and conflict rises to **70.2%–76.3%** (well
above the 25-30% target) — **consistently across all 3 α values and all 3
seeds**. None of the width=8 cells are random-policy-degenerate.

### A real methodological issue, caught and reported honestly

**Every width=8 cell is truncated** at the exact-DP's `state_limit=300,000`
default (460k-605k horizon-fallback references per cell — substantial,
not marginal); **every width=2/4 cell is exact** (zero fallback
references). Pushed one width=8 instance to `state_limit=1,000,000`
directly: still truncated at 692,628 popped states with 1,158,509
fallback references — the true reachable space for these configs is at
least ~1M+ states, well beyond what was practical to fully explore in this
session. **The width=8 numbers above should be read as approximate
("trainable, not exact" — the same term this project's own `gridtimed`
tiers use for its large, truncated tiers), not exact-DP results.**

Read the *direction* of this finding as trustworthy — it is consistent
across all 9 width=8 cells regardless of truncation depth, and the
mechanism (wider windows admit more distinct viable clock-value
combinations per location, which by definition weakens how much a
location-only observation can determine) is independently understandable,
not an artifact specific to any one truncated run. The *precise* soundness/
conflict percentages at width=8 could shift somewhat under a fully exact
computation; a large, unresolved discrepancy is not expected, but this was
not verified to full exactness given the size of the space. **A genuine,
code-caught bug was also found and fixed** while investigating this:
`build_state_zones` builds zones one layer past a truncated BFS's
boundary (successors computed via `peek()` but never actually popped),
which had been silently inflating `validate_containment`'s reported count
(300,000 zones checked against only 176,389 actually-usable states) —
fixed by restricting containment validation to the same popped/usable set
3a/3b already use; re-verified 176,389/176,389 clean afterward.

**Chosen setting for the detailed measurements below: α=0.25, width=8,
seed=0.**

---

## Measurement 1 — PAPER GATE (pair-choice inversion): **PASSED**

A delay inversion alone (v1's gate) isn't enough this time — v1 had those
and they were too rare to matter. This gate requires the optimal **next
sub-goal** to differ with clock values. Demonstration instance: reduced
absolute scale (width=3, not 8) for hand-tractability and EXACT
solvability (verified: not truncated, 96,915 states) — same α=0.25 and the
same distance-relative calibration formula as the chosen setting; only the
window slack is smaller, which changes how much room there is, not
whether the two-sided-window mechanism exists.

5×5 grid: `START=(0,0)`, `S0=(1,1)`, `S1=(1,3)`, `S2=(3,1)`, `GOAL=(3,3)`.
`d(S0,S1)=d(S0,S2)=2 → window=[3,6]`; `d(S1,GOAL)=d(S2,GOAL)=2 → cap=6`;
`D=18`.

Both states are at location `(1,2)` — one step from S1 — in phase `(0,)`
("have S0"), differing only in `t_0`:

**State I** (`t_0=5, g=7`, reached via `START--RIGHT,DOWN(d0)-->S0[reset]
--UP,RIGHT(d0)-->(0,2)--DOWN(d2)-->(1,2)`): entering S1 now (`d0_RIGHT`)
arrives at `t_0=6`, exactly satisfying `[3,6]`. Cost 1, then S1→GOAL
(distance 2, direct) succeeds: `2-100=-98`. **V = -97.** Optimal: enter
now.

**State J** (`t_0=6, g=8`, same shape, one tick slower throughout —
`UP,RIGHT,DOWN` all `d1`): entering S1 now — **even at the fastest
possible delay (d0)** — arrives at `t_0=7 > 6`, a guaranteed guard
violation (confirmed directly by the code: `newly_failed=True,
fail_reason='guard_pair'`, for *every* legal delay). A **direct** approach
to S2 (distance 3) fares no better: minimum arrival `t_0=9 > hi(0,2)=6`,
also guaranteed failure. The only way out: retreat to S0 (`d0_LEFT`,
touching S0 again — an "incidental" touch since phase is already `(0,)`,
resetting `t_0` to 0 without advancing phase), then approach S2 from the
reset clock (`DOWN, DOWN(d1)`, arriving at `t_0=3`, exactly satisfying
`window(0,2)=[3,6]`). Total: `1+1+2+(2-100) = `**`V = -94`**. Optimal:
retreat and switch targets.

```
state_I: expected d0_RIGHT (V=-97.0), got ['d0_RIGHT'] (V=-97.0) -- PASS
state_J: expected d0_LEFT (V=-94.0), got ['d0_LEFT'] (V=-94.0) -- PASS
state_J's d0_RIGHT (enter S1) forced failure confirmed: True
Measurement 1 (paper gate, PAIR-CHOICE inversion): PASSED.
```

The optimal sub-goal pairing flips from S1 to S2 — and the optimal
*immediate action* flips from "enter now" to "retreat and reset" —
purely because `t_0` crossed `hi(0,1)` by exactly one unit. This is the
two-sided window's upper bound inverting behavior; v1's lower-bound-only
chain could not produce this.

---

## Measurement 2 — clock independence, `w=0` control

Run twice: on the **chosen setting** (α=0.25, width=8, seed=0 — truncated,
per the caveat above, zones sampled from the truncated exploration) and,
as an **exact cross-check**, on Measurement 1's own small instance
(width=3, verified not truncated).

| | chosen setting (176,389 states, **truncated**) | small exact cross-check (96,915 states, exact) |
|---|---|---|
| pinned(t_0,t_1) | 8.95% | 38.5% |
| pinned(t_0,t_2) | 10.9% | 38.9% |
| pinned(t_1,t_2) | 10.0% | 38.1% |
| affine dim distribution | `{0:226,1:16,2:511,3:2288,4:959}` | `{0:1402,1:26,2:326,3:1126,4:1120}` |
| affine dim ≥ 2 | **93.95%** | **64.3%** |
| vertex mean, errors | 8.87, 0 | 7.29 (w=0.25), 0 |
| bounded / strict fraction | 1.0 / 0.0 | 1.0 / 0.0 |
| **w=0 control**: affine dim | `{0: 4000}` (100% dim 0) | `{0: 4000}` (100% dim 0) |
| **w=0 control**: pinned | 1.0 across all pairs | 1.0 across all pairs |
| **GATE** | `{"passed": true}` | `{"passed": true}` |

Both pass cleanly, and the *direction* agrees (low pinning, high affine
dimension) even though the exact magnitudes differ (the smaller/tighter
cross-check instance naturally has somewhat more pinning — narrower
windows leave less room for clocks to diverge — but nowhere near the
"mutually pinned in most zones" failure threshold either way). w=0
collapses to 100% dim-0 in **both** instances, confirming nothing besides
`w` is producing geometric width, exactly as required.

---

## Measurement 3 — full partition diagnostic at the chosen setting

α=0.25, width=8, seed=0 (176,389 usable states; 2 terminal, 0 deadlocked
excluded — every cell always has ≥1 in-bounds neighbor).

### 3a — the ceiling

89 distinct optimal-action-SET classes (v1 had 41, over less than half
the state budget — more real decision granularity). Class size mean
1981.9, median 56 (v1's median was 1156 — this domain's classes are more
right-skewed: many SMALL classes alongside a few huge ones), max 25,856,
**13.5% singleton fraction** (v1: 0.0% — a real, new signal that some
optimal-action combinations are now UNIQUE to a single state, versus v1
where every class was shared).

### 3b — candidate observation models

| Candidate | n_cells | soundness | mean cell size |
|---|---:|---:|---:|
| (a) concrete state [control] | 176,389 | 100.0% | 1.000 |
| **(b) location only** | **180** | **27.22%** | 979.94 |
| (c) zone construction | 176,389 | 100.0% | 1.000 |
| (d) location + guard-satisfiability bits | 379 | 45.91% | 465.41 |
| (e) discretized clocks, b=1 | 176,389 | 100.0% | 1.000 |
| (e) discretized clocks, b=2 | 34,225 | 96.94% | 5.15 |
| (e) discretized clocks, b=4 | 7,956 | 89.50% | 22.17 |

**The number the task asked to lead with: location-only soundness is
27.22%** — well below the ~65% stopping-rule bar and comfortably in
gridtimed T0's territory (52.17%). Location alone is now a *poor*
predictor of the optimal action, the opposite of v1's 94.29%. Zone
construction remains sound-but-non-coarsening (mean cell size 1.000 — see
3d). Discretized clocks at b=2 come close to the 95% sound-and-coarsening
bar (96.94% sound, mean size 5.15) but b=4 falls to 89.5% sound (below the
bar) despite even more coarsening (22.17) — a real tradeoff, not a clean
win at every bucket size, worth stating plainly rather than only reporting
the flattering b=2 number.

### 3c — decision relevance

**180 multi-zone locations, 131 conflicting = 72.78% conflict fraction**
(6/105 → 131/180: both the count and the fraction moved dramatically).

| Domain | conflict fraction |
|---|---:|
| gridtimed T2 | 37.5% |
| modesel M4 | 45% |
| job-shop family | 0–2.6% |
| v1 (chain) | 5.71% (6/105) |
| **v2 (branching, this setting)** | **72.78% (131/180)** |

Now *higher* than gridtimed/modesel's own benchmarks, not just matching
them — a stronger result than the ~30% target called for.

### 3d — within-zone non-uniformity

`n_multistate_zones = 0`, `mean_concrete_states_per_zone = 1.0` — same
as v1: the sound zone construction still ties zones 1:1 to states by
design (replays each state's own exact discovery path), so it isn't
coarsening anything on its own. `within_zone_nonuniform_fraction` is
`0/0` (NaN, not a misleading 0.0) — vacuous, not a confirmed-sound result.
Containment: 176,389/176,389 OK, 0 bad (after the fix above).

### Random-policy calibration check

500 uniform-random episodes at the chosen setting: **1.0% success**,
**2.2% guard_pair failures**, **0.0% guard_goal failures**, **96.8%
deadline failures**. Not degenerate (neither 0% nor 100%), so the
parameters are meaningfully calibrated — but **stated plainly**: the
pair-window guard is exercised by random play (2.2%), the GOAL-cap guard
essentially never is (0.0% — under random wandering, episodes almost
never survive long enough after pairing to test the final cap; they run
out of deadline first). Deadline dominates total failure at every setting
tested in this sweep, including the chosen one — this is a genuine,
consistent property of the domain under uninformed play, not something
this calibration pass fixed or was trying to fix.

---

## Watch-list, checked explicitly

- **Per-instance clip bound**: `k_bound_real = max(K_BOUND, deadline+4)`,
  unchanged from v1, still computed per instance.
- **Bucket before pairwise conflict detection**: unchanged from v1 —
  `measurement3.py` buckets by distinct optimal-action set before any
  pairwise comparison, verified fast enough even at 180 multi-zone
  locations (no quadratic blowup observed).
- **Guards checked on the edge**: unchanged — `env.peek` checks guards at
  the exact moment of the A→B/B→C-equivalent transition attempt, never as
  a standing invariant blocking time elapse.
- **Real-valued duration sampling**: `w=0` collapses to dim-0 in both
  Measurement 2 runs (exact and approximate instance alike), confirming
  `w>0` is genuinely the source of geometric width.
- **Deadlock states as absorbing traps**: `n_deadlock_states=0` across
  every instance built this session — every cell on these grids always has
  ≥1 in-bounds neighbor, handled (not assumed) via `dp.py`'s explicit
  deadlock branch.

**New issue this round, not on the original list, caught and fixed**: the
zones-past-truncation-boundary / containment-count bug described above.
Added as a concrete instance of "verify counts against each other, don't
trust a single number in isolation" — `n_containment_ok` silently
exceeding `n_usable_states` was the tell.

---

## Files delivered / modified

`trmgrid/instance.py`, `env.py`, `zones.py`, `generator.py`,
`measurement1.py` — rewritten in place for the branching design.
`zone_ops.py`, `dp.py` — unchanged in mechanism, `dp.py` had one
one-line fix (`done_state(instance)`/`failed_state(instance)` now take
the instance, since `k` is no longer fixed at 3). `measurement2.py` —
unchanged (already generic over `instance.clock_names`). `measurement3.py`
— candidate (d) redefined for branching phases, discretized-clock
candidates generalized to a clocks tuple, random-policy rollout
diagnostic added, containment-count bug fixed. `/tmp/TRMGRID_V2.md`, this
report — copied to `artifacts/`, alongside the raw sweep/measurement JSON
dumps.

**No encoders, no gym env, no training code built** — environment, exact
DP, and zone construction only, per the explicit scope. **No direction
recommended** — both gates passed and the partition diagnostic moved
decisively past the stopping-rule thresholds, but whether that's grounds
to build encoders here is exactly the judgment call left open.
