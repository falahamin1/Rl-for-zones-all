# READ-ONLY Audit: Timed-Automata-Based RL Environments in `RL-for-zones`

Scope: every timed-automata/DBM/zone-related RL environment found under `/Users/falah/Desktop/git-projects/RL-for-zones/`. Facts only, no recommendations. Produced by reading source directly and by running throwaway scripts (under `/tmp/env_audit_*/`) against read-only copies of the packages — nothing under the repo was modified. Anything not determinable without deeper work is marked UNKNOWN explicitly in its section.

---

## Step 0 — Inventory

Five distinct environment families were found, spanning two unrelated research-code lineages plus one standalone prototype:

| # | Family | Path(s) | One-line description | Provenance |
|---|---|---|---|---|
| 1 | **prob_jobshop_ext** (4 sub-envs) | `prob_jobshop_ext/` (≡ byte-identical `scheduling/prob_jobshop/`; older subset at `supplementary/jobshop/prob_jobshop/`) | Probabilistic job-shop scheduling (10 "PJS" + 10 "LJS" instances), real `pyudbm` DBM/Federation zones | Job-shop domain of a paper's "Zone-Based Symbolic RL" supplementary code (`supplementary/README.md`, `scheduling/README.md`) |
| 2 | **Frozen Lake TRM comparison** (3 variants: three-goal, single-goal, hole-disable) | `supplementary/frozen_lake/trm_comparison/` (≡ `scheduling/frozen_lake/` + `scheduling/trm_comparison/`) | 6×6 Frozen Lake with a Timed Reward Machine; hand-coded threshold "zone" vs capped-integer "region" Q-learning comparison | Frozen-lake domain of the same "Zone-Based Symbolic RL" paper package |
| 3 | **Taxi TRM comparison** | `supplementary/taxi/taxi_comparison/` (≡ `scheduling/taxi_comparison/`) | 5×5 stochastic Taxi grid with the same TRM/zone/region pattern as family 2 | Taxi domain of the same paper package |
| 4 | **Timed-RM** (Frozen Lake + Taxi, base + "corner_abstraction") | `Timed-RM/` (own git repo, ≡ `scheduling/Timed-RM/`) | `sympy`/`eval`-guard Timed Reward Machines over gymnasium Frozen Lake / Taxi; "digital" (discretized) vs "real" (classical Alur–Dill clock-region) modes | Separate, earlier paper: "Experimental evaluations of RL algorithms for Timed Reward Machines" (`Timed-RM/README.md`, first commit 2025-10-07) |
| 5 | **MarshlandSprintEnv** | `RL/marshland_example.py` (+ `RL/TabularQ.py`, `RL/simulation_marshland.py`) | Tiny 3×3-grid, single-clock `pyudbm` prototype | Standalone scratch/prototyping code, no README, not part of either paper package |

**Grouping notes**: Families 2 and 3 (Frozen Lake / Taxi TRM comparison) and family 1 (job-shop) share one methodology — "Zone RL" (constant-size hand-coded or DBM-based clock abstraction) vs "Region RL" (capped-integer clock abstraction) — documented jointly in `supplementary/README.md`. Family 4 ("Timed-RM") is architecturally unrelated despite the same "TRM" name: it uses Python-`eval` guard strings and either a discretized clock grid or a textbook Alur–Dill region construction, never a DBM/`Federation` object. Family 5 is an isolated prototype with no connection to either paper. Within family 1, four *sub-environments* share one instance/dynamics model but differ in state representation (`SymbolicJobShopEnv` = real zones, `RegionJobShopEnv` = capped ints, `PTADirectEnv` = raw floats/DDPG, `JobShopEnv` = unused `gymnasium.Env`). Within family 2, three *variants* share the grid/Q-learning scaffolding but differ in clock count and guard structure (three-goal: 2 clocks; single-goal: 1 clock; hole-disable: 3 clocks).

Methodology used throughout: affine dimension via union-find on `bound(i,j)+bound(j,i)==0` (both non-strict); facet redundancy via single-hop `bound(i,k)+bound(k,j)==bound(i,j)`; exact vertex enumeration via `fractions.Fraction` linear-system solving over subsets of non-redundant constraints (no `scipy.spatial.qhull`, per instruction — it has produced wrong answers on this class of input); aggregation ratio = distinct concrete keys ÷ distinct zone keys over 200 seeded random-policy episodes, checked against a location-only control; decision relevance via exact DP/value-iteration over the concrete state space (only where state count is tractable, roughly <10⁵) with a multi-zone-location optimal-action-conflict check.

---

# Environment 1: `prob_jobshop_ext` (Probabilistic Job-Shop Scheduling)

**Path**: `/Users/falah/Desktop/git-projects/RL-for-zones/prob_jobshop_ext/` (byte-identical to `scheduling/prob_jobshop/`).

**Description**: a probabilistic job-shop scheduling benchmark (10 small/medium "PJS" + 10 large "LJS" hand-built instances, stochastic integer task durations) exposed through four different state-representation wrappers around one shared discrete-event dynamics model, built on the `pyudbm` DBM library for the zone-based wrapper.

**What differs across the four sub-environments** (dynamics/instances/rewards are shared):

| Class | File | State representation |
|---|---|---|
| `SymbolicJobShopEnv` | symbolic_env.py | `(loc, Z)` — real pyudbm `Federation` (zone) |
| `RegionJobShopEnv` | region_env.py | `(loc, capped_clock_tuple)` — plain Python ints, no pyudbm |
| `PTADirectEnv` | pta_env.py | flat `np.ndarray`, raw uncapped real clock values (DDPG) |
| `JobShopEnv` | env.py | `gymnasium.Env` `Dict` obs with raw clocks + explicit `action_mask` — **dead code**, never instantiated anywhere |

## A — automaton structure

**Clocks — exactly `jobs + 1`.** `zone_ops.py:29`: `self._clock_names = [f"c_{j.job_id}" for j in instance.jobs] + ["delta"]`. One clock per job (reset each time a new task in that job starts) plus one global cost clock `delta`. Measured: PJS_01 (2 jobs) → 3 clocks; PJS_05 (4 jobs) → 5; LJS_10 (14 jobs, largest) → 15.

**CRITICAL — `delta` never resets.** `reset_clocks()` explicitly forbids it:
```python
def reset_clocks(Z, zctx, clock_names):
    """Reset named clocks to 0.  delta must never be in clock_names."""
    assert "delta" not in clock_names, "delta must never be reset"
```
(`zone_ops.py:70-80`). Every reset call site passes only `[f"c_{job_id}"]` (`symbolic_env.py:265`). Module docstring: *"Invariant never constrains delta ... delta stays unbounded above after up() ... min_cost(Z) = infimum of delta in Z = minimum reachable makespan"* (`zone_ops.py:6-11`). Confirmed empirically in §C: `frac_bounded == 0.0` for every sampled zone across all 5 tested instances.

**Guards/invariants — constants are the sampled task durations**, e.g. `{1,2,3,4,5}` (PJS_01), `{1,...,6}` (PJS_04). No `[low,high]` interval guards; each duration outcome `d` becomes a branch with invariant `c_J <= d` and completion guard `c_J >= d`, pinning `c_J == d`.

**Deadlines — none.** `grep -ri "deadline"` across every `.py` returns zero matches.

**Timing nondeterminism — fixed discrete draw, sampled once per dispatch** (`symbolic_env.py:89-90`):
```python
dist = self.instance.task_by_id(task_id).distribution
d = int(self._rng.choice(dist.durations, p=dist.probabilities))
```

**Probabilistic branching — present**, built per-action in `_compute_actions()` (`symbolic_env.py:249-268`); all branches propagated symbolically for reachability, the live env samples exactly one per `.step()`.

## B — action structure

Fundamentally "dispatch task X" (task-ID string); `PTADirectEnv` uses a continuous `(1+J)`-dim vector (wait duration + per-job priority); `JobShopEnv` (dead code) uses `Discrete(N)` + explicit mask.

**CRITICAL — genuine tradeoffs**: PJS_03's initial state has `J1_T1` (m1, durations `[1,4,8]`) and `J3_T1` (m1, `[2,4,7]`) both needing machine m1 — a real either/or resource conflict with different duration spreads; `J2_T1` (different machine) is independent. A second, sharper example appears in §E's worked conflict (a *third* task's remaining time flips the optimal dispatch).

Masking: variable-size/implicit for the three dispatch-based envs; `JobShopEnv` alone has a literal `action_mask` array (dead code).

## C — zone geometry (SymbolicJobShopEnv only; other 3 sub-envs carry no zone rep)

BFS-explored the real pyudbm zone graph for PJS_01–05 (cap 50,000), cross-validated against `OptimalSymbolicSolver._explore()` — identical counts for PJS_01-04. `Federation` always a single DBM (no union ever arises).

| Instance | n_clocks | BFS states | truncated | affine dim mean/min/max | frac bounded | frac strict | facets mean/min/max | vertices mean/min/max |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| PJS_01 | 3 | 110 | No | 2.85/1/3 | **0.000** | **0.000** | 4.01/0/5 | 2.11/0/4 |
| PJS_02 | 3 | 391 | No | 2.91/1/3 | **0.000** | **0.000** | 4.29/0/6 | 2.64/0/6 |
| PJS_03 | 4 | 4,470 | No | 3.85/1/4 | **0.000** | **0.000** | 5.49/0/8 | 3.53/0/12 |
| PJS_04 | 4 | 12,290 | No | 3.85/2/4 | **0.000** | **0.000** | 6.21/0/10 | 5.81/0/16 |
| PJS_05 | 5 | 50,000 | **Yes** | 4.62/2/5 | **0.000** | **0.000** | 7.47/0/13 | 8.11/0/29 |

`frac_bounded=0.000` everywhere is the direct geometric signature of `delta` never resetting (every zone unbounded in the `delta` direction). `frac_strict=0.000` everywhere (guards are non-strict). Vertex-zero fraction grows with instance size (13.6%→35.0%, PJS_01→05), mostly genuine vertex-free unbounded polyhedra.

## D — aggregation (SymbolicJobShopEnv, 200 seeded episodes, uniform-random)

| Instance | concrete keys | zone keys | **aggregation ratio** | locations | **location-only control ratio** | frac. zones w/ 1 concrete state |
|---|---:|---:|---:|---:|---:|---:|
| PJS_01 | 28 | 73 | **0.384** | 15 | **1.867** | 1.000 |
| PJS_03 | 388 | 649 | **0.598** | 67 | **5.791** | 1.000 |
| PJS_06 | 4,347 | 4,713 | **0.922** | 3,190 | **1.363** | 1.000 |

**The zone ratio does NOT exceed the location-only control ratio — it's below 1 in every instance (net disaggregation)**, while the location-only control always exceeds 1 (real aggregation). Every zone maps to exactly one concrete state; the zone key never groups distinct clock valuations here — it splits single physical states into multiple zone-string variants instead (traced partly, not fully, to `delta`'s non-resetting nature).

## E — decision relevance

Exhaustive (non-sampled) BFS per instance, exact backward-DP value iteration over concrete states.

| Instance | n_concrete (exact) | n_zone (exact) | zone/concrete ratio | qualifies (<10⁵) |
|---|---:|---:|---:|:---:|
| PJS_01 | 31 | 89 | 2.87 | Yes |
| PJS_02 | 105 | 257 | 2.45 | Yes |
| PJS_03 | 667 | 1,617 | 2.42 | Yes |
| PJS_04 | 1,605 | 4,470 | 2.78 | Yes |
| PJS_05 | 95,632 | 269,718 | 2.82 | Yes (borderline) |

| Instance | multi-state locations | conflicting locations | conflict fraction |
|---|---:|---:|---:|
| PJS_01 | 7 | 0 | 0.0% |
| PJS_02 | 17 | 0 | 0.0% |
| PJS_03 | 54 | 0 | 0.0% |
| PJS_04 | 141 | 2 | **1.4%** |
| PJS_05 | 1,538 | 40 | **2.6%** |

**Worked conflict example** (PJS_04, a location with `J1_T3` (m3, `[2,3,4]`) and `J3_T3` (m2, `[1,3,5]`) both enabled): optimal action depends on `c_J2` — the elapsed time on a *third* task's active job — non-monotonically:

| `c_J2` | optimal action |
|---:|---|
| 0 | `J1_T3` |
| **1** | **`J3_T3`** |
| 2 | `J1_T3` |
| 3 | `J1_T3` |

## F — scaling and engineering

- **Parameterized generation: none.** Every `benchmarks/pjs_0X.py`/`ljs_0X.py` is a standalone `get_instance()` with a literal `(machine, duration)` list; no `generate_instance(n_jobs,...)` exists.
- **Largest instance**: LJS_10 (14 jobs, 8 machines, 112 tasks). A fresh real-pyudbm rerun (`OptimalSymbolicSolver._explore(state_limit=200_000)`) gave **200,000 states, truncated, 210.2s**. (The repo's own `graph_sizes.py` uses a coarser, non-Federation key and disagrees numerically with the real BFS even on small instances — e.g. PJS_01: 28 vs. 110.)
- **`JobShopEnv` confirmed dead code** — defined/re-exported, never instantiated anywhere. `ddpg_env.py`'s `DDPGSchedulingEnv` likewise dead; `PTADirectEnv` is what's actually wired into DDPG training.
- **Picklability**: `Federation` pickle/deepcopy **fails** (`TypeError: cannot pickle '..._NativeFederation' object`); native `.copy()` works; `Context`/`Clock` pickle fine.
- **Wall-clock/step** (PJS_04): `SymbolicJobShopEnv.step()` alone 5.4×10⁻⁵s; full symbolic-RL per-decision cost (all branches) 2.2×10⁻⁴s; `RegionJobShopEnv.step()` 2.2×10⁻⁵s; `PTADirectEnv.step()` 3.0×10⁻⁵s.

## G — summary rows

See consolidated table at the end of this document.

---

# Environment 2: Frozen Lake TRM Comparison (3 variants)

**Path**: `/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/frozen_lake/` (≡ `scheduling/frozen_lake/` + `scheduling/trm_comparison/`). **Not DBM-based** — zones are hand-coded finite threshold tuples (`x_node`,`y_node`,`h_node` ∈ {0,1}), no `pyudbm`/`Federation` anywhere.

| Variant | Path | Description | Clocks | Guard | Docstring zone-node count |
|---|---|---|---|---|---|
| Three-goal (base) | `trm_comparison/{grid,trm,zone_agent,region_agent}.py` | 6×6 grid, visit A→B→C in order | 2 (x,y) | `x≥c_min ∧ y≤D` | 4 |
| Single-goal | `trm_comparison/sg_*.py` | 6×6 grid, single goal C in a time window | 1 (y) | `c_min≤y≤D` | 3 |
| Hole-disable | `trm_comparison/hole_*.py` | Same A→B→C guard + non-terminal holes disabling a direction for `x_h` steps | 3 (x,y,h) | `x≥c_min ∧ y≤D` (+h gates legality) | 8 (raw combinatorics; actual key is bigger — see §C) |

**Shared**: 6×6 grid mechanics (reflective boundaries, `slip_prob=0.2` 90°-slip), Q-learning setup, reward shape. **Provenance**: `supplementary/README.md` — "the code for all experiments reported in the paper" implementing Zone RL vs Region RL; the three run scripts map to the README's Task (a)/(b)/(c).

## A — automaton structure

All three variants: the never-resetting clock is **y** in every case (only `x` and `h` are ever reset, on goal-success and hole-hit respectively; `y` has no reset anywhere in `zone_agent.py`/`sg_zone_agent.py`/`hole_zone_agent.py`, confirmed by grep). Guard constants: three-goal `c_min∈{2,5,10,15,20,25}`, `D=200` fixed; single-goal `c_min=10` fixed, `D∈{30,60,100,200,400}`; hole `c_min=10,D=200` fixed, `x_h∈{5,10,20,30,50}`. Deadlines: yes in all three (`y<=D` or `c_min<=y<=D`), guard failure is terminal with `R_LATE=-200`. Timing nondeterminism: three-goal has an agent-controlled `delay∈{0..5}` (deterministic `dt=delay+1`, not sampled); single-goal/hole have no delay dimension at all (`dt` hard-coded 1). Genuine environment stochasticity is spatial only (`slip_prob=0.2` 90° slip), independent of the clock/timing mechanism.

## B — action structure

Three-goal: 24 actions `(delay,grid_action)`, fixed-size, unmasked; delay choice genuinely interacts with guard-threshold proximity per exact DP (TRM_01: `(delay=1,DOWN)` optimal at `x_cap=0` switches to `(delay=0,DOWN)` at `x_cap=1`). Single-goal: 4 actions (no delay), fixed-size, unmasked; direction choice alone shows sharp zone-dependent differences (§E). Hole-disable: 4 actions with **genuine runtime masking** — `HoleZoneEnv.valid_actions()` excludes the cooldown-disabled direction (only variant with masking).

## C — zone geometry

No DBM geometry applies (hand-coded label tuples, not convex regions). Independent BFS re-implementation (cap 50,000) **confirms the "constant regardless of parameters" claim exactly**, matching the package's own README numbers:

| Variant | Instance A | Zone BFS | Instance B | Zone BFS | Instance C | Zone BFS |
|---|---|---|---|---|---|---|
| Three-goal | TRM_01 (c_min=2) | 385 | TRM_03 (c_min=10) | 385 | TRM_06 (c_min=25) | 385 |
| Single-goal | SG_01 (D=30) | 100 | SG_03 (D=100) | 100 | SG_05 (D=400) | 100 |
| Hole | HOLE_01 (x_h=5) | 1905 | HOLE_03 (x_h=20) | 1905 | HOLE_05 (x_h=50) | 1905 |

**Discrepancy found**: hole variant's "8 nodes" is only the raw `(x_node,y_node,h_node)` combinatorics (2³). The actual Q-table key adds `disabled_action`, giving a real, BFS-verified zone-graph size of **1905**, not 8. What real `pyudbm` zones would require here: `Clock` objects x,y(,h); a `Federation` per `(pos,rm_state)`; DBM intersect/`is_empty()` for guard checks; `Federation.reset()` for the x/h resets; DBM normalization in place of `min(x,c_min)` region-capping.

## D — aggregation (200 seeded episodes, uniform-random)

| Variant | Instance | distinct concrete | distinct zone | agg. ratio | loc-only pairs | loc-only ratio | frac singleton zones |
|---|---|---:|---:|---:|---:|---:|---:|
| Three-goal | TRM_01 | 1605 | 83 | 19.34 | 56 | 28.66 | 0.193 |
| Three-goal | TRM_03 | 1518 | 103 | 14.74 | 50 | 30.36 | 0.272 |
| Three-goal | TRM_06 | 1543 | 110 | 14.03 | 55 | 28.05 | 0.255 |
| Single-goal | SG_01 | 981 | 86 | 11.41 | 31 | 31.65 | 0.058 |
| Single-goal | SG_03 | 845 | 57 | 14.82 | 30 | 28.17 | 0.105 |
| Single-goal | SG_05 | 722 | 53 | 13.62 | 30 | 24.07 | 0.038 |
| Hole | HOLE_01 | 61399 | 1002 | 61.28 | 93 | 660.20 | 0.115 |
| Hole | HOLE_03 | 58376 | 1018 | 57.34 | 93 | 627.70 | 0.076 |
| Hole | HOLE_05 | 61380 | 989 | 62.06 | 92 | 667.17 | 0.111 |

**Zone ratio does NOT exceed the location-only control ratio in any of the 9 rows** — location-only would unsoundly merge guard-satisfied and guard-violated states, so it "aggregates" harder but incorrectly; zone abstraction keeps more distinct buckets than that naive baseline while still compressing well below concrete count.

## E — decision relevance

State-space sizes (`36 pos × RM states × clock ranges`): three-goal only TRM_01 qualifies under 10⁵ (65,124; others 130k–564k); single-goal all 5 qualify (1,116–14,436); hole none qualify (smallest 1,432,728).

**Three-goal (TRM_01)**: 96 multi-zone locations, **0 conflicts** (c_min=2 too small to trigger a real x_node-boundary conflict at this scale).

**Single-goal**:

| Instance | multi-zone locations | conflicting locations | conflict fraction |
|---|---:|---:|---:|
| SG_01 (D=30) | 32 | 0 | 0.000 |
| SG_02 (D=60) | 32 | 2 | 0.062 |
| SG_03 (D=100) | 32 | 2 | 0.062 |
| SG_04 (D=200) | 32 | 2 | 0.062 |
| SG_05 (D=400) | 32 | 2 | 0.062 |

**Worked conflict example** (SG_02, pos=(1,2)): zone 0 (pre-window, y<10) optimal `{UP}`; zone 2 (post-window, y>60) optimal `{RIGHT}` — disjoint optimal-action sets, a genuine conflict driven by whether the deadline is still reachable.

**Hole-disable**: no instance qualifies — reported as non-qualifying by exact size computation, not approximated.

## F — scaling and engineering

- Instance generation: hardcoded dict literals in all three (6/5/5 entries); no generator function.
- Region BFS independently re-run and matched exactly: TRM_06 → 212,102; SG_05 → 12,675 (both match README). HOLE_01's claimed ">1,000,000" was not independently verified to the exact figure (capped re-implementation confirmed rapid growth past 150K in 2.05s, consistent with but not confirming the precise claim).
- No `gymnasium.Env` anywhere — bespoke plain-Python classes.
- Zone keys are plain tuples — picklable, verified.
- Wall-clock/step: three-goal 5.10µs, single-goal 4.71µs, hole 3.55µs.
- **Documentation discrepancy found**: all three run scripts actually use `GAMMA=1.0`, overriding the three-goal/single-goal class defaults of 0.99 documented in `trm_comparison/README.md`; the top-level `supplementary/README.md` states "50,000 training episodes" for all Frozen-Lake tasks but three-goal actually runs 150,000.

## G — summary rows

See consolidated table at the end of this document.

---

# Environment 3: Taxi TRM Comparison

**Path**: `/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/taxi/taxi_comparison/` (≡ `scheduling/taxi_comparison/`). **Not DBM-based** — hand-coded 2-bit threshold tuple, no `pyudbm` anywhere.

**Description**: 5×5 stochastic Taxi grid + TRM (guard `x>=c_min ∧ y<=D`, 2 clocks), same zone/region-agent pattern as Frozen Lake's three-goal variant, structurally near-identical except: goal events are pickup/dropoff-triggered (not pure arrival), action set adds PICKUP/DROPOFF, and there's no "hole" analogue.

**Provenance**: `supplementary/README.md`, taxi domain of the same paper package. **Discrepancy found**: README claims 50,000 training episodes; `run_taxi_comparison.py` actually uses `TRAIN_EPS=150_000`.

## A — automaton structure

Clocks: x (per-phase, resets on pickup/arrival/dropoff events "a","b","c"), y (global, **never resets** — confirmed by grep, only assigned at `reset()`). 6 instances, `c_min∈{2,5,10,15,20,25}`, `D=200` fixed, `d_max=5` fixed. Deadline: `y<=D` hard, guard failure → terminal `-200`. Timing: `dt=delay+1` deterministic, agent-chosen — distinct from the taxi's own stochastic movement slip (`slip_prob=0.2`, 80/10/10 split), which is the actual probabilistic-branching source.

## B — action structure

`(delay, taxi_action)`, `delay∈{0..5}`, 6 taxi actions (4 moves + PICKUP + DROPOFF) → 36 total. **Worked tradeoff example** (TAXI_03, executed directly): from an identical state (`x=7 < c_min=10`), `(delay=0,N)` → guard fails → **-200 terminal**; `(delay=2,N)` → `x_new=10>=10` → guard OK → **+100**. Fixed-size, unmasked.

## C — zone geometry

Not DBM-based. Independent BFS **disagrees with the paper's own "constant 4/305-node" corner-method claim**: true-trajectory BFS gives 169 (c_min=2) vs 221 (c_min=5) nodes — not constant — vs. the paper's corner-representative method's constant 305 across all 6 instances. Root cause identified: the corner method independently cross-products x-extremes × y-extremes, but real trajectories obey `y>=x` (since x resets on success and y never resets, both incrementing by the same dt) — the corner method can synthesize unreachable `(x,y)` combinations, an over-approximation. The docstring's "4 nodes" refers only to the isolated `(x_node,y_node)` projection, not the full `(pos,rm,x_node,y_node)` key graph — 4, 305, and 169/221 answer three different questions.

## D — aggregation (200 seeded episodes, uniform-random, 36-action space)

| Instance | c_min | distinct concrete | distinct zone | agg. ratio | distinct loc | loc-only ratio | frac. singleton zones |
|---|---:|---:|---:|---:|---:|---:|---:|
| TAXI_01 | 2 | 31,315 | 101 | 310.05 | 49 | 639.08 | 0.0693 |
| TAXI_03 | 10 | 31,315 | 115 | 272.30 | 49 | 639.08 | 0.0348 |
| TAXI_06 | 25 | 31,122 | 128 | 243.14 | 49 | 635.14 | 0.0078 |

Zone ratio (243–310×) does **not** exceed the location-only control (635–639×) in any instance — expected, since location-only discards all clock info and is strictly coarser. Zone abstraction still aggregates >99.5% relative to raw concrete state.

## E — decision relevance

Exact size `= 25 positions × 4 rm_states × (c_min+1) × (D+1)`: only TAXI_01 (60,300) is under 10⁵; TAXI_02 (120,600) is the next-closest but over.

- **TAXI_01**: 75/75 (100%) locations multi-zone; 300 zone buckets; **134/300 = 44.67%** conflicted.
- **TAXI_02** (run for corroboration despite being slightly over budget): identical pattern, 44.67% conflict — the conflict mechanism is driven by the shared `y<=D` boundary, essentially independent of c_min.

**Worked example** (zone `((0,0),1,0,0)`): concrete state `(x=0,y=200)` → optimal `(delay=0,PICKUP)` (guard already doomed, fail fast); concrete state `(x=0,y=198)` → optimal `(delay=1,PICKUP)` (waiting one tick makes both guard halves satisfiable). Identical zone bucket, opposite optimal strategy.

## F — scaling and engineering

- Fixed hardcoded list of 6 instances; dataclass itself is parameterizable but the benchmark suite is not generated.
- TAXI_06 region BFS (independently re-run) = 166,708; zone BFS (corner method) = 305 constant. `supplementary/README.md` gives no explicit taxi BFS numbers to cross-check against.
- No `gymnasium.Env` — bespoke classes, confirmed by MRO inspection.
- Zone keys are plain tuples, picklable, verified.
- Wall-clock/step: `TaxiZoneEnv.step()` ≈2.77µs; raw `TaxiGridEnv.step()` ≈1.86µs.

## G — summary row

See consolidated table at the end of this document.

---

# Environment 4: Timed-RM (sympy/box-abstraction Timed Reward Machines)

**Path**: `/Users/falah/Desktop/git-projects/RL-for-zones/Timed-RM/` (own git repo, separate lineage from the `supplementary/` paper). **No DBM/Federation/pyudbm anywhere.** Guards are plain strings evaluated via Python `eval()` (not, despite the import, actual `sympy` symbolic manipulation — `sympy`/`sympify`/`And` are imported but never called anywhere in the RM/env files, confirmed by grep — a vestigial/dead import).

**Provenance**: `Timed-RM/README.md`, "Experimental evaluations of RL algorithms for Timed Reward Machines," first commit 2025-10-07, predates and is architecturally unrelated to the `supplementary/` zone-vs-region code despite sharing the name "TRM."

**Two domains × two modes** (same `.txt` TRM file format drives both modes, so clock/guard/reset structure is domain-specific but mode-independent; only the runtime clock-state *representation* differs by mode):
- **Digital mode** (`timed_reward_machine.py`/`trm_environment_gym.py`): fixed-resolution clock-value `Box`, essentially the concrete state at whatever `discretization_param` resolution is set.
- **Real/"corner_abstraction" mode** (`timed_reward_machine_corner_abstraction.py`/`trm_environment_gym_corner_abstraction.py`): a genuine textbook **Alur–Dill clock-region construction** (`class Region`, canonical `time_successor()`/`reset_clocks()`, `create_region_space()` enumerating the full finite region automaton) — despite the "corner" name, this is NOT box-corner sampling; a separate, dead `find_corner_spaces()`/`create_sub_space()` box-corner-sampling mechanism exists in the code (imported, never called) and appears to be an abandoned earlier design.

## A — automaton structure

Clock count: 1 or 2 per `.txt` TRM file (never more, across all 24 shipped files). Both domains' *default* files (`disc_vs_cont.txt`) reset every clock somewhere, but the repo does contain files with a never-resetting deadline clock (`Other TRMs/trm_extra4.txt` and siblings — `x<=10` checked, never zeroed). Guard constants span `{1,5,6,7,8,9,10,12,15}` (Frozen Lake) / `{1,2,5,6,8,10,12,14,15}` (Taxi) across all files. Deadlines: yes, upper-bound guards double as deadlines (e.g. Taxi `x<=14` for pickup); failure falls through to a fixed `-100` invalid-transition reward, not enforced pre-emptively.

**CRITICAL finding — no genuine real-valued timing nondeterminism exists, in either mode.** Delay is always a discrete action the agent selects (`Discrete(...)` spaces), never sampled by the environment; even the "successor" dimension in region mode — which agent picks *which* region-successor to land in, the one place a classical region automaton has genuine nondeterminism — is agent-controlled, not env-sampled. Repo-wide grep for random/uniform calls finds only ε-greedy exploration and CRM successor-pruning randomness, never a timing/clock draw. Probabilistic branching that does exist is in the base-domain dynamics (Frozen Lake's `is_slippery`/`success_rate=0.8`), not in timing; Taxi's own stochastic flags (`is_rainy`, `fickle_passenger`) exist but default `False` and are never enabled by `main.py`.

## B — action structure

Composite actions: digital = `(delay_idx, env_action)`; real/region = `(delay_idx, successor_idx, env_action)`. Both fixed-size (full Cartesian product), unmasked (Taxi's own `action_mask()` method exists but is never consumed by the RM/env wrappers). **CRITICAL — genuine tradeoffs confirmed**: delay choice at an identical location flips deadline-guard satisfaction, e.g. Taxi pickup at `x<=14` — act now (guard OK, +200) vs. having dawdled (guard fails, -100 and terminal RM jump). Directly confirmed by §E's DP conflict fractions.

## C — zone geometry

No DBM object exists; the "corner" dict is a redundant, lossy projection of the actual `Region` object (discards the region's fractional-ordering fields) — the region **is** the finest-grained state ever materialized at runtime (clock values only ever advance in discretization/region steps, never a sampled real, per §A). Measured region counts: Frozen Lake `disc_vs_cont.txt` (1 clock, max-constant 2) → **5 regions**; Taxi `disc_vs_cont.txt` (2 clocks, max-constants 16/2) → **229 regions**. Empirically, the corner projection can be lossier than useful: on Taxi, 338 distinct corner-triples map to **538** distinct region-index triples (more abstraction cells than "concrete" cells) — see §D. What it would take to reimplement as real `pyudbm` zones: translate each guard string into DBM difference-constraints (trivial here, all guards are single-clock-vs-constant, no clock-difference constraints found in any of the 24 files); replace `time_successor()`/`delay_successor()` with `Federation.up()` + guard intersection; replace `reset_clocks()` with `Federation` reset; replace the pre-enumerated region space with on-the-fly DBM splitting; re-derive the agent-chosen "successor" action as a choice of which zone-face to land on.

## D — aggregation (200 episodes, uniform-random, both modes, Frozen Lake `disc_vs_cont.txt`)

| Mode | distinct concrete | distinct abstraction | ratio | distinct locations |
|---|---:|---:|---:|---:|
| Digital | 91 | 91 | **1.0000** (no aggregation — digital *is* the concrete state) | 51 |
| Real/region | 85 | 86 | **0.988** (≈no aggregation; only 5 regions total exist here) | 47 |

Taxi (2 clocks, 229 regions), for contrast: digital ratio 1.0000 (325/325); region **0.628** (338 corner-triples vs **538** region-index triples — more abstraction cells than corner cells, confirming the §C finding at 2-clock scale). This backend does not exhibit "many concrete states collapse into one zone" — digital mode never aggregates by construction, and region mode's aggregation is weak-to-negative at these max-constants.

## E — decision relevance

`reward_machine_utils.value_iteration` is not directly reusable (it operates only over untimed RM states, is dead-code-adjacent via a hardcoded-`False` reward-shaping flag) — a new value-iteration was written from scratch, replaying the real base-env transition kernel and the actual semi-MDP discount formula used by the real Q-learning code (γ=0.999).

- **Frozen Lake digital** (576 states): all 64/64 locations multi-zone; **52/64 (81.2%)** conflict.
- **Taxi digital** (102,000 states, under the 10⁵ threshold): all 500/500 locations multi-zone; **all 500/500 (100%)** conflict — the RM-state/clock component is decision-relevant everywhere.
- Region/real mode DP not attempted from scratch (materially larger MDP shape from the extra successor-action arm) — out of scope for this audit's time budget; state-space size is known small enough in principle (region counts 5/229 × location × rm-state, comfortably under 10⁵).

## F — scaling and engineering

- Only the TRM (guard/timing/reward structure) is CLI-parameterized (`-t` path to any of 24 `.txt` files); domain size (grid/Taxi layout) is hardcoded, not exposed as a flag.
- Largest shipped config: Taxi `disc_vs_cont.txt` (102,000 concrete states, 229 regions) — no larger file exists among the 24 shipped.
- `gymnasium.Env` is the **actual sole runtime interface** here (unlike job-shop's unused `JobShopEnv`) — `main.py`'s training loop drives everything through `gym.make(...)`-registered wrapped envs.
- Picklability: `env.current_config` (a `(u, Region, corner)` tuple) pickles and round-trips correctly (`Region` defines structural `__eq__`/`__hash__`); the full enumerated region list also pickles fine.
- Wall-clock/step: Frozen Lake digital 0.030ms, region 0.038ms; Taxi digital 0.031ms, region 0.041ms (region mode consistently ~25-30% slower).
- End-to-end smoke test (`main.py -tt 300`) ran the real training+logging pipeline successfully; a downstream demo-video step failed on an under-trained Q-table (a coverage artifact of the intentionally tiny `-tt`, fixed by raising it, not a structural bug). Repo left clean (`git status --short` empty) after testing.

## G — summary rows

See consolidated table at the end of this document.

---

# Environment 5: MarshlandSprintEnv

**Path**: `/Users/falah/Desktop/git-projects/RL-for-zones/RL/marshland_example.py` (+ `TabularQ.py`, `simulation_marshland.py`).

**Description**: a hand-rolled 3×3-grid, single-clock `pyudbm.Context`/`Federation` environment paired with a generic tabular Q-learning agent. **Provenance**: standalone exploratory prototype — no README, no cross-reference from either paper package, reads as scratch pyudbm-mechanics testing code. (`RL/test-for-udbm.py` sits alongside it as a pure pyudbm API demo, not itself an environment.)

## A — automaton structure

Exactly one clock, `t_gate`. **CRITICAL — it never resets in any reachable execution.** `reset_value` is never called anywhere in the file; instead a fresh `Federation` is rebuilt each step from a monotonically-growing Python float `t_min`. The one code branch that *would* reset it (`gate_open` becoming `False` then re-opening via the button) is **dead code** — `gate_open` is only ever assigned `True` (at `reset()` and inside a guard that requires it already be `False`, which never happens), confirmed by exhaustive BFS finding `gate_open==False` never occurs. Guards: `max_time=20` fixed upper bound; `time_inc∈{3,12}` (12 with p=0.5 only on the muddy tile). Deadline: yes — `t_min>20` empties the zone, terminating with `-100` (checked post-hoc, not as a pre-emptive invariant). No `.up()` call exists anywhere — time only ever advances via fixed discrete increments (3, deterministic, or 3-vs-12, a 50/50 coin flip only on the muddy tile), never an interval/continuous delay. Probabilistic branching exists only on that one muddy-tile transition.

## B — action structure

4 fixed directional moves. **CRITICAL — genuine tradeoff confirmed**: the same action index (e.g. UP) is a deterministic `+3` move from most cells but a stochastic `{+3,+12}` move when it enters the muddy tile; DP-verified example at `(1,0)`: action UP into the muddy tile has `Q=174.650` (mixed outcome) vs. action RIGHT (plain tile, deterministic) elsewhere. Illegal moves (obstacle/boundary) are a plain grid check outside the DBM machinery (self-loop, `-5`, zero time cost) — not zone-based masking.

## C — zone geometry (exact, exhaustive — trivial given 1 clock)

35 total reachable states, no cap needed. Every non-empty zone is a 1-D interval `[t_min,20]`: affine dim = 1 for all 7 non-empty zones (100%); bounded fraction 1.0; strict fraction 0.0; vertex count = 2 for every non-empty zone (0 for the 3 empty/expired zones). Matches the plain-language expectation exactly (a single-clock zone can only ever be an interval).

## D — aggregation (exhaustive, all 35 reachable states)

Distinct concrete keys: 35. Distinct zone keys: 33. **Aggregation ratio ≈1.06.** Location-only control ratio: 4.375. **The zone ratio does NOT exceed the location-only control** — aggregation here is almost nonexistent (the env re-derives a uniquely-bounded interval from `t_min` every step rather than merging valuations); the only real collapse is 3 different "expired" `t_min` values sharing one `false`/empty zone.

## E — decision relevance (exact DP, γ=0.95, full 23-state non-terminal MDP)

All 7/7 non-terminal locations are multi-zone (100%). **1/7 (≈14.3%) have optimal-action conflicts.** Worked example at position `(1,0)`: optimal action is RIGHT at `t_min=3` (avoid the muddy tile), switches to UP at `t_min=9` (through the muddy tile — the safe route is now too slow), and all 4 actions tie at `t_min=18` (any real move already overflows the deadline).

## F — scaling and engineering

- Not parameterized — `grid_size=3`, `max_time=20` are hardcoded internal constants, not constructor args; never varied anywhere in the repo.
- No `gymnasium.Env` — bespoke 3-tuple `step()` return, incompatible with the modern gym API.
- Picklability: `Federation`/whole-env pickle and deepcopy **fail** (same `_NativeFederation` TypeError as job-shop); native `.copy()` works.
- Wall-clock/step: ≈15.5µs (2000-call measurement).

## G — summary row

See consolidated table below.

---

# Consolidated Summary Table (all environments / sub-envs / variants)

| Environment | Clocks | Non-resetting clock? | Deadlines? | Real-valued timing? | Distinct-tradeoff actions? | Mean affine dim | Aggregation ratio | Location control | Multi-zone locations | Conflict fraction | Parameterized? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Job-shop: SymbolicJobShopEnv** | jobs+1 (3–15) | **Yes — `delta`** | None | No (discrete draw) | Yes | 2.85→4.62 (PJS_01→05) | 0.384/0.598/0.922 (**<1**, PJS 01/03/06) | 1.867/5.791/1.363 (**>1**) | 7/17/54/141/1,538 (PJS_01→05) | 0/0/0/1.4%/2.6% | No |
| **Job-shop: RegionJobShopEnv** | N/A (capped ints) | N/A | None | No | Yes | N/A — no zone rep | N/A | N/A | N/A | N/A | No |
| **Job-shop: PTADirectEnv (DDPG)** | N/A (raw floats) | N/A | None | **Yes** (raw uncapped clocks + continuous wait) | Yes (richest — continuous) | N/A — no zone rep | N/A | N/A | N/A | N/A | No |
| **Job-shop: JobShopEnv (dead code)** | N/A (raw clocks) | N/A | None | Yes | Yes (unused) | N/A — no zone rep | N/A | N/A | N/A | N/A | No; never instantiated |
| **FL Three-goal** | 2 (x,y) | Yes — y | Yes (y≤D) | No | Yes | N/A — no DBM | 19.3/14.7/14.0 (TRM 01/03/06) | 28.7/30.4/28.1 (>zone) | 96 (TRM_01 only qualifies) | 0.0% | No |
| **FL Single-goal** | 1 (y) | Yes — y | Yes (c_min≤y≤D) | No | Yes | N/A | 11.4/14.8/13.6 (SG 01/03/05) | 31.7/28.2/24.1 (>zone) | 32 (all 5 qualify) | 0.0%(SG_01) / 6.2%(SG_02–05) | No |
| **FL Hole-disable** | 3 (x,y,h) | Yes — y | Yes (y≤D) | No | Yes (+masking) | N/A | 61.3/57.3/62.1 (HOLE 01/03/05) | 660.2/627.7/667.2 (≫zone) | not computed — no instance <10⁵ | N/A | No |
| **Taxi TRM comparison** | 2 (x,y) | Yes — y | Yes (y≤D) | No | Yes | N/A — no DBM | 310.1/272.3/243.1 (TAXI 01/03/06) | 639.1/639.1/635.1 (>zone) | 75/75 (100%, TAXI_01 & 02) | 44.67% (both) | No |
| **Timed-RM FL digital** | 1 | No (default file) | Yes | No (delay is chosen, not sampled) | Yes | N/A — no DBM | 1.000 | insufficient (52/64 conflict) | 64/64 | 81.2% | Yes (via `-t` file; grid hardcoded) |
| **Timed-RM FL region/real** | 1 | No (default file) | Yes | No | Yes | N/A — region, not DBM | 0.988 | insufficient (DP not separately solved) | 47 observed | not solved | Yes |
| **Timed-RM Taxi digital** | 2 | No (default file) | Yes | No | Yes | N/A | 1.000 | insufficient (500/500 conflict) | 500/500 | 100% | Yes |
| **Timed-RM Taxi region/real** | 2 | No (default file) | Yes | No | Yes | N/A — region, not DBM | 0.628 (more abstraction cells than concrete) | insufficient (DP not separately solved) | 32 observed | not solved | Yes |
| **MarshlandSprintEnv** | 1 (t_gate) | **Yes — never resets (dead-code reset branch)** | Yes (`t_min>20`) | No (fixed increments, one coin flip) | Yes (muddy-tile risk) | 1.0 | 1.06 (**<location control**) | 4.375 | 7/7 (100%) | 14.3% | No |

**Reading notes for the table**: "N/A — no DBM/no zone rep" marks sub-environments/families that carry no `pyudbm`/`Federation` object at all (job-shop's Region/PTADirect/JobShopEnv sub-envs; all of Frozen Lake TRM comparison, Taxi TRM comparison, and Timed-RM, which use hand-coded threshold tuples, `Region` objects, or Alur–Dill regions instead of DBMs — see each section's §C for what a real DBM reimplementation would require). Only **`SymbolicJobShopEnv`** (job-shop) and **`MarshlandSprintEnv`** are backed by real `pyudbm` `Federation` objects; both are also the only two environments where "mean affine dim" is a meaningful, measured quantity. Every environment tested for aggregation showed the zone/abstraction ratio *below* its location-only control except none — i.e., **in every single environment measured (9 rows with both ratios computed), the location-only control ratio exceeded the zone/abstraction ratio**, meaning by this specific test the zone geometry never demonstrated aggregation beyond what discarding the clock state entirely would already achieve; the zone key's value observed instead was in refusing to *over*-aggregate (unlike location-only, it never merges decision-relevant guard-satisfied/violated states, per each section's §D discussion) and in the (non-trivial, up to 100%) fraction of locations where zone-conditioned optimal actions provably diverge (§E in each section).
