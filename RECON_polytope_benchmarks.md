# RECON: Can Zone RL vs Region RL run on the three Polytope benchmarks?

Read-only reconnaissance, 2026-10-05. Paths are relative to the repo root.
The only new file is this one. The region-size numbers in §4 come from a
scratch BFS script kept outside the repo (in the session scratchpad). It
calls the benchmarks' own pure `peek()` functions.

**Short answer.** Yes for all three, but none runs as-is. The three
Polytope benchmarks are **gridtimed** (T2, T3), **modesel** (M3, M4) and
**trmgrid** (G_SMALL, G_LARGE). Every constraint they produce is a
zone constraint (`x ~ c`, `x - y ~ c`), so nothing blocks regions. What
is missing is integration:

- Our tabular Zone RL and Region RL agents are hardcoded per environment.
  There is no generic TA/TRM interface to plug a new benchmark into.
- Our "Zone RL" on Taxi and Frozen Lake is not a DBM zone graph. It is a
  hand-derived partition of the clock space by guard thresholds.

Region RL is a small adapter for gridtimed and trmgrid. modesel needs a
choice of semantics first (its training env uses real-valued durations).
For Zone RL we first have to decide what "zone" means on these domains;
see §4 and §5.

---

## 1. Repo map

### 1.1 Top level (relevant parts only)

```
RL-for-zones/
├── scheduling/                 Zone RL vs Region RL work (our tabular agents)
│   ├── trm_comparison/         Frozen Lake TRM: 3-goal, single-goal (sg_*), hole-disable (hole_*)
│   │                           grids + TRMs + Zone/Region agents
│   ├── taxi_comparison/        Taxi TRM: env, TRM, Zone/Region agents
│   ├── prob_jobshop/           Probabilistic job shop: PTA, DBM zone ops (pyudbm),
│   │                           SymbolicRLAgent (zone RL), RegionRLAgent, DDPG
│   ├── frozen_lake/            older Frozen Lake grid (env.py, grid.py)
│   ├── run_trm_comparison.py   Zone vs Region, Frozen Lake 3-goal (TRM_01..06)
│   ├── run_sg_comparison.py    Zone vs Region, Frozen Lake single-goal (SG_01..05)
│   ├── run_hole_comparison.py  Zone vs Region, Frozen Lake hole-disable (HOLE_01..05)
│   ├── run_taxi_comparison.py  Zone vs Region, Taxi (TAXI_01..06)
│   ├── run_symbolic_rl.py / run_region_rl.py / run_comparison*.py   job-shop runners
│   ├── output/{trm,taxi,sg,hole,...}/   saved CSVs + plots
│   ├── Timed-RM/               copy of the upstream Timed-RM repo
│   └── supplementary/          paper supplementary bundle (see next line)
├── supplementary/              byte-identical to scheduling/supplementary/ (checked with diff -rq;
│                               trm_comparison/ only lacks README.md). Same agents, repackaged
│                               per domain: taxi/, frozen_lake/, jobshop/
├── Timed-RM/                   upstream github.com/ritamraha/Timed-RM: generic text-file TRMs,
│                               digital/real modes, corner abstraction (q_learn_ct*.py).
│                               NOT the Zone/Region agents
├── gridtimed/                  POLYTOPE BENCHMARK 1: grid + switches/gates (timed automaton)
├── modesel/                    POLYTOPE BENCHMARK 2: job shop with FAST/SLOW mode selection
├── trmgrid/                    POLYTOPE BENCHMARK 3: grid driven by a branching TRM
├── zoneviews/                  Zone wrapper around a pyudbm DBM: h_rep / v_rep / graph / rasters
│                               (the "polytope" views fed to encoders)
├── encoders/                   6 encoders: mlp_point, deepset_h, deepset_v, cnn_dbm, cnn_proj, gnn
├── train/                      PPO trainer (ppo.py), SyncVectorEnv (vec_env.py), job-shop gym env
├── run_one.py                  SLURM entry point for the Polytopes PPO sweep (288 runs)
├── gridtimed_eval.py, trmgrid_eval.py   greedy/stochastic policy evals used during PPO training
├── analyze.py                  builds analysis/*.png, summary.txt, steps_to_threshold.csv, encoder_stats.csv
├── results/                    PPO logs: {domain}__{inst}__{enc}__seed{k}{.jsonl,__episodes.jsonl,__manifest.json}, ckpt/
├── artifacts/                  design notes + measurement JSON (GRIDSOUND.md, MODESEL.md, TRMGRID_*.md,
│                               PARTITION.md, RESULTS_ALL_BENCHMARKS.md, ...)
├── gate_track/, deadline_track/, deadline_calib/, prob_jobshop_ext/, prob_jobshop_deadline/
│                               earlier job-shop-derived probes (not the 3 benchmarks)
├── pyudbm/                     vendored pyudbm source tree (UPPAAL UDBM bindings)
└── submit*.sbatch              SLURM scripts (Alpine)
```

### 1.2 Entry points and how they are configured

| Script | What it runs | Configuration |
|---|---|---|
| `scheduling/run_trm_comparison.py` | Zone vs Region Q-learning, Frozen Lake 3-goal | Hardcoded constants, lines 49–64: `SEEDS=[42,123,456,789,1337]`, `TRAIN_EPS=150_000`, `EVAL_INTERVAL=5_000`, `EVAL_INTERIM=30`, `EVAL_EPS=100`, `ALPHA=0.1`, `GAMMA=1.0`, `EPS_START=0.9`, `EPS_MIN=0.05`, `SLIP_PROB=0.2`, `BFS_CAP=500_000`. Instances from `trm_comparison/trm.py:107-114`. No CLI. |
| `scheduling/run_taxi_comparison.py` | Same, Taxi | Hardcoded constants, lines 57–70 (same values). Instances from `taxi_comparison/taxi_trm.py:85-92`. |
| `scheduling/run_sg_comparison.py`, `run_hole_comparison.py` | Frozen Lake variants | Hardcoded constants (`TRAIN_EPS=50_000`, `EVAL_INTERVAL=2_000`, otherwise the same). |
| `scheduling/run_symbolic_rl.py`, `run_region_rl.py`, `supplementary/jobshop/run_jobshop_comparison.py` | Job-shop Zone (symbolic) vs Region RL | Hardcoded constants at the top of each file. |
| `run_one.py` | Polytopes PPO sweep: one (domain, instance, encoder, seed) per SLURM array index | `--array-index` or `$SLURM_ARRAY_TASK_ID` indexes `TARGET_RUNS` (built at `run_one.py:195`). Flags: `--dry-run`, `--num-envs`, `--ckpt-every-steps`, `--plateau-patience-steps`. Instance constants hardcoded at lines 65–135; PPO budget at 142–157 (`TOTAL_STEPS=300_000`, `NUM_ENVS=8`). `_build_instance` (271–295) is the one place that constructs each benchmark instance. |
| `gridtimed/tiers.py`, `modesel/tiers.py`, `trmgrid/measurement{1,2,3}.py` | DP / zone / geometry diagnostics per tier | Hardcoded `TIERS` / `TIER_SPECS` and `STATE_LIMIT`. Results go to `/tmp/...`; copies are in `artifacts/`. |
| `Timed-RM/main.py` | Upstream TRM experiments | argparse: `-e taxi\|frozen_lake`, `-t <TRM .txt>`, `-m digital\|real`, `-d`, `-tt`, `-n`. |

---

## 2. Existing Zone RL and Region RL implementations

There are **two families**. Both are environment-specific and neither
consumes a generic TA/TRM spec.

### 2.1 Taxi and Frozen Lake (TRM) — the "Zone vs Region" paper agents

| Domain | Env | TRM | Zone RL | Region RL |
|---|---|---|---|---|
| Frozen Lake 3-goal | `scheduling/trm_comparison/grid.py` | `trm.py` | `zone_agent.py` (`TRMZoneEnv`, `TRMZoneAgent`) | `region_agent.py` (`TRMRegionEnv`, `TRMRegionAgent`) |
| Frozen Lake single-goal | `sg_grid.py` | `sg_trm.py` | `sg_zone_agent.py` | `sg_region_agent.py` |
| Frozen Lake hole-disable | `hole_grid.py` | `hole_trm.py` | `hole_zone_agent.py` | `hole_region_agent.py` |
| Taxi | `scheduling/taxi_comparison/taxi_env.py` | `taxi_trm.py` | `taxi_zone_agent.py` | `taxi_region_agent.py` |

**Input: hardcoded per environment, not a generic TRM.** Each TRM is a
Python class with a fixed guard predicate and a hand-written step
function. For example, `trm_comparison/trm.py:81-103`:

```python
@dataclass
class TRMInstance:
    name:  str
    c_min: int   # lower-bound guard on x   (x >= c_min to advance RM)
    D:     int   # upper-bound guard on y   (y <= D to advance RM)
    d_max: int   # maximum action delay
    def guard_satisfied(self, clocks): return clocks["x"] >= self.c_min and clocks["y"] <= self.D
```

`TRMState.step(event, trm)` (`trm.py:134-161`) hardcodes the u1→u2→u3
chain, the reset of `x`, and the rewards (+100/+500/−200/−500/−1).
Locations are ints `U1..U3`, `U_TERMINAL=0`. Clocks are a
`Dict[str,int]` `{"x","y"}`. Clock values are integers: each action
`(delay, move)` advances both clocks by `dt = delay+1`
(`zone_agent.py:64-79`).

**How "zones" are represented: a guard-threshold partition, not DBMs.**
The Zone RL state key is (`zone_agent.py:81-84`):

```python
def state_key(self) -> tuple:
    x_node = 0 if self._x < self.trm.c_min else 1
    y_node = 0 if self._y <= self.trm.D    else 1
    return (self._grid.pos(), self._trm_s.u, x_node, y_node)
```

The docstring says outright: "No pyudbm required — zone graph nodes are
determined by threshold comparisons derived directly from the guard
constants" (`zone_agent.py:17-18`). The other variants follow the same
pattern:

- `sg_zone_agent.py:56-63`: three y-nodes (`<c_min`, in window, `>D`).
- `hole_zone_agent.py:50-56`: x/y/h bits plus the disabled action.
- `taxi_zone_agent.py:70-73`: identical to Frozen Lake.

There is **no subsumption check (Z ⊆ Z')** in this family. The "zone
graph BFS" (`run_trm_comparison.py:145-210`, `count_zone_graph`) explores
these threshold keys from hand-picked corner valuations
(`zone_corners`, lines 158-162).

**Region RL: integer clocks capped at the guard constant.**
`region_agent.py:80-83`:

```python
def state_key(self) -> tuple:
    x_cap = min(self._x, self.trm.c_min)
    y_cap = min(self._y, self.trm.D)
    return (self._grid.pos(), self._trm_s.u, x_cap, y_cap)
```

No region graph is enumerated. This is the "digital clocks" abstraction,
which is exact here because all elapsed times are integers. It is not an
Alur–Dill region construction: there are no fractional parts and no
diagonal ordering. Region "graph size" is a concrete BFS over capped keys
(`run_trm_comparison.py:69-126`, `count_region_graph`). The max constant
per clock is the guard constant itself (`c_min` for x, `D` for y).

- Observation: `min(y, D)` puts `y = D` and `y > D` in the same class,
  even though the guard `y <= D` separates them. With `max_steps=400` and
  `dt` up to 6, y can reach about 2400 ≫ D=200. Capping at `D+1` would be
  the exact choice. The same pattern appears in `taxi_region_agent.py:74-77`
  and `sg_region_agent.py:57-59`.

**Learning loop.** The Zone and Region agents share an identical tabular
Q-learning loop; a diff shows only the env class differs
(`zone_agent.py:87-214` vs `region_agent.py:86-213`):

- Actions are `(delay, grid_action)` with `delay ∈ {0..d_max}` (24 actions
  for Frozen Lake, 36 for Taxi).
- Update: `Q[s][a] += α (r + γ^(delay+1) max Q[s'] − Q[s][a])`
  (`zone_agent.py:203-210`).
- Q-tables are `Dict[state_key, Dict[action, float]]`, lazily initialised
  to 0.
- ε decays linearly from 0.9 to 0.05; `max_steps=400` per episode.

**How a domain is plugged in (the template a new benchmark must follow).**
Each domain re-implements five pieces:

1. A grid env with `reset()` and `step(action) -> (pos, event, done)`.
2. A TRM class: guard predicate, step function, rewards.
3. A `*ZoneEnv` / `*RegionEnv` wrapper. It owns the integer clocks, calls
   the grid, then the TRM, applies resets, and exposes `state_key()`.
4. A `*Agent` holding the Q-table, `train(episodes, eval_interval, eval_episodes, eval_seed)`,
   `evaluate(n_episodes, seed)` and `n_states()`.
5. A runner with BFS counters and plotting.

Only `state_key()` differs between Zone and Region.

**Metrics logged** (`run_trm_comparison.py:320-432`; same in the Taxi
runner, lines 340-364):

- Per instance: `zone_mean/std` and `region_mean/std`, the final greedy
  evaluation over 100 episodes, as mean and std across 5 seeds.
- Checkpoint curves: every 5k episodes, 30 eval episodes.
- `zone_visited` / `region_visited`: `len(Q)`. **This is only the last
  seed's count** (`n_states_last`, lines 323 and 335), not an average.
- `zone_bfs` / `region_bfs`: reachable keys, plus truncation flags.
- `region_bound = (c_min+1)(D+1)`.
- Wall time is printed, not saved.

Outputs go to `scheduling/output/{trm,taxi,sg,hole}/comparison_summary.csv`
(Taxi: `taxi_summary.csv`) plus PNGs.

- Discrepancy: the README says γ=0.99 (`trm_comparison/README.md`, Q-learning
  table) but every runner uses `GAMMA = 1.0` (e.g. `run_trm_comparison.py:57`).

### 2.2 Job shop — the only real DBM-based Zone RL

| File | Contents |
|---|---|
| `scheduling/prob_jobshop/zone_ops.py` | `ZoneContext` (pyudbm `Context` over `c_J*` + cost clock `delta`); `init_zone`, `apply_guard`, `reset_clocks`, `apply_delay` (`up()` + invariant), `normalize` (`extrapolate_max_bounds`, k = M), `dominates`, `min_cost` |
| `scheduling/prob_jobshop/symbolic_env.py` | UPTA interface: `"active_d{d}"` locations, actions with per-duration `branches` |
| `scheduling/prob_jobshop/symbolic_rl.py` | `SymbolicRLAgent`: Q over `(loc, str(Z))` (lines 179, 310), probability-weighted branch values, `V_rem` table |
| `scheduling/prob_jobshop/region_env.py`, `region_rl.py` | `RegionJobShopEnv.state_key()` = `(full_loc, tuple(min(clock_j, max_constant_j)))` (lines 132–143); `RegionRLAgent` minimises cost with γ=1 |

- **Zone representation:** a pyudbm `Federation` (DBM). Identity in the
  Q-table is `str(Z)`.
- **Subsumption:** `dominates(Z_old, Z_new)` returns `Z_new <= Z_old`
  (`zone_ops.py:123-131`, pyudbm federation inclusion). **It is defined
  but never called** by `symbolic_rl.py`. A grep across the repo finds no
  call site in any RL agent. The only subsumption *use* anywhere is in the
  retired `gridtimed/zone_graph.py` (see §3.1).
- **Extrapolation:** `normalize` uses `extrapolate_max_bounds` with a
  single max constant M for all job clocks and ∞ for `delta`
  (`zone_ops.py:112-120`).

This family is the closest template for **modesel**: dispatch actions,
cost minimisation, per-job clocks.

---

## 3. The three Polytope benchmarks

Shared facts:

- All three are wired into the same PPO sweep through `run_one.py` (§3.4).
- All use pyudbm with `SCALE = 20`: every bound passed to pyudbm is
  multiplied by 20 and rounded, because the jitter `w=0.25` produces
  non-integer bounds.
- The "polytope" an encoder sees is `zoneviews.zone.Zone(Z, clock_names, k_vec).clip_to_k_box()`.
  It wraps exactly one canonical DBM (§3.5).
- None of them has a declarative TA or TRM file. The automaton is implicit
  in each package's `env.py` (concrete step) and is mirrored by hand in its
  `zones.py` / `gym_env.py` (symbolic step).

### 3.1 gridtimed (instances T2, T3)

**Files:**

- `gridtimed/instance.py`: `GridInstance` and constants.
- `generator.py`: `make_instance`.
- `env.py`: concrete `peek`/`step`.
- `zone_ops.py`: pyudbm ops.
- `zones.py`: one zone per DP state.
- `gym_env.py`: `GridTimedTrainEnv`, used for training.
- `dp.py`: exact DP.
- `tiers.py`: T0–T4.
- Diagnostics: `geometry.py`, `decision_relevance.py`, `diagnostics.py`,
  `sanity_gate.py`.
- `zone_graph.py`: retired.

**Task.** An n×n grid. Start at (0,0); GOAL at (n−1, n−1). There are K
switch cells and K gate cells. Entering switch i resets clock `s_i`. Gate i
can only be entered while `s_i ∈ [a_i, b_i]`. GOAL must be reached with
`y ≤ D`. Actions are `(delay, move)` with `delay ∈ {0,1,2}` and 4 moves
(12 actions); each move takes `delay+1` time units.

**Underlying model.** This is an implicit timed automaton:

- Locations are grid cells plus absorbing `DONE` / `FAILED`.
- Concrete semantics, `gridtimed/env.py:79-113`:

```python
elapsed = delay + 1
new_clocks = list(c + elapsed for c in state.clocks); new_y = state.y + elapsed
if instance.is_gate(target):
    lo, hi = instance.gate_window(target)
    if not (lo <= new_clocks[idx] <= hi): return StepResult(state, ILLEGAL_PENALTY, 0, legal=False, ...)  # blocked, no time passes
elif instance.is_switch(target):
    new_clocks[instance.switch_index(target)] = 0                                                     # reset
if target == instance.goal and new_y <= instance.deadline: ... DONE (reward +100 - elapsed)
if new_y > instance.deadline: ... FAILED (penalty 50)
```

- Symbolic semantics, `gridtimed/gym_env.py:132-153` (and the same in
  `zones.py:53-63`). Each edge does: reset `e` → `up()` → `e ≤ delay+1+w`
  → `e ≥ delay+1`; then, at a gate, `a ≤ s_i ≤ b`; then `y ≤ D`; then
  reset `s_i` at a switch; then reset `e`.
- **Invariants:** only the per-edge `e ≤ delay+1+w`. There are no location
  invariants.

**Clocks.** `s1..sK` (switch clocks), `y` (global, never reset; an assert
enforces this at `zone_ops.py:50-52`), and `e` (zone-only edge clock).
That gives d = K+2 zone dimensions: T2 has 5, T3 has 6 (confirmed by
`n_clocks_total` in `artifacts/gridsound_results/tiers.json`).

- Guards: `s_i ∈ [a_i, b_i]` on entering gate i, and `y ≤ D`.
- Max constants for the T2 and T3 instances actually trained on
  (generator seed 0, `w=0.25`; reproduced by calling `make_instance` with
  the `run_one.py` parameters):

| Inst | grid | switches | gates | windows on s1..sK | D | max consts (s_i, y) |
|---|---|---|---|---|---|---|
| T2 | 5×5 | (4,1),(1,1),(3,2) | (2,0),(4,0),(4,3) | (23,25),(8,10),(15,17) | 27 | 25, 10, 17, 27 |
| T3 | 6×6 | (0,3),(3,1),(1,5),(4,0) | (5,3),(5,2),(1,2),(3,4) | (19,21),(29,31),(6,8),(22,24) | 33 | 21, 31, 8, 24, 33 |

In pyudbm units, multiply by 20. The k-box used for normalize/clip is
`max(K_BOUND=20, D+4)` (`gym_env.py:107`).

**Constraint shape: zones only.** Every constraint enters through
`apply_upper_guard` / `apply_lower_guard`, which build `clock <= c` or
`clock >= c` (`zone_ops.py:59-71`), or through `up()` / `reset_value`.
The DBM then supplies the closed-form differences. No general linear
constraint is ever constructed. `zoneviews.zone.HalfSpace` can only
represent difference constraints (§3.5).

**Zone → polytope pipeline.**

1. Online, during training: `GridTimedTrainEnv.step` updates `self._Z`
   (`gym_env.py:132-153`).
2. `_current_zone()` wraps it: `Zone(self._Z, names, k_vec).clip_to_k_box()`
   (`gym_env.py:200-202`). The encoders then call `h_rep()`, `v_rep()`,
   `graph()`, or the raster views.
3. No `normalize` (extrapolation) is applied online. The zone is
   path-dependent and never merged or subsumed.
4. Offline, `zones.build_state_zones` (`zones.py:32-66`) builds **one zone
   per concrete DP state**, using the first-discovered path.
   `distinct_zones` dedupes by `(pos, str(Z))`.
5. The resulting dicts (`Dict[GridState, Federation]`) are in-memory only.
   No serializer exists; `str(Z)` is the only textual form used anywhere.
6. Measured: `n_zones_total` ≈ `n_states` (for example T2 seed 0: 76,354
   vs 76,356 in `artifacts/gridsound_results/tiers.json`). So zones are
   about 1:1 with concrete states.
7. `gridtimed/zone_graph.py` holds the older location-graph zone BFS with
   `extrapolate_max_bounds` and subsumption (`any(Z2 <= Zold ...)`,
   line 112). It is retired (`zones.py:12-20`), and its imports are stale:
   `bounded_delay`, `gate_clock`, `in_bounds` etc. no longer exist in
   `zone_ops.py` / `instance.py`, and `GridZoneContext()` now requires an
   instance. **It will not import as-is.**

**Environment interface.**

- Gymnasium env.
- Observation: `{"point", "zone", "mask"}`. `point` = position one-hot +
  `[s_i, y, 0]` + per-move features.
- Action space: `Discrete(12)`, with a legality mask.
- Reward = −cost: time cost 1 per unit; +100 success; −50 deadline
  failure; −2 for an illegal move, which costs no time and is masked
  during PPO anyway.
- Termination: `DONE` / `FAILED`.
- **Deterministic.** The concrete env always uses `elapsed = delay+1`. The
  jitter `w` exists only in the zone construction (`instance.py:20-30`,
  `gym_env.py` docstring). So the zone is a deliberately widened envelope
  around a deterministic integer trajectory.

**Natural scaling parameters.** `make_instance(grid_n, n_switches, n_gates,
window_width, D, w, seed)` (`generator.py:61-72`). Gate windows are drawn
as `a ∈ [grid_n, D−width−1]`, `b = a+width` (lines 94-98), so **D**
directly scales every max constant. `tiers.py:24-30` defines T0–T4
(grid 3→8, switches 2→5, D 17→43). `allowed_delays` scales the action
set.

**What the Polytopes project ran.** PPO with 6 encoders × 10 seeds × {T2,
T3}, 300k env steps. Results are in `results/gridtimed__*`,
`analysis/summary.txt`, and `artifacts/RESULTS_ALL_BENCHMARKS.md`; for
example, on T2 cnn_proj and gnn reach last-200 success 1.0 and mlp_point
reaches 0.05. A partition-soundness study is in `artifacts/PARTITION.md`
(hand-built T0, 3,436 states). Gate-passability bits are 43.5% sound and
location + activated-switch bitmask is 52.2% sound. Integer clocks
(b=1) and the zone construction are both 100% sound, but they coincide
with the concrete state.

### 3.2 modesel (instances M3, M4)

**Files:**

- `modesel/instance.py`: `Task`, `Job`, `ModeSelInstance`.
- `generator.py`.
- `env.py`: concrete, with `step_deterministic` and stochastic `step`.
- `zone_ops.py`.
- `zones.py`: `advance_zone_for_transition`, `build_state_zones`.
- `gym_env.py`: `ModeSelGymEnv`.
- `dp.py`, `tiers.py`, `calibration.py`, plus diagnostics.

**Task.** N jobs, each a chain of tasks routed over machines. Each machine
runs one task at a time. Every dispatch chooses FAST (duration `d_fast`,
cost 6) or SLOW (`d_slow ≈ 2·d_fast`, free). Job j has deadline `D_j`.
Time advances automatically to the next decision point. A waiting job
that can no longer finish even with all-FAST fails with penalty
3×(all-SLOW makespan). The objective is to minimise time plus FAST cost
plus failure penalties.

**Underlying model.** This is an implicit (job-shop-style) timed
automaton:

- A location is the per-job `(cursor, active_mode)` plus failed flags.
- Concrete semantics: `env.py:74-104` (legality and feasibility) and
  `env.py:132-195` (the advance loop).
- Symbolic semantics: `zones.py:35-96`. Dispatch resets `c_j` and applies
  invariant `c_j ≤ d·(1+w)`. Each micro-step does `up()` and re-applies
  the invariants of active jobs. A completion applies guard
  `c_i ≥ d·(1−w)`, then resets `c_i`. At the end, clocks of non-active
  jobs are reset.
- Feasibility is checked on the concrete edge as `g + remaining ≤ D_j`
  (`env.py:74-89`), a single-clock upper bound. **It is deliberately not
  added to the zone** (`zones.py:13-19`).

**Clocks.** `c0..c{N−1}` (one per job) plus `g` (global). That is d = 4
for M3 and d = 5 for M4. There is no `e` clock.

- Max constants, from `make_instance` with the `run_one.py` parameters
  (seed 0, tightness 0.25 / 0.5):

| Inst | jobs: (machine, d_fast, d_slow) per task | D_j | max const on c_j (nominal / with w=.25) | on g |
|---|---|---|---|---|
| M3 | J0 (M0,2,4)(M2,4,8)(M1,5,10); J1 (M0,5,10)(M2,4,8)(M1,3,6); J2 (M1,4,8)(M0,3,6)(M2,2,4) | 28, 30, 22 | 10 / 12.5 | 30 (feasibility constants are `D_j − remaining`) |
| M4 | M3's jobs + J3 (M0,3,6)(M1,4,8)(M2,2,4) | 33, 36, 27, 27 | 10 / 12.5 | 36 |

The k-box is `max(D_j)+4` (`gym_env.py:73`).

**Constraint shape: zones only.** Only `clock <= c` / `clock >= c`
(`zone_ops.py:56-67`), `up()`, and `reset_value`. Interval endpoints
`d(1±w)` are non-integers that become integers after `×SCALE`.

**Zone → polytope pipeline.** The same pattern as gridtimed:

- Online `advance_zone_for_transition` in `gym_env.py:101`.
- `Zone(...).clip_to_k_box()` in `gym_env.py:138-140`.
- No extrapolation or subsumption online.
- Offline `build_state_zones` (`zones.py:99-119`) builds one zone per DP
  state.
- No serialization.

**Environment interface.**

- Observation: `{"point": [g] ++ status one-hot ×4 per task ++ elapsed fraction per task, "zone", "mask"}`.
- Action space: `Discrete(2 × n_tasks)`, where each action is a
  `(task, mode)` pair.
- Reward = −cost. `info["success"]` means no job failed.
- **Stochastic.** Training calls `env.step`, which samples a real duration
  `U[d(1−w), d(1+w)]` (`env.py:234-249`, `gym_env.py:100`). The exact DP
  uses `step_deterministic` with nominal integer durations
  (`env.py:216-227`, `dp.py:64`).
- **Clock values in training are real-valued.**

**Scaling parameters.** `n_jobs`, `n_tasks`, `n_machines`, `tightness`
(scales every `D_j`), `fast_slow_ratio`, `w`, and `BASE_DURATION_RANGE=(2,5)`
(`generator.py:21`). Tiers M0–M4 are in `tiers.py:19-25`.

**What the Polytopes project ran.** PPO with 6 encoders × 3 seeds × {M3,
M4}. Results are in `results/modesel__*` and
`artifacts/RESULTS_ALL_BENCHMARKS.md`. Exact DP sizes (nominal,
untruncated, from `artifacts/modesel_results/tiers.json`): M3 seed 0 has
**569** states; M4 seed 0 has **8,588**. The partition study is in
`artifacts/MODESEL.md` (M4). Location-only is 60.5% sound;
discretized-g b=2 is 96.9% sound with mean cell size 2.0.

### 3.3 trmgrid (instances G_SMALL, G_LARGE) — v2, branching TRM

**Files:**

- `trmgrid/instance.py`: `TRMGridInstance`.
- `generator.py`.
- `env.py`.
- `zone_ops.py`.
- `zones.py`.
- `gym_env.py`: `TRMGridTrainEnv`.
- `dp.py`.
- `measurement1/2/3.py`.

**Task.** An n×n grid with k sub-goals and one GOAL cell. The agent must
visit any two distinct sub-goals i then j (its choice), then reach GOAL.
Actions are `(delay, move)` with delay ∈ {0,1,2} (12 actions), the same
as gridtimed.

**Underlying model: an explicit TRM in prose, implicit in code.** The
phase is a tuple: `()` → `(i,)` → `(i,j)`. The definition, from
`instance.py:7-24` and `env.py:96-150`:

```
()      --reach sub-goal i, no guard,                 reset t_i --> (i,)
(i,)    --reach sub-goal j≠i, guard lo_ij ≤ t_i ≤ hi_ij, reset t_j --> (i,j)   ; guard fail → FAILED (−50)
(i,j)   --reach GOAL, guard t_j ≤ cap_j  and g ≤ D             --> DONE (+100) ; fail → FAILED
any     --touch a sub-goal outside the active transition: reset that t_x, phase unchanged
any     --g > D --> FAILED
```

**Clocks.** `t_0..t_{k−1}`, `g` (global), and `e` (zone-only edge
clock). With k=3 that is d = 5.

- Guards: the pair windows `[lo_ij, hi_ij]` on `t_i`; `t_j ≤ cap_j`; and
  `g ≤ D`.
- Windows and caps come from Manhattan distances:
  `lo_ij = ⌈d_ij(1+α)⌉`, `hi_ij = lo_ij + width`,
  `cap_j = ⌈d_j,goal(1+α)⌉ + width` (`generator.py:72-84`).
- Instances actually trained on: α=0.25, width=8, w=0.25, layout seed 2
  (`run_one.py:118-130`).

| Inst | grid | sub-goals | goal | D | caps | windows (i,j)→(lo,hi) | max const per t_i |
|---|---|---|---|---|---|---|---|
| G_SMALL | 4×4 | (1,3),(3,3),(1,0) | (3,1) | 30 | 13,11,12 | 01:(3,11) 02:(4,12) 10:(3,11) 12:(7,15) 20:(4,12) 21:(7,15) | 13, 15, 15 |
| G_LARGE | 5×5 | (2,3),(4,3),(1,3) | (4,4) | 30 | 12,10,13 | 01:(3,11) 02:(2,10) 10:(3,11) 12:(4,12) 20:(2,10) 21:(4,12) | 12, 12, 13 |

**Constraint shape: zones only.** `apply_upper_guard` /
`apply_lower_guard` (`zone_ops.py:60-72`) and `edge_delay`
(`zone_ops.py:75-86`). The guards are applied in `zones.py:44-61` and
`gym_env.py:155+`.

**Zone → polytope pipeline.**

- Online in `gym_env.py:142-189`.
- `_current_zone()` returns `Zone(...).clip_to_k_box()` with k-box
  `max(20, D+4)·20` (`gym_env.py:129, 227-229`).
- Offline `build_state_zones` (`zones.py:19-73`), one zone per state.
- `normalize` exists (`zone_ops.py:89-91`) but is not used online.
- Measured `mean_concrete_states_per_zone = 1.0` (`artifacts/trmgrid_v2_sweep.json`).
- No serialization.

**Environment interface.**

- Observation: `point` = `[t_i…, g, 0(e)]` + position one-hot + two k-dim
  phase one-hots, plus `zone` and `mask`.
- Action space: `Discrete(12)`.
- Reward = −elapsed, +100 on success, −50 on failure.
- **Fully deterministic**, stated and verified in `gym_env.py:9-21`. As in
  gridtimed, `w` only widens the zone.

**Scaling parameters.** `grid_n`, `k`, `alpha` (forces waiting), `width`
(window slack), `w`, `allowed_delays`. D is derived from these
(`generator.py:87-103`). `artifacts/TRMGRID_SWEEP.md` swept 4×4/k=3 up to
8×8/k=4. The exact DP for 4×4/k=3 already needs 0.58M–2.0M states
depending on layout seed (G_SMALL's seed 2: 1,346,833). G_LARGE's DP was
truncated at 400k.

**What the Polytopes project ran.** PPO with 6 encoders × 10 seeds ×
{G_SMALL, G_LARGE}. Results are in `results/trmgrid__*` and
`artifacts/RESULTS_ALL_BENCHMARKS.md`; the best last-200 success rate is
about 0.73. The partition study is in `artifacts/TRMGRID_V2.md` (5×5,
k=3, seed 0, 176,389 states, truncated). Location-only is 27.2% sound.
Location + guard-satisfiability bits (the trmgrid analogue of our Frozen
Lake Zone RL key) is **45.9%** sound. Discretized clocks b=2 are 96.9%
sound.

### 3.4 How the Polytopes training is wired (common to all three)

- `run_one._build_instance` (`run_one.py:271-295`) returns `(instance,
  GymEnvClass)`.
- `build_trainer` wraps it in `train.vec_env.SyncVectorEnv` (in-process,
  because Federations cannot be pickled) and `train.ppo.PPOTrainer`.
- PPO hyperparameters (`train/ppo.py:88-111`): γ 0.99, λ 0.95, clip 0.2,
  entropy 0.01, lr 3e-4, rollout 2048, 8 envs, 4 epochs, minibatch 256,
  300k steps.
- Logs: per-update JSONL (returns, losses, timing, `domain_metric`),
  per-episode JSONL (`return`, `length`, `success`), and a manifest with
  hyperparameters, git commit and SLURM ids.
- Q-entry and visited-state metrics do not exist in this pipeline.

### 3.5 Where zones become polytopes (`zoneviews/zone.py`)

`Zone.__init__` requires exactly one DBM (`zone.py:204-222`).
`Zone.h_rep()` (`zone.py:264-331`) turns each finite DBM entry `(i,j)`
into a `HalfSpace` whose coefficient vector has at most one `+1` and one
`−1`:

```python
a = [0] * self.d
if i != 0: a[i - 1] += 1
if j != 0: a[j - 1] -= 1
result.append(HalfSpace(a=tuple(a), b=bound, strict=strict))
```

`Zone.from_h_rep` / `_halfspace_to_federation` (`zone.py:235-262,
565-592`) **reject** anything else: "a general half-space cannot be
represented as a DBM constraint". So the "polytopes" are exactly the
zones, with `clip_to_k_box` adding only box constraints. The encoders
(`encoders/*.py`) consume views of these (`h_rep`, `v_rep`, `graph`,
DBM tensor, 2-D projection rasters). They never feed a new constraint
back into the environment.

---

## 4. Gap analysis

### 4.1 What is missing, per benchmark

| | gridtimed (T2, T3) | modesel (M3, M4) | trmgrid (G_SMALL, G_LARGE) |
|---|---|---|---|
| Constraint class | zones only | zones only | zones only |
| Action shape vs our agents | `(delay, move)`, the same shape as the Frozen Lake/Taxi agents | `(task, mode)` dispatch with automatic time advance, the same shape as the job-shop agents | `(delay, move)`, same as Frozen Lake/Taxi |
| Concrete dynamics | deterministic, integer clocks | **stochastic, real-valued clocks** (nominal-integer variant available: `step_deterministic`) | deterministic, integer clocks |
| **(a) Region RL** | **Needs adapter.** A `state_key = (pos, tuple(min(s_i, b_i+1)), y)` wrapper over `env.peek` plus a copy of the tabular loop. Rewards come from `-res.cost`. | **Needs adapter + a semantics decision.** `prob_jobshop/region_env.py` is the template. Integer regions are exact only with nominal durations (`step_deterministic`). With the real `U[d(1±w)]` sampling used in PPO training, digital-clock regions are not exact, and a true region construction (fractional ordering) is needed. Alternatively, durations could be discretized. | **Needs adapter**, the same as gridtimed. Key = `(phase, pos, tuple(min(t_i, maxc_i+1)), g)`. |
| **(b) Zone RL** | **Needs a new model definition** (a choice, not code). Option 1: our Frozen Lake-style threshold key (pos + gate-passability bits), which is mechanical but measured 43–54% sound on T0. Option 2: DBM key `(pos, str(Z))` from the online zone, which works mechanically (pyudbm ops exist) but zones are about 1:1 with concrete states and path-dependent, so expect no compression. Option 3: a real forward zone graph with extrapolation + subsumption, which would mean reviving `zone_graph.py` (stale imports). | **Needs adapter.** This is the most natural fit. `SymbolicRLAgent` (`prob_jobshop/symbolic_rl.py`) keyed on `(loc, str(Z))` with the online zone from `zones.advance_zone_for_transition`. Here the zone does **not** depend on the realized duration, only on modes and completion order, so it genuinely aggregates stochastic outcomes. Branching over duration outcomes would need rework: modesel durations are a continuous interval, not a discrete distribution. | **Needs a new model definition**, the same three options as gridtimed. The threshold-bits analogue was measured 45.9% sound on 5×5/k=3. |
| pyudbm required | for Zone RL only | for Zone RL only | for Zone RL only |
| Blocked? | No | No | No |

**Common to all three:**

- None of our tabular agents is generic. Each needs a ~150-line
  env-wrapper + agent pair in the style of `trm_comparison/*_agent.py`.
- The benchmarks' pure `peek()` / `step_deterministic()` functions make
  that easy, and their DP modules already give exact optimal values for
  checking learned policies.
- Our runners' BFS counters are domain-specific and would need rewriting.
  For Region RL, the BFS is trivial with `peek`; see the script described
  below.

### 4.2 Region-graph size estimates

Method:

- Scratch BFS over the benchmarks' own `peek()`, with each clock capped at
  (its largest guard constant + 1).
- Global clocks are not capped, because they are bounded by D (exceeding
  D is terminal).
- Edge clock `e` excluded.
- No truncation, unless noted.

This is exactly the state space our existing Region RL agents would key
on (digital clocks). It is exact for gridtimed and trmgrid because their
concrete dynamics are integer and deterministic.

| Inst | clocks used for regions | max consts | reachable capped (Region RL) states | uncapped concrete states (for reference) | per-location upper bounds: Π(c_i+2) integer / Alur–Dill n!·2ⁿ·Π(2c_i+2) |
|---|---|---|---|---|---|
| T2 | s1,s2,s3,y | 25,10,17,27 | **221,309** (24 locations) | 533,149 | 1.8e5 / 8.9e8 |
| T3 | s1..s4,y | 21,31,8,24,33 | **1,956,016** (35 locations) | 6,492,818 | 6.9e6 / 6.6e11 |
| M3 (nominal) | c0..c2,g | 10,10,8,30 | **569** (exact DP; nominal durations already integer-bounded) | 569 | 4.6e4 / 2.1e8 |
| M4 (nominal) | c0..c3,g | 10,10,8,8,36 | **8,588** (exact DP) | 8,588 | 5.5e5 / 4.5e10 |
| M3 (real durations, ×20 grid) | — | 250, 600 | not computed | — | 9.6e9 / 5.8e13 |
| G_SMALL | t0,t1,t2,g | 13,15,15,30 | **490,652** (145 (pos, phase) locations) | 1,346,833 (from `run_one.py:93-96`) | 1.4e5 / 6.8e8 |
| G_LARGE | t0,t1,t2,g | 12,12,13,30 | **822,622** (235 (pos, phase) locations) | > 400,000 (DP truncated) | 9.4e4 / 4.5e8 |

For comparison, the existing Frozen Lake 3-goal Region BFS ran from
19,439 (TRM_01) to 212,102 (TRM_06) nodes
(`trm_comparison/README.md`). Multiply state counts by 12 actions for
Q-entries (24 for Frozen Lake). T2, G_SMALL and G_LARGE are therefore
1–4× the largest Frozen Lake instance. T3 is about 9× larger (about 23M
Q-entries), which is likely too big for plain tabular Q-learning at the
current episode budgets. M3 and M4 are far smaller.

**Region-count knobs:**

- gridtimed: `D` (moves every window via the generator) and `grid_n` /
  `n_switches`.
- trmgrid: `width` and `alpha` (window constants), `grid_n`, `k`.
- modesel: `tightness` (scales `D_j`), `BASE_DURATION_RANGE`,
  `fast_slow_ratio`, `n_jobs`.

The current Frozen Lake/Taxi suites sweep a single guard constant
(`c_min`). The closest analogue is gridtimed's `D` with the layout held
fixed, but the generator re-draws the windows whenever D changes. Holding
the layout fixed while scaling the constants would need a small generator
change.

**Zone-side counts.** These were not measured for this report; they are
in `MEASURE_zones.md`. (Correction: an earlier version of this section
said pyudbm does not load on this machine. That was wrong. The vendored
`pyudbm/` has a compiled extension for CPython 3.10 only,
`_udbm.cpython-310-darwin.so`; the first attempt used Python 3.11. It
works with `/Users/falah/miniforge3/bin/python3.10`.)

The offline numbers (≈1 concrete state per zone for gridtimed and trmgrid)
imply that a DBM-keyed Zone RL would have *at least* as many states as
Region RL on those two domains. Option 1 (threshold bits) would be tiny
(hundreds of cells), but it is measured unsound.

---

## 5. Open questions / ambiguities

1. **Which "Zone RL" is meant?** On Taxi and Frozen Lake it is a
   hand-derived guard-threshold partition (`x<c_min` / `x≥c_min`, ...).
   It is not built with DBMs, and it is sound there only because of that
   TRM's structure. In the job shop it is a DBM-keyed `(loc, str(Z))`
   table without subsumption (`dominates` is never called). The two
   generalise very differently to the new benchmarks (§4.1, options 1–3).
   Pick one before implementing.
2. **Which Taxi and Frozen Lake are "ours"?** The Zone/Region agents live
   in `scheduling/{taxi,trm}_comparison/` (duplicated in `supplementary/`).
   The repo also contains upstream `Timed-RM/`: different TRMs, loaded
   from `.txt` files (e.g. `env/Taxi/disc_vs_cont.txt` uses `x<=14`,
   `y>1` guards), plus digital/real modes and a corner abstraction. I
   assumed the former.
3. **The `w` jitter in gridtimed and trmgrid.** It exists only in the
   zones; concrete dynamics are integer and deterministic. For a
   Zone-vs-Region comparison, should the env run with `w=0` (zones become
   points)? Or should the concrete env sample real durations (which makes
   digital-clock Region RL inexact)? This decides whether Region RL needs
   true region construction.
4. **modesel semantics.** PPO trained on real-valued sampled durations.
   Region RL at its natural cost needs nominal durations
   (`step_deterministic`, 569 / 8,588 states). Is a deterministic modesel
   acceptable for the comparison?
5. **Online zone counts** were not measured for this report. They are now
   measured in `MEASURE_zones.md` (pyudbm works locally under Python 3.10,
   see §4.2).
6. **gridtimed DP truncation semantics.** `artifacts/gridsound_results/tiers.json`
   reports T2 seed 0 as `n_states=76,356`, `dp_truncated=True`, while
   `tiers.py:22` sets `STATE_LIMIT=150_000`. My untruncated BFS finds
   533,149 uncapped states. I did not resolve what `n_states` counts
   after truncation (possibly "usable" states only).
7. **Existing agents' minor issues** to fix if reused:
   - The region cap `min(y, D)` merges `y=D` with `y>D` (§2.1).
   - The README says γ=0.99, the code uses 1.0.
   - The `*_visited` metric reports the last seed only.
8. **Max-constant choice for Region RL with `SCALE=20`.** If regions
   should match the zones' pyudbm units (×20, with the `w` endpoints), the
   constants become 20× larger (bottom row of the §4.2 table). The table
   uses the concrete integer constants instead.
