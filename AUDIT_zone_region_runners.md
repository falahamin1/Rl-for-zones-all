# Audit of the Zone-vs-Region runners (Task A)

Branch `audit-and-zone-measure`, 2026-10-05.

**Where the edits live.** `scheduling/` is a separate git repository
nested inside this one (a gitlink). Its `main` branch already had a lot
of uncommitted work before this audit: the multi-seed refactor of
`run_trm_comparison.py`, and `run_hole_comparison.py` and other files
that were never committed. So I created the same branch name inside
`scheduling/` as well. Nothing is committed. `git diff` inside
`scheduling/` therefore shows that earlier work mixed with this audit's
edits; the edits made here are listed in §5.

No benchmark package (gridtimed/, modesel/, trmgrid/) was touched. The
paper bundles (`supplementary/`, `scheduling/supplementary/`) were also
left alone; see §5.

All Python here ran with `/Users/falah/miniforge3/bin/python3.10`. It is
the only interpreter on this machine that loads the compiled pyudbm in
`pyudbm/pyudbm/binding/`.

---

## A1. Region cap

### Per-clock justification

A capped clock is exact when every value at or above the cap behaves the
same way in every guard that reads the clock.

| Domain | Guard (source) | Clock | Test | Values that differ | Cap before | Cap after |
|---|---|---|---|---|---|---|
| Frozen Lake 3-goal | `x >= c_min and y <= D` (`trm.py:102-103`) | x | `x >= c_min` | all x ≥ c_min alike | c_min | c_min (unchanged) |
| | | y | `y <= D` | y = D passes, y > D fails | **D** | **D+1** |
| Taxi | same guard (`taxi_trm.py:80-81`) | x | `x >= c_min` | all x ≥ c_min alike | c_min | c_min (unchanged) |
| | | y | `y <= D` | D vs D+1 | **D** | **D+1** |
| Single-goal | `c_min <= y <= D` (`sg_trm.py:66-67`) | y | upper bound D | D vs D+1 (c_min < D adds nothing above D) | **D** | **D+1** |
| Hole-disable | `x >= c_min and y <= D` (`hole_trm.py:72-73`); cooldown `h >= x_h` (`hole_trm.py:121`, `hole_region_agent.py` `valid_actions`) | x | `x >= c_min` | all alike | c_min | c_min (unchanged) |
| | | y | `y <= D` | D vs D+1 | **D** | **D+1** |
| | | h | `h >= x_h` / `h < x_h` | all h ≥ x_h alike | x_h | x_h (unchanged) |

Why this matters: y keeps growing after the deadline. Episodes run up to
400 steps, and the agent can still move after `y > D`. Under the old cap,
y = D (where a goal visit still succeeds) and y = D+1 (where it fails)
were the same state.

### Code changes (all inside `scheduling/`)

- `trm_comparison/region_agent.py`, `taxi_comparison/taxi_region_agent.py`,
  `trm_comparison/sg_region_agent.py`, `trm_comparison/hole_region_agent.py`:
  - `state_key()` uses `min(y, D + 1)`.
  - A per-clock comment justifies each cap.
  - Docstrings now give D+2 y-classes.
- Region BFS counters:
  - `run_trm_comparison.py` and `run_taxi_comparison.py`
    (`count_region_graph`): `ny_cap = min(y_new, trm.D + 1)`.
  - `run_hole_comparison.py` (`rkey`): `yc = min(y, trm.D + 1)`.
  - `run_sg_comparison.py` (`count_region_graph`): the cap moved to D+1.
    This BFS also had a second issue. It only enqueued states with
    `y_new <= D` ("once y_new > D, all further steps are still capped at
    D"), so successors of y > D states were never explored. With a
    separate y > D class those states must be expanded. It now enqueues
    them, carrying the capped y.
- Theoretical bounds updated to match: `(c_min+1)×(D+2)` in the trm/taxi
  runners (print, table, `region_bound`), `D+2` clock nodes in the sg
  runner, and `HoleTRMInstance.region_bound` = `(c_min+1)(D+2)(x_h+1)`
  (`trm_comparison/hole_trm.py`).
- The zone BFS counters needed no change: their y-nodes already split at
  `y <= D` / `y > D` (e.g. `zone_corners` uses `D` and `D+1`).

### Region BFS counts, before vs after

Each runner's own `count_region_graph` / `count_zone_graph` was run for
every instance. "Before" was measured on the unmodified code before any
edit.

| Instance | Region BFS before | Region BFS after | change | Zone BFS before | Zone BFS after |
|---|---|---|---|---|---|
| TRM_01 | 19,439 | 19,538 | +99 (0.51%) | 385 | 385 |
| TRM_02 | 33,845 | 34,020 | +175 (0.52%) | 385 | 385 |
| TRM_03 | 81,077 | 81,521 | +444 (0.55%) | 385 | 385 |
| TRM_04 | 131,152 | 131,916 | +764 (0.58%) | 385 | 385 |
| TRM_05 | 174,827 | 175,911 | +1,084 (0.62%) | 385 | 385 |
| TRM_06 | 212,102 | 213,506 | +1,404 (0.66%) | 385 | 385 |
| TAXI_01 | 15,854 | 15,934 | +80 (0.50%) | 305 | 305 |
| TAXI_02 | 25,035 | 25,164 | +129 (0.52%) | 305 | 305 |
| TAXI_03 | 62,938 | 63,284 | +346 (0.55%) | 305 | 305 |
| TAXI_04 | 102,653 | 103,254 | +601 (0.59%) | 305 | 305 |
| TAXI_05 | 137,243 | 138,099 | +856 (0.62%) | 305 | 305 |
| TAXI_06 | 166,708 | 167,819 | +1,111 (0.67%) | 305 | 305 |
| SG_01 | 835 | 867 | +32 (3.83%) | 100 | 100 |
| SG_02 | 1,795 | 1,827 | +32 (1.78%) | 100 | 100 |
| SG_03 | 3,075 | 3,107 | +32 (1.04%) | 100 | 100 |
| SG_04 | 6,275 | 6,307 | +32 (0.51%) | 100 | 100 |
| SG_05 | 12,675 | 12,707 | +32 (0.25%) | 100 | 100 |
| HOLE_01 | 1,038,552 | 1,044,253 | +5,701 (0.55%) | 1,905 | 1,905 |
| HOLE_02 | 2,179,191 | 2,191,116 | +11,925 (0.55%) | 1,905 | 1,905 |
| HOLE_03 | 5,244,199 | 5,273,120 | +28,921 (0.55%) | 1,905 | 1,905 |
| HOLE_04 | 8,225,871 | 8,271,912 | +46,041 (0.56%) | 1,905 | 1,905 |
| HOLE_05 | 13,691,191 | 13,771,472 | +80,281 (0.59%) | 1,905 | 1,905 |

Notes:

- **sg gains exactly 32 states on every instance.** The new y = D+1
  class appears once per (position, RM state) pair, regardless of D.
- **The hole BFS has no real cap.** `run_hole_comparison.py` runs
  `count_region_graph` to completion and only afterwards reports
  `min(raw, BFS_CAP)` with a † flag (`BFS_CAP = 2_000_000`). The summary
  CSV therefore shows 2,000,000† for HOLE_02–05, both before and after;
  the true sizes are the ones above. The console "Ratio" column
  (region/zone) is computed from the clipped value: it prints 1049.9×
  for HOLE_02–05, where the true ratios are 1,150× to 7,229×.
- Zone BFS counts are unchanged everywhere, as expected: the zone keys
  never had the off-by-one.

---

## A2. Visited metric

`_train_multi_seed` in all four runners now keeps `len(Q)` for every seed
instead of overwriting it each time.

- Summary CSVs (`comparison_summary.csv`, `taxi_summary.csv`,
  `sg_summary.csv`, `hole_summary.csv`): the single `zone_visited` /
  `region_visited` columns (last seed only) are replaced by
  `zone_visited_mean`, `zone_visited_std`, `region_visited_mean` and
  `region_visited_std`. These are the population std across seeds, the
  same convention as the existing `zone_std` / `region_std`.
- New `visited_per_seed.csv` in each output directory. Columns:
  `instance, seed, zone_final_mean, zone_visited, region_final_mean, region_visited`.
- Console lines and summary-table figures show mean ± std.
- Nothing else in the repo reads these columns (grep over `*.py`, `*.tex`
  and `*.ipynb`; only the runners and their paper-bundle copies).

**Finding, not changed: "visited" includes terminal keys.** Every agent
calls `q_new = self._get_q(s_new)` before checking `done`. That creates a
Q-entry for the terminal state, which is never used for bootstrapping
(the target uses 0 when `done`). So `len(Q)` counts terminal keys with
their exact clock values. The BFS counters, by contrast, record some
terminal keys with the clock zeroed (e.g. holes become `(npos, TERMINAL, 0)`).
The two columns are therefore not directly comparable.

Checked on SG_01 after 2,000 training episodes (seed 42):

| Agent | len(Q) | terminal keys | non-terminal | BFS |
|---|---|---|---|---|
| Region | 957 | 127 | 830 | 867 |
| Zone | 104 | 14 | 90 | 100 |

The visited count exceeds the "reachable" BFS count only because of these
terminal keys. Counting non-terminal keys only, or making the BFS keep
the exact clock on terminal keys, would make the two comparable. Not
done here: it changes a reported metric's definition, so it is your call.

---

## A3. Subsumption and min_cost in the Q-updates

**No agent used in the paper's tables performs a Z ⊆ Z' check.**

- `dominates(Z_old, Z_new)` (`return Z_new <= Z_old`, `prob_jobshop/zone_ops.py:123-131`)
  is defined but never called. A grep for `dominates(` over all of
  `scheduling/` finds only the definition.
- **Only the job-shop `SymbolicRLAgent` uses `min_cost(Z)`, and only in
  the TD target.**
- The Taxi and Frozen Lake "zone" agents use no zones at all. They key on
  guard-threshold bits.

Exact update lines (current line numbers):

| Agent | File:line | Update |
|---|---|---|
| Frozen Lake 3-goal, Zone | `trm_comparison/zone_agent.py:208-210` | `q[action] += self.alpha * (reward + (self.gamma ** (delay + 1)) * best_next - q[action])` |
| Frozen Lake 3-goal, Region | `trm_comparison/region_agent.py:209-211` | same line |
| Taxi, Zone | `taxi_comparison/taxi_zone_agent.py:197-199` | same line |
| Taxi, Region | `taxi_comparison/taxi_region_agent.py:203-205` | same line |
| Single-goal, Zone | `trm_comparison/sg_zone_agent.py:178` | `q[action] += self.alpha * (reward + self.gamma * best_next - q[action])` |
| Single-goal, Region | `trm_comparison/sg_region_agent.py:176` | same line |
| Hole-disable, Zone | `trm_comparison/hole_zone_agent.py:166` | `q[action] += self.alpha * (reward + self.gamma * best_next - q[action])`, `best_next` over valid actions only |
| Hole-disable, Region | `trm_comparison/hole_region_agent.py:166` | same line |
| Job shop, Region | `prob_jobshop/region_rl.py:190-197` | `target = float(elapsed)` if done, else `target = elapsed + self.gamma * best_next` with `best_next = min(Q[s'])`; then `self.Q[s][action] += self.lr * (target - self.Q[s][action])` |
| Job shop, Symbolic (Zone) | `prob_jobshop/symbolic_rl.py:347, 357, 362, 365` | `r = min_cost(Z_nx, self.zctx) - prev_delta` (with `prev_delta = min_cost(Z, self.zctx)`, line 336); `v_next = self.V_rem[next_key]` (0 if done); `self.Q[key][action_id] = q_old + self.alpha * (r + self.gamma * v_next - q_old)`; `self.V_rem[key] = min(self.Q[key].values())` |

Notes on the table:

- In the Taxi/Frozen Lake rows, `reward` is the TRM reward. The zone and
  region agents differ only in `state_key()`.
- The `scheduling/supplementary/jobshop/prob_jobshop/` copies of
  `symbolic_rl.py`, `symbolic_env.py`, `region_rl.py` and `zone_ops.py`
  are byte-identical to `scheduling/prob_jobshop/` (`diff -q`).

**Where `min_cost(Z)` enters `SymbolicRLAgent`:**

- **(i) The TD target: yes.** It enters as a difference,
  `r = min_cost(Z') − min_cost(Z)`.
- **(ii) The choice of which zone to keep: no.** The next zone is the
  branch matching the sampled duration (`matching[0] if matching else chosen_branches[0]`,
  line 344). There is no dominance test and no passed list.
- **(iii)** It is also the episode's return value (lines 371, 375), used
  for logging and evaluation.

Two side observations:

- The docstring and `SYMBOLIC_RL.md` describe a probability-weighted
  target `Q[s][a] = Σ_d P(d)·(min_cost(Z_d) + V_rem)`. The code does not
  do that: it is a single-sample update, which the module docstring
  itself states.
- If the sampled duration has no matching branch, line 344 silently
  falls back to the first branch's zone.

**Trace: is accumulated cost counted more than once?**
`measurements/trace_symbolic_rl.py` runs the agent's own `_run_episode`
under `sys.settrace` and reads its locals right after the Q-update
(nothing is re-implemented). Setup: PJS_01, after 300 warm-up episodes
(so `V_rem` is non-zero), seed 42. PJS_01 has four tasks, so the episode
has four dispatches; all are shown.

```
step  action  d     delta in Z    delta in Z_ min_cost Z min_cost Z_     r V_rem[s_]   target    Q_old    Q_new
   1   J2_T1  3       [0, inf]       [0, inf]          0           0     0     6.147    6.147    5.375    5.452
   2   J1_T1  4       [0, inf]       [4, inf]          0           4     4     2.451    6.451    6.147    6.177
   3   J2_T2  2       [4, inf]       [4, inf]          4           4     0     2.912    2.912    2.451    2.497
   4   J1_T2  3       [4, inf]       [7, inf]          4           7     3     0.000    3.000    2.912    2.920
sum of r over the episode = 7   final min_cost(Z) returned = 7
```

**Accumulated cost is counted exactly once.** The per-step rewards are
differences of `delta`'s lower bound. They telescope to the final
makespan (0+4+0+3 = 7 = `min_cost` of the last zone). The absolute
`delta` value never appears in a target, and `V_rem[s']` estimates only
the cost still to come. (Step 1 has r = 0 because dispatching J2_T1
leads straight to another decision point at time 0.)

Two related facts:

- `delta` is never extrapolated (`normalize` gives it bound ∞,
  `zone_ops.py:118-119`). So `delta`'s lower bound is part of `str(Z)`,
  which means the Q-table key encodes the elapsed time.
- `delta` is unbounded above in every zone (the zone allows idling), so
  `min_cost` is the only bound used.

---

## A4. README gamma

`scheduling/trm_comparison/README.md`, "Q-learning Setup":

- γ is now 1.0, matching `GAMMA = 1.0` in every runner (the agents'
  class default of 0.99 is always overridden).
- The sentence about `γ^(delay+1)` "heavier discounting" now notes that
  with γ = 1.0 this factor is 1.
- The `Seed | 42` row, stale since the multi-seed refactor, now lists the
  five seeds.

The supplementary READMEs already said γ = 1.0.

Not updated: that README's result tables (BFS counts, region-bound
column `(c_min+1)×(D+1)`) are outputs of the old code. They should be
regenerated from the full rerun rather than edited by hand.

---

## Smoke tests

Each runner's `main()` ran with `SEEDS=[42]`, `TRAIN_EPS=2000`,
`EVAL_INTERVAL=500`. These were set on the imported module by
`smoke.py` in the session scratchpad, so no source constants changed.
Output went to the scratchpad, so `scheduling/output/` was not
overwritten.

| Runner | Result | Wall time |
|---|---|---|
| `run_trm_comparison.py` | OK. Summary CSV, `visited_per_seed.csv`, plots written; region BFS matches the table above. | 46 s |
| `run_taxi_comparison.py` | OK | 54 s |
| `run_sg_comparison.py` | OK | 4 s |
| `run_hole_comparison.py` | OK. HOLE_01 region BFS 1,044,253; HOLE_02–05 shown as 2,000,000 (the post-hoc clip, see A1). | 423 s (mostly the uncapped region BFS) |

Returns at 2,000 episodes are not meaningful: the full runs use 150k
(trm, taxi) or 50k (sg, hole). The smoke tests only show that the code
paths run and write the new CSVs.

---

## 5. Not changed, and why

- **Paper bundle copies.** `supplementary/` and `scheduling/supplementary/`
  are untouched. They still contain the old `min(y, D)` cap and the
  last-seed visited metric (e.g.
  `scheduling/supplementary/frozen_lake/trm_comparison/region_agent.py:82`).
  They are now out of sync with `scheduling/`. Whether to propagate the
  fix depends on whether the paper results will be regenerated from the
  bundle.
- **Full experiments.** Not rerun, as instructed. Every paper number that
  depends on region states (BFS counts, visited counts, Region RL returns)
  will change slightly after the rerun.
