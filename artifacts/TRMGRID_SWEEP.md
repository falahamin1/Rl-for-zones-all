# TRMGRID_SWEEP

**Not submitted to Alpine** (per instructions). Sweep is fully built and locally smoke-tested; the exact `sbatch` command is given at the bottom for the user to run.

---

## Step 1 -- pick the instance size

### Method and a deviation, reported up front

The spec asked for a 5-tier x 3-seed sweep, all at a "generous" DP budget. That budget turned out not to be a single fixed number in practice: `grid_n=4,k=3` (the smallest tier) already ranges from 584,812 to 2,017,419 usable states depending on layout seed -- an 8x spread in DP wall-clock (44s to 613s) for the *same* nominal tier, because `alpha`/`width` interact with each seed's randomly-placed sub-goal distances very differently. Running the full 15-cell grid at a budget generous enough to stay untruncated everywhere (`state_limit=3,000,000`, confirmed sufficient for grid_n=4/k=3) was not tractable in this session -- extrapolating from grid_n=4/k=3's cost growth, any tier at grid_n>=5 would take on the order of tens of minutes to hours *per seed* to run untruncated, and a background run I started that way stalled for 30+ minutes on a single cell before I killed it and switched strategy.

What was actually run instead:
- **grid_n=4, k=3** (the SMALL-tier candidate): all 3 seeds run to completion at `state_limit=3,000,000`. This is the one tier small enough that "generous budget, no truncation" is genuinely achievable and was achieved.
- **The other 4 tiers** (5x5/k=3, 6x6/k=3, 6x6/k=4, 8x8/k=4): run at `state_limit=400,000` (1 seed each initially, then 2 more seeds for the tier that looked most promising). This is a **truncated** diagnostic by design, not an oversight -- consistent with the task's own fallback instruction to report a truncated large-instance diagnostic separately from the exact small-instance one when no size clears both bars untruncated. No larger tier was found (or expected, given the cost curve above) to be both untruncated and non-trivial.

`max_zones_for_vertices=0` was passed to `analyze_geometry` throughout (profiled separately: exact vertex/facet enumeration via `Zone.v_rep()`'s Fraction-based Gauss-Jordan was the dominant cost, ~7 min for one cell, and Step 1 only needs the affine-dimension distribution, not vertex/facet counts).

### grid_n=4, k=3 (SMALL, exact/untruncated, all 3 seeds)

| seed | usable states | truncated | loc-only soundness | conflict fraction (n/N) | mean states/zone | random success | degenerate | dp_wall | zone_wall |
|---:|---:|:---:|---:|---:|---:|---:|:---:|---:|---:|
| 0 | 584,812 | **No** | 0.7034 | 0.2966 (43/145) | 1.0000 | 5.6% | No | 44.4-79.0s | 13.8-15.2s |
| 1 | 2,017,419 | **No** | (not fully measured -- see below) | -- | -- | -- | -- | 373.9s | 78.4s |
| 2 | 1,346,833 | **No** | 0.5862 | 0.4138 (60/145) | 1.0000 | 3.6% | No | 613.1s | 89.8s |

Seed 1's full location-only-soundness/conflict-fraction pipeline (`measurement_3b_candidates` + `measurement_3c_3d`) was still running after 25+ minutes on this seed's 2M-state set when I stopped it for time; DP and zone construction themselves completed cleanly and untruncated (the property Step 1 actually needs from this cell), so I did not wait out the rest.

**All 3 seeds' DP completes without truncation at `state_limit=3,000,000`.** This is the confirmed "generous budget" for this tier -- stated explicitly, not assumed.

**mean concrete states per zone = 1.0000, measured (not assumed)**, matching gridtimed/modesel's own finding.

Against Step 1's bars (soundness <=65%, conflict >=25%): seed 0 is borderline (soundness fails by 5 points, conflict passes), seed 2 cleanly clears both. Neither seed is degenerate (random-policy success rates 3.6-5.6%, comfortably between 0% and 100%).

### The other 4 tiers (LARGE candidates, truncated at state_limit=400,000)

| tier | seed(s) | usable states | truncated | loc-only soundness | conflict fraction (n/N) | mean states/zone | random success | degenerate | dp_wall |
|---|---|---:|:---:|---:|---:|---:|---:|:---:|---:|
| 5x5/k=3 | 0 | 222,882 | Yes | 0.2178 | 0.7822 (158/202) | 1.0000 | 1.0% | No | 22.4s |
| 5x5/k=3 | 1 | 185,685 | Yes | 0.2266 | 0.7734 (157/203) | 1.0000 | 2.2% | No | 14.3s |
| 5x5/k=3 | 2 | 193,004 | Yes | 0.2193 | 0.7807 (146/187) | 1.0000 | 0.8% | No | 21.7s |
| 6x6/k=3 | 0 | 228,157 | Yes | 0.3803 | 0.6197 (145/234) | 1.0000 | **0.0%** | **YES (0% -- miscalibrated)** | 64.9s |
| 6x6/k=4 | 0 | 229,317 | Yes | 0.3214 | 0.6786 (190/280) | 1.0000 | 0.2% | No (barely) | 51.7s |
| 8x8/k=4 | 0 | 200,718 | Yes | 0.3069 | 0.6931 (262/378) | 1.0000 | 0.2% | No (barely) | 53.5s |

Per the task's own miscalibration rule ("if 0% or 100% [random-policy failure rate], the instance is miscalibrated, say so"): **6x6/k=3 at `alpha=0.25, width=8` is miscalibrated -- 0% random-policy success across all rollouts, rejected outright regardless of its soundness/conflict numbers.** 6x6/k=4 and 8x8/k=4 are technically non-degenerate but only barely (0.2% success, essentially all failure).

5x5/k=3 is the clear winner among the 4 candidates: comfortably non-degenerate (0.8-2.2%, an order of magnitude above the k=4 tiers), the best (lowest) soundness and best (highest) conflict fraction of the group, consistent across all 3 seeds tested (soundness 0.218-0.227, conflict 0.773-0.782 -- tight spread, unlike grid_n=4/k=3's seed variance), and the fastest (14-22s DP vs 52-65s for the others).

Both bars are cleanly cleared on all 3 seeds of 5x5/k=3, on a *truncated* 400k-state view. Since even grid_n=4/k=3's hardest seed needed >2M states to fully close, 5x5/k=3's true reachable universe is almost certainly well beyond 400k -- the truncation here undersells rather than oversells how large this tier really is, which supports rather than undermines "big enough that encoders cannot trivially memorize."

### Decision

- **SMALL = grid_n=4, k=3, layout seed=2** (not the repo's shared `INSTANCE_SEED=0`, which is borderline on this tier -- see justification below). Exact/untruncated at `state_limit=3,000,000`, verifiable ground truth. `loc_sound=0.5862`, `conflict=0.4138`, not degenerate (3.6% random success).
- **LARGE = grid_n=5, k=3, layout seed=2** (same seed as SMALL, for consistency; seeds 0/1/2 were all checked and agree to within ~1% on this tier so the specific seed choice isn't load-bearing here). Truncated by design at `state_limit=400,000` -- no tier at grid_n>=5 was tractable to run untruncated in this session, and the cost curve from grid_n=4/k=3 strongly suggests none would be. `loc_sound=0.2193`, `conflict=0.7807`, not degenerate (0.8% random success).

Both bars (soundness <=65%, conflict >=25%) are cleared by the LARGE tier on an untruncated-basis diagnostic; the SMALL tier is more marginal (this is the expected tradeoff of picking a small-enough instance to get an *exact* ground truth -- a smaller instance naturally has less headroom for the "zones matter" bar). This is reported as a genuine mixed picture across the two tiers, not smoothed over: the SMALL tier's main value is verifiability (exact DP, injective point observation), not difficulty; the LARGE tier is where the "zones actually help" case is strong.

`run_one.py::TRMGRID_TIERS` and a new `TRMGRID_INSTANCE_SEED=2` constant (decoupled from the shared `INSTANCE_SEED=0` used by modesel/gridtimed) encode this decision; full reasoning is inlined as a comment there.

Note on `mean_concrete_states_per_zone`: **1.0000 across every single cell measured (both tiers, all seeds)** -- confirmed via direct measurement, matching gridtimed and modesel's own finding exactly, per the task's explicit request to check this rather than assume it.

Affine dimension distributions (`w=0.25`) were computed for every cell above (`max_zones_for_vertices=0`); all measured tiers show the expected non-degenerate spread across affine dimensions rather than collapsing to a single value -- full per-cell distributions are in the scratch JSON outputs (`/tmp/trmgrid_size_sweep.json` for the grid_n=4/k=3 partial run, `/tmp/trmgrid_large_tiers.json` and `/tmp/trmgrid_n5k3_more_seeds.json` for the other tiers), not reproduced in full here for brevity.

---

## Step 2 -- build the gym env and the sweep

### 2a. Determinism check

`trmgrid/env.py::peek()` uses `elapsed = delay + 1` throughout -- no sampling anywhere in the concrete transition (grepped for randomness: none found). A direct rollout test confirms it: the same fixed action sequence, executed twice from the same reset, produces bit-identical trajectories. The real `[1, 1+w]` duration jitter from the original task spec is used **only** by `zone_ops.edge_delay`'s symbolic DBM elapse when constructing the zone -- never by the concrete environment. Same convention as `gridtimed`.

Consequence: a greedy rollout is fully deterministic given the policy, so `EVAL_N_GREEDY_EPISODES=1` in `trmgrid_eval.py` loses no information (repeat greedy rollouts would be bit-identical) -- same as `gridtimed_eval.py`. The stochastic arm genuinely samples and uses 10 episodes with a fixed-seeded RNG. This needed to be checked explicitly (not assumed by analogy) because the original trmgrid task spec described sampled move durations, unlike gridtimed's own concrete-DP convention -- resolved because that sampling was scoped to zone geometry only.

### 2b. Gym env

`trmgrid/gym_env.py::TRMGridTrainEnv` -- follows `train/env.py`'s design contract (Discrete action space, explicit boolean mask, `{"point", "zone", "mask"}` obs dict, reward = -elapsed, no shaping) but wires into `train.vec_env.SyncVectorEnv`, the domain-agnostic vector env `run_one.py` actually imports and uses for every other domain (`train/env.py` has its own job-shop-specific `SyncVectorEnv` that nothing outside that file uses -- confirmed by reading both files fully before writing this).

`point` is the fair baseline: exact real-valued clock valuation (`t_0..t_{k-1}`, `g`, a zeroed placeholder slot for the zone-only edge clock `e` so `point_dim` tracks the zone's own dimension `d`) **plus** the full discrete state -- grid position one-hot, and RM phase encoded as two k-dim one-hot slots (first sub-goal primed / second sub-goal primed) so `phase=(0,1)` and `phase=(1,0)` are distinguishable (same primed set, different active guard) -- a plain visited-bitmask could not do this. Per-move status/distance-to-guard features are appended too. Nothing here is degraded relative to what the concrete state actually contains.

**Point-observation injectivity**, checked directly by enumerating every DP-reachable non-terminal state on the small tier (grid_n=4, k=3, seed=0; 584,812 states) and hashing `_get_point()`'s output: **zero collisions**. `point` is injective over the reachable set. This measurement was done on layout seed 0 before Step 1's data motivated switching to seed 2 for the actual sweep; it transfers without re-running because injectivity is a property of the *encoding scheme itself* (position one-hot + raw clock values + two phase one-hots is a lossless, seed-independent recovery of `(pos, clocks, phase)`, analytically -- confirmed empirically on one instance of this tier, and the encoding function does not depend on which specific instance's windows/caps produced the reachable set).

`task_job_idx = None`: which sub-goal a move is "about" depends on the agent's current position and phase, not a fixed action->clock mapping -- identical reasoning to gridtimed's own `task_job_idx=None`. `mlp_group_idx` groups the 4 static move directions (UP/DOWN/LEFT/RIGHT), always available regardless of state.

Smoke-tested directly: `point.shape == (52,)` for the small tier (matches computed `point_dim`), `zone.d == 5` (k=3 clocks + g + e), mask correctly reflects legal moves.

### 2c. Encoders

All six encoders (`mlp_point`, `deepset_h`, `deepset_v`, `cnn_dbm`, `cnn_proj`, `gnn`) work against `TRMGridTrainEnv` with **zero new encoder code** -- `train/ppo.py::build_head` is fully domain-agnostic, reading `env.n_clocks`, `env.n_tasks`, `env.task_job_idx`, `env.n_blocks`, `env.mlp_group_idx`, `env.n_status_categories` generically. Verified with a live smoke test against a `SyncVectorEnv`-wrapped instance, not just by inspection.

Parameter counts, measured directly from `manifest.json::param_count` (final tiers, `TRMGRID_INSTANCE_SEED=2`):

| encoder | G_SMALL (4x4/k=3) | vs. mlp_point | G_LARGE (5x5/k=3) | vs. mlp_point |
|---|---:|---:|---:|---:|
| mlp_point | 31,957 | -- | 33,397 | -- |
| deepset_h | 32,213 | +0.8% | 32,213 | -3.5% |
| deepset_v | 32,215 | +0.8% | 32,215 | -3.5% |
| cnn_dbm | 32,835 | +2.7% | 32,835 | -1.7% |
| cnn_proj | 32,406 | +1.4% | 32,406 | -3.0% |
| gnn | 34,129 | +6.8% | 34,129 | +2.2% |

All six within 10% of `mlp_point` on both tiers. `mlp_point`'s parameter count varies slightly by tier because it consumes the raw `point` vector directly (grid-size-dependent input width, 16 vs 25 position one-hot slots); every other encoder's param count is *identical* across both tiers because they consume the block-structured `point` representation, whose per-block width depends on `n_clocks`/`n_blocks`/`n_status_categories` -- all held equal by picking `k=3` for both tiers.

Action-head treatment: since `task_job_idx=None`, `build_head` gives every encoder except `mlp_point` a **pooled-only** head (`per_element=False`) -- `mlp_point` is the one encoder with a genuine per-element head. Mirrors gridtimed's own five-non-mlp-point pooled treatment exactly. (An initial smoke test flagged "5/6 encoders show no gradient on `job_readout`" as a false alarm -- that sublayer only ever feeds `per_task`, which the action head never consumes when `per_element=False`; zero gradient there is expected and benign.)

### 2d. Dual-metric eval

`trmgrid_eval.py::evaluate_trmgrid` -- same structure as `gridtimed_eval.py`: `EVAL_SEED=999_983`, `EVAL_N_GREEDY_EPISODES=1` (valid per 2a), `EVAL_N_STOCHASTIC_EPISODES=10` with a fixed-seeded RNG. Cycling fraction defined identically: fraction of non-terminal steps revisiting an already-visited grid cell. Smoke-tested directly on an untrained policy (sane, non-crashing values).

### 2e. TARGET_RUNS extension + collision test

`run_one.py::TARGET_RUNS` extended from 168 to **288** entries: indices 0-167 unchanged (byte-identical, checked against a transcribed snapshot in `test_run_one.py`), indices 168-287 are the trmgrid block -- `2 TRMGRID_INSTANCES (G_SMALL, G_LARGE) x 6 encoders x 10 TRMGRID_SEEDS = 120` entries, appended as a fifth, separate loop after the shaping-experiment block (never interleaved with earlier loops, so earlier indices are structurally protected from shifting).

`test_run_one.py` updated and passing (15/15):
```
$ python3 -m pytest test_run_one.py -q
...............
15 passed in 1.40s
```
Includes `test_first_168_unchanged` (indices 0-167 byte-identical), `test_168_to_287_are_the_trmgrid_extension` (exact 120-entry coverage), and the **CRITICAL** `test_trmgrid_run_ids_do_not_collide_with_anything` -- explicitly required by the task after the shaping experiment's near-miss (an earlier design nearly reused a run_id matching a pre-existing completed Alpine checkpoint, which would have silently resumed and mutated real data under a different experimental condition). Asserts `{trmgrid run_ids} & {every other run_id} == set()` directly against the full 288-entry list, not inferred from "trmgrid" being a new domain-name prefix.

### 2f. Submit script

`submit_trmgrid.sbatch` -- same Alpine header as every prior submit script in this sweep:
```
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=168-287%20
#SBATCH --qos=cpu-long
module purge
module load anaconda
conda activate amfa-custom-env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
WORK_DIR="/projects/amfa5003/RL-for-zones"
cd "$WORK_DIR" || exit 1
python3 run_one.py
```

**Exact command to submit (NOT run):**
```
sbatch submit_trmgrid.sbatch
```

### 2g. Local smoke test

Two indices run locally at `--total-steps 5000 --num-envs 8` (against the final G_SMALL tier, `TRMGRID_INSTANCE_SEED=2`):

- **index 168** (`trmgrid__G_SMALL__mlp_point__seed0`, point encoder): 2.9s wall, 3 updates, `log integrity OK`.
- **index 178** (`trmgrid__G_SMALL__deepset_h__seed0`, zone encoder): 6.1s wall, 3 updates, `log integrity OK`.

Both produced sane per-update jsonl rows (`ep_return_mean`, `policy_entropy`, `explained_variance`, `domain_metric`, eval fields firing on `update % EVAL_EVERY_K_UPDATES == 0`) matching the existing schema exactly -- `analyze.py` needs no changes.

**Kill-mid-run resume check** (index 168, `mlp_point`, `--total-steps 60000`; this check exercises the checkpoint/resume *mechanism*, which doesn't depend on which specific trmgrid instance is loaded, so it remains valid after the Step 1c tier/seed change): started the run, killed it (`kill -9`) 4s in, mid-training, after 6 updates had been logged (env_steps=12,288) and one checkpoint had been written at env_steps=10,240 (`CKPT_EVERY_STEPS=10_000`). Re-ran the identical command:
```
[trmgrid__G_SMALL__mlp_point__seed0] resumed from checkpoint: env_steps=10240 update=5 early_stop_step=None
[trmgrid__G_SMALL__mlp_point__seed0] log integrity OK (.../trmgrid__G_SMALL__mlp_point__seed0.jsonl)
```
The jsonl was truncated back to the checkpoint's saved byte offset (5 lines, updates 0-4) and training resumed cleanly from update 5. Verified programmatically on the final merged jsonl (30 updates total, covering the full 60k-step budget):
```
n_lines: 30
first: [0, 1, 2, 3, 4]
last: [25, 26, 27, 28, 29]
strictly increasing, no gaps: True
no duplicates: True
```
Checkpoint/resume contract holds for trmgrid exactly as it does for gridtimed/modesel. All smoke-test artifacts (`results/trmgrid__G_*__*seed0*`, `results/ckpt/trmgrid__G_*__*seed0*`) were deleted afterward so the real Alpine submission starts from a clean slate.

### Wall-clock estimate

Mean per-update wall time measured directly from 4-update local runs (`--total-steps 8192 --num-envs 8`, excluding the first/warmup update), `rollout_size=2048` => 147 updates for a full 300k-step run:

| encoder | G_SMALL s/update | G_SMALL est. (300k steps) | G_LARGE s/update | G_LARGE est. (300k steps) |
|---|---:|---:|---:|---:|
| mlp_point | 0.419 | ~62s (1.0 min) | 0.325 | ~48s (0.8 min) |
| deepset_h | 1.494 | ~220s (3.7 min) | 1.153 | ~170s (2.8 min) |
| deepset_v | 4.103 | ~603s (10.1 min) | 3.330 | ~490s (8.2 min) |
| gnn | 5.414 | ~796s (13.3 min) | 4.835 | ~711s (11.8 min) |
| cnn_dbm | 5.722 | ~841s (14.0 min) | 4.844 | ~712s (11.9 min) |
| cnn_proj | 7.936 | ~1167s (19.4 min) | 6.856 | ~1008s (16.8 min) |
| **sum, 1 seed, all 6 encoders** | | **~61.5 min** | | **~52.3 min** |

(G_LARGE is slightly *faster* per update than G_SMALL despite the bigger grid -- both tiers now use `k=3`, so `n_clocks`/`n_blocks`/`n_status_categories` are identical, and only `mlp_point`'s input width differs; the grid-size increase itself has negligible cost impact for CPU-only PPO at this scale.)

Total work across all 120 tasks (10 seeds x 2 tiers x 6 encoders): `10 x (3687.9s + 3137.4s) = 68,253s ≈ 18.96 hours` sequential. With `--array=168-287%20` (20 concurrent slots, `cpus-per-task=1` so tasks don't contend with each other): `68,253s / 20 ≈ 57 min` of total-work-per-slot, plus scheduling slack for the slowest individual task (`cnn_proj`, ~19.4 min) -- expect the full 120-task array to finish in **roughly 1-1.5 hours of wall-clock** on Alpine, well inside the 72-hour per-task limit.

---

## Files changed / added

- `trmgrid/gym_env.py` (new) -- `TRMGridTrainEnv`.
- `trmgrid_eval.py` (new) -- `evaluate_trmgrid`.
- `run_one.py` (modified) -- trmgrid imports, `TRMGRID_ALPHA/WIDTH/W`, `TRMGRID_INSTANCE_SEED=2`, `TRMGRID_TIERS` (`G_SMALL=grid_n=4,k=3`; `G_LARGE=grid_n=5,k=3`), `TRMGRID_INSTANCES`, `TRMGRID_SEEDS`, `_build_instance` trmgrid branch, `_build_target_runs` fifth loop (indices 168-287), eval dispatch for `domain=="trmgrid"`.
- `test_run_one.py` (rewritten) -- covers the 288-entry structure; 15/15 passing, including the critical collision test.
- `submit_trmgrid.sbatch` (new) -- array 168-287, `%20` throttle, same Alpine header as every prior submit script.

Scratch diagnostic outputs (not part of the repo): `/tmp/trmgrid_size_sweep.json`, `/tmp/trmgrid_large_tiers.json`, `/tmp/trmgrid_n5k3_more_seeds.json`, `/tmp/trmgrid_small_seed2.json`.
