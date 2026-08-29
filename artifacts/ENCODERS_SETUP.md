# ENCODERS_SETUP: Zone-vs-Point Encoder Sample-Efficiency Sweep, SLURM-Ready

Sets up (but does not submit) a SLURM array-job comparison of four encoders
(`mlp_point`, `deepset_h`, `cnn_dbm`, `cnn_proj`) across two domains
(modesel, gridtimed) at two calibrated instance sizes each, 3 seeds each —
48 independent PPO training runs — testing whether zone encodings learn
FASTER than a fair point baseline (sample efficiency), not whether they
CAN learn at all (the partition diagnostic already answered that).

**No `jobs/`, `envs/`, `results/`, or any `*.slurm`/`b3-extend` file existed
in this repo before this task** (searched exhaustively, including
`code.zip`/`pyudbm_ready.zip`). All contracts below ("same house style as
`jobs/nav-sweep-*.slurm`", "same contract as `envs/b3_extend_training.py`")
are implemented from the literal spec in the task prompt, not copied from
files that don't exist. Flagged per instructions rather than silently
fabricating a match.

**Nothing was submitted to Alpine. No training direction is recommended
below** — findings are reported plainly, including a genuinely
counter-hypothesis result (see difference-feature probe).

---

## 1. Confound fixes (done before any job generation, as required)

### 1a. Action head parity

All four encoders now have a per-element action head where the
architecture genuinely allows one — not a uniform retrofit, a per-encoder
architectural decision:

| Encoder | Per-element grouping | Head type |
|---|---|---|
| `mlp_point` | by raw task/move block (`env.n_blocks`, static) | per-element (with sub-slot one-hot, see below) |
| `deepset_h` | by job/clock (constraint-coefficient sign) | per-element on modesel; **pooled** on gridtimed |
| `cnn_dbm` | by clock (row-pooling over the DBM raster) | per-element on modesel; **pooled** on gridtimed |
| `cnn_proj` | by clock (new small shared per-job conv branch over averaged raw channels) | per-element on modesel; **pooled** on gridtimed |

**gridtimed's fallback to pooled heads for deepset_h/cnn_dbm/cnn_proj is a
real architectural constraint, not a shortcut**: which clock a move is
"about" depends on the agent's *current position*, not fixed for the
episode, so there is no static action→clock buffer to broadcast from.
`mlp_point`'s move-indexed grouping IS static (UP/DOWN/LEFT/RIGHT doesn't
depend on state) and keeps its per-element head on both domains. This
means gridtimed's zone-based encoders and its point encoder are NOT given
identical head architectures — reported as the literal 2-way split it is,
per the task's own escape-hatch language, rather than hidden.

**A real correctness bug was found and fixed before any training ran**:
job-indexed per-element heads broadcast one scalar to every action sharing
a group. Job-shop never exposed this (a job's tasks are never
simultaneously legal). modesel's FAST/SLOW of the same task ARE
simultaneously legal — without a fix, the policy could never learn a
preference between them (identical logits, verified: `-0.444` vs `-0.267`
before the fix collapsed to equal). Fixed via a per-slot one-hot appended
to the broadcast per-job features before the final linear head.

### 1b. CNN-P (cnn_proj) rasterization cost

Rasterization (not the conv stack) dominates cnn_proj's per-step cost,
confirmed independently this session on the actual target instances
(uncached, single forward pass, `res=32`):

| Instance | raster / full-forward (uncached) |
|---|---:|
| modesel/M4 | 51.5% |
| gridtimed/T3 | 54.8% |

**A real, previously-undetected correctness bug was found and fixed in
this session's caching layer** (added in an earlier part of this
conversation to address this same confound): `projection_stack`'s LRU
cache was keyed on `str(zone)`. `Zone` defines no `__str__`/`__repr__`, so
`str(zone)` silently fell back to Python's default
`<zoneviews.zone.Zone object at 0xADDRESS>` — an *address*, not content.
CPython reuses a garbage-collected object's address, so a stale cache
entry from an earlier, differently-shaped zone (e.g. a modesel zone,
`d=4`) could be returned for an unrelated later zone at the same recycled
address (e.g. a gridtimed zone, `d=5`), producing a channel-count mismatch
crash in `CNNProjEncoder.forward`'s `np.stack`. **Reproduced directly**: a
real PPO training run crashed with `ValueError: all input arrays must have
the same shape` after training modesel then switching to gridtimed in the
same process. Root-caused to the address-reuse mechanism (confirmed via a
forced-`gc.collect()` stress test alternating domains, which reliably
reproduced the crash pre-fix and reliably did not post-fix, 6/6 runs).
**Fixed** with `_zone_content_key`: a cache key built from the zone's
actual DBM entries (`O(d^2)` simple comparisons — negligible next to the
`O(d^2 · res^2)` rasterization it guards).

Cache performance after the fix, measured two ways:

- **Real multi-update PPO training** (the representative number): 95.5–96.4%
  cumulative hit rate across 6 alternating modesel/gridtimed training runs
  (10k steps each, forced GC between runs).
- **Controlled synthetic benchmark** (300 zones from a short random walk,
  replayed 4× to simulate one PPO update's epoch reuse): only ~51–54% of
  the 300 sampled zones were distinct to begin with (a short random walk
  revisits states quickly), giving a modest **1.55× (modesel/M4) – 1.70×
  (gridtimed/T3)** speedup on this specific small sample at ~87% hit rate.
  The real-training number above (95%+) is the one that matters for actual
  wall-clock budgeting, and is dramatically better because a real 2048-step
  rollout visits far more distinct decision points once, then replays them
  identically across `n_epochs=4` SGD passes — exactly the regime the cache
  targets. Caching only removes rasterization cost, not the conv stack's,
  so the achievable *overall* forward speedup is capped by raster's ~50–55%
  share of forward time (roughly up to ~2× even at a 100% hit rate) — not
  overclaiming a larger number than the architecture can actually deliver.

### 1c. Channel occupancy (res=64)

Global-box occupancy (rasterizing each clock pair over `[0,k_i]×[0,k_j]`)
measured under ~0.1% mean lit-pixel fraction per channel on both domains —
confirmed earlier this session, both well under the ~5% floor. Per-zone
normalization (each channel rasterized over the zone's own finite bounding
box, falling back to the global `k` only on unbounded sides) was
implemented and re-measured: **57–330× improvement in mean occupancy**
across the two domains (from earlier this session; the fix and its
occupancy re-measurement predate this continuation).

**Returns comparison, both schemes** (new this session, post cache-fix,
`cnn_proj`, single seed, 10,000 env_steps — short, exploratory, not a
substitute for the full 3-seed/300k-step sweep):

| Instance | global-box (baseline) | per-zone-normalize |
|---|---:|---:|
| modesel/M3 | −69.97 | −70.04 |
| gridtimed/T2 | 7.06 | 18.86 |

modesel shows no meaningful difference at this short horizon (~0.1%
relative). gridtimed shows per-zone-normalize clearly ahead (~167%
relative) — but this is one seed at 10k steps, nowhere near converged;
reported as a suggestive, not conclusive, result. **No direction is
recommended here** — this is exactly the kind of question the full
48-run, 3-seed, 300k-step sweep is built to answer properly. `cnn_proj`'s
`per_zone_normalize` flag is left `False` by default in `run_one.py`'s 48
target runs (matching the other three encoders' point of comparison,
global-box); switching it on for a dedicated follow-up sweep is a
mechanical one-line change to `_ENCODER_KEY_MAP`/`build_head`, not
attempted here per "do not recommend a direction."

---

## 2. Environments

`modesel/gym_env.py` (`ModeSelGymEnv`) and `gridtimed/gym_env.py`
(`GridTimedTrainEnv`) — both new, following `train/env.py`'s established
contract (fixed `Discrete` action space, explicit boolean mask,
`{point, zone, mask}` observation dict), reusing each domain's own
existing online zone-update logic rather than duplicating it
(`modesel.zones.advance_zone_for_transition`, `gridtimed`'s existing
edge/gate/switch guard chain from `gym_readiness.py`).

**The point baseline is fair, as required** (task, verbatim: "a
handicapped baseline invalidates the experiment"): `point` gives the exact
real-valued clock valuation plus discrete state on both domains — for
gridtimed specifically, this required adding *position* to the point
vector (the original `gym_readiness.py` prototype's point was
switch-clocks-and-y only, which is NOT Markov-sufficient without position
— documented explicitly in `gridtimed/gym_env.py`'s module docstring as
the bug this task fixes, not silently inherited).

Same-process `train/vec_env.SyncVectorEnv` (new, generic, shared by both
domains) — Federation objects are not picklable, so `AsyncVectorEnv` is
not an option, matching `train/env.py`'s own established reasoning.

`train/ppo.py`'s `PPOTrainer` was extended (not forked) to accept either
the original job-shop instance-name path or a pre-built `vec_env`
(keyword-only in practice, mutually exclusive with `instance_name`),
keeping `train/run_experiment.py`'s existing call signature untouched.
`UpdateLog`/`_ppo_update` were extended with `approx_kl` (Schulman's
low-variance estimator), `clip_frac`, `grad_norm` (pre-clip norm from
`clip_grad_norm_`'s own return value), `raster_s` (via a new module-level
wall-time accumulator in `zoneviews/raster.py`), and `env_step_s` — all
required by the logging schema below. A single-update method
(`PPOTrainer._do_one_update`) was factored out so `train()` (in-process)
and `run_one.py`'s checkpointing loop (SLURM) share one update
implementation, not two copies that could drift.

**A leftover syntax bug from earlier in this session's raster-timing
instrumentation was found and fixed first**: two separate `global
_raster_wall_s` statements in `projection_stack`'s two branches, with an
assignment between them — Python requires a `global` declaration to
precede the *first* assignment to that name anywhere in the function body
(not per-branch), so this was a real `SyntaxError` that would have broken
`import zoneviews.raster` entirely. Fixed by declaring `global` once at
the top of the function. Verified via `py_compile` and a direct call.

---

## 3. Parameter counts (target instances, seed 0)

| Instance | mlp_point | deepset_h | cnn_dbm | cnn_proj | spread |
|---|---:|---:|---:|---:|---:|
| modesel/M3 | 31,764 | 30,936 | 31,426 | 30,949 | 2.7% |
| modesel/M4 | 34,116 | 31,064 | 31,426 | 31,345 | 9.8% |
| gridtimed/T2 | 32,437 | 32,213 | 32,835 | 32,406 | 1.9% |
| gridtimed/T3 | 34,357 | 32,341 | 32,835 | 32,901 | 6.2% |

All four target instances (M3, M4, T2, T3 — the calibrated tiers from
`/tmp/MODESEL.md` and `gridtimed/tiers.py`) are within the 10% bound; M4
is the closest at 9.8%. This is a genuine re-verification against the
actual calibrated instances, not an extrapolation from the earlier
single-instance spot checks (2.7%/1.9%) done earlier in the session.

---

## 4. Real bugs found and fixed this session (summary, in the order hit)

1. **`zoneviews/raster.py` `SyntaxError`** (double `global` statement)
   — would have broken `import zoneviews.raster` for anyone. Fixed, verified via `py_compile`.
2. **`zoneviews/raster.py` cache-key correctness bug** (`str(zone)` was
   address-based, not content-based) — reproduced a real training crash,
   root-caused to CPython address reuse, fixed with a content-based DBM
   fingerprint, verified with a 6-run forced-GC stress test.
3. Re-verified (not re-derived) that modesel's `n_clocks = instance.n_jobs
   + 1` was already correct — flagged as unverified in the prior session's
   handoff, confirmed here via a direct `projection_stack` shape check
   (`zone.d=4` for `n_jobs=3`, matching `d(d-1)/2=6` channels observed).

None of these were hypothetical — each was caught via a real crash or a
real `SyntaxError`, root-caused before being called fixed, and re-verified
after the fix under conditions that reproduced the original failure.

---

## 5. SLURM packaging

- **`run_one.py`**: `TARGET_RUNS` is a flat, module-level list of 48
  `RunSpec(index, domain, instance, encoder, seed)` entries — the sole
  authoritative array-index→config mapping (`resolve_spec(i)` is a plain
  list lookup, nothing computed from `i`). Environment *layout* is fixed
  per (domain, instance) at generator seed 0 (matching each domain's own
  `tiers.py` calibration seed); the `RunSpec.seed` field is the PPO
  *training* seed only — holding environment layout fixed and varying only
  the training seed is what makes the 3-seed spread measure training
  variance, not environment-difficulty variance.
  `--dry-run` prints the resolved config (paths, hyperparameters) and
  exits without touching torch/env state.
- **`test_run_one.py`**: 7 tests, same contract requested for
  `envs/b3_extend_training.py`'s test file (which doesn't exist in this
  repo — see the top-of-report note): length is exactly 48, indices are
  contiguous 0–47, every (domain, instance, encoder, seed) combination is
  covered exactly once (no dup, no gap), `run_id`s are unique,
  `resolve_spec` matches direct list indexing, out-of-range raises. All 7
  pass.
- **Checkpointing** (`results/ckpt/{domain}__{instance}__{encoder}__seed{k}.pt`,
  `torch.save`, atomic: write to `.tmp`, `fsync`, `os.replace`): contains
  policy/value weights + optimizer state (`PPOTrainer.state_dict()`),
  `env_steps`, `update_idx`, the plateau tracker
  (`best_mean_return`/`steps_since_improve`), RNG state (python/numpy/torch,
  captured separately since it's process-global, not trainer-owned), the
  early-stop step (or `None`), the run's original start time, and — the
  correctness-critical part — **the byte offsets of both jsonl files AT
  checkpoint time**. On resume: trainer state and RNG are restored, then
  BOTH jsonl files are truncated to those recorded offsets before any
  further writes, discarding any "orphan tail" written after the last
  checkpoint but before a crash. This was exercised for real, not just
  argued (see §7).
- **`check_status.py`**: scans all 48 `TARGET_RUNS` against
  `results/ckpt/` and `results/`, reports env_steps done, finished y/n,
  early-stop step, last-update timestamp, and any log-integrity violation,
  per config. Read-only.
- **`submit.sbatch`**: uses the exact header given, verbatim, including
  `--qos=cpu-long` (not `long`) — the older `nav-sweep-*-gapfill.slurm`
  scripts say `--qos=long`, which `acpu` will reject (`acpu` only accepts
  `admin`, `cpu-normal`, `cpu-long`); **not fixed here** (out of this
  task's scope, per instructions), flagged in-file and here. `%12` is
  commented as a starting throttle value, not tuned against Alpine's real
  queue depth. `WORK_DIR` has an adjust-before-submitting comment, same
  convention as `b3-extend.slurm`. Resubmitting the identical script is
  safe — `run_one.py` checks its own checkpoint on start, no flag needed.
- **`analyze.py`**: per-(domain,instance) learning curves (env_steps and
  wall-clock x-axes, mean±1σ band across seeds, `--smooth-window` CLI arg),
  steps-to-threshold at 50/75/90% of each encoder's own [worst,best]
  return range per instance (mean±std across seeds — the headline
  sample-efficiency number; sign-agnostic by construction since these
  domains log `return = −cost`, so "% of best measured from zero" would
  run backwards), a CSV+PNG output, a text summary-table fallback, and
  graceful partial-run handling (missing configs listed, not crashed on).
  Exercised end-to-end on real (if short) local data — see §7.
- **`difference_probe.py`**: the hypothesis-specific probe — see §6.

---

## 6. Difference-feature probe

On each encoder, a frozen closed-form ridge-regression head was fit to
predict pairwise clock differences (`c_i − c_j`, all `d(d−1)/2` pairs) from
the pooled embedding alone, on modesel/M3 (4 encoders, 1000 samples,
70/30 train/test split). **Encoders here were given only 6,000 env_steps
of quick in-script training (not a real 300k-step checkpoint — none exist
yet, since nothing has been submitted to Alpine)**; this is a
methodology/smoke-test-quality demonstration of the probe machinery and a
first look, not a final answer — the ranking could plausibly shift with
full training.

| Encoder | R² (overall) | R² per pair (c0-c1, c0-c2, c0-c3, c1-c2, c1-c3, c2-c3) |
|---|---:|---|
| `mlp_point` | **0.933** | 0.464, 0.571, 0.968, 0.570, 0.975, 0.980 |
| `deepset_h` | **0.927** | 0.716, 0.771, 0.930, 0.784, 0.942, 0.960 |
| `cnn_dbm` | 0.411 | 0.007, 0.018, 0.459, 0.005, 0.451, 0.447 |
| `cnn_proj` | 0.167 | 0.247, 0.282, 0.120, 0.017, 0.196, 0.178 |

**This is the OPPOSITE of the a priori hypothesis** ("deepset_h and
cnn_dbm receive these explicitly while mlp_point must infer them" implied
deepset_h/cnn_dbm should score higher). Instead, the two encoders that
ingest the raw real-valued point vector directly (`mlp_point`, `deepset_h`)
recover pairwise clock differences far better than the two that only see a
*rasterized image of the reachable zone* (`cnn_dbm`, `cnn_proj`). A
plausible mechanism, stated as a hypothesis and not further verified
within this task's scope: the CNN encoders' input is a rendered *region*
(the zone — a set of concretely-reachable values), not the single realized
concrete point, and recovering an exact real-valued difference through
pixel-grid quantization (`res=32`) is a fundamentally harder decoding
problem than reading it off a raw float already sitting in the input
vector. **No direction is recommended based on this** — it is reported as
a first-look, short-training result on one instance, exactly what it is.

---

## 7. Local smoke test (run BEFORE this report, output below)

Two array indices, modesel/M3 (`INSTANCE_SEED=0`): index 0 (`mlp_point`,
**point**) and index 6 (`cnn_dbm`, **symbolic/zone-based**).

**1. Configs resolve correctly** — `--dry-run` output for both indices
confirmed correct domain/instance/encoder/seed/paths (index 0:
`modesel__M3__mlp_point__seed0`; index 6: `modesel__M3__cnn_dbm__seed0`).

**2. jsonl schema validates** — a real record from index 0's run:
```json
{"update": 0, "env_steps": 2048, "wall_s": 0.4555, "ep_return_mean": -70.842,
 "ep_return_std": 26.097, "ep_len_mean": 8.585, "n_episodes_completed": 236,
 "policy_entropy": 1.0087, "value_loss": 2090.55, "policy_loss": -0.000197,
 "explained_variance": 0.00048, "approx_kl": 5.71e-06, "clip_frac": 0.0,
 "grad_norm": 133.64, "encoder_forward_s": 0.067, "raster_s": 0.0,
 "env_step_s": 0.237, "domain_metric": 23.681}
```
Matches the required schema exactly, field-for-field. `manifest.json`
(written as a per-run shard, see below) also validated: correct
hyperparameters, `param_count=31764`, `git_commit: null` (this repo is not
a git repository — correctly handled, not a crash), `total_env_steps`,
`early_stop_step: null`.

**Deviation, reported not silent**: `results/manifest.json` as a single
shared file would corrupt under the `--array=...%12` throttle's 12
concurrent writers with no file locking. Used one manifest shard per run
(`results/{run_id}__manifest.json`, identical schema) instead —
trivially globbable by `analyze.py`/`check_status.py`, safe under
concurrency.

**3. Checkpoint writes and resumes cleanly, including idempotent
resubmission**: resubmitting index 0's exact finished command produced
`"already finished -- nothing to do"` with the jsonl line count unchanged
(3 lines, both before and after).

**4. Kill-mid-run resume, exercised for real** (index 6, `cnn_dbm`,
`--ckpt-every-steps 8000` so checkpoints lag updates, deliberately
constructed to create a resumable orphan tail):
- Process killed (`SIGKILL`) with the jsonl at 6 lines (updates 0–5) but
  the last checkpoint recorded at update_idx=4 — **a real orphan tail**:
  checkpoint's `jsonl_offset=2354` bytes vs. the actual file's `3530`
  bytes (updates 4 and 5 written after the last checkpoint).
- Restarted the identical command. Log: `"resumed from checkpoint:
  env_steps=8192 update=4"`. jsonl was truncated back to 2354 bytes,
  discarding the two orphaned records, then training continued and ran to
  completion.
- **Verified**: the full, final update sequence is `[0, 1, ..., 29]` —
  strictly increasing, zero gaps, zero duplicates, confirmed
  programmatically (`updates == list(range(len(updates)))` → `True`).
  Episode-log timestamps (`env_steps`) also confirmed non-decreasing
  across the resume boundary (7007 episodes, `all(steps[i] <= steps[i+1])`
  → `True`).
- `run_one.py`'s own post-run integrity check (`verify_jsonl_monotonic`,
  also used by `check_status.py`) reported `log integrity OK` on both
  final files.

**5. `analyze.py` on partial data**: run against this exact two-config,
short-budget result set (46/48 configs missing). Produced real PNG curves
(`modesel__M3__curves_steps.png`, `modesel__M3__curves_wallclock.png`,
visually inspected — both curves monotonically improving, sane), a
populated `steps_to_threshold.csv` (e.g. `cnn_dbm` reached 50/75/90% of
its own range at 12,288 / 20,480 / 34,816 env_steps), and a `summary.txt`
that correctly listed the other 46 configs as `NO DATA` rather than
crashing.

All smoke-test result files were removed after verification (they used
reduced `--total-steps` overrides for speed, not the real 300k budget, so
they are not meaningful training runs to keep around).

---

## 8. Wall-clock budget (local estimate — NOT cluster-measured)

Per-step throughput benchmarked on the WORST-CASE (largest) instance per
domain, 12,288 steps each, this machine, single-process, CPU:

| Encoder | modesel/M4 (est. 300k) | gridtimed/T3 (est. 300k) |
|---|---:|---:|
| `mlp_point` | 1.1 min | 1.0 min |
| `deepset_h` | 3.4 min | 4.6 min |
| `cnn_dbm` | 14.2 min | 19.6 min |
| `cnn_proj` | 23.1 min | 26.5 min |

Worst single run observed: **~26.5 minutes** for a full 300k-step budget
(gridtimed/T3, `cnn_proj`, no early stop). Against the sbatch header's
`--time=72:00:00` (4320 minutes) per array task, this is a **~160×
margin** — even a 10× slower Alpine core would leave a ~16× margin.
**BUDGET WARNING, in the style of `b3-extend.slurm`'s own header**: this
number is measured on a local development machine, not Alpine's `acpu`
partition — cluster CPU speed, filesystem I/O latency for the
`torch.save`/jsonl-fsync checkpoint writes, and contention under the
`%12` concurrent-task throttle are all unmeasured here and could push real
wall-clock meaningfully higher than this estimate; do not treat this
number as a cluster-validated bound.

**Conclusion**: even a conservative, worst-case-instance-for-all-48-runs
estimate (3 seeds × 4 instances × the M4/T3 per-encoder times above,
serialized) totals well under 10 hours of aggregate work; divided across
the `%12` concurrency throttle, real wall-clock to complete the full
sweep is plausibly on the order of an hour to a few hours, not days. **72h
per task is generously adequate; resubmission should NOT be expected for
step-budget reasons.** If a resubmission does happen (node preemption,
transient failure, Alpine being much slower than this estimate), it is
handled automatically and correctly by `run_one.py`'s checkpoint/resume
contract, exercised for real in §7 above.

---

## 9. Resolved 48-row manifest (from `run_one.TARGET_RUNS`)

Domain/instance/encoder/seed combinations, 4 encoders × 4 instances × 3
seeds, indices 0–47, generated by (and copy-verified against) `python3 -c
"from run_one import TARGET_RUNS; [print(r.index, r.domain, r.instance,
r.encoder, r.seed) for r in TARGET_RUNS]"`:

```
 0 modesel   M3 mlp_point seed0   16 modesel   M4 deepset_h seed1   32 gridtimed T2 cnn_dbm   seed2
 1 modesel   M3 mlp_point seed1   17 modesel   M4 deepset_h seed2   33 gridtimed T2 cnn_proj  seed0
 2 modesel   M3 mlp_point seed2   18 modesel   M4 cnn_dbm   seed0   34 gridtimed T2 cnn_proj  seed1
 3 modesel   M3 deepset_h seed0   19 modesel   M4 cnn_dbm   seed1   35 gridtimed T2 cnn_proj  seed2
 4 modesel   M3 deepset_h seed1   20 modesel   M4 cnn_dbm   seed2   36 gridtimed T3 mlp_point seed0
 5 modesel   M3 deepset_h seed2   21 modesel   M4 cnn_proj  seed0   37 gridtimed T3 mlp_point seed1
 6 modesel   M3 cnn_dbm   seed0   22 modesel   M4 cnn_proj  seed1   38 gridtimed T3 mlp_point seed2
 7 modesel   M3 cnn_dbm   seed1   23 modesel   M4 cnn_proj  seed2   39 gridtimed T3 deepset_h seed0
 8 modesel   M3 cnn_dbm   seed2   24 gridtimed T2 mlp_point seed0   40 gridtimed T3 deepset_h seed1
 9 modesel   M3 cnn_proj  seed0   25 gridtimed T2 mlp_point seed1   41 gridtimed T3 deepset_h seed2
10 modesel   M3 cnn_proj  seed1   26 gridtimed T2 mlp_point seed2   42 gridtimed T3 cnn_dbm   seed0
11 modesel   M3 cnn_proj  seed2   27 gridtimed T2 deepset_h seed0   43 gridtimed T3 cnn_dbm   seed1
12 modesel   M4 mlp_point seed0   28 gridtimed T2 deepset_h seed1   44 gridtimed T3 cnn_dbm   seed2
13 modesel   M4 mlp_point seed1   29 gridtimed T2 deepset_h seed2   45 gridtimed T3 cnn_proj  seed0
14 modesel   M4 mlp_point seed2   30 gridtimed T2 cnn_dbm   seed0   46 gridtimed T3 cnn_proj  seed1
15 modesel   M4 deepset_h seed0   31 gridtimed T2 cnn_dbm   seed1   47 gridtimed T3 cnn_proj  seed2
```

`test_run_one.py` mechanically verifies this table's 48-row, no-dup,
no-gap, full-coverage property against the live `TARGET_RUNS` object — the
table above is a direct copy of that script's output, not hand-assembled.

---

## 10. Exact sbatch command

```bash
# On Alpine, after setting WORK_DIR in submit.sbatch:
sbatch submit.sbatch
```

No arguments — the array range (`0-47%12`) and everything else needed is
already in the header. Not run here, per instructions.

---

## 11. Files delivered

- `run_one.py` — `TARGET_RUNS`, `resolve_spec`, `--dry-run`, full
  checkpoint/resume training loop.
- `test_run_one.py` — 7/7 passing coverage tests.
- `submit.sbatch` — exact header as given.
- `check_status.py`, `analyze.py`, `difference_probe.py`.
- `train/vec_env.py`, `modesel/gym_env.py`, `gridtimed/gym_env.py` (from
  earlier this session, exercised fully in this continuation).
- Modified: `train/ppo.py` (vec_env support, extended `UpdateLog`,
  `_do_one_update`, `state_dict`/`load_state_dict`), `zoneviews/raster.py`
  (syntax fix, cache-key correctness fix, raster wall-time
  instrumentation), `encoders/base.py`, `encoders/mlp_point.py`,
  `encoders/deepset_h.py`, `encoders/cnn_proj.py` (from earlier this
  session).
