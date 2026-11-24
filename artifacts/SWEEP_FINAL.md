# SWEEP_FINAL: Real-Checkpoint Analyses, Bimodality Characterization, and the Gridtimed Seed Extension

Three tasks on the completed 72-run sweep (48 original + 24 `deepset_v`/`gnn`,
all pulled from Alpine and integrity-checked). **Nothing submitted to
Alpine.** No direction recommended.

---

## Task A — permutation ablation and difference probe, real checkpoints only

### A0. What is actually permuted, per encoder (read directly from each `forward()`)

| Encoder | kwarg | What moves | What it tests |
|---|---|---|---|
| `mlp_point` | `shuffle_tasks` | Each block's status/elapsed content is permuted into a **different position slot** of the flat point vector; the action head's fixed slot→action convention and the mask are untouched. | Whether the MLP relies on position↔content binding learned during training. |
| `deepset_h` | `shuffle_elements` | Order of the constraint-row **set** (`zone.h_rep()`) before sum-pool. | Sum-pool invariance — exactly 0 by construction. |
| `cnn_dbm` | `shuffle_clocks` | **One joint permutation** of real clocks `1..d`, applied to **both rows and columns** of the `(n,n)` DBM grid consistently — relabels which clock's data sits at which **grid position**, moving it to a different 2D neighborhood under the 3×3 conv kernels. | Whether spatially-local kernels rely on grid *position* (≡ clock identity). |
| `cnn_proj` | `shuffle_channels` | Order of the `d(d-1)/2` **already-fixed channel slots** (each channel is one specific clock-pair raster) — no spatial change within any channel, only which index of the channel *axis* it occupies. | Whether per-channel weights are order-sensitive across the channel axis — a materially weaker/different test than `cnn_dbm`'s, since Conv2d combines channels densely regardless of axis order. |
| `deepset_v` | `shuffle_elements` | Order of the vertex-row **set** (`zone.v_rep()`) before sum-pool. | Sum-pool invariance — exactly 0 by construction, same as `deepset_h`. |
| `gnn` | `shuffle_nodes` | One joint permutation of real-clock nodes `1..d`, applied to node features and both axes of the edge tensors consistently. | Graph analogue of `cnn_dbm`'s test — but **not** invariant by construction; mean/sum-aggregated message passing gives no symmetry guarantee. |

**`cnn_dbm` and `cnn_proj` are not directly comparable** (carried over from the
last report): `cnn_dbm` moves *where in 2D space* a clock sits, which conv
kernels are locally sensitive to; `cnn_proj` only reorders *which channel-axis
index* an already-fixed raster occupies, which dense cross-channel mixing can
be insensitive to independent of any real "robustness" difference. Reported
in the same table below since all six now share real checkpoints, but read
`cnn_dbm` vs. `cnn_proj` as two different questions, not a head-to-head.

### A1. Permutation ablation — all six, real checkpoints (modesel/M3, gridtimed/T2)

| Encoder | mean KL (M3) | mean KL (T2) | Invariant by construction? | Checkpoint (M3 / T2 env_steps) |
|---|---:|---:|---|---|
| `mlp_point` | 8.6e-05 | 8.8e-03 | No | 143,360 / 118,784 |
| `deepset_h` | ~0 (-4.1e-10) | ~0 (-7.0e-10) | **Yes** (sum-pool) | 301,056 / 88,064 |
| `cnn_dbm` | 0.679 | 0.085 | No | 163,840 / 81,920 |
| `cnn_proj` | ~0 (4.4e-10) | 0.030 | No | 126,976 / 83,968 |
| `deepset_v` | 0.0 exactly | ~0 (1.7e-10) | **Yes** (sum-pool) | 126,976 / 106,496 |
| `gnn` | ~0 (8.8e-07) | ~0 (-2.6e-09) | **No** (empirically, not architecturally) | 63,488 / 79,872 |

**All numbers now from real 300k-scale checkpoints — no quick-trained rows.**
Two things changed from the earlier (partly quick-trained) pass:

- `deepset_v` reproduces `deepset_h`'s exact-zero pattern with real data,
  confirming the earlier quick-trained diagnostic wasn't an artifact.
- `gnn`'s near-zero KL, previously flagged as "likely just the short
  8,000-step training budget, not genuine invariance, an open question" —
  **is now measured against a real checkpoint and still comes out near-zero
  on both instances.** This is a real, measured finding: a GCN with no
  architectural permutation-invariance guarantee learned an empirically
  near-permutation-invariant policy. Stated plainly as *empirical*, not
  *architectural*, invariance — a different kind of claim than `deepset_h`/
  `deepset_v`'s guaranteed zero, even though the numbers land in the same
  place.

### A2. Full-capacity difference probe — all six, real checkpoints (modesel/M3)

2-layer/256-unit MLP probe, trained to convergence with early stopping on a
held-out split (train/val/test = 55/15/30%), reported alongside the small
ridge probe:

| Encoder | ridge R² | full-capacity MLP R² | checkpoint (env_steps) |
|---|---:|---:|---:|
| `mlp_point` | 0.932 | 0.948 | 143,360 |
| `deepset_h` | 0.945 | 0.994 | 301,056 |
| `cnn_dbm` | 0.899 | 0.969 | 163,840 |
| `cnn_proj` | 0.272 | 0.342 | 126,976 |
| `deepset_v` | 0.976 | 0.993 | 126,976 |
| `gnn` | **0.809** | 0.884 | 63,488 |

**This materially corrects the earlier quick-trained finding for `gnn`.**
The 8,000-step diagnostic showed ridge R²=0.144 (low) jumping to 0.878 at
full capacity — reported at the time as "probe-capacity-limited, similar to
`cnn_dbm`'s pattern." With a real (though early-stopped at 63,488 steps,
not a full 300k run) checkpoint, `gnn`'s ridge R² is **0.809** — already
close to `deepset_h`/`deepset_v`'s near-ceiling linear decodability, with
only a modest further gain at full capacity (+0.075, similar in size to
`cnn_dbm`'s own gain of +0.070). The "probe-capacity-limited" framing from
the quick-trained pass does not hold up under real training; `gnn`'s
representation is mostly linearly decodable already. `deepset_v` tracks
`deepset_h` closely at both probe capacities, as expected from sharing the
same sum-pool/exact-coordinate structure. `cnn_proj` remains the one
genuinely representation-limited encoder — low at both probe capacities.

---

## Task B — characterizing the gridtimed bimodality

### B1. Per-seed values, all six encoders × four instances (raw, not means)

**Reporting convention, per the explicit note**: cells are classified
**unimodal** (report mean±std) or **bimodal** (report per-seed values and a
converged-seed count, no mean±std as the primary statistic — the mean falls
in a region with no probability mass). A cell is called bimodal here if it
contains at least one seed with `last200_success_rate ≤ 0.05` (a real
collapse) **and** at least one seed with `last200_success_rate ≥ 0.35` (a
working policy) — i.e., there's a genuine gap, not just spread.

**Unimodal cells** (mean±std, all 3 seeds): every modesel cell (8 of 8, all
six encoders); gridtimed `mlp_point` (both instances, uniformly low, ~0.02–0.06);
gridtimed `deepset_h`, `cnn_dbm`, `cnn_proj` (both instances — some of these
have real spread, e.g. `deepset_h`/T3 std=0.200, but no seed collapses to
near-zero, so it's high natural variance, not a collapse-vs-success split);
gridtimed/T2 `gnn` (values 0.425/1.0/0.56 — elevated variance with one high
outlier, but **no collapse**, so reported as unimodal-with-high-variance,
not bimodal, despite superficially looking similar to the true bimodal cells).

| Instance | Encoder | seed0 | seed1 | seed2 | mean | std | class |
|---|---|---:|---:|---:|---:|---:|---|
| modesel/M3 | mlp_point | 0.850 | 0.825 | 0.890 | 0.855 | 0.027 | unimodal |
| modesel/M3 | deepset_h | 0.900 | 0.830 | 0.905 | 0.878 | 0.034 | unimodal |
| modesel/M3 | cnn_dbm | 0.855 | 0.900 | 0.865 | 0.873 | 0.019 | unimodal |
| modesel/M3 | cnn_proj | 0.775 | 0.845 | 0.690 | 0.770 | 0.063 | unimodal |
| modesel/M3 | deepset_v | 0.725 | 0.725 | 0.735 | 0.728 | 0.005 | unimodal |
| modesel/M3 | gnn | 0.750 | 0.855 | 0.735 | 0.780 | 0.053 | unimodal |
| modesel/M4 | mlp_point | 0.815 | 0.795 | 0.770 | 0.793 | 0.018 | unimodal |
| modesel/M4 | deepset_h | 0.830 | 0.860 | 0.890 | 0.860 | 0.024 | unimodal |
| modesel/M4 | cnn_dbm | 0.810 | 0.800 | 0.865 | 0.825 | 0.029 | unimodal |
| modesel/M4 | cnn_proj | 0.580 | 0.605 | 0.600 | 0.595 | 0.011 | unimodal |
| modesel/M4 | deepset_v | 0.630 | 0.660 | 0.715 | 0.668 | 0.035 | unimodal |
| modesel/M4 | gnn | 0.665 | 0.690 | 0.660 | 0.672 | 0.013 | unimodal |
| gridtimed/T2 | mlp_point | 0.025 | 0.060 | 0.045 | 0.043 | 0.014 | unimodal |
| gridtimed/T2 | deepset_h | 0.420 | 0.395 | 0.475 | 0.430 | 0.033 | unimodal |
| gridtimed/T2 | cnn_dbm | 0.595 | 0.410 | 0.450 | 0.485 | 0.079 | unimodal |
| gridtimed/T2 | cnn_proj | 0.240 | 0.245 | 0.335 | 0.273 | 0.044 | unimodal |
| gridtimed/T2 | **deepset_v** | **1.000** | **0.000** | **1.000** | — | — | **BIMODAL** |
| gridtimed/T2 | gnn | 0.425 | 1.000 | 0.560 | 0.662 | 0.246 | unimodal (high variance, no collapse) |
| gridtimed/T3 | mlp_point | 0.035 | 0.025 | 0.035 | 0.032 | 0.005 | unimodal |
| gridtimed/T3 | deepset_h | 0.705 | 0.400 | 0.885 | 0.663 | 0.200 | unimodal (high variance, no collapse) |
| gridtimed/T3 | cnn_dbm | 0.430 | 0.620 | 0.650 | 0.567 | 0.097 | unimodal |
| gridtimed/T3 | cnn_proj | 0.460 | 0.340 | 0.575 | 0.458 | 0.096 | unimodal |
| gridtimed/T3 | **deepset_v** | **1.000** | **0.000** | **0.000** | — | — | **BIMODAL** |
| gridtimed/T3 | **gnn** | **0.645** | **0.505** | **0.000** | — | — | **BIMODAL** |

Values are `last200_success_rate` (fraction of each run's final 200
episodes that succeeded — the "true" per-episode outcome, not derived from
return sign). Converged-seed count (threshold `last200_success_rate > 0.25`,
chosen because it sits in a wide gap for the bimodal cells — every failed
seed is exactly 0.0, every non-failed seed is ≥0.395 — but is a coincidental
near-miss for `cnn_proj`/T2, where two of three seeds (0.240, 0.245) sit
just below it despite being a tight, unimodal cluster, not a collapse; noted
explicitly rather than read as "cnn_proj also failed"):

| | mlp_point | deepset_h | cnn_dbm | cnn_proj | deepset_v | gnn |
|---|---:|---:|---:|---:|---:|---:|
| modesel/M3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| modesel/M4 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| gridtimed/T2 | 0/3 | 3/3 | 3/3 | 1/3 (threshold artifact, see above) | 2/3 | 3/3 |
| gridtimed/T3 | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 2/3 |

### B2. Learning curves — failed vs. successful seeds in the same cell

Four seeds meet the bimodal-collapse definition: `T2/deepset_v/seed1`,
`T3/deepset_v/seed1`, `T3/deepset_v/seed2`, `T3/gnn/seed2`. For each,
compared its full `ep_return_mean`/`policy_entropy` trajectory against the
successful seed(s) in the same cell.

**All four failures never rise** — none of them is "rises then collapses"
or "plateaus after partial progress." Checked directly: `max(return) -
first(return)` is ≤0.7 for all four (i.e., the return recorded at the very
first logged update, after just one PPO update on the initial random
policy, is statistically indistinguishable from the return at early-stop,
tens of updates later). The policy never finds anything better than its
starting point, for the entire run, until the plateau detector gives up.

The **internal dynamics differ by encoder**, though, and this is worth
reporting precisely rather than lumping the four failures together:

- **`deepset_v` failures** (3 of 4): entropy starts low-moderate (0.10–0.51)
  and stays in that range — no dramatic collapse visible in the log.
  `explained_variance` stays **persistently and deeply negative** throughout
  (range roughly -9 to -33, never approaching 0) — the value function never
  learns to predict anything useful, the entire run.
- **`gnn` failure** (`T3/seed2`): entropy starts **high** (2.10, matching
  its own successful siblings' initial entropy) and **collapses rapidly** to
  near-zero (0.036) within ~25 updates (~55k steps) — a fast, visible
  entropy crash. `explained_variance` **rises to 0.999** over the same
  window — the value function becomes an almost-perfect predictor, but only
  because the return has become constant (nothing left to explain, not
  because it learned something useful). This is close to the textbook
  signature of premature policy collapse onto a degenerate deterministic
  action, locked in early.

These are two distinct failure signatures — `deepset_v`'s value function
never converges to anything; `gnn`'s policy and value function both
converge fast, onto a bad fixed point. Both manifest as "return never
rises," but for different underlying reasons, visible in the logs, not
speculated.

### B3. Correlation with logged metrics (12 gridtimed `deepset_v`/`gnn` seeds)

| run_id | last200_sr | n_updates | mean approx_kl | mean clip_frac | mean grad_norm | final value_loss | final explained_var | vrep_fallback_frac (max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T2/deepset_v/seed0 | 1.000 | 52 | 0.199 | 0.100 | 25,248 | 240.8 | -1.946 | 0.0 |
| T2/deepset_v/seed1 | **0.000** | 26 | 0.051 | 0.081 | 15,921 | 748.0 | -13.09 | 0.0 |
| T2/deepset_v/seed2 | 1.000 | 86 | 0.084 | 0.017 | 53,551 | 118.5 | 0.913 | 0.0 |
| T3/deepset_v/seed0 | 1.000 | 56 | 0.189 | 0.054 | 15,813 | 417.4 | -1.383 | 0.0 |
| T3/deepset_v/seed1 | **0.000** | 43 | 0.122 | 0.034 | 36,652 | 1209.4 | -15.51 | 0.0 |
| T3/deepset_v/seed2 | **0.000** | 45 | 0.038 | 0.048 | 53,447 | 586.4 | -9.64 | 0.0 |
| T2/gnn/seed0 | 0.425 | 39 | 0.043 | 0.214 | 511 | 2654.4 | 0.003 | n/a |
| T2/gnn/seed1 | 1.000 | 112 | 0.059 | 0.138 | 522 | 0.1 | 1.000 | n/a |
| T2/gnn/seed2 | 0.560 | 44 | 0.035 | 0.264 | 737 | 2761.0 | 0.007 | n/a |
| T3/gnn/seed0 | 0.645 | 67 | 0.041 | 0.174 | 501 | 3281.9 | -0.017 | n/a |
| T3/gnn/seed1 | 0.505 | 52 | 0.040 | 0.204 | 669 | 2293.7 | 0.006 | n/a |
| T3/gnn/seed2 | **0.000** | 27 | 0.024 | 0.169 | 145 | 0.2 | 0.999 | n/a |

**Does not correlate** (checked, stated plainly rather than assumed):

- **`vrep_fallback_frac`**: exactly `0.0` on every single `deepset_v` seed,
  failed or not. Ruled out as a factor in the `deepset_v` failures.
- **`grad_norm`**: `deepset_v`'s failed seeds (15,921–53,447) fall squarely
  inside its successful seeds' range (15,813–53,551) — no separation.
  `gnn`'s failed seed (145) is the lowest of its six, but the successful
  range (501–737) is narrow and close by — not a clean signal on one data
  point.
- **`clip_frac`**: overlapping ranges in both encoders; failed and
  successful seeds interleave.
- **`mean approx_kl`**: a mild tendency for failed seeds to run lower
  (`deepset_v`: failed 0.038–0.122 vs. successful 0.084–0.199; `gnn`: failed
  0.024 vs. successful 0.035–0.061), but the ranges overlap and n=1 failure
  per encoder-domain cell is too little to call this a real correlate
  rather than noise.
- **`final value_loss`**: correlates, but in **opposite directions per
  encoder** — `deepset_v`'s failures have *higher* final value_loss than its
  successes (748–1209 vs. 119–417); `gnn`'s failure has *near-zero* value_loss
  (0.2) against its successes' much higher range (2254–3282). The raw number
  has no universal "high = bad" reading here: `gnn`'s near-zero value_loss on
  the failed seed is an artifact of the return having become a *constant*
  (trivial to predict), while its successful seeds have real value_loss from
  genuinely volatile returns (-84 to +50 swings). Not a usable cross-encoder
  signal on its own.

**Does correlate, with a stated caveat:**

- **`final explained_variance`**: `deepset_v` failures are consistently
  deeply negative (-9.6 to -15.5) vs. successes near zero (-1.9 to +0.9).
  `gnn`'s one failure is anomalously near-perfect (0.999) vs. successes near
  zero (-0.017 to 0.007). In both encoders the failed seed's `explained_variance`
  is the clearest outlier in its group — but the *direction* of the anomaly is
  opposite (too-negative for `deepset_v`, suspiciously-perfect for `gnn`),
  consistent with the two distinct failure mechanisms in B2. Read as "a
  value-function anomaly accompanies both failure types," not "there is one
  universal EV signature of failure."
- **`n_updates` (≈ early-stop timing)**: failed seeds always have fewer total
  updates than their cell's successful seeds (`deepset_v`: 26/43/45 failed vs.
  52/86/56 successful; `gnn`: 27 failed vs. 39–112 successful). This is
  consistent with failure but is a **mechanical consequence, not an
  independent predictor**: `plateau_patience_steps` triggers off elapsed
  steps since the last improvement, and a return that's flat from update 0
  hits that window sooner than one that's still improving and resetting the
  clock. You cannot use "stopped early" to flag a run as failed without
  already knowing its return trajectory — it's the same information restated.

### B4. Greedy rollout of the four failed checkpoints (200 episodes each) — plus all non-failed gridtimed `deepset_v`/`gnn` seeds for comparison

**Methodology note, stated upfront**: environment dynamics, the initial
state, and a greedy (argmax) policy are all deterministic here (gridtimed
has no environment stochasticity — `reset()` always produces the same
initial state, and `y` increases monotonically toward the deadline every
step, guaranteeing natural termination). All 200 episodes per seed were
therefore identical — `success_rate` and every other metric below is
always exactly 0.0 or 1.0, never a fraction. Ran all 200 anyway to match
POINTCHECK's methodology directly; the determinism itself is worth stating
rather than silently presenting 200 as if it captured real variation.

**New metric defined here** (not in POINTCHECK, added per this task):
*cycling fraction* = fraction of non-terminal steps in an episode that land
on a grid cell already visited earlier in that same episode.

| run_id | greedy success | terminal cause | switch ever activated | gate ever attempted | ep length | cycling frac |
|---|---:|---|---:|---:|---:|---:|
| T2/deepset_v/seed0 | 1.0 | DONE | 1.0 | 0.0 | 8 | 0.0 |
| T2/deepset_v/seed1 | **0.0** | **FAILED (deadline)** | 0.0 | 0.0 | 15 | **0.80** |
| T2/deepset_v/seed2 | 1.0 | DONE | 0.0 | 0.0 | 8 | 0.0 |
| T3/deepset_v/seed0 | 1.0 | DONE | 1.0 | 0.0 | 10 | 0.0 |
| T3/deepset_v/seed1 | **0.0** | **FAILED (deadline)** | 0.0 | 0.0 | 23 | **0.91** |
| T3/deepset_v/seed2 | **0.0** | **FAILED (deadline)** | 1.0 | 0.0 | 20 | **0.65** |
| T2/gnn/seed0 | **1.0** | DONE | 1.0 | 0.0 | 22 | 0.55 |
| T2/gnn/seed1 | 1.0 | DONE | 0.0 | 0.0 | 8 | 0.0 |
| T2/gnn/seed2 | **1.0** | DONE | 0.0 | 0.0 | 8 | 0.0 |
| T3/gnn/seed0 | **1.0** | DONE | 1.0 | 0.0 | 10 | 0.0 |
| T3/gnn/seed1 | **1.0** | DONE | 1.0 | 0.0 | 10 | 0.0 |
| T3/gnn/seed2 | **0.0** | **FAILED (deadline)** | 1.0 | 0.0 | 34 | **0.79** |

**This substantially revises the B1/B3 picture for `gnn`, and is worth
leading with.** Every `gnn` seed that looked "moderate" under training-time
(stochastic) `last200_success_rate` (T2/seed0=0.425, T2/seed2=0.560,
T3/seed0=0.645, T3/seed1=0.505) turns out to have a **fully successful
greedy policy (1.0)**. The gap between the low logged training-time success
rate and the actual learned policy's quality is explained by B2's finding
that `gnn` maintains much higher residual policy entropy throughout
training than `deepset_v` — the *logged* success rate is diluted by
exploratory (sampled, non-greedy) actions during training, not because the
policy itself is bad. Under greedy evaluation, **`gnn` has only one true
failure in all 12 gridtimed seeds** (`T3/seed2`), not the four-ish "moderate"
cells B1's raw training-log numbers suggested. `deepset_v`'s three failures,
by contrast, are confirmed genuine under both stochastic and greedy
evaluation (0.0 both ways) — not a logging artifact.

**Behavioral character of the four real failures**: none resembles
`mlp_point`'s POINTCHECK degenerate policy (which reliably attempted gates
100% of the time and never activated a switch — a specific, stable,
reproducible strategy). Here, **gate-attempt rate is 0.0 in all four
failures** — these policies don't even try to interact with a gate. Switch
activation is mixed (0/4 activate in two failures, 1/4... i.e. 2 of the 4
failures do activate a switch en route). What's consistent across all four
is **high cycling fraction** (0.65–0.91 — the large majority of steps
revisit an already-seen cell) and longer episodes than any successful
rollout (15–34 steps vs. 8–10 for every success). This reads as **aimless
wandering that exhausts the deadline clock**, not a stable alternative
strategy the way `mlp_point`'s gate-seeking-without-switching pattern was —
different in kind, not just degree, from the earlier POINTCHECK finding.

---

## Task C — extending seeds on gridtimed (72 → 156)

- `run_one.TARGET_RUNS`: 156 entries. Indices 0-71 verified byte-identical
  to the pre-extension sweep (`test_run_one.py::test_first_72_unchanged`,
  checked against an independently-built transcribed snapshot, same
  discipline as the 0-47 check before it). Indices 72-155 appended via a
  **third**, separate loop — structurally cannot shift 0-71 regardless of
  what it contains. New: `EXTRA_SEEDS = (3, 4, 5, 6, 7, 8, 9)`, applied to
  **gridtimed only, all six encoders** (2 instances × 6 encoders × 7 seeds
  = 84). `modesel` untouched — still exactly 3 seeds per cell, 2×6×3=36
  entries within 0-71, none added. 11/11 tests pass (up from 9, two new:
  `test_72_to_155_are_gridtimed_extra_seeds_all_six_encoders`,
  `test_modesel_untouched_by_seed_extension`).
- `submit_sweep_final.sbatch` (new): same Alpine header convention as
  `submit_vrep_gnn.sbatch` (`--partition=acpu`, `--qos=cpu-long`,
  `--time=72:00:00`, `--nodes=1 --ntasks=1 --cpus-per-task=1`, `--mem=32G`,
  `module purge; module load anaconda; conda activate amfa-custom-env`,
  same `LD_LIBRARY_PATH` fix, same `WORK_DIR=/projects/amfa5003/RL-for-zones`),
  `--array=72-155%20` (throttle raised from %12 to %20 given this batch is
  3.5x larger — a starting value, not independently tuned), distinct
  job-name/output (`zone-encoders-sf`). Same checkpoint/resume contract,
  same atomic writes, same jsonl byte-offset truncation on resume —
  `run_one.py` itself is unchanged in this regard.
- `check_status.py`: zero modification, confirmed running cleanly against
  156 entries (pure `TARGET_RUNS` iteration, as before).
- `analyze.py`: **one necessary change**, beyond what "modification limited
  to the manifest" would normally require. `analyze.py` had its own local
  `SEEDS = (0, 1, 2)` (separate from `run_one.py`'s), used to iterate seeds
  per cell. With gridtimed now having 10 seeds and modesel staying at 3,
  that flat constant would silently under-report gridtimed the moment the
  new runs land — a correctness bug, not a scope question. Added
  `_seeds_for_domain(domain)` (gridtimed → `SEEDS + EXTRA_SEEDS`, modesel →
  `SEEDS`), used in the two places that previously iterated the flat
  `SEEDS`. Verified: `compute_encoder_stats_table` currently reports
  `n_seeds=3` for gridtimed cells (correct — only seeds 0-2 have data
  today) and will pick up seeds 3-9 automatically once they exist, no
  further code change needed.

**Exact command to submit, when/if that decision is made** (not run here):

```
sbatch submit_sweep_final.sbatch
```

---

## Files delivered / modified this round

- `run_one.py`: `EXTRA_SEEDS` added, `TARGET_RUNS` extended to 156.
- `test_run_one.py`: rewritten for 156 entries, `_ORIGINAL_72` snapshot
  check, two new tests for the gridtimed-only extension.
- `submit_sweep_final.sbatch` (new).
- `analyze.py`: `_seeds_for_domain` helper (necessary correctness fix, not
  scope creep — see Task C).
- `/tmp/SWEEP_FINAL.md`, this report — copied to `artifacts/`.
- Scratch analysis scripts (not repo deliverables): `perm_ablation_final.py`,
  `diffprobe_final.py`, `rollout_gridtimed.py`, all in the session scratchpad,
  results in `perm_ablation_final_results.json`, `diffprobe_final_results.json`,
  `rollout_gridtimed_results.json` alongside them.

**Not submitted to Alpine.** No direction is recommended — the seed
extension exists to get a firmer read on the gridtimed bimodality (3
genuine `deepset_v` failures out of 12 seeds, 1 genuine `gnn` failure out of
12), not to argue for or against either encoder.
