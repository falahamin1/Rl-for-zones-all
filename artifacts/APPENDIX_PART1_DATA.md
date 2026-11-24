# Appendix data report — sourced, with citations

Companion to `figures/part1_curves.pdf`, `figures/part1_curves_appendix.pdf`,
`figures/part1_per_seed.pdf`. Every value below is either a `file:line`
citation, `(resolved by execution)`, or `UNKNOWN`. Nothing is fabricated or
interpolated. Machine-readable companion: `artifacts/appendix_part1_data.json`.

**Read this first — one finding contradicts the paper's current claim and
needs a decision, not a workaround (item 6 below).**

---

## Flagged up front

### Item 6: GridTimed T2 and T3 ARE solvable without entering any gate — contradicts the stated claim

The brief states: *"The paper states that the baseline, which never
presses a switch, never reaches a state from which the task can be
completed; this depends on the answer being no."* **The answer is yes, on
both instances, with slack to spare:**

| Instance | reachable non-gate cells | GOAL reachable avoiding all gates | shortest gate-free path (delay=0 throughout) | deadline D | slack |
|---|---:|---|---:|---:|---:|
| T2 | 22 of 25 | **True** | 8 steps | 27 | 19 steps |
| T3 | 32 of 36 | **True** | 10 steps | 33 | 23 steps |

(resolved by execution — grid BFS with gate cells removed as walls,
against `run_one._build_instance("gridtimed", "T2"/"T3")`'s actual
deployed instance, `INSTANCE_SEED=0`; shortest-path length computed the
same way, and since every grid move's minimum cost is `elapsed=delay+1`
with `delay=0` available (`gridtimed/instance.py:74`,
`DEFAULT_DELAYS=(0,1,2)`), path length in cells equals minimum possible
`y` cost.)

This means a policy that never activates a switch is **not** physically
barred from reaching GOAL — there is a short detour around every gate on
both instances, well inside the deadline. The actual reason `mlp_point`
never succeeds (see item 7) is that its greedy policy cycles between two
cells instead of ever finding this detour or the gated shortcut — a
behavioral failure, not a topological impossibility. The paper's claim as
stated is incorrect and needs rewording (something like "never finds a
path to GOAL," not "never reaches a state from which the task can be
completed") — I did not rewrite it myself, since that's a paper-content
decision, not a data question.

### Item 5: the 76,354 figure is not a complete reachable-state count — provenance unresolved

`artifacts/POINTCHECK.md:27` reports gridtimed/T2 "usable states" = 76,354
= distinct point vectors (used to claim `point` is injective). I attempted
to reproduce a full concrete-state reachability count independently and
could not, within tractable time, get anywhere close to a stable number
at or below 76,354 — the true count appears to be substantially larger:

- `gridtimed.dp.explore_and_solve(inst, state_limit=200_000)`
  (`gridtimed/dp.py:54`, the codebase's own existing, already-optimized
  BFS — reused directly, not reimplemented) on T2: **107,543 states fully
  processed in 10.00s, and still `truncated=True`** — the frontier
  (`seen`) exceeded 200,000 before the queue drained. Raising
  `state_limit` to 800,000 did not finish within 100s. (resolved by
  execution)
- `artifacts/GRIDSOUND.md:186` independently reports T2's exploration
  discovering ~80,819 states (also truncated, `state_limit=150,000` in
  that report's convention) — consistent with mine (107,543 > 80,819,
  same direction, different limit/build), **also larger than 76,354**.

So two independent runs of the codebase's own reachability BFS both
process *more* than 76,354 states before truncating, meaning 76,354 is
neither the full state count nor an intermediate count either of these
runs passed through and stopped at. **I could not determine what
enumeration in `POINTCHECK.md` actually produced 76,354** — no script
backing that number is present in the repo, and its own methodology text
("Grouped every DP-reachable, non-terminal, non-deadlock state...") does
not specify a `state_limit` or otherwise explain how it differs from the
BFS above. Marking the provenance of 76,354 **UNKNOWN**, and flagging
`POINTCHECK.md §1`'s "point is injective" conclusion as **checked against
an incomplete/unidentified sample, not the true reachable state space** —
that conclusion should not be cited as exhaustive without re-deriving it
properly (e.g., rerun the DP with a state_limit large enough to actually
converge, which may need a faster-than-pure-Python implementation given
what's observed here).

---

## Item 1: instance parameters — reconfirmed, not re-derived

No source file changed since `artifacts/SPECS.md` was written (`find
modesel gridtimed trmgrid -newer artifacts/SPECS.md -name "*.py"` returns
nothing). All values in `SPECS.md`'s `GRIDTIMED`/`TRMGRID`/`MODESEL`
sections (clocks, action spaces, deadlines, windows, generator seeds) are
reconfirmed current. See `artifacts/SPECS.md` for the full file:line
citations — not reproduced here to avoid a second copy drifting out of
sync with the first.

## Item 2: hyperparameters, widths, param counts — reconfirmed

Same reconfirmation basis (`train/ppo.py`, `run_one.py` unchanged since
`SPECS.md` §6 was written). Param-count table there already checks the
"matched within 10%" claim per instance (worst case 9.82%, modesel/M4) —
still current.

## Item 3: exploration-off (greedy) success rate — GridTimed reconfirmed, TRMGrid computed fresh

GridTimed's greedy SR already exists via `analyze.py`'s
`greedy_success_rate_mean` (`analysis/encoder_stats.csv`) — the *last
logged in-training eval* (update ~145 of 146), not literally the final
saved checkpoint. TRMGrid's was missing entirely before this pass.

**Computed fresh for all 240 GridTimed+TRMGrid deepset_v/all-encoder
final checkpoints combined below is TRMGrid's; see the note on GridTimed
below** (resolved by execution: numpy shim from
`RL-for-Polytopes/tangram-intermediate/train_single.py` to load
Alpine-written checkpoints locally, `build_trainer(spec)` to reconstruct
the exact architecture, one greedy [argmax] episode per checkpoint —
exact given both env and policy are deterministic, per
`trmgrid_eval.py:5-15`'s own determinism argument, independently verified
by rerunning 5 reps per checkpoint in the item-3 gridtimed cross-check
during the prior session (`artifacts/RESULTS_CHECK.md` §1) and finding
zero variance):

| Instance | mlp_point | deepset_h | cnn_dbm | cnn_proj | deepset_v | gnn |
|---|---:|---:|---:|---:|---:|---:|
| G_SMALL | 0.0 | 0.8 | 0.6 | 0.3 | 0.0 | 0.4 |
| G_LARGE | 0.0 | 0.8 | 0.2 | 0.5 | 0.0 | 0.4 |

n=10 seeds per cell, all checkpoints present, no missing runs. Per-seed
raw booleans in `appendix_part1_data.json` → `trmgrid_greedy`.

**Note, worth flagging**: these fresh-final-checkpoint numbers differ
from the already-logged near-final numbers in `analysis/encoder_stats.csv`
in several cells (e.g. G_LARGE/deepset_h: logged 0.600 vs fresh 0.800;
G_SMALL/cnn_dbm: logged 0.700 vs fresh 0.600) — expected, since these are
single-episode (n=1) measurements one gradient update apart (2048 steps),
and several of these policies are still fluctuating at that granularity.
**Use the fresh final-checkpoint numbers above as authoritative** for
anything citing "the deployed checkpoint's" greedy performance; the CSV's
logged column is a near-final training-time snapshot, not the same thing.
I did not rerun GridTimed's fresh-checkpoint numbers for all 120 runs
(only the 3-config T2 spot-check already in `RESULTS_CHECK.md`) since the
brief states GridTimed's values already exist — say if you want the full
120-run fresh recomputation too, for exact parity with TRMGrid's
methodology above; it's a straightforward extension of the same script.

## Item 4: diagnostics, bucketed soundness — NOT DONE (lowest priority per the brief)

Location-only soundness exists for gridtimed (`artifacts/GRIDSOUND.md`)
and trmgrid (`artifacts/TRMGRID_SWEEP.md`, `TRMGRID_V2.md`). Bucketed
(location + bucketed-global-clock) soundness exists **only** for
modesel/M4 (`artifacts/MODESEL.md:95-99`, slack-bucket b=1/2/4). Computing
the identical bucketing for gridtimed/trmgrid is genuinely new work, not
started — deprioritized per the brief's own ordering ("if time allows").
Given item 5's finding above (T2's exact-DP doesn't converge in
tractable time even unbucketed), bucketed soundness for gridtimed/trmgrid
should be expected to face the same scaling issue and may need the same
"report the timeout, don't approximate" treatment rather than a clean
number.

## Item 5 — see "Flagged up front" above.

## Item 6 — see "Flagged up front" above.

## Item 7: mlp_point failure analysis — extended to all 10 seeds x both instances

Resolves the "0/200 vs. ten seeds x two instances" discrepancy directly:
`POINTCHECK.md §4`'s "200" was **200 *repeated* greedy episodes on the
*same* checkpoint** (T2 only, 3 seeds, predates the seed-extension to 10)
— deterministic env + deterministic policy means all 200 reps are
bit-identical, confirmed there (identical mean episode length across all
200). It is not 200 independent draws. Below is the correctly-scoped
version for the current sweep: **10 seeds x 2 instances x 1 greedy
episode each = 20 independent-checkpoint rollouts** (sufficient given
determinism — running more reps of the same checkpoint adds no
information, as `POINTCHECK.md` itself already demonstrated).

(resolved by execution — same final-checkpoint methodology as item 3,
extended with direct instrumentation of gate-attempt detection via
`instance.is_gate()` on the action's target cell, and switch-activation
detection via `state.clocks[i] == 0 and pos == switches[i]`, both against
`gridtimed/instance.py`'s own `GridInstance` methods, not a
reimplementation of `gridtimed/env.py`'s transition logic):

| Instance | seeds tested | success | episode length | cycling fraction | gate attempts (rate) | switch ever activated |
|---|---:|---:|---:|---:|---:|---:|
| T2 | 10/10 | **0/10** | 27 (all 10, identical) | 0.852 (all 10, identical) | 2 (0.074/step, all 10, identical) | **0/10** |
| T3 | 10/10 | **0/10** | 34 (all 10, identical) | 0.912 (all 10, identical) | 0 (0.000/step, all 10, identical) | **0/10** |

**All 10 independently-trained seeds converge to a bit-identical
degenerate policy, on both instances** — not just consistent across 3
seeds (`POINTCHECK.md`'s original scope) but across all 10, exactly. A
real T2-vs-T3 behavioral difference: on T2 the degenerate policy still
probes a gate twice before settling into cycling; on **T3 it never
attempts a gate at all** (0 attempts, all 10 seeds) — worth noting in the
writeup as a genuine cross-instance difference, not sampling noise (both
figures are exact, not means with variance — every seed produced the
identical count).

## Item 8: ablations on mlp_point — exists, but not as a clean 2-condition split

`artifacts/POINTCHECK.md §5` (`artifacts/POINTCHECK.md:125-139`) is what
exists: **one combined run**, capacity 5.4x (172,117 params vs. ~32k in
the sweep) **and** `entropy_coef=0.05` (5x the sweep's 0.01) applied
**together**, not as two separate ablations. Configuration: gridtimed/T2,
**single seed**, 50,000 env_steps (not the sweep's full 300,000-step
budget). Result: `final_mean_return=-74.24`,
`greedy_rollout_success_rate=0.0`, `greedy_rollout_switch_activation_rate=0.0`
— identical to the unmodified sweep result. **Flagging plainly**: this
does not let you separately attribute the null result to capacity vs.
entropy (confounded), rests on one seed (no variance estimate), and used
1/6 of the sweep's training budget. If the paper needs two clean,
separately-seeded, full-budget ablations, that's new training (12+ runs:
2 conditions x >=3 seeds x 300k steps) — not something I ran here, given
the priority ordering puts this below items 3/5/6/7. Say the word if you
want it queued.

## Item 9: zero-width control — exists, complete

`artifacts/POINTABLATE.md`, "Separate control" section
(`artifacts/POINTABLATE.md:76-94`). Configuration: `deepset_h` fed a
degenerate zone (DBM hull of the single exact concrete point — every
constraint tight, zero width) instead of the real w-jitter-widened zone;
gridtimed/T2; **3 seeds**; full 300,000-step budget (same as the sweep).
Results: **100% greedy success, all 3 seeds**; 52.8% ± 4.4% / 40.2% ±
3.5% stochastic success (all-episodes / last-200) — matching or exceeding
the real wide-zone `deepset_h` from the sweep (50.9% per
`ENCODERS_SETUP.md`, cited in `POINTABLATE.md:32-36`). Rules out zone
width as the source of `deepset_h`'s advantage over `mlp_point`.

## Item 10: cnn_proj information loss — pixel occupancy exists for 4/6 instances; R² now computed for all 6 encoders x 6 instances

**Pixel occupancy at resolution**, `artifacts/PRESUBMIT.md:120-168`
(quoted exactly, not paraphrased, since it's a small table):

| Instance | res=64 | res=128 | res=256 | res=512 | res=1024 |
|---|---:|---:|---:|---:|---:|
| modesel/M3 | 0.115% | 0.106% | 0.102%\* | 0.100% | 0.098% |
| modesel/M4 | 0.087% | 0.081% | 0.040%\* | 0.039% | 0.038% |
| gridtimed/T2 | 0.056% | 0.035% | 0.010%\* | 0.005% | 0.0025% |
| gridtimed/T3 | 0.058% | 0.031% | 0.028%\* | 0.022% | 0.020% |

(\*res=256 column: the 512/1024 sweep used a separate, smaller 25-zone
sample at a different seed than the res=64/128/256 table, per
`PRESUBMIT.md:147-151`'s own explicit caveat — the two res=256 values in
that source aren't from the same sample; I've used the second (512/1024)
sweep's own res=256 for continuity with 512/1024, and it's clearly
labelled that way in the source. None of the four instances reach 10%
occupancy anywhere tested; trend is flat-to-declining, not resolution-
limited.) Per-rasterization cost at res=1024: modesel/M3 0.33s, M4 0.57s,
gridtimed/T2 0.57s, T3 0.88s (`PRESUBMIT.md:168-170`, single-threaded,
same local machine).

**TRMGrid is not covered by this table** — `PRESUBMIT.md` predates
trmgrid's addition to the repo. Genuinely missing, not computed here
(flagging rather than fabricating a plausible-looking number).

**R² probe (pairwise clock-difference decodability), extended to all six
encoders and trmgrid** — `difference_probe.py` previously only covered
modesel+gridtimed x the original 4 encoders (`ENCODERS`, not `+
NEW_ENCODERS`), which is exactly why deepset_v's R² was missing from
every prior note, as you said. Extended: added a `trmgrid` branch to
`concrete_clocks()` (`difference_probe.py`, mirroring gridtimed's
switches+g+e pattern, confirmed against `trmgrid/env.py:48-52`'s
`TRMState` fields), changed the default encoder set to `ENCODERS +
NEW_ENCODERS`, added the numpy-shim fix needed to load checkpoints
locally (same shim as item 3). Ran fresh, all 36 (instance x encoder)
cells, seed=0, n=2000 samples, ridge probe, held out 30% for R²:

| Instance | mlp_point | deepset_h | cnn_dbm | cnn_proj | deepset_v | gnn |
|---|---:|---:|---:|---:|---:|---:|
| modesel/M3 | 0.948 | 0.961 | 0.949 | 0.212 | 0.986 | 0.875 |
| modesel/M4 | 0.936 | 0.932 | 0.927 | 0.285 | 0.940 | 0.837 |
| gridtimed/T2 | 0.998 | 0.882 | 0.737 | 0.504 | **-1.751** | 0.675 |
| gridtimed/T3 | 0.999 | 0.856 | 0.696 | 0.385 | **-7.879** | 0.436 |
| trmgrid/G_SMALL | 0.998 | 0.869 | 0.965 | 0.418 | **-0.187** | 0.563 |
| trmgrid/G_LARGE | 0.997 | 0.910 | 0.956 | 0.408 | **-1.347** | 0.711 |

(resolved by execution, `difference_probe.py --out
analysis/difference_probe.json`, all default settings; full per-pair R²
and raw config in that JSON file.) `cnn_proj`'s R² is low on every
instance (0.21–0.50) relative to `mlp_point`/`deepset_h`/`cnn_dbm`/`gnn` —
consistent with the "cnn_proj loses pairwise-clock information through
its rasterization" hypothesis this probe was built to test, and now
covers trmgrid too, not just modesel/gridtimed. **`deepset_v`'s R² is
negative on gridtimed and trmgrid** (worse than predicting the mean — the
embedding actively anti-correlates with the true pairwise differences on
held-out data, not just "uninformative"), while strongly positive on
modesel (0.94–0.99) — this tracks exactly the same domain-specific
collapse pattern established in `RESULTS_CHECK.md` §2 (0% vertex-fallback
everywhere, so this isn't a fallback artifact either) and is a striking,
directly citable number for the paper's deepset_v discussion.

## Item 11: deepset_v checks — fallback fraction confirmed exhaustive; max affine dim is a sample estimate, not exhaustive

**Vertex-enumeration fallback fraction**: exactly 0.0 across all 240
gridtimed+trmgrid deepset_v runs, every logged update
(`artifacts/RESULTS_CHECK.md` §2 — this figure IS exhaustive, since
`vrep_fallback_frac` is computed from a live per-update counter
instrumented into every real training run, not a resample).

**Maximum affine dimension encountered**: this is **not** exhaustive —
re-instrumenting all 240 already-completed training runs to log every
zone's affine dimension would mean rerunning them, which is out of scope
here. What exists is a **~450-520 zone random-rollout sample per
instance** (`RESULTS_CHECK.md` §2): observed max affine_dim was **4** on
gridtimed/T2, **3** on gridtimed/T3, trmgrid/G_SMALL, and trmgrid/G_LARGE.
Ambient dimension `d` (the hard upper bound) is 5 (T2, both trmgrid
tiers) or 6 (T3) per `SPECS.md` §1. Report the sampled max with this
caveat, not as a proven ceiling.

## Item 12: computational cost — computed fresh from existing per-update logs

Every run already logs `wall_s`, `encoder_forward_s`, `raster_s` per PPO
update (`run_one.py:592-594`). Aggregated per (domain, instance, encoder),
seed0 (resolved by execution, mean across all ~147 updates of that run):

| | mlp_point | deepset_h | cnn_dbm | cnn_proj | deepset_v | gnn |
|---|---:|---:|---:|---:|---:|---:|
| **gridtimed/T2** wall_s/update | 0.69 | 2.27 | 4.84 | 6.00 | 3.68 | 7.07 |
| raster frac | 0.000 | 0.000 | 0.783 | 0.064 | 0.008 | 0.000 |
| enc-fwd frac | 0.179 | 0.668 | 0.851 | 0.851 | 0.869 | 0.879 |
| **trmgrid/G_SMALL** wall_s/update | 0.61 | 1.54 | 4.88 | 6.76 | 4.56 | 6.85 |
| raster frac | 0.000 | 0.000 | 0.831 | 0.080 | 0.005 | 0.000 |
| enc-fwd frac | 0.170 | 0.689 | 0.882 | 0.873 | 0.880 | 0.869 |
| **modesel/M3** wall_s/update | 0.66 | 1.25 | 2.81 | 3.40 | 2.84 | 4.56 |
| raster frac | 0.000 | 0.000 | 0.698 | 0.030 | 0.046 | 0.000 |
| enc-fwd frac | 0.178 | 0.560 | 0.784 | 0.761 | 0.795 | 0.827 |

(Full 6-instance x 6-encoder table, all three fractions, in
`appendix_part1_data.json` → `computational_cost`; T3/G_LARGE/M4 follow
the same pattern, omitted here for space.) `cnn_dbm` spends 70–85% of its
wall-clock on rasterization alone (vs. `cnn_proj`'s 3–8%) — a genuine
representation-cost asymmetry between the two rasterization encoders, not
noise. `mlp_point` is consistently cheapest (encoder-forward is 16–24% of
its wall-clock vs. 56–91% for every zone-aware encoder) — expected, it's
the only encoder with no zone-construction cost at all.

**Hardware**: `hyperparameters.device` in every manifest is `"cpu"`
(`results/*__manifest.json`) — confirmed, not assumed. No CPU model, core
count, or hostname is logged anywhere in any manifest or jsonl. From this
session's own deployment context (not from result files): the sweep ran
on CU Boulder's Alpine cluster, partition `acpu`, QOS `cpu-long`
(`submit_trmgrid.sbatch`). Exact CPU model/core count: **UNKNOWN** — not
recorded anywhere in the repo.

## Item 13: shaping experiment — exists, results complete, summarized fresh

Design: `artifacts/PART1_FINAL.md` item 3 (`PART1_FINAL.md:141-186`).
Results: 12 real completed runs, `results/gridtimed__T2__{mlp_point,deepset_h}__seed{0,1,2}__shapeexp_{shaped,unshaped}*` —
all 12 finished the full 300,000-step budget (resolved by execution,
checked `total_env_steps=301056` on every one). No longer in the paper
per the brief, reported for completeness:

| Encoder | condition | final return (mean, 3 seeds) | last-200 success rate (mean, 3 seeds) |
|---|---|---:|---:|
| mlp_point | unshaped | -72.14 | 0.053 |
| mlp_point | shaped (r_shape=10.0) | -57.11 | 0.050 |
| deepset_h | unshaped | 83.17 | 1.000 |
| deepset_h | shaped (r_shape=10.0) | 99.11 | 0.983 |

Shaping raises `mlp_point`'s return substantially (-72 → -57, consistent
with the shaping bonus rewarding switch activations it wasn't getting
credit for) but **does not move its success rate at all** (0.053 →
0.050) — the same "changes the reward signal without closing the
task-completion gap" pattern as `POINTABLATE`'s V1/V2 features (item 8's
companion finding, different intervention, same qualitative outcome).
`deepset_h` is already near-saturated either way.

---

*Figures referenced: `figures/part1_curves.pdf`, `part1_curves_appendix.pdf`,
`part1_per_seed.pdf` (all regenerated/created this session — see D1/D2 in
this session's log). `figures/part1_instances.pdf` (A1) not yet built —
lowest priority per the brief's ordering, will follow.*
