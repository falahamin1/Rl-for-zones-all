# PART1_FINAL: Uniform Budget, Dual-Metric Eval, and the Shaping Experiment

Three items, gated on item 1's verification. **Nothing submitted to Alpine.**
No direction recommended.

---

## Also re-run, cheap: mlp_point cycling on existing checkpoints

Greedy rollout (200 episodes, all 6 gridtimed `mlp_point` checkpoints — T2
and T3, seeds 0-2), reusing `SWEEP_FINAL.md`'s B4 cycling definition
(fraction of non-terminal steps that revisit an already-visited grid cell).

| Instance | greedy success | switch activated | gate attempted | mean ep length | cycling frac |
|---|---:|---:|---:|---:|---:|
| T2 (all 3 seeds identical) | 0/200 | 0/200 | 200/200 | 27.0 | **0.889** |
| T3 (all 3 seeds identical) | 0/200 | 0/200 | 0/200 | 34.0 | **0.941** |

Reproduces POINTCHECK's T2 numbers exactly (0% success, 27.0 ep length, 0%
switch activation, 100% gate attempts). **Answer to the question asked:
yes, `mlp_point`'s greedy failures are predominantly cycling** — 89-94% of
non-terminal steps revisit an already-seen cell, higher than any of the
`deepset_v`/`gnn` failures characterized in `SWEEP_FINAL.md` (65-91%). The
caveat belongs next to the greedy numbers in the paper: greedy evaluation
here is measuring a substantial amount of deterministic self-looping
pathology, not purely "the policy never learned anything useful." One
side finding: T3's degenerate policy **never attempts a gate at all**
(0% vs. T2's 100%) — the "stable degenerate strategy" POINTCHECK described
is real and reproducible, but not instance-invariant; T2 and T3 converge to
different degenerate behaviors.

---

## Item 1: uniform budget (the gate)

**Code change** (`train/ppo.py`, `run_one.py`): `PPOConfig.plateau_enabled:
bool = False` (new field, default off) — the `should_stop = True` line in
`_do_one_update` is now gated on it; the plateau-tracking machinery itself
(`_steps_since_improve`, `_best_mean_return`) is untouched, still runs,
just no longer allowed to stop training unless explicitly opted into via
`run_one.py --enable-early-stop`. Every run now trains to exactly
`total_steps` by default. One necessary companion fix: a resumed
checkpoint can carry a stale `early_stop_step` recorded back when plateau
detection *was* enabled — without clearing it, the `already_done` check
would treat a previously-plateaued run as finished and never actually
extend it. Fixed: resuming with plateau disabled now discards that stale
marker and continues toward `total_steps`.

### VERIFICATION GATE

Picked `gridtimed/T3/deepset_h/seed1` — stopped at 59,392 steps (target
300,000), a genuinely still-learning run (40% `last200_success_rate`, high
return variance), not one of the degenerate/collapsed cells. Resumed
locally with plateau disabled, ran to completion (301,056 steps, 147
updates, ~3.5 minutes wall-clock).

- jsonl integrity: `update` sequence `[0..146]`, strictly increasing, no
  gaps, across the full pre+post-resume file.
- Discontinuity check, 10 updates each side of the resume boundary
  (update 28 → 29), boundary-crossing `|delta|` vs. the typical
  step-to-step `|delta|` already present within those same 20 updates:

  | Metric | boundary Δ | mean within-window Δ | max within-window Δ | ratio |
  |---|---:|---:|---:|---:|
  | `ep_return_mean` | 58.94 | 54.51 | 90.65 | 1.08x |
  | `policy_entropy` | 0.081 | 0.083 | 0.320 | 0.98x |
  | `value_loss` | 675.6 | 864.3 | 1917.5 | 0.78x |
  | `grad_norm` | 175.0 | 373.4 | 907.5 | 0.47x |

  Every boundary delta falls **within** the range of ordinary update-to-
  update noise already present on both sides — no discontinuity. **Gate
  passes.**

Per the explicit instruction, this means: **extend, don't re-run from
scratch.** This one run (`gridtimed/T3/deepset_h/seed1`) is now genuinely
complete at 301,056 steps as a byproduct of the verification itself — it
does not need to be resubmitted.

### Extending the remaining 71 runs

Since the new default is already "no early stopping," extending the other
71 existing runs requires **no new script and no flag** — resubmitting the
unmodified `submit.sbatch` (indices 0-47) and `submit_vrep_gnn.sbatch`
(48-71) picks up the new default automatically and resumes each from its
checkpoint. Of the 72 original runs, **7 already reached ≥300,000 steps**
(no work needed); **65 need extension** (64 remaining after the one done
above). Cost broken down in the wall-clock section below.

---

## Item 2: dual-metric eval

**New module** `gridtimed_eval.py`: `evaluate_gridtimed(head, instance_obj,
ctor)` → `(greedy_success_rate, greedy_cycling_frac, stochastic_success_rate)`.
Gridtimed has no environment stochasticity (fixed initial state, fixed
dynamics — established in `SWEEP_FINAL.md` B4), so a greedy rollout is
fully deterministic given the policy: **1 greedy episode carries exactly as
much information as 200 would** (all identical), so
`EVAL_N_GREEDY_EPISODES = 1`. The stochastic pass genuinely samples, so it
uses `EVAL_N_STOCHASTIC_EPISODES = 10` with a fixed-seeded RNG
(`EVAL_SEED = 999983`, same every call, every encoder, every seed) for a
controlled, reproducible comparison.

**Cadence**: benchmarked real eval cost directly against all six real
checkpoints (both T2 and T3) rather than estimating:

| Encoder | worst-case eval cost | mean wall_s/update (gridtimed) | overhead at K=5 |
|---|---:|---:|---:|
| `cnn_proj` | 0.544s | 6.082s | **1.79%** |
| `mlp_point` | 0.050s | 0.582s | 1.72% |
| `deepset_h` | 0.150s | 1.848s | 1.60% |
| `cnn_dbm` | 0.169s | 5.101s | 0.66% |
| `gnn` | 0.187s | 7.747s | 0.48% |
| `deepset_v` | 0.113s | 5.619s | 0.40% |

**Chose K=5** (eval every 5th update): worst case 1.79% overhead, all six
comfortably under the ~5% budget, ~29 eval points across a full 147-update
run. Gated on `domain == "gridtimed"` only — `greedy_cycling_frac` has no
modesel analogue (no grid positions to revisit). As a side effect, this
also adds the three fields to every gridtimed run within item 1's
extension (same jsonl schema, same codepath) — modesel rows stay null.

New jsonl fields: `greedy_success_rate`, `greedy_cycling_frac`,
`stochastic_success_rate` — null except every 5th update, matching the
existing `vrep_fallback_frac` "null where undefined" convention. Verified
end-to-end with a local smoke run (fields populate at updates 0 and 5,
null elsewhere; confirmed absent entirely — not just null — on modesel).

**`analyze.py`**: `load_run` now tolerates these fields via `.get(...) is
not None else nan`, same pattern as `vrep_fallback_frac` — old jsonl files
written before this field existed load cleanly with all-NaN columns, no
`KeyError`. `compute_encoder_stats_table` reports the **last** valid eval
(not a mean across training — success rate improves over training, so
averaging early bad-policy evals in with the final one would understate
the trained result), `None` when a run has no eval data at all. Verified
against current (pre-item-2) data: no crash, correctly reports `None`
throughout since no jsonl written before this change has these keys.

---

## Item 3: shaping experiment

**Mechanism**: `GridTimedTrainEnv.__init__` gained
`switch_shaping_reward: float = 0.0`. Implemented as genuine potential-
based shaping — `Φ(s) = |distinct switches activated so far this
episode|`, `shaping = switch_shaping_reward * (Φ(s') − Φ(s))` — which,
since `Φ` only increments the first time a *new* switch is hit, reduces
exactly to "+`r_shape` the first time each switch is activated," matching
the operational spec precisely while keeping the `Φ(s')−Φ(s)` form.
Default `0.0`: contributes nothing regardless of episode history, so the
unshaped path is byte-identical to before this change by construction (not
by keeping a separate code path in sync). `r_shape = 10.0` (10% of
`gridtimed.instance.REWARD_SUCCESS = 100.0`).

**Verified directly** (not just by inspection): manually drove an env
toward switch `(1,1)` in both configurations. Unshaped: every legal move
costs exactly `-1.0` (`TIME_COST`). Shaped: first arrival at `(1,1)` gives
`reward = -1.0 + 10.0 = 9.0`; moving away and back to the *same* switch
gives `-1.0` again (no repeat bonus) — confirms both the bonus and the
once-per-switch-per-episode behavior.

**Wiring** (`run_one.py`): `RunSpec.shaped: Optional[bool] = None` (`None`
= not part of this experiment, unaffected — every one of indices 0-155;
`False`/`True` = the 12 shaping-experiment configs at indices 156-167:
`SHAPING_ENCODERS = (mlp_point, deepset_h)` × `SEEDS (0,1,2)` × `{unshaped,
shaped}`, gridtimed/T2 only).

**A real bug caught before it could touch data**: the first version of
this used a plain `shaped: bool = False`, giving the unshaped control the
*same run_id* as the pre-existing `gridtimed/T2/mlp_point` (and
`deepset_h`) runs already sitting in `results/` from indices 0-47 — real,
already-trained Alpine data. Since `run_one.py`'s resume logic loads
`results/ckpt/{run_id}.pt` unconditionally, submitting the "unshaped
control" as designed would have **silently resumed and mutated that real
checkpoint** under a training regime (item 1's early-stop-then-resume
history) that doesn't match a clean 300k run — corrupting both the
original sweep's data and invalidating the shaping comparison at once.
Fixed: `run_id` now gets an explicit `__shapeexp_unshaped` /
`__shapeexp_shaped` suffix for *both* arms of the shaping experiment, so
neither collides with anything in indices 0-155. Verified: all 168
`run_id`s unique, and the 12 shaping-experiment IDs share no intersection
with the original sweep's `gridtimed/T2/{mlp_point,deepset_h}` IDs.

`test_run_one.py`: 13 tests (up from 11), including
`test_shaping_run_ids_distinct_from_original_sweep` and
`test_156_to_167_are_the_shaping_experiment`; `test_first_156_unchanged`
checks indices 0-155 against an independently-transcribed snapshot, same
discipline as every prior extension. All pass.

---

## Wall-clock estimate

Computed from **real Alpine `wall_s`/update data**, not benchmarked
locally — every encoder×domain combination has 245-758 real logged
updates to average over.

**Two corrections to the prompt's premises, found while computing this,
stated plainly rather than silently worked around:**

1. **"cnn_proj is the expensive one (52-55% rasterization)"** — checked
   directly: it's `cnn_dbm` that has 80-88% `raster_frac` on gridtimed;
   `cnn_proj`'s is 7-13%. `cnn_proj` *is* one of the pricier encoders in
   absolute wall-clock (6.08s/update mean, 2nd-highest after `gnn`'s
   7.75s), but the mechanism isn't rasterization — its `encoder_forward_frac`
   is 86-92%, similar to `cnn_dbm`'s, consistent with per-forward-pass conv
   cost rather than raster time specifically.
2. **"mlp_point on gridtimed stopped earliest so it has the most to make
   up"** — checked directly against all 6×3 gridtimed manifests: `mlp_point`
   actually has the **highest** mean stopping point (155,307 steps) of all
   six encoders; `cnn_dbm` stopped earliest (83,627 steps mean). Combined
   with `mlp_point` being by far the cheapest per-update (0.582s vs.
   `gnn`'s 7.747s), it needs the **fewest** total remaining updates (427,
   vs. `cnn_dbm`'s 637) and the least item-1 compute of any encoder
   (0.07h). The opposite of "most to make up."

### Item 1 (extend 65 runs — 7 already at 300k, 1 done during verification)

| (domain, encoder) | compute-hours |
|---|---:|
| gridtimed/gnn | 1.16h |
| gridtimed/cnn_proj | 1.03h |
| gridtimed/cnn_dbm | 0.90h |
| gridtimed/deepset_v | 0.90h |
| modesel/deepset_v | 0.77h |
| modesel/cnn_proj | 0.75h |
| modesel/gnn | 0.64h |
| gridtimed/deepset_h | 0.26h |
| modesel/cnn_dbm | 0.37h |
| gridtimed/mlp_point | 0.07h |
| modesel/mlp_point | 0.07h |
| modesel/deepset_h | 0.05h |
| **Total** | **6.98h** |

### Item 2 (84 new gridtimed runs -- 2 instances x 7 seeds each encoder, full 300k, includes eval overhead)

| Encoder | compute-hours |
|---|---:|
| gnn | 4.43h |
| cnn_proj | 3.48h |
| deepset_v | 3.21h |
| cnn_dbm | 2.92h |
| deepset_h | 1.06h |
| mlp_point | 0.33h |
| **Total** | **15.42h** |

### Item 3 (12 shaping runs, gridtimed/T2, full 300k each)

`mlp_point`: 0.14h, `deepset_h`: 0.45h — **total 0.60h**.

### Grand total

**22.99 compute-hours** (~0.96 days serial). At `--array=...%20`:
**≈1.15h wall-clock**; even at the more conservative `%12` used by the
original sweep: **≈1.92h**. **Well under the 3-day threshold** — no
throttle reduction or staging needed on compute grounds. This is a
best-case estimate assuming perfect scheduling and no queue wait — the
maintenance-window block encountered earlier in this project (a full day
of `PD` before the array could even start) is a reminder that real
wall-clock can differ substantially from this number for reasons entirely
outside the compute itself; nothing in this estimate accounts for queue
depth or reservations at submission time.

---

## Exact sbatch commands, in submission order

**Not run.** All four scripts share the same Alpine header convention
(`--partition=acpu`, `--qos=cpu-long`, `--time=72:00:00`, `--nodes=1
--ntasks=1 --cpus-per-task=1`, `--mem=32G`, `module purge; module load
anaconda; conda activate amfa-custom-env`, the `LD_LIBRARY_PATH` fix,
`WORK_DIR=/projects/amfa5003/RL-for-zones`), same checkpoint/resume
contract, same atomic writes, same jsonl byte-offset truncation on resume.

```
# 1. Item 1 -- extend the original 72 (unmodified scripts; the new
#    plateau-disabled-by-default resumes and continues each automatically)
sbatch submit.sbatch              # indices 0-47,  --array=0-47%12
sbatch submit_vrep_gnn.sbatch     # indices 48-71, --array=48-71%12

# 2. Item 2 -- the 84 new gridtimed seeds (dual-metric eval included)
sbatch submit_sweep_final.sbatch  # indices 72-155, --array=72-155%20

# 3. Item 3 -- the shaping experiment
sbatch submit_shaping.sbatch      # indices 156-167, --array=156-167%12
```

---

## Files delivered / modified

- `train/ppo.py`: `PPOConfig.plateau_enabled` (new, default `False`).
- `run_one.py`: `--enable-early-stop` CLI flag; stale-`early_stop_step`
  resume fix; `RunSpec.shaped`; `TARGET_RUNS` extended to 168 (indices
  156-167, the shaping experiment); dual-metric eval wired into the
  training loop; `SWITCH_SHAPING_REWARD`, `SHAPING_ENCODERS`,
  `SHAPING_INSTANCE`, `EVAL_EVERY_K_UPDATES` constants.
- `gridtimed_eval.py` (new): `evaluate_gridtimed`.
- `gridtimed/gym_env.py`: `GridTimedTrainEnv.switch_shaping_reward`
  (potential-based, default `0.0` — no behavior change unless set).
- `analyze.py`: `load_run`/`compute_encoder_stats_table` tolerate the
  three new fields, null-safe against older jsonl files.
- `test_run_one.py`: 13 tests, 168-entry structure, `_ORIGINAL_156`
  snapshot check, shaping-experiment coverage and non-collision tests.
- `submit_shaping.sbatch` (new).
- One real run genuinely extended during verification:
  `gridtimed/T3/deepset_h/seed1`, now at 301,056 steps.
- `/tmp/PART1_FINAL.md`, this report — copied to `artifacts/`.

**Not submitted to Alpine.** No direction is recommended — item 3's actual
result (does `mlp_point` succeed when shaped) isn't known yet; that's what
submitting would tell us.
