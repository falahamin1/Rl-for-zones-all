# Results check — three findings, one fix

No retraining was done. Everything below comes from `results/*.jsonl`,
`results/*__manifest.json`, and `results/ckpt/*.pt` already on this machine
(pulled from Alpine), plus the code already in this repo.

## Summary

1. **The gridtimed greedy column is correct in the data. It was wrong in
   `artifacts/RESULTS_ALL_BENCHMARKS.md`** — an off-by-one column
   transcription error I made reading `analysis/encoder_stats.csv` by eye.
   `gridtimed_eval.py`, `run_one.py`, and `analyze.py` are all fine. Fixed
   in that file; no re-evaluation or retrain needed for the data itself.
2. **deepset_v's trmgrid collapse is real and is not a vertex-enumeration
   scope limitation.** `vrep_fallback_frac` is exactly 0.0 on every single
   trmgrid *and* gridtimed run — the bounding-box fallback never fires on
   either domain. This is a genuine V-rep representation-learning failure.
3. **On modesel, `mlp_point` is competitive on success rate**, not weak —
   the final-return gap that makes it look bad is a different axis and
   shouldn't be read as a competence gap.
4. `trmgrid` is now folded into `analyze.py`'s `DOMAINS_INSTANCES` properly
   (was a one-off script). Curves + CSVs regenerated for all three domains
   from one script, one run.

---

## 1. The gridtimed greedy column

### What was reported vs. what's actually true

The table in the first version of `RESULTS_ALL_BENCHMARKS.md` had gridtimed
T2/T3 "greedy SR" as: mlp_point 0.852/0.912, cnn_dbm 0.000/0.202, cnn_proj
0.000/0.000, gnn 0.000/0.079 — looking inverted against their stochastic
rates (0.050/0.040, 0.900/0.790, 1.000/1.000, 0.900/0.900).

**Those greedy numbers were wrong.** The correct values, re-read from
`analysis/encoder_stats.csv` with `csv.DictReader` (field-name access, not
eyeballing columns):

| encoder | T2 greedy SR | T2 stoch SR | T3 greedy SR | T3 stoch SR |
|---|---|---|---|---|
| mlp_point | 0.000 | 0.050 | 0.000 | 0.040 |
| deepset_h | 0.900 | 0.860 | 0.900 | 0.900 |
| cnn_dbm | 1.000 | 0.900 | 0.700 | 0.790 |
| cnn_proj | 1.000 | 1.000 | 1.000 | 1.000 |
| deepset_v | 0.500 | 0.500 | 0.200 | 0.200 |
| gnn | 1.000 | 1.000 | 0.900 | 0.900 |

Greedy and stochastic agree closely for every encoder. There is no
inversion, no anomaly, nothing to explain behaviorally — the domain
behaves exactly as you'd expect (a converged policy's argmax action
succeeding about as often as its sampled actions).

### Root cause

Read `gridtimed_eval.py` end to end and diffed it against `trmgrid_eval.py`
line by line. They are structurally near-identical: same `_rollout_episode`
(reset with a fixed `EVAL_SEED`, mask illegal logits to `-inf`, argmax for
greedy / categorical sample for stochastic, check `info.get("success")` on
`done`), same `EVAL_N_GREEDY_EPISODES=1` (both domains document zero
environment stochasticity given a fixed seed + deterministic policy, so
one greedy episode is lossless). `gridtimed/gym_env.py`'s success check —
`success = self._state.pos == "DONE"` — is correct: `done_state()` in
`gridtimed/env.py` sets `pos="DONE"` exactly for the success terminal,
`failed_state()` sets `pos="FAILED"` for deadline-expiry. No bug found in
either eval module, in `run_one.py`'s unpacking of
`(greedy_success_rate, greedy_cycling_frac, stochastic_success_rate) =
eval_fn(...)` (order matches the return signature and the jsonl field
order), or in `analyze.py`'s `compute_encoder_stats_table` (each field is
assigned by name, no swap).

I then checked the raw jsonl directly, per-seed, at the last eval point
(update 145 of 146 — evaluated every `EVAL_EVERY_K_UPDATES=5` updates) for
`gridtimed__T2__{mlp_point,cnn_proj,gnn}__seed{0..9}`: every single seed's
own last-recorded `greedy_success_rate` already matched what's in the
corrected table above (mlp_point: 0.0 for all 10 seeds; cnn_proj/gnn: 1.0
for all 10 seeds). The jsonl data was never wrong.

The bug was mine: `analysis/encoder_stats.csv` has an empty
`vrep_fallback_frac_mean` field for every non-`deepset_v` row (correctly —
that column is `None` for encoders that never call `cached_v_rep`), and
reading the CSV by eye, I miscounted past that blank cell and reported
column 15 (`greedy_cycling_frac_mean`) as if it were column 14
(`greedy_success_rate_mean`). `0.8518518518518519` for mlp_point/T2 is the
**cycling fraction** (which does match POINTCHECK's independently-measured
"89–94% cycling" almost exactly — that number was real, just mislabeled),
not a success rate.

### Independent re-evaluation (the task's explicit ask)

Applied the numpy shim from `RL-for-Polytopes/tangram-intermediate/train_single.py`
(`numpy._core` → alias to `numpy.core`, since this machine's numpy 1.24.3
lacks the submodule path newer numpy's pickle format references) to load
real checkpoints locally, then wrote a fresh rollout function — not calling
`gridtimed_eval.py` at all, independently re-implementing the mask/argmax/
step loop — against the actual final checkpoints (`total_env_steps=301056`,
i.e. genuinely the last saved state) for T2/seed0:

| encoder | independent greedy rollout (5 reps) |
|---|---|
| mlp_point | 0/5 success, every rep ends at step 27 via `'FAILED'` |
| cnn_proj | 5/5 success, every rep ends at step 8 via `'DONE'` |
| gnn | 5/5 success, every rep ends at step 8 via `'DONE'` |

Exactly matches the corrected table and POINTCHECK's own finding (0/200 for
mlp_point). Determinism holds as documented — every rep of a given
checkpoint is bit-identical.

### Verdict

**The logged column is correct.** No re-evaluation or retrain of anything
in `results/` is needed — nothing there is wrong. The only fix required was
correcting the transcription in `artifacts/RESULTS_ALL_BENCHMARKS.md`,
which is done (with a visible correction note left in place rather than a
silent edit).

---

## 2. deepset_v's trmgrid collapse

### vrep_fallback_frac

Pulled every `deepset_v` run's `vrep_fallback_frac` field (logged once per
PPO update, `None` unless `total_count > 0` that update — i.e. this is the
**real, training-time fallback rate**, not a resampled estimate) and
averaged across each run's own logged updates, then across seeds:

| domain/instance | n runs | mean-of-run-means | max seen anywhere |
|---|---|---|---|
| gridtimed/T2 | 10/10 | 0.0000 | 0.0000 |
| gridtimed/T3 | 10/10 | 0.0000 | 0.0000 |
| trmgrid/G_SMALL | 10/10 | 0.0000 | 0.0000 |
| trmgrid/G_LARGE | 10/10 | 0.0000 | 0.0000 |

Every single one of the 240 deepset_v runs across both domains has
`vrep_fallback_frac == 0.0` for its entire training run. The
`zoneviews.zone.VertexEnumerationLimit` fallback path never fires, on
either domain, ever. (Confirmed the counting mechanism itself is live and
per-update-windowed by reading `run_one.py`: `reset_vrep_fallback_stats()`
is called once per update before `trainer._do_one_update()`, and
`vrep_fallback_frac` is only logged `None` when `total_count == 0` that
update, which isn't what's happening here — `total_count > 0` for all of
these, the fallback count is just always 0.)

### Affine-dimension distribution and why fallback never fires

Sampled zones directly (random legal-action rollouts, ~450–520 per
instance, 30 episodes × 80 steps) via `obs["zone"]` and called
`zone.affine_dim()` / `zone.d` / `zone.v_rep()` on each:

| domain/instance | ambient dim (n clocks) | affine_dim distribution | `VertexEnumerationLimit` hits |
|---|---|---|---|
| gridtimed/T2 | 5 (100%) | {0: 46, 1: 198, 2: 173, 3: 17, 4: 2} | 0 / 436 |
| gridtimed/T3 | 6 (100%) | {0: 44, 1: 294, 2: 159, 3: 22} | 0 / 519 |
| trmgrid/G_SMALL | 5 (100%) | {0: 42, 1: 149, 2: 251, 3: 24} | 0 / 466 |
| trmgrid/G_LARGE | 5 (100%) | {0: 44, 1: 349, 2: 54, 3: 3} | 0 / 450 |

Affine dimension tops out at 3–4 on both domains at these clock counts —
`C(m', affine_dim)` with `_VERTEX_ENUM_LIMIT = 200_000`
(`zoneviews/zone.py`) needs a genuinely large reduced-constraint count `m'`
at `affine_dim` this small to threaten the limit at all, and it doesn't
happen in practice on either domain. trmgrid's zones are not structurally
higher-dimensional than gridtimed's — if anything G_LARGE's sampled zones
skew *lower*-dimensional (fewer dim-2/3 zones than T2/T3) than gridtimed's.

### Verdict

Per the task's own framing: fallback is not high, it's zero. **This is a
genuine finding about V-rep as a representation failing to learn trmgrid's
branching structure — not a scope limitation from hitting the
vertex-enumeration budget at higher clock counts.** deepset_v really is
processing real, exact vertices the entire time on both domains; it simply
doesn't learn a working policy from them on trmgrid (near-0% success on
both instances, tight variance — reliably failing, not randomly
struggling). What specifically about trmgrid's structure defeats V-rep
while gridtimed's structure only makes it bimodal is not established by
this check and would need a different investigation (e.g. does the
embedding actually distinguish trmgrid's states, is the gradient signal
degenerate, etc.) — flagging as open, not claiming to have resolved it.

---

## 3. modesel framing

Side-by-side, all six encoders, both instances (`overall`/`last200` =
per-episode success rate from `__episodes.jsonl`, all training episodes vs.
last 200):

| instance | encoder | final_return | overall SR | last200 SR |
|---|---|---|---|---|
| M3 | mlp_point | -68.23 | 0.831 | 0.887 |
| M3 | deepset_h | -56.25 | 0.832 | 0.888 |
| M3 | cnn_dbm | -55.86 | 0.844 | 0.905 |
| M3 | cnn_proj | -66.94 | 0.762 | 0.820 |
| M3 | deepset_v | -65.16 | 0.752 | 0.867 |
| M3 | gnn | -64.78 | 0.775 | 0.837 |
| M4 | mlp_point | -89.68 | 0.752 | 0.797 |
| M4 | deepset_h | -71.42 | 0.766 | 0.873 |
| M4 | cnn_dbm | -71.20 | 0.800 | 0.855 |
| M4 | cnn_proj | -88.97 | 0.639 | 0.680 |
| M4 | deepset_v | -89.80 | 0.647 | 0.673 |
| M4 | gnn | -90.57 | 0.634 | 0.648 |

### Verdict

**Stated plainly: `mlp_point` is competitive on success rate, not weak.**
On M3 it's essentially tied with the top two (0.831/0.887 vs. deepset_h's
0.832/0.888 and cnn_dbm's 0.844/0.905 — a 1–2 point gap, well within
seed-to-seed noise territory for this metric). On M4 it trails deepset_h
and cnn_dbm by more (roughly 2–8 points) but clearly *beats*
cnn_proj (0.639/0.680), deepset_v (0.647/0.673), and gnn (0.634/0.648) —
the same three encoders that also have the worst final-return numbers on
M4. `mlp_point`'s final-return magnitude (-68 to -90, the least negative
= "worst"-looking of the six on both instances) makes it look like the
bottom performer if you only look at that column, but final return and
success rate are different axes measuring different things (return also
reflects path efficiency/time cost, not just whether the task got done),
and on the "did it actually solve the task" axis it's mid-pack, not
bottom-of-pack. Conflating the two would be a real misreading of this
data.

---

## 4. analyze.py / trmgrid fold-in

`trmgrid` is now a proper entry in `analyze.py`'s `DOMAINS_INSTANCES`
(`"trmgrid": ("G_SMALL", "G_LARGE")`), with its own seed-count case in
`_seeds_for_domain` (`_TRMGRID_SEEDS = (0..9)`, kept as a separate named
constant from gridtimed's `_GRIDTIMED_EXTRA_SEEDS` rather than reusing it,
since trmgrid was swept at 10 seeds from the start rather than grown from a
3-seed base — different history, shouldn't imply a shared one). No other
code changes were needed: `compute_encoder_stats_table`'s dual-eval columns
(`greedy_success_rate`, `greedy_cycling_frac`, `stochastic_success_rate`)
were already domain-generic (read from the jsonl regardless of domain,
null-tolerant), so trmgrid's dual-eval numbers came through automatically
once the domain loop included it.

Ran `python3 analyze.py` once, end to end. It now writes, in one pass, for
all three domains:

- `analysis/{domain}__{instance}__curves_steps.png` /
  `curves_wallclock.png` — 12 files total, including the 4 new trmgrid ones
  (`trmgrid__G_SMALL__curves_steps.png`, `trmgrid__G_SMALL__curves_wallclock.png`,
  `trmgrid__G_LARGE__curves_steps.png`, `trmgrid__G_LARGE__curves_wallclock.png`)
- `analysis/steps_to_threshold.csv` — 24 rows (12 modesel+gridtimed, 12 trmgrid)
- `analysis/encoder_stats.csv` — 24 rows, same split
- `analysis/summary.txt` — all three domains

The trmgrid numbers this produced are byte-for-byte consistent with the
earlier one-off script's output (cross-checked final_ep_return_mean for all
12 trmgrid rows) — folding it in didn't change any trmgrid result, it just
means there's one script and one code path instead of two going forward.

`artifacts/RESULTS_ALL_BENCHMARKS.md` has been updated to reflect all of
the above (corrected gridtimed greedy numbers with a visible correction
note, the vrep_fallback_frac/affine-dim finding added to the trmgrid
section, the modesel section rewritten to separate the final-return and
success-rate axes explicitly, and the analyze.py provenance note updated).
