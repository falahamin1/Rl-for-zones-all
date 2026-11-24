# POINTABLATE: Sharpening the gridtimed mlp_point Finding

New training runs added (as explicitly requested — this is not the
"diagnosis only" mode of `PRESUBMIT`/`POINTCHECK`). `run_one.py`,
`encoders/`, and the sweep are untouched: three new `GridTimedTrainEnv`
subclasses and one degenerate-zone variant live in new scratch files, and
training reuses `train.ppo.build_head`/`PPOTrainer` exactly as `run_one.py`
does. All 12 runs (4 conditions × 3 seeds, gridtimed/T2, 300k steps,
identical hyperparameters to the sweep) logged to the sweep's exact jsonl
schema — verified against `analyze.py`'s own `load_run()` before running
anything at scale.

**Bottom line: neither (a) nor (b) as originally framed is a clean fit —
the real result is more precise than either.** Stochastic (training-time)
success rate strongly supports (b): both V1 and V2 close nearly the entire
gap to zone-encoder performance. But greedy (deterministic) evaluation
reveals a further, unanticipated split that the (a)/(b) framing didn't
have room for — reported honestly below rather than forced into one bucket.
The separate width-vs-structure control is unambiguous: width is ruled out.

---

## Results (mean ± std across 3 seeds, gridtimed/T2, 300k-step budget)

| Condition | stochastic success (all episodes) | stochastic success (last 200) | greedy success | greedy switch-activation | final entropy |
|---|---:|---:|---:|---:|---:|
| V0 (control, sweep's point) | 3.3% ± 0.2% | 3.5% ± 0.8% | 0% | 0% | 2.120 ± 0.036 |
| V1 (+ switch-activated bitmask) | 37.1% ± 3.6% | **48.2% ± 3.1%** | 0% | **100%** | 1.542 ± 0.009 |
| V2 (+ gate-reachability bits) | 34.9% ± 1.5% | **50.7% ± 1.2%** | 0% | 0% | 1.440 ± 0.018 |
| deepset_h + degenerate (zero-width) zone | 52.8% ± 4.4% | 40.2% ± 3.5% | **100%** | 33.3% | 2.000 ± 0.004 |

For reference, the original sweep's zone encoders on gridtimed/T2 (from
`/tmp/ENCODERS_SETUP.md`'s data, stochastic/episode-level, same metric as
the "stochastic" columns here): `deepset_h` 50.9%, `cnn_dbm` 49.8%,
`cnn_proj` 48.5%. V0's 3.3% matches the originally-reported 3.7% closely
(different training run, same seed set, expected minor variance).

Learning curves: `pointablate_results/pointablate_curves.png` (kept with
the run data, not copied into the repo — this is ablation output, not a
sweep deliverable). V0 stays flat near -75 for the full run. V1 and V2
climb together, converging to roughly the same asymptote by ~150-160k
steps. `deepset_h`-degenerate rises fastest and earliest (positive by
~10-15k steps) and oscillates around a higher plateau (+10 to +20) for
most of training.

## V1 / V2 vs the (a)/(b) framing

**Stochastic success rate says (b), clearly.** V0's ~3.3% jumps to
37-51% for both V1 and V2 — in the same range as the zone encoders'
48-57%. Adding either an explicit switch-activation memory (V1) or an
explicit gate-reachability signal (V2) closes nearly the entire gap
between `mlp_point` and the zone-based encoders, as measured by what the
policy PPO actually trains against.

**Greedy evaluation complicates this, and does not fit cleanly into
either (a) or (b).** Despite the stochastic gains, greedy/argmax success
stays at exactly 0% for both V1 and V2, on every seed — no different from
V0 on this specific measure. V1's greedy policy does reliably touch a
switch now (100%, up from 0%) but still doesn't convert that into
reaching the goal; V2's greedy policy doesn't even reliably touch a
switch (0%, like V0), despite its stochastic success rate being just as
high as V1's. So the added feature clearly changes what the STOCHASTIC
policy learns to do (confirmed two different ways, V1 and V2, converging
to a similar stochastic success rate by two different mechanisms), but
neither closes the gap for the DETERMINISTIC policy at all. This is a
real, seed-consistent split the original two-hypothesis framing didn't
anticipate — reported as its own finding rather than squeezed into (a) or
(b).

Contrast with the separate `deepset_h`-degenerate control below, which
DOES achieve 100% greedy success — so "learns something under stochastic
sampling that greedy can't reliably use" is not an inherent property of
this task or of PPO in general here; it's specific to what V1/V2 add to
`mlp_point`.

## Separate control: does the zone need its width?

`deepset_h` fed a degenerate zone (DBM hull of a single point — the exact
concrete valuation, every constraint tight, zero width) instead of the
real w-jitter-widened zone: **100% greedy success, all 3 seeds**, and
52.8%/40.2% stochastic success — matching or exceeding the real
(wide-zone) zone encoders from the sweep.

**This rules out zone width as the source of the advantage.** A
zero-width zone is architecturally identical to the real one from
`deepset_h`'s perspective (same `d`, same `phi`/pooling), just fed a
degenerate input — and it still succeeds cleanly and fast (positive
return by ~10-15k steps, versus ~130k+ for V1/V2 to reach comparable
stochastic success). Whatever `deepset_h` needs, it isn't the
zone's uncertainty geometry. One scope note: this isolates width from
"everything else deepset_h's architecture + h_rep format together
provide" — it does not, on its own, further separate "the set-structured
constraint format" from "this specific architecture's pooling/inductive
bias," since only one architecture was tested here.

## What's ruled out, stated plainly

- Zone width is ruled out as the source of `deepset_h`'s advantage
  (100% greedy success with a zero-width zone).
- "Point observation is simply missing all useful signal" is ruled out —
  V1/V2 show that two different explicit additions each substantially
  raise stochastic success.
- "Adding the missing feature fully closes the gap" is also ruled out —
  greedy evaluation stays at 0% for both V1 and V2, unlike the
  `deepset_h` control's clean 100%.

No direction is recommended.
