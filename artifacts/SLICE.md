# SLICE: shared PPO trainer + 4 encoders, preliminary de-risking slice

Scope: `train/env.py` (delta-free `SymbolicJobShopEnv` gym wrapper), `train/ppo.py` (PPO+GAE), `encoders/{mlp_point,deepset_h,cnn_dbm,cnn_proj}.py`, and the full 3-instance x 4-encoder x 3-seed experiment matrix (36 runs). `prob_jobshop_ext` was not modified.

## Parameter counts (actual, per instance)

| Instance | MLP-Point | DeepSet-H | CNN-DBM (CNN-M) | CNN-Proj (CNN-P) | max/min ratio |
|---|---|---|---|---|---|
| PJS_04 | 29470 | 30634 | 31420 | 30568 | 1.07x |
| PJS_05 | 31541 | 31053 | 31420 | 31138 | 1.02x |
| LJS_01 | 39601 | 32473 | 31420 | 33121 | 1.26x |

Per-element action head: only **CNN-DBM** has one (per-clock/per-job rows of `dbm_tensor` broadcast to that job's task slot(s), since clock==job in this domain). MLP-Point, DeepSet-H, and CNN-Proj all read all N task logits off a single pooled embedding via a shared linear layer -- MLP by design (flat baseline), DeepSet-H because its elements are constraints with no fixed task correspondence, CNN-Proj because its channels are indexed by clock *pairs* rather than single clocks/jobs. See each encoder module's docstring for the full justification.

## Final and best mean return per encoder per instance (mean +/- std across seeds)

| Instance | Encoder | seeds | final return | best return | final makespan | total env steps (mean) | wall-clock (mean, s) |
|---|---|---|---|---|---|---|---|
| PJS_04 | MLP-Point | 3 | -11.79 +/- 0.20 | -11.76 +/- 0.19 | 11.79 | 217088 | 46.8 |
| PJS_04 | DeepSet-H | 3 | -12.10 +/- 0.08 | -11.96 +/- 0.02 | 12.10 | 209579 | 89.9 |
| PJS_04 | CNN-DBM (CNN-M) | 3 | -12.53 +/- 0.14 | -12.38 +/- 0.12 | 12.53 | 122880 | 140.3 |
| PJS_04 | CNN-Proj (CNN-P) | 3 | -11.72 +/- 0.08 | -11.71 +/- 0.07 | 11.72 | 256000 | 660.6 |
| PJS_05 | MLP-Point | 3 | -19.75 +/- 0.28 | -19.73 +/- 0.28 | 19.75 | 215040 | 54.1 |
| PJS_05 | DeepSet-H | 3 | -20.07 +/- 0.26 | -19.93 +/- 0.17 | 20.07 | 213675 | 139.3 |
| PJS_05 | CNN-DBM (CNN-M) | 3 | -22.13 +/- 0.73 | -21.96 +/- 0.85 | 22.13 | 155648 | 290.5 |
| PJS_05 | CNN-Proj (CNN-P) | 3 | -19.82 +/- 0.50 | -19.76 +/- 0.52 | 19.82 | 260779 | 1316.7 |
| LJS_01 | MLP-Point | 3 | -86.46 +/- 0.92 | -86.19 +/- 1.02 | 86.46 | 218453 | 84.3 |
| LJS_01 | DeepSet-H | 3 | -93.93 +/- 2.17 | -91.27 +/- 1.16 | 93.93 | 133120 | 187.3 |
| LJS_01 | CNN-DBM (CNN-M) | 3 | -97.97 +/- 0.11 | -96.71 +/- 0.92 | 97.97 | 152235 | 722.6 |
| LJS_01 | CNN-Proj (CNN-P) | 3 | -89.77 +/- 2.45 | -88.65 +/- 2.41 | 89.77 | 160427 | 1938.3 |

## Sample-matched comparison (return at a fixed env-step budget)

Mean return interpolated (forward-filled) at fixed env-step checkpoints, mean +/- std across seeds.

### PJS_04

| Encoder | 50k steps | 100k steps | 200k steps | 300k steps |
|---|---|---|---|---|
| MLP-Point | -12.01+/-0.15 | -11.93+/-0.21 | -11.80+/-0.21 | -11.79+/-0.20 |
| DeepSet-H | -12.26+/-0.14 | -12.14+/-0.07 | -12.07+/-0.03 | -12.10+/-0.08 |
| CNN-DBM (CNN-M) | -12.60+/-0.16 | -12.53+/-0.16 | -12.55+/-0.13 | -12.53+/-0.14 |
| CNN-Proj (CNN-P) | -12.11+/-0.18 | -11.99+/-0.16 | -11.81+/-0.14 | -11.72+/-0.08 |

### PJS_05

| Encoder | 50k steps | 100k steps | 200k steps | 300k steps |
|---|---|---|---|---|
| MLP-Point | -20.18+/-0.29 | -19.90+/-0.17 | -19.78+/-0.26 | -19.76+/-0.27 |
| DeepSet-H | -20.75+/-0.22 | -20.42+/-0.32 | -20.14+/-0.21 | -20.05+/-0.27 |
| CNN-DBM (CNN-M) | -22.73+/-0.74 | -22.32+/-0.62 | -22.21+/-0.67 | -22.13+/-0.73 |
| CNN-Proj (CNN-P) | -21.19+/-0.74 | -20.27+/-0.47 | -19.85+/-0.51 | -19.82+/-0.50 |

### LJS_01

| Encoder | 50k steps | 100k steps | 200k steps | 300k steps |
|---|---|---|---|---|
| MLP-Point | -89.24+/-1.06 | -87.38+/-1.18 | -86.58+/-1.09 | -86.44+/-0.90 |
| DeepSet-H | -95.06+/-0.59 | -94.07+/-2.25 | -93.93+/-2.17 | -93.93+/-2.17 |
| CNN-DBM (CNN-M) | -98.72+/-1.44 | -99.62+/-0.97 | -97.97+/-0.11 | -97.97+/-0.11 |
| CNN-Proj (CNN-P) | -92.45+/-0.87 | -91.90+/-0.57 | -89.99+/-2.14 | -89.70+/-2.56 |

## Wall-clock-matched comparison (return at a fixed wall-clock budget)

Same idea, but the x-axis is cumulative wall-clock seconds rather than env steps -- this is where the raster encoders' per-step cost shows up directly: they get far fewer env steps in for the same time budget.

### PJS_04

| Encoder | 60s | 300s | 900s | 1800s |
|---|---|---|---|---|
| MLP-Point | -11.79+/-0.20 | -11.79+/-0.20 | -11.79+/-0.20 | -11.79+/-0.20 |
| DeepSet-H | -12.07+/-0.05 | -12.10+/-0.08 | -12.10+/-0.08 | -12.10+/-0.08 |
| CNN-DBM (CNN-M) | -12.63+/-0.08 | -12.53+/-0.14 | -12.53+/-0.14 | -12.53+/-0.14 |
| CNN-Proj (CNN-P) | -12.42+/-0.05 | -11.93+/-0.19 | -11.72+/-0.08 | -11.72+/-0.08 |

### PJS_05

| Encoder | 60s | 300s | 900s | 1800s |
|---|---|---|---|---|
| MLP-Point | -19.76+/-0.27 | -19.75+/-0.28 | -19.75+/-0.28 | -19.75+/-0.28 |
| DeepSet-H | -20.29+/-0.18 | -20.07+/-0.26 | -20.07+/-0.26 | -20.07+/-0.26 |
| CNN-DBM (CNN-M) | -22.75+/-0.68 | -22.21+/-0.67 | -22.13+/-0.73 | -22.13+/-0.73 |
| CNN-Proj (CNN-P) | -22.35+/-0.78 | -20.75+/-0.50 | -19.87+/-0.49 | -19.82+/-0.50 |

### LJS_01

| Encoder | 60s | 300s | 900s | 1800s |
|---|---|---|---|---|
| MLP-Point | -86.44+/-1.24 | -86.46+/-0.92 | -86.46+/-0.92 | -86.46+/-0.92 |
| DeepSet-H | -94.66+/-0.55 | -93.93+/-2.17 | -93.93+/-2.17 | -93.93+/-2.17 |
| CNN-DBM (CNN-M) | -100.69+/-0.84 | -99.10+/-0.29 | -97.97+/-0.11 | -97.97+/-0.11 |
| CNN-Proj (CNN-P) | -101.05+/-1.44 | -94.28+/-1.51 | -91.36+/-0.48 | -90.46+/-1.48 |

## Encoder forward time as a fraction of total step time

| Instance | Encoder | mean encoder_forward_frac | mean wall-clock/update (s) |
|---|---|---|---|
| PJS_04 | MLP-Point | 8.4% | 0.44 |
| PJS_04 | DeepSet-H | 44.3% | 0.88 |
| PJS_04 | CNN-DBM (CNN-M) | 63.0% | 2.34 |
| PJS_04 | CNN-Proj (CNN-P) | 82.8% | 5.28 |
| PJS_05 | MLP-Point | 8.2% | 0.51 |
| PJS_05 | DeepSet-H | 53.3% | 1.33 |
| PJS_05 | CNN-DBM (CNN-M) | 70.5% | 3.82 |
| PJS_05 | CNN-Proj (CNN-P) | 89.2% | 10.34 |
| LJS_01 | MLP-Point | 6.3% | 0.79 |
| LJS_01 | DeepSet-H | 68.0% | 2.88 |
| LJS_01 | CNN-DBM (CNN-M) | 82.7% | 9.72 |
| LJS_01 | CNN-Proj (CNN-P) | 94.1% | 24.74 |

## Permutation ablation (evaluation-time only)

Each encoder's own 'set of pieces' shuffled at eval time (see `train/permutation_ablation.py` docstring for exactly what is shuffled per encoder); greedy policy, 20 episodes/checkpoint, same env seeds for baseline and shuffled runs. Gap = baseline - shuffled mean return (positive gap = shuffling hurt).

| Instance | Encoder | seeds | baseline return | shuffled return | mean gap +/- std |
|---|---|---|---|---|---|
| PJS_04 | MLP-Point | 3 | -12.18 | -12.25 | 0.07 +/- 0.09 |
| PJS_04 | DeepSet-H | 3 | -12.32 | -12.32 | 0.00 +/- 0.00 |
| PJS_04 | CNN-DBM (CNN-M) | 3 | -12.12 | -11.98 | -0.13 +/- 0.06 |
| PJS_04 | CNN-Proj (CNN-P) | 3 | -11.90 | -11.85 | -0.05 +/- 0.04 |
| PJS_05 | MLP-Point | 3 | -23.27 | -23.23 | -0.03 +/- 0.02 |
| PJS_05 | DeepSet-H | 3 | -23.45 | -23.45 | 0.00 +/- 0.00 |
| PJS_05 | CNN-DBM (CNN-M) | 3 | -23.03 | -24.72 | 1.68 +/- 0.91 |
| PJS_05 | CNN-Proj (CNN-P) | 3 | -23.30 | -23.52 | 0.22 +/- 0.27 |
| LJS_01 | MLP-Point | 3 | -90.20 | -90.25 | 0.05 +/- 0.07 |
| LJS_01 | DeepSet-H | 3 | -88.72 | -88.72 | 0.00 +/- 0.00 |
| LJS_01 | CNN-DBM (CNN-M) | 3 | -97.72 | -100.43 | 2.72 +/- 0.84 |
| LJS_01 | CNN-Proj (CNN-P) | 3 | -88.60 | -89.03 | 0.43 +/- 0.61 |

Expected pattern: DeepSet-H's gap should be ~0 (sum-pool is invariant to element order by construction); MLP-Point and both CNNs should show a nonzero (likely positive) gap, quantifying how much of their return depends on position/order rather than content.

**Observed vs. expected**: DeepSet-H matches exactly (gap 0.00 +/- 0.00 on every instance -- sum-pool really is invariant, confirmed rather than assumed). CNN-DBM shows the clearest, largest, and most consistent positive gap (1.68 on PJS_05, 2.72 on LJS_01) -- expected, since its per-element head depends on a *fixed* row-index-means-job correspondence that clock relabeling directly breaks. CNN-Proj shows smaller, less consistent gaps (slightly negative on PJS_04). MLP-Point's gap is close to zero on all three instances, smaller than the "should degrade" expectation would suggest -- but this is a real property of *this* domain's `point` encoding, not evidence the ablation failed to bite: at any decision point only up to `d` tasks can be active/done at once (one per job, precedence-sequential), so most of the N per-task feature blocks being shuffled are numerically *identical* placeholders (`waiting`, elapsed=0) -- shuffling among duplicates is a no-op. The permutation genuinely perturbs the logit vector (verified directly, not just via episode return, during development: nonzero per-logit differences up to ~0.08 mid-episode), it just rarely changes which of the few genuinely-distinguishable tasks has the max logit. A stricter version of this ablation for MLP specifically would need to restrict comparison to states with several simultaneously-active/done tasks rather than averaging over whole episodes.

## Anomalies

- PJS_04/cnn_dbm/seed1: no improvement over training (first-third mean -12.46 -> last-third mean -12.50)
- CNN-Proj channel emptiness (fraction of projection_stack channels with zero coverage, sampled over 300 random-policy decision points per instance, checked independently of training): PJS_04 1.3%, PJS_05 3.9%, LJS_01 17.3%. Growing with instance size but not "mostly empty" at any instance tested (would need >50%) -- reported as a trend to watch, not flagged as a failure.
- Early stopping triggered for at least one seed in most (instance, encoder) groups (see "total env steps (mean)" in the return table above, most below the 300k budget) -- this is the plateau mechanism working as specified, not an anomaly, but it means "final return" and "best return" are close for most rows by construction.
