# POINTCHECK: Verifying the gridtimed mlp_point Finding

Diagnosis only — no retraining of the sweep, no changes to `encoders/` or
`run_one.py`. One new short training run was explicitly requested by this
task (§5) and is reported as such, not treated as a sweep result.

**Bottom line: the competing explanation (malformed/non-Markov `point`
observation) does not hold.** Sections 1, 3, and 5 each independently rule
out a distinct version of "it's an observation/setup problem" — sufficiency,
scaling, and capacity/exploration. Section 4 shows what the trained policy
is actually doing: a specific, exactly reproducible degenerate behavior
(never activates a switch), not noise or a training-pipeline failure.

---

## 1. Is `point` sufficient for optimal control?

Grouped every DP-reachable, non-terminal, non-deadlock state by its exact
`point` vector (via the real `_get_point()` method, not a reimplementation),
then checked whether any group's members have disjoint DP-optimal-action
sets (same method as `gridtimed/decision_relevance.py`'s
`multi_zone_conflict_analysis`, applied to point-vector groups instead of
zones).

| Instance | usable states | distinct point vectors | groups with 2+ states | conflicting groups | conflict fraction |
|---|---:|---:|---:|---:|---:|
| gridtimed/T2 | 76,354 | **76,354** | **0** | 0 | **0%** |
| modesel/M3 (control) | 543 | 369 | 154 | **4** | **2.6%** (0.74% of all usable states) |

**gridtimed's `point` is not just sufficient — it's injective.** Every
single reachable state maps to a distinct vector; there is no aliasing at
all, let alone conflicting aliasing. This directly rules out "point is
non-Markov" for gridtimed: there's no missing information, full stop.

**The control came back the opposite of the ~0% expected.** modesel has a
small but real sufficiency gap — 4 groups where two states share a point
vector but need different actions. Root cause, confirmed from the examples:
`elapsed_frac = min(1.0, c/active_duration)` is 0 immediately after any
dispatch regardless of mode, and the status one-hot only encodes
waiting/active/done/failed — not FAST vs SLOW. So "just dispatched FAST"
and "just dispatched SLOW" are indistinguishable in `point` even though
they need different follow-up decisions. This is real, small
(0.74% of states), and on the domain where `mlp_point` performs *well*
(70-88% success) — the opposite of what "point insufficiency causes poor
mlp_point performance" would predict.

## 2. What's in the vector?

Both domains happen to have `point_dim=46` (confirmed against
`observation_space['point'].shape`, matches exactly on both). Two examples,
values confirmed to match the concrete state exactly (not stale/zeroed):

**gridtimed/T2**, state `GridState(pos=(2,2), clocks=(8,25,10), y=25)`:
```
pos_onehot[cell12] = 1.0
switch_clock[s1] = 8.0   switch_clock[s2] = 25.0   switch_clock[s3] = 10.0
y (global deadline clock) = 25.0
move[UP/DOWN/LEFT/RIGHT].status[plain_or_switch] = 1.0  (all 4 directions open)
```
(all other components — the other 20 position cells, the e_slot, other move-status/distance features — correctly 0)

**modesel/M3**, state with jobs at various cursors, `g=12`:
```
g (global clock) = 12.0
task[J0_T0/J0_T1/J1_T0/J2_T0/J2_T1].status[done] = 1.0
task[J0_T2/J1_T1/J1_T2/J2_T2].status[waiting] = 1.0
```
Both check out: position one-hot, all three switch clocks, and `y` are
present and correctly valued. Note there is no separate "which switches
are activated" bitmask — a switch's clock only encodes time-since-reset —
but §1 already confirms this never causes an actual sufficiency problem
(0% conflict), so it's a representational detail, not a gap.

## 3. Scaling

No normalization anywhere — confirmed by reading both `_get_point()`
implementations and `encoders/mlp_point.py` (`grep` for `BatchNorm`,
`LayerNorm`, or manual rescaling: no matches). Raw `point` goes straight
into the first `nn.Linear`.

| Component type | gridtimed/T2 range | modesel/M3 range |
|---|---|---|
| Position/status one-hot | [0, 1] | [0, 1] |
| Raw clock(s) | switch clocks & `y`: [0, 27], mean 9-20, std 4-7 | `g`: [0, 25], mean 12.2, std 4.4 |
| Elapsed/distance feature | signed gate-window distance: [-7.3, 5.7] | `elapsed_frac`: [0, 1] (explicitly clipped) |

The two domains' raw-clock components are on **comparable scales** (0-25
vs 0-27) — this isn't a case of one domain's inputs being orders of
magnitude larger. The one real difference: gridtimed's gate-window
distance feature is *signed* (can be negative), modesel's `elapsed_frac`
is not. Scaling looks like a minor, not dominant, candidate explanation.

## 4. Behavioral check (trained checkpoints, greedy rollout)

200 greedy (argmax) episodes per seed, all 3 gridtimed/T2 `mlp_point`
seeds, using the actual trained checkpoints pulled from Alpine:

| Seed | trained env_steps | success | deadline-expired | switch ever activated | gate ever attempted | mean ep length |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 118,784 | 0/200 | 200/200 | **0/200** | 200/200 | 27.0 |
| 1 | 225,280 | 0/200 | 200/200 | **0/200** | 200/200 | 27.0 |
| 2 | 210,944 | 0/200 | 200/200 | **0/200** | 200/200 | 27.0 |

Identical across all three independently-trained seeds, down to the exact
mean episode length. **This is a specific, fully reproducible degenerate
policy** — it consistently attempts gates (so it's not doing nothing) but
never once activates a switch, which the domain requires to open a gate's
window in time. Not random flailing, not "never found any signal" — it
found *a* stable behavior, just not the one that works.

Final policy entropy (avg of last 5 updates, avg across 3 seeds), gridtimed/T2:

| Encoder | first-update entropy | final entropy |
|---|---:|---:|
| `mlp_point` | 2.160 | **2.099** (highest of the four) |
| `deepset_h` | 2.004 | 1.941 |
| `cnn_dbm` | 2.156 | 1.976 |
| `cnn_proj` | 2.158 | 1.892 |

`mlp_point`'s entropy barely moved and stayed the *highest* — it did not
collapse to a narrow deterministic policy. Its stochastic policy still has
meaningful residual randomness; it's specifically the greedy/mode action
that consistently omits switch activation, not an entropy collapse.

## 5. Sanity control: capacity and exploration

`mlp_point` on gridtimed/T2, 50k steps, single seed, 3x hidden width
(172,117 params vs ~32k in the sweep) and `entropy_coef=0.05` (vs 0.01):

```
n_params: 172117 (5.4x the sweep encoder)
final_mean_return (avg last 3 updates): -74.24
greedy_rollout_success_rate: 0.0
greedy_rollout_switch_activation_rate: 0.0
```

**0% success, 0% switch activation — identical to the original result.**
Per the task's own threshold ("if success rate stays under ~5%, capacity
and exploration are ruled out"): both are ruled out.

## What's ruled out, stated plainly

- §1: gridtimed's `point` is not just sufficient, it's injective — not an
  observation bug.
- §3: no scaling/normalization anomaly large enough to explain a 3.7% vs
  ~50% success gap.
- §5: neither undersized network capacity nor insufficient exploration
  (5.4x params, 5x entropy coefficient) changes the outcome at all.
- §4 shows the actual, consistent, reproducible behavior: the trained
  policy never learns to activate switches, across every seed and every
  capacity/entropy setting tested here.

No direction is recommended. The evidence rules out three specific
competing explanations; it does not, on its own, establish what the right
next step is.
