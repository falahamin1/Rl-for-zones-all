# Track 2: Per-Job Deadlines with Slack-Driven Scheduling — Calibration Report

Scratch fork of `prob_jobshop_ext` in a new `deadline_track/` package (nothing under `prob_jobshop_ext` was modified). Calibration only — no encoders, no PPO, no training. All numbers below are from actually running the code, not estimated.

## 1. Model change

- **Clocks**: `2 * n_jobs` per instance — the existing per-task-resetting `c_J` (unchanged) plus a new `g_J` per job that starts at 0, advances with every time-elapse, and **never resets**. Enforced in `deadline_track/zone_ops.py::reset_clocks` with an actual assertion (`assert cname not in zctx.g_names`), the same pattern `prob_jobshop_ext/zone_ops.py:76` uses for `delta` — not a comment.
- **Instance model**: `DeadlineJobShopInstance` (subclass of `ProbJobShopInstance`) adds `deadlines: Dict[job_id, D_J]` and `release: Dict[job_id, r_J]` (default 0). `D_J = nominal_work(job) * (1+t)`, where "nominal duration" of a task is defined as `task.distribution.expected_duration()` (there is no other directly-available "unperturbed" duration field on `TaskDistribution` — stated as a modeling choice, not a given).
- **Deadlines are measured against `g_J`** (the non-resetting clock), not `c_J` — checked directly in the feasibility-check code and cross-validated (§ below), not just asserted.
- **Deadline variants of PJS_01–05 and LJS_01 are generated programmatically** (`deadline_track/instance.py::make_deadline_variant`) from the existing base instances — no hand-written instance files.
- **SCALE=20** integer encoding is used throughout the zone-construction path (both `c_J` and `g_J` bounds) since `D_J` can be fractional while pyudbm rejects non-integer bounds; the concrete simulation itself stays in exact real units throughout (task durations are always integers, so every reachable `g_J`/`c_J` value is already an exact integer — SCALE only matters at the pyudbm-constraint boundary).

## 2. Deadline semantics

- Feasibility is checked **on the dispatch edge only**, never as a blanket per-location invariant (which would block time-elapse for every unrelated job the moment one deadline expires and deadlock the whole automaton).
- Infeasible jobs auto-fail (all their still-`"waiting"` tasks become `"failed"`) **eagerly**, as an environment-side transition, not an agent-chosen action — this is what keeps the environment from ever reaching a decision point with zero available actions purely because of an expired deadline.
- Reward: unchanged `-elapsed` convention (elapsed read from the concrete side, sign convention identical to `region_env.py:86-111`), plus `-P` per newly-failed job this step. **P = 3× the instance's nominal makespan** (`ProbJobShopInstance.expected_makespan_lower_bound()`), computed per instance — reported per instance in §3's table (18.0–228.0 across the 6 instances).

**Real bug found and fixed before any calibration numbers were trusted**: the first implementation called `_advance_to_decision` once, then `_resolve_actions` once, per `step()`. `_advance_to_decision` only checks precedence/machine readiness (not deadline feasibility), so it can correctly stop with some task "ready" that `_resolve_actions` then discovers is infeasible and auto-fails. If *every* currently-ready task got auto-failed in that single pass while other machines still had tasks actively running, the environment reported zero available actions at a non-terminal state — a real deadlock (**41/200 episodes**, ~20%, in the first stress test on PJS_04 at t=0.0). Fixed by looping advance↔resolve until either a real action exists or the episode is terminal (`_advance_and_resolve`, capped at `len(task_ids)+1` iterations with a hard `RuntimeError` if exceeded, not a silent retry). **After the fix: 0 deadlocks in 15,000 calibration episodes and 0 in every other run in this report.**

## 3. Calibration — 500 seeded episodes/instance, uniform-random policy over available actions, paired seeds across t (same 500 episode seeds reused for every t value of a given instance, for a clean within-instance comparison)

| Instance | t=0.0 | t=0.1 | t=0.25 | t=0.5 | t=1.0 | P (penalty) |
|---|---:|---:|---:|---:|---:|---:|
| PJS_01 | 13.2% | 13.2% | 0.0% | 0.0% | 0.0% | 18.0 |
| PJS_02 | 33.2% | 7.6% | 7.6% | 0.0% | 0.0% | 24.0 |
| PJS_03 | 87.6% | 87.6% | 71.8% | 48.0% | 16.0% | 39.3 |
| PJS_04 | 62.4% | 62.2% | 13.6% | 2.2% | 0.0% | 30.0 |
| PJS_05 | 99.6% | 99.2% | 96.0% | 79.0% | 25.6% | 50.4 |
| LJS_01 | 100.0% | 100.0% | 100.0% | 99.0% | 39.6% | 228.0 |

(Cell values = fraction of episodes with ≥1 deadline violation. `deadlock_fraction = 0.0000` for every one of these 30 cells — 15,000 episodes total.)

**Calibrated t (closest to 25% violation) per instance, all within the requested [5%,50%] target band:**

| Instance | Calibrated t | Violation fraction | Mean violated jobs/episode |
|---|---:|---:|---:|
| PJS_01 | 0.0 | 13.2% | 0.132 |
| PJS_02 | 0.0 | 33.2% | 0.332 |
| PJS_03 | 1.0 | 16.0% | 0.166 |
| PJS_04 | 0.25 | 13.6% | 0.136 |
| PJS_05 | 1.0 | 25.6% | 0.278 |
| LJS_01 | 1.0 | 39.6% | 0.454 |

No instance needed rescaling outside the requested t grid to land in [5%,50%].

## 4. Geometry — mutual-pinning check first, as instructed

**Headline finding**: across all 6 calibrated instances, **both** the uniform-release (r_J=0 for every job) **and** the staggered-release (r_J ∈ {0,1,2,...} spread across jobs) variants show **mutual_pinning_fraction = 1.000, exactly, with zero exceptions**, over up to 2000 sampled clipped zones per (instance, variant) — 24 cells, all 1.000.

This is not a coincidence to be improved by tuning — it is a mathematical certainty given the model as specified: every `g_J` starts at 0 and advances by the identical `dt` at every micro-step regardless of which job's task is active (UDBM's `up()` delay operator advances *all* clocks in a Federation uniformly; there is no per-clock rate), and no `g_J` is ever reset. So `g_J1 == g_J2 == ... == g_Jn` holds identically for the entire episode, for every episode, regardless of instance or release-time spread. **Staggered release does not fix this and cannot**: `r_J` only gates *when a job's first task may be dispatched* (a precondition on the dispatch edge); it does not touch `g_J`'s own growth trajectory at all. Both variants are reported side-by-side above precisely because the task asked to test this rather than assume a fix would work — it doesn't, and the mechanism why is exact, not approximate.

**Consequence for zone dimensionality**: with `delta` removed (per the encoder-slice track's convention) and `g_J`'s collapsing to a single pinned group, **every sampled zone, on every instance, had affine dimension exactly 0** — a single point:

| Instance | t | affine_dim (mean, both variants) | facets (mean) | bounded_frac | vertices (mean) |
|---|---:|---:|---:|---:|---:|
| PJS_01 | 0.0 | 0.00 | 8.00 | 1.000 | 1.00 |
| PJS_02 | 0.0 | 0.00 | 8.00 | 1.000 | 1.00 |
| PJS_03 | 1.0 | 0.00 | 12.00 | 1.000 | 1.00 |
| PJS_04 | 0.25 | 0.00 | 12.00 | 1.000 | 1.00 |
| PJS_05 | 1.0 | 0.00 | 16.00 | 1.000 | 1.00 |
| LJS_01 | 1.0 | 0.00 | 24.00 | 1.000 | **error** (see anomaly) |

This holds not just for the near-deterministic small instances but universally across all 6, including the larger/more-probabilistic ones. Mechanism (verified, not assumed): a "waiting" job's own `c_J` is *also* never reset until its first dispatch, so it too tracks raw elapsed time exactly like `g_J` until that point — meaning every clock in the system is, at every decision point, either exactly pinned by a completion guard (active/done tasks) or exactly tied to the single shared elapsed-time value (waiting tasks + all `g_J`'s). There is never any remaining genuine geometric freedom once the discrete location is fixed. This is the "slack comparison collapses to a scalar" case the task named as the risk, occurring universally rather than only under high pinning risk.

## 5. Decision relevance (exact DP, uniform-release variant — staggered gave identical pinning in §4, so there was no variant with *lower* pinning to prefer; stated explicitly rather than picked silently)

| Instance | t | Concrete states | Truncated | Multi-zone locations | Conflicting locations | Conflict fraction | No-deadline baseline |
|---|---:|---:|:---:|---:|---:|---:|---:|
| PJS_01 | 0.0 | 31 | No | 7 | 0 | 0.0% | 0.0% |
| PJS_02 | 0.0 | 151 | No | 17 | 0 | 0.0% | 0.0% |
| PJS_03 | 1.0 | 1,030 | No | 54 | 1 | **1.85%** | 0.0% |
| PJS_04 | 0.25 | 2,242 | No | 135 | 0 | 0.0% | **1.4%** |

Note the flip relative to baseline: PJS_03 gains a conflict under deadlines that didn't exist without them; PJS_04 loses the one it had. Both are genuine — deadlines change *which* states are decision-relevant, not just how many exist — but statistical power is thin either way (1 conflict out of 54 multi-zone locations for PJS_03; 0 out of 135 for PJS_04), consistent with how little power the no-deadline baselines had too.

### Worked example (PJS_03, the one conflict found)

**Location**: `J1_T1=done, J1_T2=active(d=8), J2_T1=done, J2_T2=waiting, J3_T1=waiting, J3_T2=waiting` — J1's second task is mid-flight with a realized (long) duration of 8; J2_T2 and J3_T1 are both otherwise-ready and compete for dispatch.

| | Zone I (g=1) | Zone J (g=8) |
|---|---|---|
| **DP-optimal action** | `J3_T1` | `J2_T2` |
| job J1: g, D, remaining, W, **slack** | 1, 18.00, [J1_T2], 5.00, **12.00** | 8, 18.00, [J1_T2], 5.00, **5.00** |
| job J2: g, D, remaining, W, **slack** | 1, 16.60, [J2_T2], 5.00, **10.60** | 8, 16.60, [J2_T2], 5.00, **3.60** |
| job J3: g, D, remaining, W, **slack** | 1, 16.20, [J3_T1,J3_T2], 8.10, **7.10** | 8, 16.20, [J3_T1,J3_T2], 8.10, **0.10** |

**Tested, not assumed**: does the action flip coincide with a slack-ordering inversion? **No.** J3 has the strictly lowest slack in *both* zones (ranking `J3 < J2 < J1` in Zone I, and still `J3 < J2 < J1` in Zone J — `ranking_inverted = False`, checked programmatically). A naive "always dispatch minimum-slack job first" heuristic would predict `J3_T1` in both zones; the DP instead flips to `J2_T2` in Zone J even though J3 is still (nominally) the most urgent job there. The likely mechanism, stated as a hypothesis rather than further verified within this task's scope: by Zone J, J3's nominal slack (0.10) is so thin that dispatching J3_T1 immediately doesn't meaningfully change J3's odds (nominal slack uses *expected* durations, while the actual pass/fail boundary — §2 — uses *minimum* durations, so a job can sit at near-zero nominal slack while still being genuinely feasible or already doomed regardless of dispatch order); J2, at slack 3.60, is in the region where dispatch timing still has real leverage, making it the better target for the DP's actual objective (minimize expected cost) even though it isn't the nominally most urgent job. This means the nominal-slack heuristic is a useful *lens* on this model but is demonstrably not what the DP is actually optimizing.

## 6. Sanity gate (blocking — run before anything above was trusted)

Hand-built 2-job, 1-task-each instance sharing one machine (durations 3 and 5, deterministic), in two deadline configurations chosen so the optimal dispatch order is computable on paper via minimum-slack-first, with `P` set far larger than the 8-unit makespan itself so there's no numeric ambiguity:

| | slack_A | slack_B | Expected order | DP result |
|---|---:|---:|---|---|
| "A tight" (D_A=3, D_B=8) | 0 | 3 | A first | `['A_T1']` ✓, V(root)=8.00 |
| "B tight" (D_A=8, D_B=5) | 5 | 0 | B first | `['B_T1']` ✓, V(root)=8.00 |

**PASSED** on both. `V(root)=8.00` (the bare makespan, no penalty) in both cases confirms the DP correctly avoids the strictly-dominated wrong-order choice.

## Anomalies / things flagged rather than worked around

1. **Deadlock bug (found and fixed)** — §2, 41/200 episodes on the first stress test before the advance↔resolve loop fix; 0/15,000+ after.
2. **A second real bug, caught by running the geometry sweep, not assumed away**: the first version of the zone-clipping bound (`m_c`, used for `c_J`'s `clip_to_k_box`) was sized to a single task's max duration. That's wrong for a *waiting* job's `c_J`, which (§4) tracks raw elapsed time exactly like `g_J` until its first dispatch and can therefore be far larger — `clip_to_k_box`'s hard intersection emptied the zone the instant this happened (crashed on PJS_03). Fixed by using the instance's worst-case-makespan bound for both clock families uniformly.
3. **Feasibility-check equivalence validated, not assumed**: the live environment's feasibility gate uses an arithmetic shortcut (fast; needed for 15,000+ calibration episodes) claimed equivalent to a real zone-based check (`zone_ops.feasible_best_case`, genuine DBM reset/invariant/delay/guard chain) because `g_J` is provably shared across jobs (§4). Cross-checked directly: **2,484/2,484 sampled decision points agree (100.00%)** across PJS_04 (t=0.1, t=0.5) and LJS_01 (t=0.25).
4. **`v_rep()` exact vertex enumeration failed for all 200/200 sampled LJS_01 zones** — not a correctness bug: `VertexEnumerationLimit: C(24,12) = 2,704,156 exceeds the 200,000 exact-enumeration limit`. Every one of these zones is provably a single point (affine dim 0, verified independently via the union-find pinning check), so the true vertex count is trivially 1 — but the general-purpose exact-enumeration implementation enumerates `C(facets, full_clock_dim)` subsets rather than `C(facets, affine_dim)`, so it doesn't special-case an already-degenerate zone and the combinatorics blow up at LJS_01's 12-clock count even though the geometric answer is trivial. Reported as a limitation of the vertex-counting method at high clock counts on fully-degenerate input, not a disagreement between counters (no second counter was available to disagree — this is the only exact-enumeration implementation used throughout, no `scipy.spatial.qhull` anywhere in this codebase).
5. No `pyudbm` rejection occurred anywhere in this run (all `Federation`/`Context`/`Clock` operations succeeded once the two bugs above were fixed); no zone was unexpectedly empty except the ones diagnosed in (2), which were fixed at the root cause.

## Files

`deadline_track/{instance,zone_ops,symbolic_env,dp,collect,decision_relevance}.py`, `deadline_track/{sanity_gate,validate_feasibility,calibration,geometry,run_section5}.py` (entry points). Result JSON: `/tmp/deadline_results/{calibration,geometry,section5}.json`. `prob_jobshop_ext/` was not modified.
