# Benchmark specifications for the paper — sourced facts

READ-ONLY extraction, no code modified. Every number below is either a direct
`file:line` citation, or (where marked) a value I resolved by *running* the
repo's own generator/DP code with the exact kwargs `run_one.py` uses for the
sweep — those are marked "(resolved by execution)" rather than claimed as a
literal line in the file, since they're not hardcoded anywhere as a single
constant. `UNKNOWN` means genuinely not determinable from what's in this
repo — I did not fill any gap with a plausible-sounding number.

---

## GRIDTIMED (T2, T3)

### 1. Clocks — the arithmetic discrepancy, resolved

Your notes undercounted by exactly one clock: **the internal edge clock `e`
is real and is part of the ambient dimension, and your arithmetic left it
out.**

- `instance.clock_names` (`gridtimed/instance.py:119-120`) returns **only**
  the switch clocks `s1..sK` — for T2 (`n_switches=3`) that's `('s1','s2','s3')`,
  a 3-tuple. This is *not* the full clock set used for zone construction.
- The zone/DBM context adds two more: `gridtimed/zone_ops.py:41` —
  `self._clock_names = list(instance.clock_names) + [GLOBAL_CLOCK, EDGE_CLOCK]`,
  with `GLOBAL_CLOCK = "y"`, `EDGE_CLOCK = "e"` (`gridtimed/zone_ops.py:31-32`).
- `zoneviews/zone.py:225` — `self.d = len(clock_names)` — so the ambient
  dimension `d` used by every `Zone` **is** `n_switches + 2` (switches + `y`
  + `e`), not `n_switches + 1`.

| Instance | `n_switches` (`gridtimed/tiers.py:27-28`) | clocks = switches + y + e | ambient `d` |
|---|---|---|---|
| T2 | 3 | s1, s2, s3, y, e | **5** |
| T3 | 4 | s1, s2, s3, s4, y, e | **6** |

This matches your notes' totals (5 and 6) exactly once `e` is counted — the
"three switches + one global = 4" arithmetic simply omitted `e`.

**Reset behavior**, each cited:
- `s_i`: resets to 0 on entering switch `i`'s cell, unconditionally —
  `gridtimed/gym_env.py:142`, `reset_clocks(Z2, self._zctx, [self.instance.switch_clock(target)])`.
- `y`: **never** resets — enforced by an assertion, not just documented:
  `gridtimed/zone_ops.py:52`, `assert name != GLOBAL_CLOCK, ...`.
- `e`: resets **twice per move** — once at the start of `edge_delay()`
  before the elapsed-time invariant/guard is applied (`gridtimed/zone_ops.py:83`),
  and again unconditionally right after, every single move
  (`gridtimed/gym_env.py:152`, `reset_clocks(Z2, self._zctx, [EDGE_CLOCK])`).
  Per the module docstring (`gridtimed/zone_ops.py:15-17`): *"e is reset
  again immediately after being used (so every REPORTED zone has e pinned
  to a constant)."*

**Is `e` part of the automaton's discrete state, or an implementation
device?** Present it as **an implementation device of the zone/DBM
construction, not a state variable of the automaton**. The concrete,
DP-level state (`GridState`, `gridtimed/env.py`, fields `pos`, `clocks`,
`y` — no `e` field at all) never carries `e`; the discrete transition
function `step()`/`peek()` (`gridtimed/env.py:79-113`) never references
`e` either — `elapsed = delay + 1` (`gridtimed/env.py:91`) is computed and
consumed directly, with no clock object involved. `e` exists solely inside
`GridZoneContext` to give the *online zone-tracking* (used by zone-aware
encoder observations: `deepset_h`/`cnn_dbm`/`cnn_proj`/`deepset_v`/`gnn`)
a scratch clock that can bound one edge's real, unobserved duration on
`[delay+1, delay+1+w]` without that uncertainty leaking into any other
clock's own accumulated bound.

**Does it appear in `d`, `h_rep`, `v_rep`, the graph view, affine
dimension?** Yes to `d` (shown above) and yes, mechanically, to `h_rep()`
(which iterates the DBM's full `d`-dimensional constraint set) and to
`v_rep()`'s **output** (vertices are expanded back to full `d`-dimensional
points before being returned, `zoneviews/zone.py`'s `v_rep()` docstring at
`zoneviews/zone.py:347-352`) — every reported vertex carries an `e`
coordinate. But because `e` is always reset to a pinned constant by the
time any zone is actually observed (both resets above happen *before*
the zone is read for an observation), it contributes **0 to the free/affine
dimension**: `affine_dim()` (`zoneviews/zone.py:501-`) counts components
after collapsing exactly-pinned clocks, and `e` is always in a
pinned-to-reference-clock equivalence class at observation time. So: `e`
is real in `d` and in the raw half-space/vertex machinery, but it never
contributes to `affine_dim()` and its value is a constant, not a free
coordinate, in every zone the encoders actually see.

### 2. Action space

- Moves: `MOVES = ("UP", "DOWN", "LEFT", "RIGHT")` (`gridtimed/instance.py:66`) — 4.
- Delays: `DEFAULT_DELAYS = (0, 1, 2)` (`gridtimed/instance.py:74`); confirmed
  **not overridden** for T2/T3 — `run_one.py`'s `_build_instance` call to
  `grid_make_instance` (`run_one.py:281-286`) passes no `allowed_delays`
  kwarg, so the generator's own default (`gridtimed/generator.py:69`,
  `= DEFAULT_DELAYS`) applies; resolved-by-execution instance objects for
  both T2 and T3 print `allowed_delays=(0, 1, 2)`.
- **No STAY/no-op action** — the action set is exactly
  `{d{delay}_{move} : delay in allowed_delays, move in MOVES}`
  (`gridtimed/env.py:56`, `all_actions`). A wall bump or closed-gate attempt
  self-loops behaviorally, but it is not a distinct "stay" action — it's a
  masked-out directional move (see below).
- **|A| = 4 moves × 3 delays = 12**, for both T2 and T3 (delay set is the
  same for all tiers; grid size doesn't change the action set, only the
  state space).
- **Masking**: `action_mask()` (`gridtimed/env.py:116-117`) = `{aid:
  peek(instance, state, aid).legal for aid in all_actions(instance)}`,
  exposed to the policy as `obs["mask"]`. An action is `legal=False`
  (masked) in exactly two cases, both in `peek()`:
  1. target cell out of bounds (`gridtimed/env.py:88-89`);
  2. target is a gate cell and the paired switch clock is outside the
     gate's window at that instant (`gridtimed/env.py:95-100`).

  Masked actions are never selectable by the policy (`gridtimed_eval.py`'s
  and training's own logit-masking both set masked logits to `-inf`); the
  underlying `step()`/`peek()` still *supports* calling them directly
  (self-loop, `ILLEGAL_PENALTY`, `elapsed=0` — see §3), which is what the
  DP/BFS solvability check uses, but the trained policy's action
  distribution excludes them.

### 3. Reward function and success criteria

Constants, `gridtimed/instance.py:61-64`:
```
REWARD_SUCCESS = 100.0
TIME_COST = 1.0          # per unit of elapsed y on any legal (moving) action
ILLEGAL_PENALTY = 2.0    # cost of taking an illegal action (wall / closed gate); no time elapses
DEADLINE_FAIL_PENALTY = 50.0  # terminal cost when y > DEADLINE before reaching GOAL
```
Cost assembly, `gridtimed/env.py:79-113` (`peek()`, also used unmodified as
`step()`, `gridtimed/env.py:120-122`):
```python
if not instance.in_bounds(target):
    return StepResult(state, ILLEGAL_PENALTY, 0, legal=False, ...)          # env.py:88-89
...
if instance.is_gate(target):
    ...
    if not (lo <= val <= hi):
        return StepResult(state, ILLEGAL_PENALTY, 0, legal=False, ...)      # env.py:95-100
...
if target == instance.goal and new_y <= instance.deadline:
    cost = TIME_COST * elapsed - REWARD_SUCCESS
    return StepResult(DONE, cost, elapsed, legal=True, newly_done=True, ...)  # env.py:105-107
if new_y > instance.deadline:
    cost = TIME_COST * elapsed + DEADLINE_FAIL_PENALTY
    return StepResult(FAILED, cost, elapsed, legal=True, newly_done=False, newly_failed=True)  # env.py:108-110
...
return StepResult(next_state, TIME_COST * elapsed, elapsed, legal=True, ...)  # env.py:112-113 (ordinary move)
```
`reward = -res.cost [+ shaping_bonus]` (`gridtimed/gym_env.py:162`) — the
shaping term (`switch_shaping_reward * 1{first activation of this switch
this episode}`) is a **separate, opt-in experiment only** (`RunSpec.shaped`,
`run_one.py:169-182`); for the 168 non-shaping-experiment sweep configs it's
hardcoded to `0.0` (`run_one.py:313`, `env_kwargs["switch_shaping_reward"]
= SWITCH_SHAPING_REWARD if spec.shaped else 0.0`), so for T2/T3's own
120+ sweep runs it is byte-identically absent.

**Success criterion**: `info["success"] = self._state.pos == "DONE"`
(`gridtimed/gym_env.py:165`), set only when `done`. `DONE`/`FAILED` are the
two terminal sentinels (`gridtimed/env.py:39-44`, `pos="DONE"` /
`pos="FAILED"`).

**Episode horizon / termination**: **no `TimeLimit`/step-count wrapper
anywhere** in the training path (grepped `gridtimed/gym_env.py`, `train/`,
`run_one.py` for `TimeLimit`/`max_episode_steps`/`max_steps` —
zero matches). Termination is purely a consequence of the deadline
mechanic: `y` strictly increases by `elapsed = delay+1 >= 1` on every
*selectable* (i.e. legal, unmasked) move, so the episode is guaranteed to
hit `FAILED` (if it hasn't already reached `DONE`) once `y > deadline`,
within a number of steps bounded by the deadline. (Illegal actions don't
advance `y`, but they're masked out of the policy's action set entirely —
see §2 — so they can't be used to stall.)

**Discount factor**: `gamma = 0.99`, `PPOConfig` default
(`train/ppo.py:89`), not overridden per-domain in `run_one.py`'s
`build_trainer` (`run_one.py:297-309` only overrides `total_steps`,
`plateau_patience_steps`, `plateau_enabled`, `seed`, `num_envs`) —
confirmed against a real manifest's recorded hyperparameters
(`results/modesel__M3__mlp_point__seed0__manifest.json`, `gamma` field
matches 0.99; same `PPOConfig` path is shared by every domain).

### 4. Instance parameters, resolved (T2, T3)

Tier definitions, `gridtimed/tiers.py:27-28`:
```
T2: grid_n=5, n_switches=3, n_gates=3, window_width=2, D=27
T3: grid_n=6, n_switches=4, n_gates=4, window_width=2, D=33
```
Call site, `run_one.py:281-286`: `grid_make_instance(grid_n=cfg["grid_n"],
n_switches=cfg["n_switches"], n_gates=cfg["n_gates"],
window_width=cfg["window_width"], D=cfg["D"], w=0.25, seed=INSTANCE_SEED,
label=instance)`, with `INSTANCE_SEED = 0` (`run_one.py:65`).

**Placement/window generation formula** (`gridtimed/generator.py:78-107`):
switch and gate cells are `n_switches + n_gates` cells drawn by shuffling
all free (non-start, non-goal) grid cells under `random.Random(sub_seed)`
where `sub_seed = seed * 1_000_003 + attempt` (`gridtimed/generator.py:88-89`);
each gate's window is `(a, a+window_width)` with `a = randint(grid_n,
max(grid_n, D - window_width - 1))` (`gridtimed/generator.py:100-102`); the
candidate is accepted on the first attempt for which a concrete BFS
(`is_solvable`, `gridtimed/generator.py:33-58`) confirms GOAL is reachable
within `D`.

**Resolved values** (by executing `gridtimed.generator.make_instance` with
the exact kwargs above — deterministic given `seed=0`):

| | T2 | T3 |
|---|---|---|
| grid | 5×5 | 6×6 |
| start / goal | (0,0) / (4,4) | (0,0) / (5,5) |
| switches | (4,1), (1,1), (3,2) | (0,3), (3,1), (1,5), (4,0) |
| gates | (2,0), (4,0), (4,3) | (5,3), (5,2), (1,2), (3,4) |
| gate windows `[a_i,b_i]` | (23,25), (8,10), (15,17) | (19,21), (29,31), (6,8), (22,24) |
| deadline D | 27 | 33 |
| w | 0.25 | 0.25 |
| allowed_delays | (0,1,2) | (0,1,2) |
| generation: sub_seed / attempts | 1 / 1 (accepted first try) | 1 / 1 |
| ambient clocks d | 5 | 6 |
| \|A\| | 12 | 12 |

`w=0.25` traces to `gridtimed/tiers.py:21` (`W = 0.25`, "Task A validated
soundness at this w") and is passed as a literal `0.25` at the `run_one.py`
call site — same numeric value, not coincidence, but two separate
hardcodes (worth flagging: nothing forces these to stay in sync if either
is edited independently — see §Flags).

**Are durations sampled at rollout time, or fixed?** **Fixed/deterministic
at rollout time.** `elapsed = delay + 1` (`gridtimed/env.py:91`) — no `w`,
no RNG, no `random`/`np.random` call anywhere in `gridtimed/env.py` (grepped,
zero matches). `w` appears **only** inside `zone_ops.py`'s `edge_delay()`
(`gridtimed/zone_ops.py:74-87`), which constrains the *zone's* invariant/guard
to `[delay+1, delay+1+w]` for the online DBM used by zone-aware encoder
observations — it never touches the actual transition, reward, or episode
outcome. This confirms the framing in your notes is correct for gridtimed
too, not just trmgrid (see next section).

---

## TRMGRID (G_SMALL, G_LARGE)

### 1. Clocks

- `TRMGridInstance.clock_names` (`trmgrid/instance.py:107-108`) = `t_0 ..
  t_{k-1}`, one per sub-goal — for both G_SMALL and G_LARGE, `k=3`
  (`run_one.py:127-128`, `dict(grid_n=4, k=3)` / `dict(grid_n=5, k=3)`), so
  3 clocks from this property alone.
- Zone context adds two more, identically to gridtimed's pattern:
  `trmgrid/zone_ops.py:42` — `self._clock_names = list(instance.clock_names)
  + [GLOBAL_CLOCK, EDGE_CLOCK]`, `GLOBAL_CLOCK = "g"`, `EDGE_CLOCK = "e"`
  (`trmgrid/zone_ops.py:32-33`).
- `zoneviews/zone.py:225` — same `self.d = len(clock_names)` — so **ambient
  `d = k + 2 = 5` for both G_SMALL and G_LARGE** (unlike gridtimed, where
  T2/T3 differ in clock count because `n_switches` differs — trmgrid's two
  tiers both use `k=3`, only `grid_n` differs between them; see §Flags on
  why G_LARGE ended up `k=3` not the originally-planned `k=4`).

**Reset behavior**:
- `t_i`: resets to 0 on physical arrival at sub-goal `i`'s cell,
  unconditionally — three branches in `peek()` all zero the relevant
  `new_clocks[idx]` on arrival: `trmgrid/env.py:115` (first sub-goal touched),
  `trmgrid/env.py:125` (second, phase-advancing touch), `trmgrid/env.py:142`
  ("incidental touch" — touching a sub-goal that doesn't advance phase
  still resets its own clock unconditionally, per the inline comment at
  that line).
- `g`: never resets — `trmgrid/zone_ops.py:53`, same assertion pattern as
  gridtimed's `y`.
- `e`: reset twice per move, identically to gridtimed —
  `trmgrid/zone_ops.py:82` (inside `edge_delay()`) and
  `trmgrid/gym_env.py:172` (unconditional second reset after use).

**Same verdict as gridtimed on `e`**: implementation device of the zone
construction, not part of the automaton's discrete state (`TRMState`,
`trmgrid/env.py`, has fields `phase`, `pos`, `clocks`, `g` — confirmed at
the constructor call `trmgrid/env.py:149`, no `e` field), not referenced by
`step()`/`peek()`'s transition or reward logic at all. It is part of `d`
and of `v_rep()`'s output coordinates, contributes 0 to `affine_dim()` at
observation time, for the same reason as gridtimed (always pinned by the
time a zone is read).

### 2. Action space

- Same `MOVES` (4) and `DEFAULT_DELAYS = (0,1,2)` (`trmgrid/instance.py:45,58`),
  not overridden at the `run_one.py:288-291` call site — resolved instances
  for both G_SMALL/G_LARGE print `allowed_delays=(0, 1, 2)`.
- **No STAY/no-op action** — same action-set construction as gridtimed
  (`trmgrid/env.py`, `all_actions`, mirrors `gridtimed/env.py:56`).
- **|A| = 4 × 3 = 12**, both instances.
- **Masking differs from gridtimed** — this is a real domain-behavior
  difference, not a copy-paste-identical mechanism. `action_mask()`
  (`trmgrid/env.py:153-154`) mirrors gridtimed's `{aid: peek(...).legal
  for aid in all_actions}` structure, but `peek()`'s only `legal=False`
  case is **out-of-bounds** (`trmgrid/env.py:101-102`). Guard-window
  violations (visiting the second sub-goal too early/late,
  `trmgrid/env.py:119-123`; reaching GOAL after the second sub-goal's cap,
  `trmgrid/env.py:126-132`) and deadline expiry
  (`trmgrid/env.py:133-136`, `144-147`) are **not masked** — they're
  `legal=True` but terminal failures (`newly_failed=True`,
  `fail_reason` in `{"guard_pair","guard_goal","deadline"}`). This matches
  the explicit design note in `trmgrid/instance.py:26-28`: *"Guard
  violations... are TERMINAL FAILURES with a penalty, checked ON THE EDGE
  -- unchanged from v1, not a blocked/illegal move."* So a trmgrid policy
  genuinely can select an action that ends the episode in failure; a
  gridtimed policy cannot select a gate-window violation at all (it's
  masked out before selection).

### 3. Reward function and success criteria

Constants, `trmgrid/instance.py:53-56`:
```
REWARD_SUCCESS = 100.0
TIME_COST = 1.0           # per unit of elapsed g on any legal (moving) action
ILLEGAL_PENALTY = 2.0     # cost of an out-of-bounds move; no time elapses
FAIL_PENALTY = 50.0       # terminal cost: guard violation OR deadline expiry
```
Cost assembly, `trmgrid/env.py:96-150` (`peek()`, reused as `step()`):
```python
if not instance.in_bounds(target):
    return StepResult(state, ILLEGAL_PENALTY, 0, legal=False, ...)                    # env.py:101-102
...
# second sub-goal visited outside its ordered-pair window:
cost = TIME_COST * elapsed + FAIL_PENALTY
return StepResult(failed_state(...), cost, ..., fail_reason="guard_pair")             # env.py:119-123
...
# reach GOAL with clock_j > cap_j:
cost = TIME_COST * elapsed + FAIL_PENALTY
return StepResult(failed_state(...), cost, ..., fail_reason="guard_goal")             # env.py:126-132
...
if new_g > instance.deadline:                                                          # checked twice: env.py:133 and env.py:144
    cost = TIME_COST * elapsed + FAIL_PENALTY
    return StepResult(failed_state(...), cost, ..., fail_reason="deadline")
...
# success: GOAL reached, phase=(i,j), clock_j <= cap_j, g <= deadline:
cost = TIME_COST * elapsed - REWARD_SUCCESS
return StepResult(done_state(...), cost, ..., newly_done=True)                        # env.py:137-138
...
return StepResult(next_state, TIME_COST * elapsed, elapsed, legal=True, ...)          # env.py:150 (ordinary move)
```
`reward = -res.cost` (`trmgrid/gym_env.py:180`) — **no shaping term at
all** for trmgrid (no `switch_shaping_reward`-equivalent kwarg exists on
`TRMGridTrainEnv`; grepped, not present).

**Success criterion**: `info["success"] = self._state.pos == "DONE"`
(`trmgrid/gym_env.py:183`), identical pattern to gridtimed.

**Episode horizon**: same as gridtimed — no step-count wrapper found;
bounded naturally by `g` strictly increasing on every selectable move and
the deadline check.

**Discount factor**: `gamma = 0.99`, same shared `PPOConfig` default, not
overridden for trmgrid.

### 4. Instance parameters, resolved (G_SMALL, G_LARGE)

Tier dict, `run_one.py:126-128`: `TRMGRID_TIERS = {"G_SMALL": dict(grid_n=4,
k=3), "G_LARGE": dict(grid_n=5, k=3)}`. Shared constants,
`run_one.py:118-120,125`: `TRMGRID_ALPHA = 0.25`, `TRMGRID_WIDTH = 8`,
`TRMGRID_W = 0.25`, `TRMGRID_INSTANCE_SEED = 2`. Call site,
`run_one.py:288-291`.

**Formulas** (`trmgrid/generator.py:69-89`, distance-relative, per the
module docstring):
```
lo_ij = ceil(d_ij * (1 + alpha)),   hi_ij = lo_ij + width      (per ordered pair i != j)
cap_j = ceil(d_j_goal * (1 + alpha)) + width                    (j = second sub-goal visited)
```
where `d_ij`/`d_j_goal` are Manhattan distances between the actual placed
cells. Deadline `D` is a generous, then-verified estimate
(`trmgrid/generator.py:92-107`, `_estimate_deadline`): the best-case ordered
pair's `(start→i distance) + hi_ij + cap_j`, plus a fixed margin of 4.
Placement uses the same `sub_seed = seed * 1_000_003 + attempt` shuffle
idiom as gridtimed (`trmgrid/generator.py:121-122`).

**Resolved values** (by executing `trmgrid.generator.make_instance` with
the exact kwargs above):

| | G_SMALL | G_LARGE |
|---|---|---|
| grid | 4×4 | 5×5 |
| start / goal | (0,0) / (3,1) | (0,0) / (4,4) |
| sub-goals (i=0,1,2) | (1,3), (3,3), (1,0) | (2,3), (4,3), (1,3) |
| windows `[lo_ij,hi_ij]` | (0,1):(3,11) (0,2):(4,12) (1,0):(3,11) (1,2):(7,15) (2,0):(4,12) (2,1):(7,15) | (0,1):(3,11) (0,2):(2,10) (1,0):(3,11) (1,2):(4,12) (2,0):(2,10) (2,1):(4,12) |
| caps (cap_0,cap_1,cap_2) | 13, 11, 12 | 12, 10, 13 |
| alpha | 0.25 | 0.25 |
| width | 8 | 8 |
| deadline D | 30 | 30 |
| w | 0.25 | 0.25 |
| allowed_delays | (0,1,2) | (0,1,2) |
| generation: sub_seed / attempts | 2000007 / 1 | 2000007 / 1 |
| ambient clocks d | 5 (k=3 + g + e) | 5 (k=3 + g + e) |
| \|A\| | 12 | 12 |

**Are move durations stochastic at rollout time?** **No — genuinely
deterministic, exactly like gridtimed, confirmed the same way.**
`elapsed = delay + 1` (`trmgrid/env.py:104`), and grepping the entire file
for `random`/`np.random`/`rng`/`uniform` returns **zero matches**. `w` is
used **only** in `trmgrid/zone_ops.py`'s `edge_delay()`
(`trmgrid/zone_ops.py:75-86`), for the zone/DBM invariant-guard chain, not
the concrete transition. `trmgrid/gym_env.py:149` calls `peek()` (the same
function DP uses) for the actual step, not any sampling path. **This
confirms the earlier report's claim exactly**: the concrete DP/training
transition is deterministic on `elapsed=delay+1`; `w` enters only through
zone construction. Since environment stochasticity is genuinely zero given
a fixed seed and a deterministic (greedy/argmax) policy, the `n=1` greedy
evaluation used throughout this project (`EVAL_N_GREEDY_EPISODES=1`,
`trmgrid_eval.py:31`) is valid for exactly the reason its own docstring
states (`trmgrid_eval.py:5-15`) — this was independently verified here by
reading `env.py`, not just trusted from the docstring.

---

## MODESEL (M3, M4)

*(Extracted by a subagent; I did not personally re-verify every line
number in this section, only spot-checked the headline numbers against
the manifest files, which matched.)*

### 1. Clocks

- `modesel/zone_ops.py:32-37` (`ModeSelZoneContext.__init__`): one clock
  `c_i` per **job** (elapsed time since that job's active task was
  dispatched), plus one global clock `g` — `self.job_clock_names =
  [f"c{i}" for i in range(instance.n_jobs)]`;
  `self._clock_names = self.job_clock_names + [GLOBAL_CLOCK]`.
- Reset: `c_i` resets at dispatch (`modesel/zones.py:57`) **and again**
  when that task completes (`modesel/zones.py:73`). `g` never resets
  (assertion, `modesel/zone_ops.py:49`).
- **Total clocks = `n_jobs + 1`**, explicit at `modesel/gym_env.py:53`
  (`self.n_clocks = instance.n_jobs + 1`).
- **No auxiliary edge/bookkeeping clock analogous to `e`** — explicitly
  stated, `modesel/zone_ops.py:6-11`: *"No extra bookkeeping clock is
  needed here the way gridtimed needed e... the per-job clock IS the
  task-specific clock the duration interval naturally bounds."* Ambient
  `d = n_jobs + 1` exactly, nothing folded in beyond the reported clocks
  (`modesel/gym_env.py:138-140`, `Zone(self._Z, self._zctx.clock_names,
  k_vec)`; `zoneviews/zone.py:225`).

| Instance | n_jobs | clocks | ambient d |
|---|---|---|---|
| M3 | 3 | c0,c1,c2,g | **4** |
| M4 | 4 | c0,c1,c2,c3,g | **5** |

### 2. Action space

- Actions are **(task, mode) dispatch pairs only** —
  `modesel/env.py:70-71`, `MODES = (FAST, SLOW)` (`modesel/instance.py:20-22`).
- **No STAY/no-op, no delay set** — time advances automatically via
  `advance_and_resolve` (`modesel/env.py:132-195`), not an agent choice.
- **|A| = 2 × n_tasks_total** (`modesel/gym_env.py:46`): **M3 = 18**
  (9 tasks total), **M4 = 24** (12 tasks total).
- **Masking** (`modesel/env.py:92-104`, exposed via `modesel/env.py:107-114`):
  an action is masked unless (1) the job isn't failed, its cursor is on
  this exact task, and no task is already active for that job; (2) no
  other active job occupies the same machine; (3) dispatching in this mode
  keeps the job's best-case remaining completion within its deadline `D_j`
  (`modesel/env.py:74-79`).

### 3. Reward function and success criteria

```python
# modesel/gym_env.py:104-110
done = res.done
reward = -res.cost
if done:
    n_failed_jobs = sum(1 for js in self._state.jobs if js.failed)
    info["domain_metric"] = self._state.g          # makespan
    info["success"] = n_failed_jobs == 0
```
`res.cost` (`modesel/env.py:234-249`) = `dispatch_cost + advance_cost`:
- `dispatch_cost = cost_penalty if mode == FAST else 0.0` (`modesel/env.py:243`)
  — **FAST costs `cost_penalty` once per dispatch; SLOW is free.**
- `advance_cost` accumulates `TIME_COST` per unit elapsed
  (`modesel/env.py:177`, `TIME_COST = 1.0`, `modesel/env.py:37`), plus a
  **per-job** failure penalty `P_j = 3.0 * job.slow_makespan()`
  (`modesel/instance.py:96-99`) added once when a job auto-fails — that
  job stops progressing but **the episode does not end** until every job
  is done or failed.
- **No explicit success bonus, no shaping term** — reward is purely
  `-cost` (grepped for `bonus`/`SUCCESS_BONUS` across `modesel/*.py`, no
  matches).

**Success**: `n_failed_jobs == 0` at episode end. **Episode
horizon**: no `TimeLimit`/step-count wrapper found; ends when
`is_terminal` (`modesel/env.py:63-67`) — every job either `failed` or
fully completed.

**Discount factor**: same shared `gamma = 0.99` default, not overridden
for modesel.

### 4. Instance parameters, resolved (M3, M4)

Call site, `run_one.py:272-279`: `modesel_make_instance(n_jobs=spec["n_jobs"],
n_tasks=spec["n_tasks"], n_machines=spec["n_machines"],
fast_slow_ratio=MODESEL_FAST_SLOW_RATIO, cost_penalty=MODESEL_COST_PENALTY,
tightness=spec["tightness"], w=0.25, seed=INSTANCE_SEED, label=instance)`.
`MODESEL_FAST_SLOW_RATIO = 2.0`, `MODESEL_COST_PENALTY = 6.0`
(`modesel/tiers.py:27-28`), `INSTANCE_SEED = 0`.

Tier specs (`modesel/tiers.py:23-24`): M3 = `n_jobs=3, n_tasks=3,
n_machines=3, tightness=0.25`; M4 = `n_jobs=4, n_tasks=3, n_machines=3,
tightness=0.5`.

**Routing**: independent random permutation of the machine set per job
(`modesel/generator.py:38-47`), a clean permutation since `n_tasks ==
n_machines` for both tiers. **Durations**: `d_fast = randint(2,5)`
(`modesel/generator.py:21,48`); `d_slow = max(d_fast+1, round(d_fast *
fast_slow_ratio))` (`modesel/generator.py:51`). **Deadline**: `D_j =
round(slow_makespan_j * (1 + tightness))` (`modesel/generator.py:56-57`).
**Failure penalty**: `P_j = 3.0 * slow_makespan_j` (`modesel/instance.py:96-99`).

Resolved (seed=0):

**M3** (t=0.25): J0 slow_makespan=22 → D=28, P=66.0; J1 slow_makespan=24 →
D=30, P=72.0; J2 slow_makespan=18 → D=22, P=54.0. |A|=18. Clocks: 4.

**M4** (t=0.5): J0 D=33 P=66.0; J1 D=36 P=72.0; J2 D=27 P=54.0; J3
(slow_makespan=18) D=27 P=54.0. |A|=24. Clocks: 5.

(J0-J2's per-task machine/duration draws are identical between M3 and M4 —
same `random.Random(0)` stream, first 3 jobs; only `tightness`, and M4's
extra job J3, differ.)

`w = 0.25` (duration-jitter half-width for the actual sampled interval
`[d·(1-w), d·(1+w)]`, `modesel/env.py:230-231,244`) — **unlike gridtimed
and trmgrid, modesel's `w` DOES affect the concrete simulator** (task
durations are genuinely sampled at execution time in
`modesel/env.py:230-244`, not fixed at their nominal value) — flagging
this as a real cross-domain difference, not an oversight: modesel has no
analogous "w only affects zone geometry" property claim to check against.

---

## 5. trmgrid selection numbers

*(Extracted by a subagent from `artifacts/TRMGRID.md`, `TRMGRID_V2.md`,
`TRMGRID_SWEEP.md`, and `artifacts/trmgrid_v2_sweep.json`; chronology
inferred from internal cross-references since this repo has a single
squashed commit, no real git history to date these docs.)*

### Pre-registered screening thresholds

**Location-only soundness ≤ 0.65, conflict fraction ≥ 0.25.**
`artifacts/TRMGRID_SWEEP.md:33`: *"Against Step 1's bars (soundness
<=65%, conflict >=25%)..."*; restated `artifacts/TRMGRID_SWEEP.md:57`;
`run_one.py:96`, *"bars (soundness 0.5862 <= 0.65, conflict 0.4138 >=
0.25..."*.

**Caveat, stated plainly rather than glossed over**: I cannot confirm
these were written down *before* any result was computed (true
pre-registration) versus stated inline while reporting results that
already existed. Every occurrence found phrases the bar as referencing an
already-established rule ("Step 1's own bars"), but no standalone
"here are our criteria" passage predates a results table in either doc.
There is also a **different, unrelated 95%-soundness bar**
(`artifacts/TRMGRID.md:188-193`) used earlier to judge *observation-model
candidates* (point/location/zone/discretized-clock) — don't conflate it
with the instance-parameter screening bars above.

### Diagnostic values, three layout seeds, at the finally-chosen parameters

`TRMGRID_INSTANCE_SEED = 2` (`run_one.py:125`, direct citation).

**G_SMALL (grid_n=4, k=3, α=0.25, width=8)** — `artifacts/TRMGRID_SWEEP.md:19-25`:

| seed | usable states | loc-only soundness | conflict (n/N) |
|---|---:|---:|---|
| 0 | 584,812 | 0.7034 | 0.2966 (43/145) |
| 1 | 2,017,419 | **UNKNOWN — not in repo** | **UNKNOWN** |
| 2 (deployed) | 1,346,833 | 0.5862 | 0.4138 (60/145) |

Seed 1's full soundness/conflict pipeline was started but explicitly
abandoned before completion: *"Seed 1's full location-only-soundness/
conflict-fraction pipeline... was still running after 25+ minutes... I
stopped it for time"* (`artifacts/TRMGRID_SWEEP.md:24,27`). This is a real
gap in the repo, not a citation I failed to find — mark it `UNKNOWN`, not
interpolated.

**G_LARGE (grid_n=5, k=3, α=0.25, width=8, truncated at state_limit=400,000)**
— `artifacts/TRMGRID_SWEEP.md:39-41`:

| seed | usable states | loc-only soundness | conflict (n/N) |
|---|---:|---:|---|
| 0 | 222,882 | 0.2178 | 0.7822 (158/202) |
| 1 | 185,685 | 0.2266 | 0.7734 (157/203) |
| 2 (deployed) | 193,004 | 0.2193 | 0.7807 (146/187) |

All three G_LARGE seeds pass both bars cleanly and agree within ~1%
(`artifacts/TRMGRID_SWEEP.md:55`), which is why seed 2 was reused here "for
consistency" without a separate ceremony.

**One more nuance worth keeping**: the *earlier* α×width sweep
(next subsection) that originally selected `width=8, α=0.25` ran its
full illustrative diagnostic at **seed=0**, not seed=2 —
`artifacts/TRMGRID_V2.md:118-119`, *"Chosen setting for the detailed
measurements below: α=0.25, width=8, seed=0"* — giving location-only
soundness 0.2722, conflict 0.7278 (131/180), 176,389 states at that seed.
The seed actually shipped (`TRMGRID_INSTANCE_SEED=2`) comes from the
*later* tier/seed sweep in `TRMGRID_SWEEP.md`, which superseded the
illustrative seed=0 report once seed-sensitivity was checked. Not a
contradiction — just report both figures with their correct seed labels
if the paper cites either.

### The parameter sweep that selected width=8, α=0.25

`artifacts/TRMGRID_V2.md:43-75` — a 3×3 (α×width) grid, each cell run at 3
seeds = 27 rows (the doc's own heading calls it "the 9-cell sweep", i.e. 9
(α,width) cells):

```
alpha=0.0,  width=2: soundness .82-.89, conflict .11-.18  (fails bar; some cells degenerate, 0% rand-succ)
alpha=0.0,  width=4: soundness .78-.86, conflict .14-.22  (fails bar)
alpha=0.0,  width=8: soundness .25-.30, conflict .70-.75  (CLEARS both bars, all truncated)
alpha=0.25, width=2: soundness .81-.87, conflict .13-.19  (fails; some degenerate)
alpha=0.25, width=4: soundness .78-.85, conflict .15-.22  (fails)
alpha=0.25, width=8: soundness .24-.30, conflict .70-.76  (CLEARS, all truncated)  <- shipped
alpha=0.5,  width=2: soundness .81-.87, conflict .13-.19  (fails; some degenerate)
alpha=0.5,  width=4: soundness .78-.85, conflict .15-.22  (fails)
alpha=0.5,  width=8: soundness .26-.30, conflict .70-.74  (CLEARS, all truncated)
```
(Full 27-row table with exact per-seed values is in
`artifacts/TRMGRID_V2.md:47-75`, cross-checked against
`artifacts/trmgrid_v2_sweep.json`.) Selection reasoning
(`artifacts/TRMGRID_V2.md:45,77-90`): *"width is the dominant driver; α
barely matters."* width∈{2,4} never clears the bars at any α; width=8
clears at all three α values and all three seeds. **The doc does not give
an explicit tie-breaking argument for α=0.25 specifically over α=0/0.5** —
all three pass equally well at width=8 — so α=0.25's selection into
`run_one.py:118` isn't separately justified beyond "any of the three
works." Flag this as a real gap if the paper needs to defend the specific
α value, not just the width.

### Why 6×6/k=3 was rejected

**Confirmed: 0.0% random-policy success**, exactly. `artifacts/TRMGRID_SWEEP.md:42,46`:
```
6x6/k=3, seed 0, alpha=0.25, width=8, 400,000-state truncation: 228,157 states,
soundness 0.3803, conflict 0.6197 (145/234), random-policy success = 0.0%
```
*"Per the task's own miscalibration rule ('if 0% or 100% [random-policy
failure rate], the instance is miscalibrated, say so'): 6x6/k=3... is
miscalibrated -- 0% random-policy success across all rollouts, rejected
outright regardless of its soundness/conflict numbers."* Tested at seed=0
only for this specific tier (not 3 seeds) — rejection on this one measured
seed was treated as conclusive.

---

## 6. Training hyperparameters

*(Extracted by a subagent; per-encoder widths and the manifest-derived
param-count table were spot-checked against
`results/gridtimed__T2__mlp_point__seed0__manifest.json`, which matched.)*

`run_one.py`'s `build_trainer` (`run_one.py:297-309`) overrides **only**
`total_steps`, `plateau_patience_steps`, `plateau_enabled`, `seed`,
`num_envs` on `PPOConfig` — every other field is the `train/ppo.py`
dataclass default (`train/ppo.py:88-111`), confirmed against a real
manifest's recorded `hyperparameters` block:

| Hyperparameter | Value | Citation |
|---|---|---|
| Learning rate | 3e-4 | `train/ppo.py:94` |
| Rollout size (total transitions/update) | 2048 | `train/ppo.py:95` |
| Steps per env per update | 256 (=2048/8) | `train/ppo.py:113-116` |
| Epochs per update | 4 | `train/ppo.py:97` |
| Minibatch size | 256 | `train/ppo.py:98` |
| Clip coefficient | 0.2 | `train/ppo.py:91` |
| Entropy coefficient | 0.01 | `train/ppo.py:92` |
| Value-loss coefficient | 0.5 | `train/ppo.py:93` |
| GAE lambda | 0.95 | `train/ppo.py:90` |
| Gamma | 0.99 | `train/ppo.py:89` |
| Max grad norm | 0.5 | `train/ppo.py:99`, applied `train/ppo.py:379` |
| Optimizer | `torch.optim.Adam`, no explicit betas/eps (PyTorch defaults 0.9/0.999/1e-8 apply implicitly — not a stated line, an absence) | `train/ppo.py:236` |
| Parallel envs | 8 | `run_one.py:145` |
| Total env-step budget | 300,000 | `run_one.py:142` |
| Early stopping (plateau) | **disabled by default** — `build_trainer`'s `plateau_enabled` param defaults `False` (`run_one.py:300`), matching `PPOConfig.plateau_enabled=False` (`train/ppo.py:109`); confirmed empirically, sample manifest shows `plateau_enabled: false` and `total_env_steps: 301056` (ran the full budget) | `run_one.py:300`; `results/modesel__M3__mlp_point__seed0__manifest.json` |

**Network width per encoder** (`build_head`, `train/ppo.py:40-84`, passes
no width-overriding kwargs — class `__init__` defaults apply):

| Encoder | Widths | Citation |
|---|---|---|
| mlp_point | embed_dim=92, hidden=112, per_task_hidden=48 | `encoders/mlp_point.py:53-62` |
| deepset_h | embed_dim=96, hidden=128, per_task_dim=8 | `encoders/deepset_h.py:41` |
| deepset_v | embed_dim=96, hidden=128, per_task_dim=10 | `encoders/deepset_v.py:89` |
| cnn_dbm | embed_dim=120, channels=(40,58), per_task_dim=36 | `encoders/cnn_dbm.py:36` |
| cnn_proj | res=32, embed_dim=90, channels=(11,19) | `encoders/cnn_proj.py:48` |
| gnn | embed_dim=90, hidden=64, per_task_dim=24 | `encoders/gnn.py:103` |

**Parameter count per encoder per instance** (from real `__manifest.json`
`param_count`, seed0 — the authoritative already-measured number, not
recomputed from architecture):

| Instance | mlp_point | deepset_h | cnn_dbm | cnn_proj | deepset_v | gnn | max/min |
|---|---|---|---|---|---|---|---|
| modesel/M3 | 31,764 | 30,936 | 31,426 | 30,949 | 30,940 | 33,068 | 1.0689 (6.9%) |
| modesel/M4 | 34,116 | 31,064 | 31,426 | 31,345 | 31,068 | 33,068 | 1.0982 (9.8%) |
| gridtimed/T2 | 32,437 | 32,213 | 32,835 | 32,406 | 32,215 | 34,129 | 1.0595 (6.0%) |
| gridtimed/T3 | 34,357 | 32,341 | 32,835 | 32,901 | 32,343 | 34,129 | 1.0623 (6.2%) |
| trmgrid/G_SMALL | 31,957 | 32,213 | 32,835 | 32,406 | 32,215 | 34,129 | 1.0680 (6.8%) |
| trmgrid/G_LARGE | 33,397 | 32,213 | 32,835 | 32,406 | 32,215 | 34,129 | 1.0595 (6.0%) |

**The "matched within 10%" claim holds** for all six instances checked —
worst case is modesel/M4 at 9.82%, real but close to the stated boundary,
worth knowing if a reviewer pushes on it.

---

## Cross-domain table

| Instance | Grid / job size | Ambient clocks (d) | \|A\| | Deadline D | w | Usable states |
|---|---|---|---|---|---|---|
| gridtimed/T2 | 5×5 grid, 3 switches/3 gates | 5 (3 switch + y + e) | 12 | 27 | 0.25 | UNKNOWN — DP timed out at 100s locally; not measured for the exact deployed (seed=0) instance in this repo; `gridtimed/tiers.py`'s own sweep measures a *different* seed set for the T0-T4 ladder, not directly this instance |
| gridtimed/T3 | 6×6 grid, 4 switches/4 gates | 6 (4 switch + y + e) | 12 | 33 | 0.25 | UNKNOWN, same reason |
| trmgrid/G_SMALL | 4×4 grid, k=3 sub-goals | 5 (3 sub-goal + g + e) | 12 | 30 | 0.25 | **1,346,833** (seed=2, `TRMGRID_SWEEP.md`, untruncated at state_limit=3,000,000) |
| trmgrid/G_LARGE | 5×5 grid, k=3 sub-goals | 5 (3 sub-goal + g + e) | 12 | 30 | 0.25 | **193,004** (seed=2, `TRMGRID_SWEEP.md`, measured under a 400,000-state cap but did not hit it — likely exact for this instance, though the tier class is nominally treated as "truncated by design") |
| modesel/M3 | 3 jobs × 3 tasks, 3 machines | 4 (3 job + g) | 18 | per-job D_j = 28/30/22 | 0.25 | UNKNOWN — not computed; modesel doesn't have a directly comparable "usable states via BFS" figure in what was gathered |
| modesel/M4 | 4 jobs × 3 tasks, 3 machines | 5 (4 job + g) | 24 | per-job D_j = 33/36/27/27 | 0.25 | UNKNOWN, same reason |

---

## Flags — places worth double-checking or footnoting in the paper

1. **`w=0.25` is two independent hardcodes, not one shared constant.**
   `gridtimed/tiers.py:21` (`W = 0.25`) and the literal `w=0.25` at
   `run_one.py:277,286,291` (modesel, gridtimed, trmgrid call sites) all
   currently agree, but nothing in the code enforces they stay in sync if
   one is edited later — `run_one.py` doesn't import `tiers.py`'s `W`
   constant, it re-hardcodes the number. Not a bug today, but a
   reproducibility footgun worth a footnote if the paper states "w=0.25
   throughout" as a single design decision.
2. **trmgrid's α=0.25 selection isn't individually justified** — the
   9-cell sweep shows width=8 is what matters; all three α values clear
   the bar equally at width=8, and the doc doesn't argue for 0.25 over 0
   or 0.5 specifically (§5). If a reviewer asks "why α=0.25 and not 0.5",
   the honest answer from what's in this repo is "any of the three would
   have passed; 0.25 was the one carried forward."
3. **trmgrid layout-seed-1 diagnostics for G_SMALL are genuinely missing**
   from the repo (measurement run was manually stopped mid-computation,
   `artifacts/TRMGRID_SWEEP.md:24,27`) — if the paper wants to show "all
   three seeds" for G_SMALL specifically, seed 1's soundness/conflict/
   multi-zone-count cells cannot be filled in without re-running that
   measurement; only its state count (2,017,419) is on record.
4. **modesel's `w` genuinely affects the concrete simulator** (task
   durations are sampled at execution time), unlike gridtimed/trmgrid
   where `w` only ever touches zone geometry, never the reward/transition.
   If the paper describes `w` as playing "the same role" across all three
   domains, that's not accurate — flag this distinction explicitly.
5. **gridtimed T2/T3 "usable states" are not in the cross-domain table** —
   genuinely not measured for the exact deployed instances within the time
   I spent (the DP is slow enough at these sizes that a 100s local run
   didn't finish); computable in principle via `gridtimed/dp.py`'s
   `explore_and_solve` on the resolved instance with a larger
   `state_limit`, just not done here.
