# Audit: `prob_jobshop_ext` — pyudbm-based probabilistic job-shop benchmark

Scope: `prob_jobshop_ext/` (verified byte-identical to `scheduling/prob_jobshop/`,
confirmed via `diff` on every file checked). All line references are to
`prob_jobshop_ext/*.py`. Facts only; no recommendations. Anything not
determinable from the code is marked **UNKNOWN**.

---

## 1. Zone/DBM layer

**pyudbm version**: `0.0.1` (`pip show pyudbm` → `Version: 0.0.1`), editable
install pointing at `RL-for-zones/pyudbm`, a Python wrapper (via pybind11)
around the UPPAAL UDBM C++ library. Import note: `import pyudbm` resolves
correctly only when the current working directory (or an earlier `sys.path`
entry) does not itself contain a literal directory named `pyudbm` — the repo
root does (`RL-for-zones/pyudbm/`), which shadows the installed package as an
empty namespace package and breaks `from pyudbm import Context`. This is a
Python import-path collision, not a bug in `zone_ops.py` itself.

**API calls actually used**, all in `zone_ops.py`:

| Operation | Call | Where |
|---|---|---|
| Zero zone | `ctx.get_zero_federation()` | `init_zone` (zone_ops.py:48) |
| Guard/invariant intersection | `Z & (clock <= bound)` (`Federation.__and__`) | `apply_guard`, `apply_invariant` (zone_ops.py:64, 108) |
| Completion guard (lower bound) | `Z & (clock >= exact_d)` | `symbolic_rl.py:279`, `symbolic_optimal.py:186` |
| Reset | `Z.reset_value(clock)` | `reset_clocks` (zone_ops.py:79) |
| Delay / time-elapse | `Z.up()` | `apply_delay` (zone_ops.py:96) |
| Emptiness | `Z.is_empty()` | `apply_guard` (zone_ops.py:65), and directly in `symbolic_rl.py`/`symbolic_optimal.py` after each completion guard |
| Extrapolation | `Z.extrapolate_max_bounds({clock: M, ...})` | `normalize` (zone_ops.py:120) |
| Readout | `Z.to_dbm_list()`, then `dbm.bound(0, idx)` | `min_cost` (zone_ops.py:143-149) |
| Inclusion | `Z_new <= Z_old` (`Federation.__le__`) | `dominates` (zone_ops.py:131) — **note: `dominates` is defined but never called anywhere in the package** (`grep` for `dominates(` finds only its own definition) |

**Not used anywhere in this package** (confirmed present on the `Federation`/`DBM`
API via `dir()` + `help()`, but zero call sites in `zone_ops.py`,
`symbolic_env.py`, `symbolic_rl.py`, or `symbolic_optimal.py`): `reduce()`,
`intern()`, `convex_hull()`, `predt()`, `down()`, `free_clock()`,
`update_value()` (only its wrapper `reset_value` is used), `set_init()`,
`set_zero()`, `has_zero()`, `is_zero()`, `to_cdd()`, `to_cdd_context()`, `plot()`.
There is no explicit canonicalize/close call anywhere — canonical form is
maintained implicitly by the native DBM operations (`&`, `up()`,
`reset_value()`), which is consistent with pyudbm's own documented behavior;
this repo never calls a canonicalization primitive directly.

**Extrapolation**: yes, applied — `Z.extrapolate_max_bounds(bounds)`, called
once per transition in `normalize()` (zone_ops.py:112-120). Per pyudbm's own
docstring this is "the classical DBM `k`-normalization ... `extrapolateMaxBounds`"
— i.e. **max-bound k-normalization**, one scalar bound per clock. It is **not**
LU-extrapolation (which would need separate lower/upper max constants per
clock) — the API (`Mapping[Clock, int]`) only accepts a single bound per clock,
so LU-extrapolation is not expressible through this call, and nothing in the
codebase implements it manually either. `zone_ops.normalize()` gives every job
clock the same bound `M` and gives `delta` `pyudbm`'s own infinity sentinel
(`_UDBM_INF = 1073741823`), i.e. `delta` is explicitly excluded from
extrapolation (zone_ops.py:118-119).

**Reading a DBM out into Python**: `Federation.to_dbm_list()` returns a list of
`DBM` objects (a `Federation` is a union of convex `DBM`s; every code path in
this repo only ever produces **singleton** federations — see §3). Each `DBM`
exposes:
```python
dbm.bound(i, j)      # -> int, decoded bound for constraint x_i - x_j <= bound(i,j)
dbm.is_strict(i, j)   # -> bool, whether the inequality is strict (<)
dbm.is_infinity(i, j) # -> bool
dbm.raw(i, j)         # -> int, raw encoded UDBM `raw_t` cell (undecoded)
dbm.to_matrix(mode="tuple")  # -> nested list of ('<=', 3) / ('<', 3) / ('<', inf) per cell
```
`i`/`j` accept either integer indices (0 = reference clock) or clock-name
strings. This *is* the accessor giving raw (bound, strictness) pairs the audit
asked about — `bound(i,j)` + `is_strict(i,j)` together, or `to_matrix()` for
the whole matrix at once. `zone_ops.min_cost()` (zone_ops.py:134-149) is the
only place in this repo that actually reads cell values out of a `DBM`; it
reads exactly one cell per DBM: `dbm.bound(0, delta_idx)`.

**Bound type**: integers. `DBM.bound()` is typed and documented as returning
`int`; `DBM.raw()` similarly returns `int` (UDBM's native `raw_t` encoding).
Verified empirically — every `bound()` call observed in this repo's
executions returns a Python `int`, never a float. This matches the instance
model, where all task durations are validated as positive integers
(`verification.py:31`: `if not isinstance(d, int) or d <= 0`).

---

## 2. Automaton and instance model

**Clocks per instance**: exactly `(number of jobs) + 1` — one clock per job
(`c_{job_id}`, shared by all tasks of that job, reset each time a new task in
that job starts) plus the single global cost clock `delta`
(`zone_ops.py:29`: `self._clock_names = [f"c_{j.job_id}" for j in instance.jobs] + ["delta"]`).
It is **not** one clock per task. Measured across all 20 shipped instances,
this ranges from 3 clocks (PJS_01, 2 jobs) to 15 clocks (LJS_10, 14 jobs) —
see §4 table.

**Task durations are DETERMINISTIC, not intervals** — but deterministic
*conditioned on a discrete probabilistic choice*, not a single fixed number
per task. This is the most important question in the audit, and the answer
requires two parts to be precise:

- Each `Task` carries a `TaskDistribution(durations: List[int], probabilities: List[float])`
  (instance.py:7-9) — a **discrete probability distribution over a finite set
  of possible exact integer durations**, e.g. `TaskDistribution([3,4,5],[0.1,0.8,0.1])`.
  This is quoted directly from `instance.py:6-9`:
  ```python
  @dataclass
  class TaskDistribution:
      durations: List[int]
      probabilities: List[float]
  ```
- At execution time (`symbolic_env.py:83-96`, the `.step()` method), one exact
  duration is sampled: `d = int(self._rng.choice(dist.durations, p=dist.probabilities))`,
  and the location is annotated with that exact value: `loc[idx] = f"active_d{d}"`
  (symbolic_env.py:92). Once sampled, `d` is an exact integer, not an interval.
- In the **symbolic** (zone-graph) view (`symbolic_env.py:129-141`, `invariant()`),
  each of the discrete duration outcomes becomes its own **branch** with its
  own guard: `invariant c_J <= d` combined with a later completion guard
  `c_J >= d` (`_compute_path`, symbolic_env.py:174-224) pins the clock to
  `c_J == d` exactly in that branch's zone. Quoted directly, symbolic_env.py:41-43:
  ```
  invariant(loc) returns [(clock_name, exact_d)] for each active_d{X} task —
  the exact upper bound equals the sampled duration, so combined with the
  completion guard c_J >= exact_d we get c_J = exact_d in the zone.
  ```
  So there is no `[l, u]` interval-guard construction anywhere in this
  codebase; uncertainty is represented purely as **branching over a
  finite set of exact-duration outcomes**, each of which is a single point in
  clock space once resolved, not a genuine bounded interval of nondeterministic
  real-valued duration.
- Dead code note: `pta.py`'s `build_task_automaton()` (pta.py:61-104) documents
  states named `..._active_d{d}` with per-branch `invariant f"{clock} <= {d}"`
  and end-transition guard `f"{clock} = {d}"` — consistent with the above — but
  this function is **never called** anywhere in the package (confirmed by
  `grep -rn "build_task_automaton"`: only its own definition and its
  re-export in `__init__.py`; no call sites). Only `TaskState`/`GlobalState`
  from `pta.py` are actually used, by the plain discrete-event `PTASimulator`
  (`simulator.py`), which does not touch pyudbm at all.

**Probabilistic branching**: yes, genuinely probabilistic, not merely
deterministic-given-the-action. Each dispatch action in the symbolic env
carries a `branches` list, one entry per possible duration with its own
`prob` (symbolic_env.py:24-32, 249-260):
```python
"branches":  [
  {"d": int, "prob": float, "next_loc": tuple, "path": [...], "decision_loc": tuple},
  ...
],
```
`SymbolicRLAgent` and `OptimalSymbolicSolver` propagate all branches
symbolically for reachability (`_propagate_zone`), but the actual environment
transition samples exactly one (`step()` calls `self._rng.choice(dist.durations, p=dist.probabilities)`,
symbolic_env.py:90) — model-free Q-learning from a single sampled outcome per
step, as previously established for `symbolic_rl.py`.

**Concurrency / serialization**: multiple machines run tasks **concurrently**
(not serialized) — each task occupies its own machine and independent job
clock; dispatch of an enabled task on a free machine does not wait for other
active tasks. Enforced by `enabled_tasks()`/`_compute_actions()` checking only
that the task's *own* machine is not in the currently-active-machines set
(pta.py:33-49, symbolic_env.py:226-269, region_env.py:113-130):
```python
if task.machine in active_machines:
    continue
```
Mutual exclusion is enforced *per machine only* (not globally): concurrently
active tasks on different machines race to completion, and the environment
correctly resolves "whichever finishes first" by taking the **minimum**
remaining time across all currently active tasks, not just the most recently
dispatched one — quoted from `symbolic_env.py:174-224` (`_compute_path`):
```python
remaining = {}
for t_id, status in zip(self._task_ids, sim_loc):
    if status.startswith("active_d"):
        ...
        rem = exact_d - virtual_clocks.get(jid, 0.0)
        remaining[t_id] = max(0.0, rem)
...
dt = min(remaining.values())
```
The same min-of-remaining pattern appears independently in `region_env.py:168-196`
(`_advance`), `pta_env.py:225-266` (`_advance_time_limited`), and
`simulator.py:51-75` (`advance_time`) — four separate re-implementations of
the same concurrency-resolution logic, not one shared function (see §7).

**Deadlines**: none. **Machine failures / breakdowns / maintenance**: none.
`grep -ri "deadline\|failure\|breakdown\|maintenance"` across every `.py` file
in `prob_jobshop_ext/` returns zero matches. The only termination/deadline-like
condition is "all tasks done" (`is_terminal()` / `all(s == "done" ...)`), and
`simulator.py:58-61` raises `RuntimeError("... possible deadlock")` if
`advance_time` is ever called with no active tasks, which is a defensive
assertion against a modeling bug, not a deadlock *feature*. **Precedence
constraints**: yes — each task's predecessor (previous task in the same job)
must be `"done"` before it becomes enabled (instance.py:57,
`predecessor_of`; enforced in every env's `enabled_tasks`/`_compute_actions`).

---

## 3. Empirical zone structure

**Method**: a throwaway script (`/tmp/zone_audit.py`, not committed) drove
`SymbolicJobShopEnv` + `zone_ops.py` directly (the same call sequence as
`symbolic_rl.py`/`symbolic_optimal.py`'s `_propagate_zone`) to BFS-explore the
real pyudbm-backed zone graph for all 20 shipped instances (10 PJS + 10 LJS),
capped at 200,000 states. For each instance: a sample of up to 2,000
reachable zones was drawn (all of them, if fewer than 2,000 exist) and, for
each sampled zone's single `DBM`:
- **(a)** off-diagonal count = number of `(i,j)`, `i≠j`, both `i,j ≥ 1` (i.e.
  excluding the reference clock 0), with `dbm.is_infinity(i,j) == False` —
  genuine finite difference constraints between two real clocks.
- **(b)** box = `(a) == 0`.
- **(c)** vertex count: an LP-based extreme-point search (`scipy.optimize.linprog`,
  HiGHS), probing 2×n_clocks axis-aligned directions plus 24 random Gaussian
  directions per zone, collecting distinct optimal solutions. This is
  **exact for box zones** (whose vertex count is independently cross-checked
  against 2^(number of non-degenerate clock dimensions)) and a **documented
  lower-bound approximation** for non-box zones (a bounded direction budget
  can miss vertices of a genuinely high-facet-count polytope). For
  tractability, metric (c) was computed on a sub-sample of up to 200 zones
  per instance (metrics (a)/(b) used the full up-to-2000 sample).

**Results** (`n_states`/`truncated`/`max_depth`/`wall_s` are exact — not
sampled; `box_frac`/`off_diag`/`vertices` are over the samples described
above):

| Instance | n_states | truncated@200k | max_depth | wall (s) | box_frac | off_diag mean/max | vertices mean/max |
|---|---:|:---:|---:|---:|---:|---:|---:|
| PJS_01 | 110 | No | 3 | 0.01 | **0.000** | 3.66 / 6 | 1.66 / 2 |
| PJS_02 | 402 | No | 4 | 0.02 | **0.000** | 3.68 / 6 | 1.77 / 3 |
| PJS_03 | 4,283 | No | 5 | 0.30 | **0.000** | 7.20 / 12 | 2.32 / 5 |
| PJS_04 | 15,767 | No | 8 | 1.83 | **0.000** | 7.52 / 9 | 2.35 / 6 |
| PJS_05 | 200,000 | **Yes** | 8 | 27.40 | **0.000** | 12.78 / 16 | 2.90 / 7 |
| PJS_06 | 200,000 | **Yes** | 8 | 39.97 | **0.000** | 23.36 / 25 | 4.00 / 10 |
| PJS_07 | 200,000 | **Yes** | 9 | 31.83 | **0.000** | 22.03 / 25 | 4.04 / 11 |
| PJS_08 | 200,000 | **Yes** | 6 | 38.25 | **0.000** | 29.18 / 32 | 3.46 / 9 |
| PJS_09 | 200,000 | **Yes** | 4 | 45.87 | **0.000** | 52.08 / 58 | 3.53 / 6 |
| PJS_10 | 200,000 | **Yes** | 5 | 140.90 | **0.000** | 80.78 / 92 | 4.43 / 9 |
| LJS_01 | 200,000 | **Yes** | 7 | 37.41 | **0.000** | 32.43 / 36 | 4.68 / 11 |
| LJS_02 | 200,000 | **Yes** | 5 | 44.46 | **0.000** | 41.49 / 44 | 4.18 / 10 |
| LJS_03 | 200,000 | **Yes** | 4 | 47.69 | **0.000** | 52.09 / 58 | 3.58 / 6 |
| LJS_04 | 200,000 | **Yes** | 4 | 46.37 | **0.000** | 52.09 / 58 | 3.48 / 7 |
| LJS_05 | 200,000 | **Yes** | 4 | 46.48 | **0.000** | 65.85 / 74 | 3.37 / 7 |
| LJS_06 | 200,000 | **Yes** | 3 | 52.94 | **0.000** | 82.32 / 92 | 3.11 / 5 |
| LJS_07 | 200,000 | **Yes** | 3 | 70.90 | **0.000** | 82.35 / 100 | 3.24 / 6 |
| LJS_08 | 200,000 | **Yes** | 3 | 78.21 | **0.000** | 100.66 / 112 | 2.98 / 6 |
| LJS_09 | 200,000 | **Yes** | 3 | 106.10 | **0.000** | 120.97 / 134 | 3.08 / 6 |
| LJS_10 | 200,000 | **Yes** | 3 | 146.81 | **0.000** | 168.09 / 184 | 3.06 / 6 |

**Box fraction is 0.000 for every single instance, with no exceptions —
loudly: across ~35,900 sampled zones total (up to 2,000 per instance × 20
instances, fewer for the 4 small uncapped instances), not one was a pure
box.** Every sampled zone had at least 2-6 genuine finite difference
constraints between real (non-reference) clocks, growing with instance size
(mean off-diagonal count climbs from 3.66 at PJS_01 to 168.09 at LJS_10). This
directly contradicts the premise written into `graph_sizes.py`'s own
docstring (graph_sizes.py:18-20): *"Since all task durations are integers,
every clock value at a decision point is an exact integer, so zones reduce to
single points"* — that claim is **not borne out** by directly querying the
real pyudbm `Federation` objects; genuine non-degenerate difference
constraints between two real (not-yet-pinned) clocks are the norm at every
instance size measured, not single-point degeneracies. (`graph_sizes.py`'s
`count_zone_graph` does not actually build any `Federation` — see §4 — so
this discrepancy was never checked against real DBMs before this audit.)

Every sampled `Federation` across all 20 instances was a **single DBM**
(`n_fed_dbms_max = 1` in every case, checked over the first 200 zones per
instance) — the union-of-DBMs case that `Federation` supports in general
never arises here, consistent with `zone_ops.py` never calling `|`/`__or__`.

Vertex counts stay small (means 1.66–4.68, maxima 2–11) even as off-diagonal
constraint counts grow into the hundreds (LJS_10 mean 168, max 184) — i.e.
these are non-box but still low-complexity polytopes (few active facets are
doing the work; most of the "off-diagonal count" growth reflects clocks that
have already been pinned to exact values by completion guards, which
contributes a finite, non-infinity DBM cell without adding a genuine new
vertex-producing dimension). This is consistent with the LP-vertex-search
method being a **lower bound** for high-off-diagonal-count zones (§ methodology
above) — the true vertex counts for the largest instances may be
undercounted by the fixed 24-random-direction budget; **UNKNOWN** exactly how
much, since no exact DBM vertex-enumeration algorithm was implemented for
this audit.

---

## 4. Scaling

**Instance table** (machines, tasks, clocks = jobs+1) with zone-graph size
from §3:

| Instance | Jobs | Machines | Tasks | Clocks | Zone-graph n_states (≥, capped at 200k) |
|---|---:|---:|---:|---:|---:|
| PJS_01 | 2 | 2 | 4 | 3 | 110 (exact) |
| PJS_02 | 2 | 3 | 5 | 3 | 402 (exact) |
| PJS_03 | 3 | 2 | 6 | 4 | 4,283 (exact) |
| PJS_04 | 3 | 3 | 9 | 4 | 15,767 (exact) |
| PJS_05 | 4 | 3 | 12 | 5 | ≥200,000 |
| PJS_06 | 5 | 5 | 25 | 6 | ≥200,000 |
| PJS_07 | 5 | 5 | 25 | 6 | ≥200,000 |
| PJS_08 | 6 | 6 | 36 | 7 | ≥200,000 |
| PJS_09 | 8 | 6 | 48 | 9 | ≥200,000 |
| PJS_10 | 10 | 10 | 100 | 11 | ≥200,000 |
| LJS_01 | 6 | 4 | 24 | 7 | ≥200,000 |
| LJS_02 | 7 | 5 | 35 | 8 | ≥200,000 |
| LJS_03 | 8 | 5 | 40 | 9 | ≥200,000 |
| LJS_04 | 8 | 6 | 48 | 9 | ≥200,000 |
| LJS_05 | 9 | 6 | 54 | 10 | ≥200,000 |
| LJS_06 | 10 | 6 | 60 | 11 | ≥200,000 |
| LJS_07 | 10 | 7 | 70 | 11 | ≥200,000 |
| LJS_08 | 11 | 7 | 77 | 12 | ≥200,000 |
| LJS_09 | 12 | 8 | 96 | 13 | ≥200,000 |
| LJS_10 | 14 | 8 | 112 | 15 | ≥200,000 |

**Instance-generation code**: hard-coded, not parameterized. Every
`benchmarks/pjs_0X.py` / `benchmarks/ljs_0X.py` file (`pjs_01.py` through
`ljs_10.py`) is a standalone function `get_instance()` containing a literal
Python list of `(machine, nominal_duration)` pairs per job — quoted from
`ljs_01.py:8-21`, e.g. `[("m1", 14), ("m2", 10), ("m3", 12), ("m4", 9)]` for
job 1. There is no `generate_instance(n_jobs, n_machines, ...)` function
anywhere in the package. The only parameterization present is at the
**distribution** level: `benchmarks/_utils.py` provides two reusable formulas
— `apply_formula_070_050_140(d)` (`{0.7d: 0.25, d: 0.50, 1.4d: 0.25}`) and
`apply_formula_spread4(d)` (`{d-3: 0.35, d: 0.30, d+4: 0.35}`) — that turn a
hard-coded nominal duration into a 3-point `TaskDistribution`; the
`(machine, nominal_duration)` job/machine structure itself is always
hand-written per instance file. `ALL_INSTANCE_NAMES`/`ALL_LARGE_INSTANCE_NAMES`
(benchmarks/all_instances.py, all_large_instances.py) are fixed 10-element
lists; `get_all_instances()`/`get_all_large_instances()` just calls each
file's `get_instance()` in turn — no loop over a size parameter anywhere.

**Extrapolation — where does the zone graph exceed 10⁶ states?**

Two independent pieces of evidence, both reported (they answer slightly
different questions and should not be conflated):

1. **Direct observation**: 16 of the 20 shipped instances (PJS_05 through
   LJS_10 — i.e. everything at or above 4 jobs / 3 machines / 12 tasks)
   **already exceed the 200,000-state cap**, reaching it within the first
   3-9 BFS layers (`max_depth` column, §3) out of what would be many more
   layers before any of these instances terminates. The **10⁶ threshold is
   almost certainly already crossed by instances already in this benchmark
   suite** — not at some larger hypothetical instance — but the exact
   uncensored size for any of these 16 instances is **UNKNOWN**: the BFS was
   capped at 200,000 states by design (per the audit's own instructions) and
   was not run to completion for any instance beyond PJS_04.
2. **Extrapolation fit** from the 4 instances small enough to explore to
   completion (PJS_01-04, tasks = 4, 5, 6, 9; n_states = 110, 402, 4283,
   15767) — a least-squares fit of `ln(n_states)` against task count gives:
   ```
   n_states ≈ 3.87 · exp(0.971 · n_tasks)
   ```
   which crosses 10⁶ at **n_tasks ≈ 12.8** — consistent with direct
   observation #1, since PJS_05 (12 tasks) is exactly where the cap starts
   being hit. The same fit against clock count (3, 3, 4, 4 clocks) gives
   `n_states ≈ 0.0035 · exp(3.67 · n_clocks)`, crossing 10⁶ at **≈5.3
   clocks** (≈ 4-5 jobs) — again consistent, since PJS_05 has 4 jobs / 5
   clocks. **Caveat stated explicitly, as instructed**: this fit rests on
   only 4 data points spanning a narrow range (4-9 tasks, 3-4 clocks); it is
   presented as the requested estimate, not a rigorously validated growth
   law. A secondary, independent check — the *implied per-layer branching
   factor* `200000^(1/max_depth)` at the instances that did hit the cap —
   ranges from ≈4.6 (PJS_05/06, 4-5 jobs) up to ≈58.5 (LJS_06-10, 10-14
   jobs), i.e. the branching factor itself grows with instance size, so a
   single fixed exponential-in-tasks model is at best a rough first-order
   fit, not the true asymptotic law; the true growth rate is faster than a
   single global exponential in task count, since the effective per-step
   branching factor is itself increasing.

For comparison, the codebase's own analytic bound for the (smaller) **region**
graph — `region_theoretical_bound()` in `run_comparison_all.py:79-83`,
`∏ⱼ (max_task_duration_j + 2)` over jobs — gives a location-independent
per-job multiplicative bound; no equivalent closed-form bound for the zone
graph exists anywhere in the codebase (the zone graph's true size for any
instance beyond PJS_04 is simply unmeasured, per point 1 above).

---

## 5. RL interface

There are **four different environment classes**, not one, with materially
different state representations, and only one of them is a real
`gymnasium.Env`:

| Class | File | Gym? | State representation | Used by |
|---|---|---|---|---|
| `SymbolicJobShopEnv` | symbolic_env.py | No (bespoke) | `(loc, zone)` — pyudbm `Federation` | `SymbolicRLAgent`, `OptimalSymbolicSolver` |
| `RegionJobShopEnv` | region_env.py | No (bespoke) | `(loc, capped_clock_tuple)` — plain Python ints, no pyudbm | `RegionRLAgent` |
| `PTADirectEnv` | pta_env.py | No (bespoke) | flat `np.ndarray` of raw (uncapped, unabstracted) clock values | `DDPGAgent` (via `run_ddpg_comparison.py`) |
| `JobShopEnv` | env.py | **Yes** — `gym.Env` subclass, `Discrete(N)` action space, `Dict` observation space with `action_mask` | — | **nothing** (see below) |

`JobShopEnv` (env.py:16-153) is the only Gymnasium-compliant environment in
the package. Its observation space (env.py:58-65):
```python
self.observation_space = spaces.Dict({
    "task_status":       spaces.Box(0, 2,   shape=(n,), dtype=np.int32),
    "time_remaining":    spaces.Box(0.0, max_dur, shape=(n,), dtype=np.float32),
    "expected_duration": spaces.Box(0.0, max_dur, shape=(n,), dtype=np.float32),
    "clock_values":      spaces.Box(0.0, ub,      shape=(j,), dtype=np.float32),
    "current_time":      spaces.Box(0.0, ub,      shape=(1,), dtype=np.float32),
    "action_mask":       spaces.Box(0, 1,   shape=(n,), dtype=np.int8),
})
self.action_space = spaces.Discrete(n)
```
It wraps `PTASimulator` directly with **raw, unabstracted real clock
values** — no zone, no region cap. It does explicit action masking
(`_action_mask()`, env.py:112-117) and raises `ValueError` on an invalid
action rather than silently ignoring it (env.py:87-91). However, **it is
dead code**: `grep -rn "JobShopEnv\b|make_env\b"` across every `.py` file
(both in `prob_jobshop_ext/` and every `run_*.py` script) finds it only
imported/re-exported in `__init__.py` (`from .env import JobShopEnv, make_env`)
and never instantiated anywhere. Same status for `ddpg_env.py`'s
`DDPGSchedulingEnv` class (a third, hand-engineered 6J+2M-feature continuous
env) — defined, documented, but not imported by `run_ddpg_comparison.py`,
which uses `PTADirectEnv` from `pta_env.py` instead (verified: `ddpg_env.py`
has zero import references anywhere outside its own file).

**What the tabular agents actually observe** (state key, not a hashed
canonical form, not an integer index — a plain Python tuple used directly as
a dict key):
- `SymbolicRLAgent` / `OptimalSymbolicSolver`: `(loc, str(Z))` — the **string
  representation of the `Federation` object** is the hash/equality key
  (`symbolic_rl.py:310`, `key = (loc, str(Z))`). This means two zones are
  treated as the same state iff their `str()` output is character-identical;
  there is no explicit canonical-form or hash-based dedup step beyond
  whatever `Federation.__str__` produces (which is itself backed by the
  native canonical DBM, so this is sound, just worth noting it's string
  equality doing the work, not a numeric zone index).
- `RegionRLAgent`: `(full_loc, capped_clock_tuple)` — plain integers
  (region_env.py:132-146).
- `DDPGAgent`: a continuous `np.ndarray` (`state_dim = 4*n_tasks + n_jobs`,
  pta_env.py:89, 159-181) — no discretization at all.

**Action space**: task-ID dispatch (a string), effectively variable-size —
`_compute_actions`/`enabled_tasks` returns however many tasks are currently
dispatchable (0 to min(waiting-with-satisfied-precedence, free-machines)),
and every tabular agent selects among exactly that returned set
(`min(enabled, key=...)` or ε-greedy over `enabled`). This **is** action
masking, just implemented as "only ever consider the legal set" rather than
a fixed-size space with an explicit mask vector — except in `JobShopEnv`
(the unused gym.Env), which does use a fixed `Discrete(N)` space with an
explicit `action_mask` array (env.py:64, 109-117). `DDPGAgent`'s action space
is fixed-size and continuous (`action_dim = 1 + n_jobs`, one delay scalar +
one continuous priority score per job, pta_env.py:90) with a masking step
applied only implicitly (`_dispatch` always picks the highest-priority
*enabled* job — infeasible jobs are excluded from the `max()` call, not
masked in the network output).

**Termination**: `all(status == "done" for every task)` — every env
implements this identically (`is_terminal()` in pta.py:27-28, or the
equivalent inline `all(s == "done" for s in loc)` check in every other env).

**Reward function** — differs by agent family (quoted exactly):
- `region_rl.py` (cost-min): `elapsed` returned by `env.step()` is literally
  the wall-clock time advanced this step (region_env.py:86-111, "Returns ...
  elapsed: integer time elapsed until the next decision point (the cost)");
  summed over an episode this equals total makespan; the agent minimizes it.
- `symbolic_rl.py` (cost-min, zone-based): `r = min_cost(Z_nx, self.zctx) - prev_delta`
  (symbolic_rl.py:347) — the increase in the zone's minimum-elapsed-time bound
  this step (`delta` is a never-reset clock tracking cumulative elapsed time
  since episode start; `min_cost` reads its infimum out of the zone — see §1).
- `env.py`'s unused `JobShopEnv`: two selectable modes (env.py:31-34, 95-106):
  `"sparse"` → `0` every step, `-makespan` at the terminal step only;
  `"dense"` → `-dt` at every internal `advance_time` call.
- `pta_env.py`'s `PTADirectEnv` (used by DDPG): `reward = -(elapsed time this step)`
  (pta_env.py:136), same sign convention as `"dense"` above.

All reward conventions are cost-minimization framed as reward-maximization via
negation (`-elapsed`) or literal cost-minimization with `min` in the Bellman
update (`region_rl.py`, `symbolic_rl.py`) — never a positive/shaped reward for
progress.

---

## 6. Existing results

Stored results live under `scheduling/output/<run>/*.csv` (verified content,
`scheduling/` is a separate nested git repo from the top-level directory):

| File | Columns (relevant) | Contents |
|---|---|---|
| `output/all/comparison_summary.csv`, `output/selected/comparison_summary.csv` | `best_baseline_mean/std`, `symbolic_rl_mean/std`, `zone_graph_bfs`, `region_rl_mean/std`, `region_graph_bfs`, `region_states_theory` | Best-of-5-heuristics baseline, **learned** symbolic-RL and region-RL policy returns (500-1000 real `PTASimulator` episodes), plus BFS state counts from `graph_sizes.py` |
| `output/ddpg/comparison_summary.csv`, `output/ddpg_all/results_summary.csv` | adds `ddpg_mean/std/min/max`, `ddpg_params` | Same, plus learned DDPG returns |
| `output-old/benchmark_summary.csv` | adds `optimal_dp_mean/std`, `opt_states`, `opt_truncated` | The **only** file anywhere in the repo containing output from `OptimalSymbolicSolver` (exhaustive zone-graph DP) — and it contains exactly **one row** (`PJS_06`), with `opt_truncated=True` (BFS capped at `OPTIMAL_STATE_LIMIT = 20_000` states in `run_benchmark.py:81`). `output-old/` is a separate, apparently superseded directory from the canonical `output/` — its name and its single-row content suggest an incomplete/stale run, not a maintained result set. |

**Do we have per-instance OPTIMAL values?** Only for **one instance**
(PJS_06), from a **truncated** (hence not provably optimal — the code's own
comment at `run_benchmark.py:132-134` says exactly this: `"policy is
partial (best-effort, not provably optimal)"`) BFS, stored in the
non-canonical `output-old/` directory. For all other 19 instances (10 PJS +
10 LJS) and both current `output/` directories, **there is no optimal-value
column at all** — only learned-policy returns (symbolic-RL, region-RL, DDPG)
and the 5 fixed heuristic baselines. `OptimalSymbolicSolver` itself
(`symbolic_optimal.py`) is imported by exactly one script,
`run_benchmark.py`, out of the ~10 `run_*.py` scripts in `scheduling/`.

**UNKNOWN**: whether `output-old/`'s single truncated PJS_06 row was ever
cross-checked against `instance.expected_makespan_lower_bound()` or a re-run;
no such comparison artifact exists in the repo.

---

## 7. Reusability verdict (facts only)

**Modules with no coupling to the tabular/finite-state assumption** (pure
data or pure simulation, reusable unchanged by construction):
- `instance.py` — dataclasses only, no RL assumptions.
- `pta.py`'s `TaskState`/`GlobalState` — plain dataclasses (though
  `build_task_automaton` is dead code, see §2).
- `simulator.py`'s `PTASimulator` — real-valued discrete-event simulator, no
  finite-state enumeration.
- `strategies.py` — pure functions of `GlobalState`, no state-table.
- `benchmarks/*.py` — static instance data.

**Modules tightly coupled to a finite/enumerable-state assumption:**
- `graph_sizes.py` — its entire purpose is enumerating all reachable states
  via BFS into a Python `set`/`dict`; not applicable to a continuous or
  function-approximated state space by construction.
- `symbolic_optimal.py` — `OptimalSymbolicSolver` stores `self._states`,
  `self._transitions`, `self.Q`, `self.V` as plain dicts keyed by
  `(loc, str(Z))`; backward DP requires the full graph materialized in memory.
- `region_rl.py` / `symbolic_rl.py` — `self.Q: Dict[tuple, Dict[str, float]]`,
  a literal Python dict growing one entry per visited `(state, action)` pair;
  no function approximation anywhere, so this is the definition of "assumes a
  finite state-index table."
- `zone_ops.normalize()` — the `M`-bound extrapolation exists specifically to
  keep the zone graph finite for exhaustive BFS/DP (`symbolic_optimal.py`)
  and for a dict-keyed Q-table (`symbolic_rl.py`); a function-approximation
  agent would have no intrinsic need for this bound (though it could still
  choose to use it for state normalization).

**Concurrency-resolution logic duplicated four times**, independently
maintained, not shared (relevant to reuse because a deep-RL environment
rewrite would need to either keep re-deriving this fourth time or factor it
out): `symbolic_env.py:_compute_path`/`_advance_to_decision`,
`region_env.py:_advance`, `pta_env.py:_advance_time_limited`/`_advance_to_dispatch`,
`simulator.py:advance_time` — each independently computes `dt = min(remaining task times)`.

**Hard blockers, empirically verified in this session:**
- `pyudbm.Federation` objects are **not picklable and not deep-copyable**:
  ```
  >>> pickle.dumps(Z)
  TypeError: cannot pickle 'pyudbm.binding._udbm._NativeFederation' object
  >>> copy.deepcopy(Z)
  TypeError: cannot pickle 'pyudbm.binding._udbm._NativeFederation' object
  ```
  (`Context` and `Clock` objects, by contrast, **are** picklable/deep-copyable —
  verified.) This blocks any vectorized-env approach (e.g. subprocess-based
  `SubprocVecEnv`) that would need to serialize a `SymbolicJobShopEnv`-derived
  state containing a live `Federation` across a process boundary. A
  same-process/thread vectorized env, or one that reconstructs the `Federation`
  from `(loc, action-history)` in each worker rather than passing it directly,
  would not hit this.
- `Federation.copy()` (the native pyudbm method, not Python's `copy` module)
  does work and returns an independent `Federation` — verified.
- Import-path collision (§1): any tooling that runs with the repo root on
  `sys.path` before `pyudbm` has been imported once already will fail to
  import pyudbm at all. Not itself a concurrency/vectorization blocker, but a
  reproducibility hazard for any new script/env added to this repo.
- `SymbolicJobShopEnv.save_state()`/`restore_state()` (symbolic_env.py:119-127)
  snapshot `(self._loc, dict(self._virtual_clocks), dict(self._active_d))` —
  **plain Python objects, no `Federation`** — so env-state snapshotting itself
  is fine; the picklability problem only arises if a `Federation` object is
  kept as part of an externally-serialized state (as `symbolic_rl.py` and
  `symbolic_optimal.py` do, keeping `Z` in local variables / dict values
  across a training loop within one process).
- No global/module-level mutable state was found in any of the audited files
  (`grep` for module-level `global` statements or mutable module-level
  containers written to at import time: none found) — each `Context`/env is
  constructed fresh per instantiation.

---
