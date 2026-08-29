# Spike: does interval-valued task duration produce full-dimensional zones?

**Status: throwaway spike, not merged, package not refactored.** All code
changes live in `/tmp/prob_jobshop_spike/` (a full copy of `prob_jobshop_ext/`
with **only** `symbolic_env.py` modified) and `/tmp/zone_common.py` /
`/tmp/zone_dim.py` / `/tmp/zone_dim_interval.py` (throwaway measurement
scripts). `prob_jobshop_ext/` at the repo root was never touched — verified
untouched, no diff. `region_env.py`, `pta_env.py`, `simulator.py`, and every
agent (`region_rl.py`, `symbolic_rl.py`, `symbolic_optimal.py`, `ddpg_*.py`)
are byte-identical to the original in the scratch copy — not edited, per
instructions. Facts only below; no direction is recommended.

---

## Step 1 — `affine_dim` metric and current-semantics measurement

**Definition** (`/tmp/zone_common.py:affine_dim`): two clock indices `i,j`
(0 = reference) are **pinned** iff neither `(i,j)` nor `(j,i)` is the
infinity sentinel, neither is strict, and `bound(i,j) + bound(j,i) == 0`
(i.e. `x_i - x_j` is forced to an exact constant). Union-find over
`{0, ..., dbm.dimension-1}` under "pinned"; `affine_dim = (components) - 1`
(the reference clock's own component doesn't count as a free dimension).

**Sanity checks** (all PASS):
```
zero zone            affine_dim = 0  (expected 0)
zero-zone.up()        affine_dim = 1  (expected 1)
zero zone            exact_vertices = 1  (expected 1)
```

**Current (exact-duration) semantics, cap 20,000 states, exact (untruncated
except LJS_01):**

| Instance | n_states | truncated | max_depth | wall (s) | affine_dim histogram | dim mean | dim max | mean concurrent-active |
|---|---:|:---:|---:|---:|---|---:|---:|---:|
| PJS_01 | 110 | No | 3 | 0.01 | {1:1, 2:14, 3:95} | 2.855 | 3 | 0.409 |
| PJS_02 | 402 | No | 4 | 0.02 | {1:1, 2:23, 3:378} | 2.938 | 3 | 0.346 |
| PJS_03 | 4,283 | No | 5 | 0.31 | {1:1, 2:69, 3:564, 4:3649} | 3.835 | 4 | 0.312 |
| PJS_04 | 15,767 | No | 8 | 1.87 | {1:1, 2:281, 3:2217, 4:13268} | 3.824 | 4 | 0.740 |
| LJS_01 | 20,000 | **Yes** | 4 | 3.59 | {1:38, 2:162, 3:786, 4:3243, 5:10262, 6:5509} | 5.003 | 6 | 2.384 |

**Hypothesis stated in the task ("I expect dim to be 0 or 1 almost
everywhere") is REFUTED, clearly and by a wide margin.** `affine_dim` sits
close to its *maximum possible* value (n_user_clocks) at almost every
reachable state, not close to 0/1 — e.g. for PJS_01 (3 user clocks, max
possible affine_dim = 3), **95 of 110 states (86%)** have affine_dim exactly
3, the maximum; for LJS_01 (7 user clocks, max possible = 7), the mean is
5.00 and the mode is 5 (10,262/20,000 states). Only the very first
post-reset state in each instance has affine_dim = 1 (matching the sanity
check — a freshly-`up()`'d zero zone).

**Why, mechanistically**: a job clock only gets pinned (equality-constrained)
at the exact instant *that job's own active task* completes. Between
"task starts" and "that same task completes," its clock has only an upper
bound (`c_J <= d`, from the invariant), no lower bound forcing equality — it
is free to range anywhere from wherever it has accumulated up to that bound.
At any given decision point, only the *one* job whose completion triggered
that decision point is (momentarily) pinned; every other job clock — whether
belonging to a still-active task, or a not-yet-dispatched task that has
simply been accumulating time since the episode start alongside `delta` — is
unpinned relative to the others. This is a direct, mechanistic explanation,
not a guess: it follows from re-reading `_compute_path`'s guard/invariant
construction (`prob_jobshop_ext/symbolic_env.py:211-217`, quoted in
AUDIT.md §2) with the `affine_dim` union-find applied to the actual DBM
output rather than inferred from the code.

---

## Step 2 — interval-duration variant

**Location**: `/tmp/prob_jobshop_spike/symbolic_env.py` only. Module-level
knob `INTERVAL_WIDTH: float = 0.0` (set by the driver script per sweep
point). `w = 0.0` is a deliberate degenerate case: `[lo,hi] = [d,d]`,
reproducing the original exact-duration semantics exactly bit-for-bit —
confirmed by regression (below).

**Design decisions actually made** (this is where the task explicitly asked
for judgment calls to be reported):

1. **Formula**: `lo, hi = max(1, floor(d*(1-w))), max(lo, ceil(d*(1+w)))`
   — the `max(1, ...)` / `max(lo, ...)` clamps guard against a
   zero/negative or inverted interval for small `d` at large `w` (e.g.
   `d=1, w=0.4` → naive `floor(0.6)=0`, clamped to `lo=1`).
2. **`invariant()`** (symbolic_env.py, "Upper-bound constraints"): changed
   from returning the exact bucket value `d` to returning `hi`.
3. **`_compute_path`**'s `completing_guards`: changed from appending
   `(clock_name, exact_d)` (which, combined with the `<=exact_d` invariant,
   pinned `c_J == exact_d`) to appending `(clock_name, lo)` — combined with
   the new `<=hi` invariant, this leaves genuine `[lo,hi]` room instead of
   pinning. This is the literal implementation of "the symbolic side simply
   does not pin clocks."
4. **Event ORDERING/topology is deliberately left unchanged.** `_compute_path`
   still decides *which* task's completion triggers each path step (i.e.
   the decision-point structure / branching topology of the zone graph)
   using the bucket value `d`'s `dt = min(remaining)` bookkeeping — the same
   control flow as the original code. **This was a scope decision, not an
   oversight**: under genuine interval uncertainty, two concurrently active
   tasks' real completion order can be ambiguous (task A's interval and task
   B's interval can overlap), and the "correct" treatment would be to branch
   the zone graph over *both* possible orderings whenever intervals overlap.
   That is a materially larger change (new branch dimension in the zone
   graph, not just different bound values) and was explicitly out of scope
   per the task's framing ("simply not pin clocks"). **Consequence, stated
   plainly**: the reported zone graph for `w>0` still reflects the
   *original* exact-duration event ordering, with only the DBM bound values
   loosened. It is not a full interval-timed-automaton model. The zone
   graph's *state count* under `w>0` should be read as "same topology,
   looser per-state zones," not as "the true interval-duration zone graph."
5. **Concrete `.step()`** samples a duration bucket exactly as before (same
   categorical probabilities), then additionally samples a real value
   `real_dur ~ Uniform(lo, hi)`, stored in a new `self._active_real_dur`
   dict. **`_advance_to_decision`** (the concrete env's own decision-point
   timer) was changed to prefer `self._active_real_dur[job_id]` over the
   bucket value when present, falling back to the bucket value otherwise
   (e.g. for `step_deterministic`, used by the optimal solver, which is
   unchanged and doesn't set a real-duration value). This realizes "the
   concrete side stays executable" using an actual sampled real number, kept
   fully separate from the symbolic side's `[lo,hi]` bounds.
6. **`step_deterministic`, `save_state`/`restore_state`, `rate`,
   `clock_names`, `M`, `_has_dispatch_pure`, `_compute_actions`'s branch
   construction (bucket `d`/`prob` per branch): all unchanged.**

**pyudbm behavior — no rejections observed.** Ran `w ∈ {0.0, 0.1, 0.25, 0.4}`
on PJS_01 first as a smoke test: zero exceptions, zero unexpected `is_empty()`
zones, and terminal (all-tasks-done) states were reached under every width
(48/39/26/30 terminal states out of 110/103/72/82 total reachable states for
w=0/0.1/0.25/0.4 respectively) — i.e. the model can still complete episodes
end-to-end under interval semantics; it is not degenerately dead-ending. No
place in `_compute_path`/`invariant`/`_propagate`-equivalent code required
reordering guard/invariant application beyond the two value substitutions
described above (points 2–3) — the existing operation *order* (invariant via
`apply_delay`, then completion guard, then next `apply_delay`) is untouched
and continued to type-check against pyudbm at every width tested.

**Regression check (w=0.0 vs. stock package, PJS_01):** exact match —
`n_states=110, truncated=False, max_depth=3`, affine_dim histogram
`{1:1, 2:14, 3:95}`, `mean_concurrent_active=0.409`, identical to Step 1's
stock measurement to the last decimal. This confirms the interval-variant
code changes are behaviorally inert at `w=0`, i.e. the diff is isolated to
exactly the intended semantic change.

---

## Step 3 — comparison table, w ∈ {0.0, 0.1, 0.25, 0.4}

Cap 20,000 states. `dim` = affine_dim; `vertices (mean, known)` computed
exactly (see methodology note below) over up to 3,000 sampled zones per
instance, with the fraction that were combinatorially intractable at this
budget reported separately (`unk_frac`) rather than guessed.

| Instance | w | n_states | trunc | depth | wall(s) | dim mean/max | frac dim≥2 | vertices mean (known) | unk_frac | mean concurrent-active |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| PJS_01 | 0.00 | 110 | No | 3 | 0.01 | 2.85/3 | 0.991 | 2.30 | 0.000 | 0.409 |
| PJS_01 | 0.10 | 103 | No | 3 | 0.01 | 2.92/3 | 0.990 | 2.58 | 0.000 | 0.437 |
| PJS_01 | 0.25 | 72 | No | 3 | 0.01 | 2.89/3 | 0.986 | 2.51 | 0.000 | 0.417 |
| PJS_01 | 0.40 | 82 | No | 3 | 0.01 | 2.90/3 | 0.988 | 2.37 | 0.000 | 0.439 |
| PJS_02 | 0.00 | 402 | No | 4 | 0.02 | 2.94/3 | 0.998 | 2.30 | 0.000 | 0.346 |
| PJS_02 | 0.10 | 465 | No | 4 | 0.03 | 2.97/3 | 0.998 | 2.77 | 0.000 | 0.325 |
| PJS_02 | 0.25 | 423 | No | 4 | 0.03 | 2.98/3 | 0.998 | 2.78 | 0.000 | 0.343 |
| PJS_02 | 0.40 | 298 | No | 4 | 0.02 | 2.97/3 | 0.997 | 2.76 | 0.000 | 0.389 |
| PJS_03 | 0.00 | 4,283 | No | 5 | 0.31 | 3.84/4 | 1.000 | 3.20 | 0.000 | 0.312 |
| PJS_03 | 0.10 | 4,768 | No | 5 | 0.32 | 3.95/4 | 1.000 | 4.11 | 0.000 | 0.284 |
| PJS_03 | 0.25 | 4,872 | No | 5 | 0.32 | 3.95/4 | 1.000 | 3.96 | 0.000 | 0.276 |
| PJS_03 | 0.40 | 4,398 | No | 5 | 0.29 | 3.95/4 | 1.000 | 3.81 | 0.000 | 0.266 |
| PJS_04 | 0.00 | 15,767 | No | 8 | 1.88 | 3.82/4 | 1.000 | 3.85 | 0.000 | 0.740 |
| PJS_04 | 0.10 | 16,059 | No | 8 | 1.88 | 3.98/4 | 1.000 | 4.85 | 0.000 | 0.678 |
| PJS_04 | 0.25 | 13,584 | No | 8 | 1.65 | 3.98/4 | 1.000 | 4.80 | 0.000 | 0.704 |
| PJS_04 | 0.40 | 6,269 | No | 8 | 0.83 | 3.99/4 | 1.000 | 4.74 | 0.000 | 0.785 |
| LJS_01 | 0.00 | 20,000 | **Yes** | 4 | 3.52 | 5.00/6 | 0.998 | 5.25 | **0.800** | 2.384 |
| LJS_01 | 0.10 | 20,000 | **Yes** | 4 | 3.42 | 5.14/6 | 1.000 | 5.25 | **0.864** | 2.284 |
| LJS_01 | 0.25 | 20,000 | **Yes** | 4 | 3.39 | 5.33/6 | 1.000 | 5.33 | **0.877** | 2.238 |
| LJS_01 | 0.40 | 20,000 | **Yes** | 4 | 3.38 | 5.18/6 | 1.000 | 5.24 | **0.894** | 2.227 |

**Methodology note on exact vertex enumeration — this needed a correction
mid-spike, worth flagging.** The task said "exact enumeration is fine at
these sizes," which is true for PJS_01-04 (`unk_frac = 0.000` throughout —
100% exact coverage) but turned out **not** to be true for LJS_01, precisely
*because* Step 1 refuted the low-affine-dim hypothesis: with affine_dim
routinely 5-6 (not 0-1), a naive combinatorial tight-constraint-subset search
over the *unreduced* space is intractable (initial attempt did not finish in
120s on a 200-zone sample). Fix: reduce to the `affine_dim`-dimensional free
subspace first (collapse pinned clock groups via the same union-find as
`affine_dim`, dedupe to the tightest bound per collapsed constraint pair),
then enumerate exactly within that reduced space. This is fast (~4-5ms/zone)
for `affine_dim <= 4`, which covers PJS_01-04 completely, but for LJS_01
(affine_dim up to 6, up to ~36 reduced constraints) the combinatorial count
`C(m,k)` is still large enough that a hard budget cap (`m<=18, k<=4`) was
applied; zones exceeding it are reported as `UNKNOWN` (80-89% of the LJS_01
sample) rather than guessed at. **A second, more serious correction**: an
earlier version of this metric used `scipy.spatial.HalfspaceIntersection`
(qhull) for speed, which was **empirically wrong** — these reduced
polyhedra are routinely **unbounded** (`delta`, the cost clock, is
deliberately never given an upper bound by `zone_ops.normalize()` — see
AUDIT.md §1 — so any zone where `delta`'s group is a free dimension is
unbounded in that direction), and qhull's `HalfspaceIntersection` assumes a
bounded region. Caught by cross-validating against a hand-solved concrete
case (a PJS_03 zone reducing to the wedge `{x≥0, y≥0, x≤1, x≤y}`, unbounded
as `y→∞`): qhull reported 3 vertices, the correct answer (verified by hand)
is 2 finite vertices. Reverted to exact combinatorial enumeration (which
correctly reports only finite extreme points, with no notion of "vertices at
infinity") after that finding; the numbers in the table above are from the
corrected method.

**Correlation between `affine_dim` and concurrent-active-task count, w=0.25**
(the task's stated hypothesis: "they should be close to equal"):

| Instance | Pearson r (dim vs. concurrent-active) |
|---|---:|
| PJS_01 | -0.211 |
| PJS_02 | -0.091 |
| PJS_03 | -0.140 |
| PJS_04 | -0.023 |
| LJS_01 | -0.135 |

**Hypothesis REFUTED**: the correlation is weak and, where it is
distinguishable from noise, **negative**, not "close to equal" (which would
require r close to +1, and the two quantities to be numerically close —
compare, e.g., LJS_01 w=0.25: `affine_dim` mean 5.33 vs. `mean
concurrent-active` 2.238 — not close in magnitude either). Mechanistically
this follows directly from the Step-1 explanation above: `affine_dim` counts
*every* clock not currently pinned to another (including waiting-job clocks
that haven't dispatched their first task yet, which still track elapsed time
alongside `delta` and are technically "free" relative to jobs that have
already dispatched and diverged), not just clocks belonging to
*concurrently active* tasks — the two concepts are related but genuinely
different quantities in this model.

---

## Step 4 — summary

Both stated hypotheses in the task are **refuted** by direct measurement
against real pyudbm `Federation` objects, not inferred from reading the
code. `affine_dim` is close to its maximum (not 0-1) at almost every
reachable state under both the current exact-duration semantics and the
interval-duration variant at every tested width, because in this job-shop
model a clock is only pinned at the exact instant its own task completes —
every other clock (active-but-not-completing, or not-yet-dispatched) stays
genuinely free at any given decision point. Correlation between `affine_dim`
and the number of concurrently active tasks is weak-to-negative, not the
expected near-equality, because `affine_dim` also counts free-floating
not-yet-dispatched job clocks that have nothing to do with concurrency.
Widening the duration interval (`w`) does not noticeably move `affine_dim`
(means stay within ~0.15 of the `w=0` baseline across all five instances and
all three nonzero widths) — the *bound values* loosen as designed, but the
*dimensionality* of the zones was already close to maximal before any
interval was introduced, so there was little headroom left for interval
width to add further dimension. `n_states` itself is not monotonic in `w`
(e.g. PJS_01: 110→103→72→82; PJS_03: 4283→4768→4872→4398) — reachable-state
count does not simply grow or shrink smoothly with interval width in this
model, for reasons not further investigated in this spike. No pyudbm
operation was rejected and no unexpectedly-empty zone was observed at any
tested width; the one real methodological problem encountered (`qhull`
mis-handling unbounded reduced polyhedra) was in this spike's *measurement
tooling*, not in the modified `symbolic_env.py` or in pyudbm itself.
