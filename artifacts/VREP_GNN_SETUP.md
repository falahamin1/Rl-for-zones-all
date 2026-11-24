# VREP_GNN_SETUP: Extending the Sweep with deepset_v and gnn

Two new encoders added to the existing 48-run sweep: `deepset_v` (DeepSets
over `Zone.v_rep()`) and `gnn` (2-layer GCN over `Zone.graph()`). 24 new
array tasks (indices 48-71), appended to `run_one.TARGET_RUNS` — the
original 48 are untouched and verified byte-identical. **Nothing submitted
to Alpine.** No direction recommended.

**Revision note**: after the first pass of this report, `gnn`'s global
pooling was switched from mean to sum to match `deepset_h`/`deepset_v`, so
all three zone-set encoders share the same aggregation and any gap between
them isolates representation, not pooling choice. This revision re-verifies
everything downstream of that change (parameters, smoke test, resume
correctness) and tightens the Step 3 reporting: `deepset_v`/`gnn` are
dropped from the difference-probe table (an 8k-step quick-trained row next
to a 300k-step real-checkpoint row invited misreading even with a caveat),
the `gnn` permutation-ablation row is removed pending a real checkpoint, and
a precise per-encoder definition of what is actually permuted is included,
since `cnn_dbm` and `cnn_proj`'s numbers turned out not to be measuring
comparable things.

---

## Step 0: feasibility (full detail in `/tmp/VREP_FEASIBILITY.md`)

**Both gates passed with wide margin, no array split needed.** Unaffected
by the pooling change (Step 0 doesn't touch encoder architecture).

- **v_rep cost**: worst case 0.63 projected hours per 300k-step run,
  *uncached* (gridtimed/T3) — three orders of magnitude under the 20-hour
  stop threshold. Zero `VertexEnumerationLimit` hits across 2000 sampled
  zones (500 × 4 instances). The content-keyed cache
  (`zoneviews.raster.cached_v_rep`, reusing the `_zone_content_key`
  fingerprint from `PRESUBMIT` — not `str(zone)`) measured 85-93% hit rate
  under realistic 4-epoch replay, cutting projected cost to under 6
  minutes per run.
- **Zone degeneracy**: no instance is majority dim-0. Highest is
  modesel/M3 at 25.2% — flagged plainly since a dim-0 zone's `v_rep()` is
  a single point carrying almost no shape information. gridtimed's tiers
  are the least degenerate (9-10%).

---

## Step 1: the two encoders

**`encoders/deepset_v.py`** — architecturally identical to `deepset_h`
(same hidden width, sum-pool, per-job broadcast mechanism) so any H-vs-V
gap isolates the representation, not an incidental architecture
difference. `VertexEnumerationLimit` triggers a bounding-box-corner
fallback (`2**d` points, read directly off per-clock DBM bounds), counted
via `vrep_fallback_frac` (null for the other five encoders, never a
misleading 0.0). Unchanged by this revision.

**`encoders/gnn.py`** — 2-layer message-passing GCN over dense
per-zone node/edge tensors (no `torch_geometric`/`dgl` in this
environment; exact here, not an approximation, since every zone in one
instance has the same fixed node count `d+1`, only which edges are finite
varies — handled with an edge mask). Node features are each clock's own
bounds against the reference; edge features are `(bound, strict,
finiteness)`, the same 3-channel convention `cnn_dbm` uses per pixel.
**Global pooling is now SUM, matching `deepset_h`/`deepset_v`** (changed
from the initial mean-pool choice): `pooled = self.readout(h.sum(dim=1))`.
`cnn_dbm`/`cnn_proj` keep their own mean pool unchanged — they read off a
fixed-size tensor, not a variable zone-derived set, so sum-vs-mean doesn't
carry the same "set cardinality" meaning there; only the three genuinely
set-structured encoders (`deepset_h`, `deepset_v`, `gnn`) needed to be
brought into alignment.

**Action-head treatment** (both new encoders, unaffected by the pooling
change — the action head reads `per_task`, not `pooled`):

| Domain | deepset_v | gnn |
|---|---|---|
| modesel | per-element (`task_job_idx` available) | per-element |
| gridtimed | **pooled fallback** (`task_job_idx=None`, same reason as `deepset_h`/`cnn_dbm`/`cnn_proj`) | **pooled fallback** |

**Parameter counts, re-verified after the pooling change** (pooling is a
parameter-free op, so no shift was expected — confirmed by re-running the
count, not assumed):

| Instance | mlp | deepset_h | cnn_dbm | cnn_proj | deepset_v | gnn | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| modesel/M3 | 31,764 | 30,936 | 31,426 | 30,949 | 30,940 | 33,068 | 6.9% |
| modesel/M4 | 34,116 | 31,064 | 31,426 | 31,345 | 31,068 | 33,068 | 9.8% |
| gridtimed/T2 | 32,437 | 32,213 | 32,835 | 32,406 | 32,215 | 34,129 | 5.9% |
| gridtimed/T3 | 34,357 | 32,341 | 32,835 | 32,901 | 32,343 | 34,129 | 6.2% |

Identical to the pre-change table, all four instances under 10% (M4 the
tightest at 9.8%).

---

## Step 2: SLURM

- `run_one.TARGET_RUNS`: 72 entries. Indices 0-47 verified byte-identical
  to the pre-extension sweep (`test_run_one.py::test_first_48_unchanged`,
  checked against a transcribed snapshot, not re-derived from the same
  code that produces them). Indices 48-71 appended via a **separate**
  loop, not interleaved into the original one. **9/9 tests re-run after
  the pooling change, still pass** (the change is internal to
  `GNNEncoder.forward`; `TARGET_RUNS` construction doesn't touch encoder
  internals).
- `submit_vrep_gnn.sbatch` (unchanged by this revision — verified by
  re-reading, not re-derived):

  ```
  #SBATCH --nodes=1
  #SBATCH --time=72:00:00
  #SBATCH --partition=acpu
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=1
  #SBATCH --mem=32G
  #SBATCH --array=48-71%12
  #SBATCH --job-name=zone-encoders-vg
  #SBATCH --output=zone-encoders-vg.%A_%a.out
  #SBATCH --qos=cpu-long

  module purge
  module load anaconda
  conda activate amfa-custom-env
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  cd "$WORK_DIR"
  python3 run_one.py
  ```

  **Exact command to submit, when/if the user decides to** (not run here):

  ```
  sbatch submit_vrep_gnn.sbatch
  ```

- `check_status.py`: zero modification, re-confirmed running cleanly —
  reports all 72 configs (48 pre-existing + 24 new, all "not started"
  locally, as expected since nothing has been submitted).
- `analyze.py`: `ENCODERS` extended to six, imports and runs cleanly;
  `encoder_stats.csv` generation confirmed unaffected by the pooling
  change (it reads jsonl records, not encoder internals).

**Local smoke test, re-run after the pooling change** (2 array indices,
one per action-head regime — same two used in the original pass):

- `--array-index 48` (modesel/M3/deepset_v/seed0, per-element head, not
  touched by the pooling change): short real run (5000 steps, 3 updates)
  — `ep_return_mean` -76.6 → -72.3 → -74.0, `policy_loss`/`value_loss`/
  `grad_norm` all in normal ranges, `vrep_fallback_frac: 0.0` on every
  record (consistent with §0's zero-fallback finding).
- `--array-index 63` (gridtimed/T2/gnn/seed0, pooled head, **the changed
  architecture**): short real run (5000 steps, 3 updates) —
  `ep_return_mean` -77.6 → -76.3 → -66.4, `policy_loss` ranges
  -0.0015..0.051, `value_loss` 738..2002, `grad_norm` 357..1119, entropy
  decreasing smoothly (2.13 → 1.69). **No destabilization observed** —
  sum-pool changes activation scale in principle (summing `d` node
  embeddings instead of averaging them), but no NaN/Inf, no
  loss/gradient blowup, no runaway entropy collapse showed up in this
  smoke-scale run. `vrep_fallback_frac: null` throughout, correctly
  omitted (gnn never touches `cached_v_rep`).
- **Kill-mid-run resume test, run specifically against `gnn`** (the
  changed encoder, not `deepset_v` as in the first pass): killed at 30s
  into a run with `--ckpt-every-steps 10000 --total-steps 100000`,
  landing between the first checkpoint (after update 4, env_steps=10240)
  and the next — a genuine orphan tail. Checkpoint recorded
  `jsonl_offset=2972`; the jsonl file had grown to 3576 bytes (6 records,
  updates 0-5). Verified directly the checkpoint's offset lands exactly
  at the end of record 4, making record 5 the true orphan. Resumed with
  `run_one.py`'s own log confirming
  `resumed from checkpoint: env_steps=10240 update=5`; final jsonl update
  sequence `[0..19]`, strictly increasing, no gaps, no duplicates. Confirms
  the sum-pool architecture change didn't affect `state_dict()`
  serialization or the truncate-on-resume contract.

Local test artifacts (checkpoints/jsonl for indices 48 and 63) were
deleted after verification — no leftover state affects a real submission.

**One real bug found and fixed in the original pass, unaffected by this
revision**: `difference_probe.py`'s `load_frozen_head` was missing the
`weights_only=False` fix `run_one.py`/`check_status.py` already had for
PyTorch ≥2.6's stricter `torch.load` default.

---

## Step 3: reporting

### Success rate, final return, timing (six encoders, `analyze.py`'s `encoder_stats.csv`)

Populated automatically for any run with data. Currently only the
original 48 have real (300k-step) numbers, since none of the 24 new
configs (including the now-changed `gnn`) have been submitted.
`encoder_forward_frac`, `raster_frac` (v_rep enumeration is captured by
the existing `raster_s` field — `cached_v_rep` accumulates into the same
timer `dbm_tensor`/`projection_stack` use), and `vrep_fallback_frac`
columns are wired and confirmed schema-correct against the new smoke-test
jsonl output above.

### Permutation ablation: what is actually permuted, per encoder

Before presenting any KL numbers, the six encoders' `shuffle_*` hooks do
**not** all test the same kind of sensitivity — read literally from each
encoder's `forward()`:

| Encoder | kwarg | What moves | What it tests |
|---|---|---|---|
| `mlp_point` | `shuffle_tasks` | Each block's status/elapsed content is permuted into a **different position slot** of the flat point vector; the action head's fixed slot-i→action-i convention and the mask are untouched. | Whether the MLP relies on position↔content binding learned during training, independent of genuine block identity. |
| `deepset_h` | `shuffle_elements` | Order of the constraint-row **set** (`zone.h_rep()`) before sum-pool. | Sum-pool invariance — expected exactly 0 by construction. |
| `cnn_dbm` | `shuffle_clocks` | **One joint permutation** of real clocks `1..d`, applied to **both rows and columns** of the `(n,n)` DBM grid consistently (reference clock 0 fixed) — relabels which clock's data sits at which **grid position**, moving it to a different 2D neighborhood under the 3×3 conv kernels. | Whether the CNN's spatially-local kernels rely on which grid *position* (≡ which clock identity) content sits at. |
| `cnn_proj` | `shuffle_channels` | Order of the `d(d-1)/2` **already-fixed channel slots** (each channel is one specific clock-pair raster) — no spatial change within any channel, only which index of the channel *axis* a given raster occupies. | Whether the conv stack's per-channel weights are order-sensitive across the channel axis. Conv2d combines all input channels densely at every spatial location regardless of axis order — so this is only informative if the *learned* per-channel weights differ substantially, which is a different (weaker) condition than `cnn_dbm`'s spatial test. |
| `deepset_v` | `shuffle_elements` | Order of the vertex-row **set** (`zone.v_rep()`) before sum-pool. | Sum-pool invariance — expected exactly 0 by construction, same as `deepset_h`. |
| `gnn` | `shuffle_nodes` | One joint permutation of real-clock nodes `1..d`, applied to node features and both axes of the edge tensors consistently (reference node 0 fixed). | Structurally the graph analogue of `cnn_dbm`'s test (node-position≡clock-identity), but **not invariant by construction** — mean/sum-aggregated message passing gives no symmetry guarantee against it, unlike the DeepSets pair. |

**`cnn_dbm` and `cnn_proj`'s numbers are not directly comparable**:
`cnn_dbm`'s shuffle moves *where in 2D space* a clock's information sits,
which conv kernels are locally sensitive to by construction; `cnn_proj`'s
shuffle only reorders *which channel-axis index* holds an already-fixed
raster, which conv layers process via full dense cross-channel mixing
regardless of order. A large KL for `cnn_dbm` and a near-zero KL for
`cnn_proj` on the same instance is therefore not evidence that one CNN is
more "order-robust" than the other in a comparable sense — the two are
being asked different questions. Presented separately below, not as a
head-to-head CNN comparison.

### Permutation ablation results (modesel/M3, gridtimed/T2)

`gnn` is **excluded from this table** — no real checkpoint exists for it
yet (nothing submitted), and unlike the DeepSets pair it has no
architectural invariance guarantee, so a quick-trained (8,000-step) number
would not be informative about its trained behavior either way. It will
be measured once the array actually runs. (A quick-trained diagnostic
value is still sitting in the raw
`perm_ablation_results.json` file for reference, but is deliberately left
out of this report's table to avoid it being read as a finding.)

| Encoder | mean KL (M3) | mean KL (T2) | Invariant by construction? |
|---|---:|---:|---|
| `mlp_point` | 8.6e-05 | 8.8e-03 | No |
| `deepset_h` | ~0 (-6e-10) | ~0 (-2e-09) | **Yes** (sum-pool) |
| `cnn_dbm` | 0.679 | 0.085 | No |
| `cnn_proj` | ~0 (4e-10) | 0.030 | No |
| `deepset_v` | ~0 (0.0 exactly) | ~0 (1e-09) | **Yes** (sum-pool) |

`deepset_v` reproduces `deepset_h`'s exact-zero pattern, as expected from
identical sum-pool. `cnn_dbm` vs. `cnn_proj`: see the "not directly
comparable" note above — read each in isolation, not against the other.

### Difference-feature probe, full capacity

**`deepset_v` and `gnn` are dropped from this table entirely.** The four
original encoders' numbers come from real, fully-trained checkpoints (up
to 301k steps); the two new encoders have no real checkpoint yet, so their
only available numbers were from an 8,000-step quick-training fallback —
two orders of magnitude less training. Placing that row next to a
300k-step row, even with a caveat attached, invites exactly the kind of
misreading this report is trying to avoid. These will be measured for real
once the array (indices 48-71) actually completes on Alpine.

2-layer/256-unit MLP probe (the `PRESUBMIT` A2 protocol, trained to
convergence with early stopping), for the four original encoders only:

| Encoder | ridge R² | full-capacity MLP R² | source |
|---|---:|---:|---|
| `mlp_point` | 0.932 | 0.948 | real checkpoint, 143k steps |
| `deepset_h` | 0.945 | 0.994 | real checkpoint, 301k steps |
| `cnn_dbm` | 0.899 | 0.969 | real checkpoint, 164k steps |
| `cnn_proj` | 0.272 | 0.342 | real checkpoint, 127k steps |

(The 8k-step `deepset_v`/`gnn` numbers from the quick-training fallback
remain in the raw `diffprobe_vg_results.json` file for reference, but are
excluded from this report's table for the same reason as above.)

---

## Files delivered / modified in this revision

- `encoders/gnn.py`: pooling changed from mean to sum
  (`h.mean(dim=1)` → `h.sum(dim=1)`), docstring updated to match.
- Everything else from the original pass is unchanged; see the file list
  below.

**Full file list**:
- `encoders/deepset_v.py`, `encoders/gnn.py` (new).
- `zoneviews/raster.py`: `cached_v_rep`/`vrep_cache_stats`/`clear_vrep_cache`.
- `train/ppo.py`: `ENCODER_NAMES` extended, `build_head` handles `deepset_v`/`gnn`.
- `run_one.py`: `NEW_ENCODERS`, `TARGET_RUNS` extended to 72,
  `vrep_fallback_frac` wired into the jsonl schema.
- `test_run_one.py`: 72 entries + explicit 0-47-unchanged assertion.
- `submit_vrep_gnn.sbatch` (new).
- `analyze.py`: `ENCODERS` extended to six, `encoder_stats.csv` output.
- `difference_probe.py`: `weights_only=False` bug fix.
- `/tmp/VREP_FEASIBILITY.md`, this report — both copied to `artifacts/`.

**Not submitted to Alpine.** The exact command to submit, if and when
that decision is made, is `sbatch submit_vrep_gnn.sbatch` from the repo
root with `WORK_DIR` pointed at the Alpine checkout. No direction is
recommended here — real trained results for `deepset_v` and `gnn` don't
exist yet.
