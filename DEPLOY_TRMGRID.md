# Deploying the trmgrid extension (array indices 168-287) to Alpine

This is a packaging/instructions doc only. No code was changed to produce
it. Follow the checklist top to bottom.

Placeholders you need to fill in:
- `<ALPINE_HOST>` — your SSH config alias for the Alpine login node
- `<LOCAL_REPO>` — absolute local path to this repo (on this machine:
  `/Users/falah/Desktop/git-projects/RL-for-zones`)

---

## 1. What goes, what's protected

### Ships to Alpine (≈1.1 MB total, verified by `du -sh` locally)

| Path | Why |
|---|---|
| `trmgrid/` (180K) | New benchmark package. `run_one.py` imports `trmgrid.generator` and `trmgrid.gym_env`; those import `.env`, `.instance`, `.zone_ops` in turn. `zone_ops.py` imports `pyudbm` (via `zoneviews`' bootstrap) and `zones.py`/`dp.py`/`measurement*.py` aren't imported at runtime by the sweep but are small enough to ship as one package. |
| `run_one.py` (36K) | Extended to 288 `TARGET_RUNS` entries (indices 168-287 are the new trmgrid block). This is the file the array actually runs. |
| `gridtimed_eval.py`, `trmgrid_eval.py` | `run_one.py` imports `evaluate_gridtimed` / `evaluate_trmgrid` from these directly. |
| `test_run_one.py` | Covers the full 288-entry table, including the collision check for the new domain (see §3). |
| `check_status.py` | Imports `CKPT_DIR`, `RESULTS_DIR`, `TARGET_RUNS`, `TOTAL_STEPS` from `run_one` — needed for monitoring in §4. |
| `submit_trmgrid.sbatch` | The array script itself. |
| `encoders/` (112K) | `run_one.py` imports `encoders.deepset_v` directly; all 6 encoder modules (`mlp_point`, `deepset_h`, `cnn_dbm`, `cnn_proj`, `deepset_v`, `gnn`) are referenced by the sweep's `_build_target_runs`. `deepset_v.py` additionally imports `zoneviews.raster.cached_v_rep`. |
| `train/` (160K) | `run_one.py` imports `train.ppo` and `train.vec_env`. `train/ppo.py` shows modified in local git status — the Alpine copy needs to match. |
| `zoneviews/` (236K) | `encoders/deepset_v.py` and `trmgrid/gym_env.py`/`trmgrid/zone_ops.py` import it, including the `pyudbm` sys.path bootstrap in `zoneviews/__init__.py`. `zoneviews/raster.py` shows modified locally — must sync. |
| `gridtimed/` (264K) | `run_one.py` imports `gridtimed.generator`, `gridtimed.gym_env`, `gridtimed.tiers`. `gridtimed/gym_env.py` shows modified locally — must sync even though this extension doesn't touch gridtimed's own indices. |
| `modesel/` (176K) | `run_one.py` imports `modesel.generator`, `modesel.gym_env`, `modesel.tiers`. Shows unmodified locally (almost certainly already correct on Alpine from prior batches) — included anyway so the sync is self-contained; rsync will no-op on it if it's already identical. |

### Explicitly does NOT go

- **`results/` and `results/ckpt/`** — never a source path in any command
  below. `results/` holds 336 completed run files (≈366M total incl.
  checkpoints, ≈66M for `results/ckpt/` alone) pulled from/written on
  Alpine. Overwriting these from your Mac would destroy real training
  data or resurrect stale local copies over newer cluster state.
- **`pyudbm/`** — this is a git submodule containing pyudbm's native
  build. Alpine's copy was compiled there specifically (via
  `build_pyudbm_alpine.sh`, against Alpine's `gcc/14.2.0` + conda's
  `libstdcxx-ng`) and prior batches already run successfully against it.
  Your local copy is a **macOS** build — syncing it over would either
  waste bandwidth (it's large — `pyudbm_ready.zip` alone is 411M) or, if
  you synced a built `.so`, silently break `import pyudbm` on Alpine with
  a platform mismatch. Don't touch it. If `import pyudbm` turns out to be
  broken on Alpine during §3, that's a signal to re-run
  `build_pyudbm_alpine.sh` **on Alpine**, not to sync from here.
- **`RL/`, `scheduling/`** — separate git submodules, unrelated to this
  sweep.
- **Everything else at repo root** not listed in the table above:
  `artifacts/`, `analysis/`, `graphs/`, `supplementary/`,
  `prob_jobshop_*/`, `deadline_*/`, `gate_track/`, `Timed-RM/`,
  `code.zip`, `pyudbm_ready.zip`, `AUDIT.md`, `ENV_AUDIT.md`,
  `__pycache__/`, `.git/`. None of these are imported by `run_one.py` or
  `submit_trmgrid.sbatch`.

### Checked and clean

- No hardcoded local (`/Users/...`) paths in any file being shipped —
  only some one-off scripts under `artifacts/` (not shipped) reference
  local paths.
- No new third-party dependency: `encoders/gnn.py` has a docstring note
  confirming it deliberately avoids `torch_geometric`/`dgl` and uses only
  `torch`/`numpy`, already required by the existing sweep.
- `submit_trmgrid.sbatch` **already contains** the `LD_LIBRARY_PATH` fix
  for pyudbm's `GLIBCXX_3.4.32` issue (lines 37-43: exports
  `$CONDA_PREFIX/lib` onto `LD_LIBRARY_PATH` before `python3 run_one.py`
  runs). You do not need to add this yourself.
- No separate trmgrid config/instance file to ship — `G_SMALL`
  (`grid_n=4, k=3`) and `G_LARGE` (`grid_n=5, k=3`) are defined inline in
  `run_one.py` itself, not imported from elsewhere.
- No trmgrid-specific test file exists outside `test_run_one.py` — that's
  the one test file that matters here.

---

## 2. rsync: push code to Alpine

Run these from your local repo root.

```
cd <LOCAL_REPO>
```

### Step 2a — dry run (see what would transfer, nothing is sent)

```
rsync -avzn --relative \
  ./trmgrid \
  ./encoders \
  ./train \
  ./zoneviews \
  ./gridtimed \
  ./modesel \
  ./run_one.py \
  ./gridtimed_eval.py \
  ./trmgrid_eval.py \
  ./test_run_one.py \
  ./check_status.py \
  ./submit_trmgrid.sbatch \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='results/' \
  --exclude='results/ckpt/' \
  --exclude='logs/' \
  --exclude='.git/' \
  <ALPINE_HOST>:/projects/amfa5003/RL-for-zones/
```

Read the output. It should list only files under the directories/files
named above — nothing under `results/`, `results/ckpt/`, or `pyudbm/`.
The `--exclude` flags for `results/`/`logs/`/`.git/` are defense-in-depth
(none of those are actual source args here, so they shouldn't appear
regardless) — if you see any of them in the dry-run output, stop and
investigate before proceeding.

### Step 2b — real transfer

Identical command with `n` dropped from `-avzn`:

```
rsync -avz --relative \
  ./trmgrid \
  ./encoders \
  ./train \
  ./zoneviews \
  ./gridtimed \
  ./modesel \
  ./run_one.py \
  ./gridtimed_eval.py \
  ./trmgrid_eval.py \
  ./test_run_one.py \
  ./check_status.py \
  ./submit_trmgrid.sbatch \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='results/' \
  --exclude='results/ckpt/' \
  --exclude='logs/' \
  --exclude='.git/' \
  <ALPINE_HOST>:/projects/amfa5003/RL-for-zones/
```

`-a` (archive mode) already preserves permissions/mtimes on everything,
including `submit_trmgrid.sbatch` — no separate flag needed. Note the
`.sbatch` file doesn't need the executable bit; it's submitted via
`sbatch submit_trmgrid.sbatch`, not run directly.

---

## 3. Verify on Alpine, before submitting

SSH in first:

```
ssh <ALPINE_HOST>
cd /projects/amfa5003/RL-for-zones
```

### 3a — conda env and python

```
module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env
which python3
python3 --version
```

Confirm `which python3` resolves inside the `amfa-custom-env` conda
prefix, not a system Python.

### 3b — pyudbm / trmgrid import check

```
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 -c "import zoneviews; import pyudbm; print(pyudbm.__file__); from pyudbm import Context; import trmgrid; from trmgrid.gym_env import TRMGridTrainEnv; print('OK')"
```

As noted in §1, `submit_trmgrid.sbatch` already sets this
`LD_LIBRARY_PATH` export itself (lines 37-43) — you're just reproducing
it here interactively so a broken pyudbm build fails fast, before you've
burned queue time on 120 tasks. If this import fails with a
`GLIBCXX_3.4.32 not found` or similar, the fix is `./build_pyudbm_alpine.sh`
**run on Alpine** (it's not part of the sync above since pyudbm's native
build shouldn't come from your Mac — copy just that one script over
separately if you need it).

### 3c — test suite

```
python3 -m pytest test_run_one.py -q
```

Specifically confirms (among other things):
- `test_length_is_288` — table is the full 288 entries
- `test_first_168_unchanged` — indices 0-167 are byte-identical to before
  this extension
- `test_168_to_287_are_the_trmgrid_extension`
- `test_trmgrid_run_ids_do_not_collide_with_anything` — the check called
  out in the task: trmgrid's 120 run_ids can't collide with any of the
  other 168 entries' run_ids, checked explicitly in code (not just
  reasoned about), because a prior batch (the shaping experiment) nearly
  resumed and mutated real Alpine checkpoints over exactly this kind of
  collision.

All should pass. If anything in indices 0-167 fails, stop — that means
the synced `run_one.py` diverges from what actually produced the
completed runs in `results/`.

### 3d — dry-run resolve a few array indices

First, a middle, and the last of the new 168-287 range:

```
python3 run_one.py --array-index 168 --dry-run
python3 run_one.py --array-index 228 --dry-run
python3 run_one.py --array-index 287 --dry-run
```

Expected resolved configs (from `run_one.py`'s own construction — verify
the printed JSON matches):
- `168` → `trmgrid` / `G_SMALL` / `mlp_point` / seed `0` (first entry)
- `228` → `trmgrid` / `G_LARGE` / `mlp_point` / seed `0` (first entry of
  the second instance block — a good boundary check)
- `287` → `trmgrid` / `G_LARGE` / `gnn` / seed `9` (last entry)

### 3e — run_id collision check against what's actually on disk

This is the check that matters most given the prior near-miss. Every
existing file in `results/`/`results/ckpt/` starts with `modesel__` or
`gridtimed__`; trmgrid's run_ids all start with `trmgrid__` by
construction, so this should always come back empty — but confirm it,
don't assume it:

```
ls results/ 2>/dev/null | grep '^trmgrid__'
ls results/ckpt/ 2>/dev/null | grep '^trmgrid__'
```

Both commands must print **nothing**. If either prints anything, stop —
do not submit the array — and figure out where that file came from
before proceeding.

---

## 4. Submission and monitoring

### Submit

```
sbatch submit_trmgrid.sbatch
```

The array range (`168-287%20`) and job name (`zone-encoders-trmgrid`)
are already embedded in the script's `#SBATCH` directives — no extra
flags needed for the initial submission.

### Check queue status for this array

```
squeue -u $USER --name=zone-encoders-trmgrid
```

Or, once you have the job ID printed by `sbatch` (`<jobid>`):

```
squeue -j <jobid>
```

For finer per-task detail (state, elapsed time, exit code) including
tasks that have already left the queue:

```
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode -X
```

### Per-run training progress

```
python3 check_status.py --verbose
```

This reads `results/ckpt/` and `results/` directly (read-only, doesn't
touch anything) and reports env steps completed, finished y/n, and any
log-integrity violation per run.

### If a task hits the 72h wall

`submit_trmgrid.sbatch` requests `--time=72:00:00`. If a task times out
before its 300k steps finish, the resume contract is: **resubmitting the
identical script is always safe** — `run_one.py` checks
`results/ckpt/{run_id}.pt` on start and continues from the last
checkpoint automatically (byte-offset jsonl truncation, RNG restore,
atomic checkpoint writes — no flag needed).

Identify the timed-out array indices:

```
sacct -j <jobid> --format=JobID,State -X | grep TIMEOUT
```

Resubmit just those indices (comma-separated, or a sub-range) rather than
the full 120:

```
sbatch --array=<comma-or-range-of-timed-out-indices> submit_trmgrid.sbatch
```

Resubmitting the full `168-287` range instead is also safe (idempotent)
if you'd rather not hand-pick indices — every task that already finished
will just no-op past its checkpoint quickly.

---

## 5. Pull results back

Checkpoints for the *existing* 168 runs total ~66M for all of
`results/ckpt/` (336 files, ~366M total across `results/` including
episode logs) — small enough that pulling everything, checkpoints
included, is likely fine. The 120 new trmgrid runs will add roughly
another 60M of checkpoints if the per-run size holds (~400-450K/`.pt`
file). Decide based on that; the exclude line below is there if you want
manifests/jsonl only.

Pull everything (including checkpoints):

```
rsync -avzn <ALPINE_HOST>:/projects/amfa5003/RL-for-zones/results/ <LOCAL_REPO>/results/
```

(dry run first, drop the `n` in `-avzn` once you've reviewed it — same
pattern as §2.)

Pull everything **except** checkpoints:

```
rsync -avzn --exclude='ckpt/' <ALPINE_HOST>:/projects/amfa5003/RL-for-zones/results/ <LOCAL_REPO>/results/
```
