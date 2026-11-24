"""SLURM array-task entry point for the zone-vs-point encoder sample-
efficiency sweep (/tmp/ENCODERS_SETUP.md). One call = one (domain,
instance, encoder, seed) training run.

``TARGET_RUNS`` is the single AUTHORITATIVE array-index -> config mapping
(same contract requested for envs/b3_extend_training.py's TARGET_RUNS,
which does not exist in this repo -- see test_run_one.py's docstring).
Index resolution is a plain list lookup: SLURM_ARRAY_TASK_ID (or
--array-index) selects ``TARGET_RUNS[i]``, nothing computed from it.

IDEMPOTENT / RESUMABLE: resubmitting the identical command continues from
the last checkpoint under results/ckpt/ -- no flag needed. See
_run_training's docstring for the exact resume contract (byte-offset jsonl
truncation, RNG restore, atomic checkpoint writes).

Instance construction: the environment LAYOUT is fixed per (domain,
instance) using generator seed 0 (matching modesel/tiers.py's and
gridtimed/tiers.py's own calibration seed, so the param-count check done
against these exact instances in /tmp/ENCODERS_SETUP.md stays valid). The
RunSpec's ``seed`` field is the PPO TRAINING seed only (weight init,
rollout/minibatch randomness, per-env reset-seed offsets) -- this is what
varies across the 3 seeds per (instance, encoder), not the environment
itself; holding the environment fixed and varying only the training seed
is what makes the 3-seed spread a measure of training variance rather
than environment-difficulty variance.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from encoders.deepset_v import reset_vrep_fallback_stats, vrep_fallback_stats
from gridtimed_eval import evaluate_gridtimed
from gridtimed.generator import make_instance as grid_make_instance
from gridtimed.gym_env import GridTimedTrainEnv
from gridtimed.tiers import TIERS as GRIDTIMED_TIERS
from modesel.generator import make_instance as modesel_make_instance
from modesel.gym_env import ModeSelGymEnv
from modesel.tiers import COST_PENALTY as MODESEL_COST_PENALTY
from modesel.tiers import FAST_SLOW_RATIO as MODESEL_FAST_SLOW_RATIO
from modesel.tiers import TIER_SPECS as MODESEL_TIER_SPECS
from train.ppo import PPOConfig, PPOTrainer
from train.vec_env import SyncVectorEnv
from trmgrid.generator import make_instance as trmgrid_make_instance
from trmgrid.gym_env import TRMGridTrainEnv
from trmgrid_eval import evaluate_trmgrid

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
CKPT_DIR = RESULTS_DIR / "ckpt"

#: Generator seed for INSTANCE construction (layout), held fixed across all
#: 3 training seeds -- see module docstring.
INSTANCE_SEED = 0

MODESEL_INSTANCES = ("M3", "M4")
GRIDTIMED_INSTANCES = ("T2", "T3")
#: Task-facing encoder names (module basenames under encoders/); mapped to
#: train.ppo.build_head's short internal names in _build_env_and_head.
ENCODERS = ("mlp_point", "deepset_h", "cnn_dbm", "cnn_proj")
#: Added for the v_rep/GNN extension (/tmp/VREP_FEASIBILITY.md) -- kept as
#: a SEPARATE tuple, not merged into ENCODERS, specifically so
#: _build_target_runs's original 4-encoder loop (indices 0-47) is
#: untouched by construction, not just by coincidence of ordering.
NEW_ENCODERS = ("deepset_v", "gnn")
SEEDS = (0, 1, 2)
#: Added for the seed-extension on gridtimed (/tmp/SWEEP_FINAL.md Task C)
#: -- gridtimed/T2 and T3 showed per-seed bimodal outcomes for deepset_v
#: and gnn at 3 seeds; extended to 10 total seeds (0-2 existing + 3-9 new)
#: on gridtimed ONLY, for ALL SIX encoders (not just the two bimodal ones)
#: so the comparison stays on equal footing. modesel is NOT extended --
#: its variance was already tight at 3 seeds (std <= 0.056 everywhere),
#: so the extra compute is spent where the headline result actually is.
EXTRA_SEEDS = (3, 4, 5, 6, 7, 8, 9)

#: trmgrid extension (/tmp/TRMGRID_SWEEP.md) -- two tiers picked from the
#: Step 1 size sweep (alpha=0.25, width=8 fixed throughout, the only
#: setting that cleared v2's stopping rule):
#:   G_SMALL = grid_n=4, k=3 -- the exact (untruncated) tier, verified
#:     across 3 layout seeds at state_limit=3,000,000 (584,812 / 2,017,419
#:     / 1,346,833 usable states respectively, ALL untruncated). Diagnostic
#:     values vary noticeably by seed (soundness 0.586-0.703, conflict
#:     0.297-0.414) -- TRMGRID_INSTANCE_SEED=2 is used rather than the
#:     shared INSTANCE_SEED=0 because it cleanly clears both of Step 1's
#:     bars (soundness 0.5862 <= 0.65, conflict 0.4138 >= 0.25, not
#:     degenerate: random-policy success 3.6%), where seed 0 is borderline
#:     (soundness 0.7034, just above the 0.65 ceiling).
#:   G_LARGE = grid_n=5, k=3 -- NOT the originally-placeholdered
#:     grid_n=6,k=4. The exact DP is impractical to run untruncated beyond
#:     the 4x4/k=3 tier within tractable wall-clock (grid_n=4,k=3 alone
#:     ranged 44s-613s across seeds; larger tiers explode further), so
#:     G_LARGE's diagnostics are TRUNCATED BY DESIGN at
#:     state_limit=400,000, per the task's own allowance for a "truncated
#:     large-instance diagnostic" distinct from the exact small one.
#:     grid_n=5,k=3 beat the other 3 candidate tiers at this truncation
#:     budget: 6x6/k=3 is DEGENERATE (0% random-policy success -- rejected
#:     outright per Step 1's own miscalibration rule), 6x6/k=4 and 8x8/k=4
#:     are technically non-degenerate but only barely (0.2% random
#:     success, vs. 5x5/k=3's 0.8-2.2% across 3 seeds). 5x5/k=3 also
#:     cleanly clears both bars on all 3 seeds tested (soundness
#:     0.218-0.227, conflict 0.773-0.782) and is the fastest of the four
#:     candidates. Uses TRMGRID_INSTANCE_SEED (same seed as G_SMALL, for
#:     consistency -- 5x5/k=3/seed=2 was not individually re-verified but
#:     seeds 0/1/2 were all consistent to within ~1% on this tier).
#: Both use w=0.25 (the real operating-point duration jitter used
#: throughout Measurement 2/3).
TRMGRID_ALPHA = 0.25
TRMGRID_WIDTH = 8
TRMGRID_W = 0.25
#: Decoupled from the shared INSTANCE_SEED=0 (used by modesel/gridtimed)
#: because Step 1's own calibration data showed seed-to-seed variation
#: large enough to flip the soundness bar (see justification above) --
#: nothing about modesel/gridtimed's instance generation is affected.
TRMGRID_INSTANCE_SEED = 2
TRMGRID_TIERS: Dict[str, Dict[str, int]] = {
    "G_SMALL": dict(grid_n=4, k=3),
    "G_LARGE": dict(grid_n=5, k=3),
}
TRMGRID_INSTANCES = ("G_SMALL", "G_LARGE")
#: 10 seeds from the start (not a later extension) -- trmgrid is a new
#: domain with no prior 3-seed baseline to extend from, and gridtimed's
#: own bimodality finding (SWEEP_FINAL.md) is exactly why a new domain
#: sweep starts at the seed count that already proved necessary there.
TRMGRID_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

_ENCODER_KEY_MAP = {
    "mlp_point": "mlp", "deepset_h": "deepset", "cnn_dbm": "cnn_dbm", "cnn_proj": "cnn_proj",
    "deepset_v": "deepset_v", "gnn": "gnn",
}

TOTAL_STEPS = 300_000
PLATEAU_PATIENCE_STEPS = 50_000
CKPT_EVERY_STEPS = 10_000
NUM_ENVS = 8
#: Dual-metric eval cadence (/tmp/PART1_FINAL.md item 2), gridtimed only.
#: Benchmarked directly against real checkpoints for all six encoders: the
#: single most expensive eval call (cnn_proj/T3, 0.544s) against the
#: cheapest per-update wall-clock (mlp_point, 0.582s/update mean on
#: Alpine) gives 1.72% overhead at K=5 -- the tightest of all six, every
#: other encoder is well under that. K=5 -> ~29 eval points across a full
#: 147-update run.
EVAL_EVERY_K_UPDATES = 5
#: Shaping experiment (/tmp/PART1_FINAL.md item 3) -- the deferred-credit-
#: assignment mechanism test. gridtimed.instance.REWARD_SUCCESS is 100.0;
#: "roughly 10% of the success bonus" -> 10.0.
SWITCH_SHAPING_REWARD = 10.0
SHAPING_ENCODERS = ("mlp_point", "deepset_h")  # mlp_point is the subject, deepset_h the control
SHAPING_INSTANCE = ("gridtimed", "T2")


@dataclass(frozen=True)
class RunSpec:
    index: int
    domain: str  # "modesel" | "gridtimed"
    instance: str  # e.g. "M3"
    encoder: str  # e.g. "mlp_point"
    seed: int
    #: Shaping experiment (/tmp/PART1_FINAL.md item 3), gridtimed only --
    #: None (default) for every pre-existing config (indices 0-155): NOT
    #: part of the shaping experiment, no run_id suffix, unaffected by
    #: this field's mere existence. True/False for the 12 shaping-
    #: experiment configs (156-167): shaped vs. unshaped CONTROL. Both
    #: get an explicit run_id suffix -- critically, the unshaped control
    #: is NOT the same run_id as the pre-existing gridtimed/T2/mlp_point
    #: (or deepset_h) runs at indices 0-47, even though same (domain,
    #: instance, encoder, seed): those went through item 1's
    #: early-stop-then-resume path, this is a fresh, uninterrupted 300k
    #: run -- reusing the old run_id would silently resume (and mutate)
    #: real completed Alpine data under a different experimental
    #: condition than what actually produced it.
    shaped: Optional[bool] = None

    @property
    def run_id(self) -> str:
        if self.shaped is None:
            suffix = ""
        elif self.shaped:
            suffix = "__shapeexp_shaped"
        else:
            suffix = "__shapeexp_unshaped"
        return f"{self.domain}__{self.instance}__{self.encoder}__seed{self.seed}{suffix}"


def _build_target_runs() -> List[RunSpec]:
    """Indices 0-47: original 4 encoders, unchanged construction (byte-
    identical to before the v_rep/GNN extension -- verified by
    test_run_one.py). Indices 48-71: NEW_ENCODERS appended afterward, same
    instance/seed structure, continuing the index counter -- never
    interleaved with the original loop, so 0-47 can't shift regardless of
    what NEW_ENCODERS contains. Indices 72-155: gridtimed-only seed
    extension (EXTRA_SEEDS) across all six encoders, appended as a THIRD,
    separate loop -- same reasoning, 0-71 can't shift regardless of what
    this loop contains. Indices 156-167: the shaping experiment
    (/tmp/PART1_FINAL.md item 3), a FOURTH separate loop -- gridtimed/T2
    only, SHAPING_ENCODERS (mlp_point + deepset_h), SEEDS (0-2, not
    EXTRA_SEEDS -- this is a fresh 3-seed comparison, not part of the
    variance-characterization extension), shaped in (False, True)."""
    instances: List[Tuple[str, str]] = [("modesel", i) for i in MODESEL_INSTANCES] + [
        ("gridtimed", i) for i in GRIDTIMED_INSTANCES
    ]
    gridtimed_instances: List[Tuple[str, str]] = [("gridtimed", i) for i in GRIDTIMED_INSTANCES]
    runs = []
    idx = 0
    for domain, instance in instances:
        for encoder in ENCODERS:
            for seed in SEEDS:
                runs.append(RunSpec(idx, domain, instance, encoder, seed))
                idx += 1
    for domain, instance in instances:
        for encoder in NEW_ENCODERS:
            for seed in SEEDS:
                runs.append(RunSpec(idx, domain, instance, encoder, seed))
                idx += 1
    for domain, instance in gridtimed_instances:
        for encoder in ENCODERS + NEW_ENCODERS:
            for seed in EXTRA_SEEDS:
                runs.append(RunSpec(idx, domain, instance, encoder, seed))
                idx += 1
    shaping_domain, shaping_instance = SHAPING_INSTANCE
    for encoder in SHAPING_ENCODERS:
        for seed in SEEDS:
            for shaped in (False, True):
                runs.append(RunSpec(idx, shaping_domain, shaping_instance, encoder, seed, shaped=shaped))
                idx += 1
    # trmgrid extension (/tmp/TRMGRID_SWEEP.md), a FIFTH separate loop --
    # a brand-new domain string ("trmgrid"), so no run_id here can collide
    # with anything in indices 0-167 by construction (every existing
    # run_id starts with "modesel__" or "gridtimed__"); still checked
    # explicitly by test_run_one.py, not just reasoned about, per the
    # shaping experiment's own near-miss.
    for instance in TRMGRID_INSTANCES:
        for encoder in ENCODERS + NEW_ENCODERS:
            for seed in TRMGRID_SEEDS:
                runs.append(RunSpec(idx, "trmgrid", instance, encoder, seed))
                idx += 1
    return runs


#: THE authoritative array-index -> config mapping. len == 288:
#: indices 0-47 = (2 instances/domain * 2 domains) * 4 original encoders *
#: 3 seeds (unchanged from before the v_rep/GNN extension); indices 48-71 =
#: (2 instances/domain * 2 domains) * 2 NEW_ENCODERS * 3 seeds (unchanged
#: from before this seed extension); indices 72-155 = 2 gridtimed
#: instances * 6 encoders (ENCODERS + NEW_ENCODERS) * 7 EXTRA_SEEDS
#: (unchanged from before this shaping extension); indices 156-167 = the
#: shaping experiment, 2 SHAPING_ENCODERS * 3 SEEDS * 2 shaped variants
#: (unchanged from before this trmgrid extension); indices 168-287 = the
#: trmgrid extension, 2 TRMGRID_INSTANCES * 6 encoders * 10 TRMGRID_SEEDS,
#: appended. modesel/gridtimed are untouched by this extension.
TARGET_RUNS: List[RunSpec] = _build_target_runs()


def resolve_spec(array_index: int) -> RunSpec:
    if not (0 <= array_index < len(TARGET_RUNS)):
        raise ValueError(f"array_index {array_index} out of range [0, {len(TARGET_RUNS)})")
    return TARGET_RUNS[array_index]


# ----------------------------------------------------------------------
def _build_instance(domain: str, instance: str):
    if domain == "modesel":
        spec = MODESEL_TIER_SPECS[instance]
        inst = modesel_make_instance(
            n_jobs=spec["n_jobs"], n_tasks=spec["n_tasks"], n_machines=spec["n_machines"],
            fast_slow_ratio=MODESEL_FAST_SLOW_RATIO, cost_penalty=MODESEL_COST_PENALTY,
            tightness=spec["tightness"], w=0.25, seed=INSTANCE_SEED, label=instance,
        )
        return inst, ModeSelGymEnv
    if domain == "gridtimed":
        cfg = next(t for t in GRIDTIMED_TIERS if t["name"] == instance)
        inst, _info = grid_make_instance(
            grid_n=cfg["grid_n"], n_switches=cfg["n_switches"], n_gates=cfg["n_gates"],
            window_width=cfg["window_width"], D=cfg["D"], w=0.25, seed=INSTANCE_SEED, label=instance,
        )
        return inst, GridTimedTrainEnv
    if domain == "trmgrid":
        cfg = TRMGRID_TIERS[instance]
        inst, _info = trmgrid_make_instance(
            grid_n=cfg["grid_n"], k=cfg["k"], alpha=TRMGRID_ALPHA, width=TRMGRID_WIDTH,
            w=TRMGRID_W, seed=TRMGRID_INSTANCE_SEED, label=instance,
        )
        return inst, TRMGridTrainEnv
    raise ValueError(f"unknown domain {domain!r}")


def build_trainer(
    spec: RunSpec, num_envs: int = NUM_ENVS,
    total_steps: int = TOTAL_STEPS, plateau_patience_steps: int = PLATEAU_PATIENCE_STEPS,
    plateau_enabled: bool = False,
) -> PPOTrainer:
    inst, ctor = _build_instance(spec.domain, spec.instance)
    cfg = PPOConfig(
        total_steps=total_steps,
        plateau_patience_steps=plateau_patience_steps,
        plateau_enabled=plateau_enabled,
        seed=spec.seed,
        num_envs=num_envs,
    )
    # Shaping experiment (item 3): switch_shaping_reward is a
    # gridtimed-only GridTimedTrainEnv kwarg -- ModeSelGymEnv doesn't
    # accept it, so it's only ever passed for spec.domain == "gridtimed".
    # spec.shaped is False for every config outside the shaping experiment
    # (default field value), so this is 0.0 -- byte-identical env
    # construction -- for every run except the 12 shaped ones.
    env_kwargs: Dict[str, Any] = {}
    if spec.domain == "gridtimed":
        env_kwargs["switch_shaping_reward"] = SWITCH_SHAPING_REWARD if spec.shaped else 0.0
    vec_env = SyncVectorEnv(
        [
            (lambda s=cfg.seed + i, inst=inst, ctor=ctor, kw=env_kwargs: ctor(inst, seed=s, **kw))
            for i in range(cfg.num_envs)
        ]
    )
    encoder_key = _ENCODER_KEY_MAP[spec.encoder]
    return PPOTrainer(None, encoder_key, cfg, vec_env=vec_env)


# ----------------------------------------------------------------------
# Checkpoint / RNG / jsonl helpers.
def _capture_rng_state() -> Dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _atomic_torch_save(state: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _atomic_json_dump(obj: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _truncate_to_offset(path: Path, offset: int) -> None:
    if not path.exists():
        return
    with open(path, "r+b") as f:
        f.truncate(offset)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    line = json.dumps(record) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def verify_jsonl_monotonic(jsonl_path: Path) -> List[str]:
    """Returns a list of violation descriptions (empty == clean). Checks
    BOTH strictly increasing AND no gaps (update N+1 must follow N exactly)
    -- the correctness property required after any resume."""
    violations: List[str] = []
    if not jsonl_path.exists():
        return violations
    last_update: Optional[int] = None
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            u = rec["update"]
            if last_update is not None:
                if u <= last_update:
                    violations.append(f"line {lineno}: update={u} not strictly greater than previous={last_update}")
                elif u != last_update + 1:
                    violations.append(f"line {lineno}: gap, expected update={last_update + 1}, got update={u}")
            last_update = u
    return violations


def _save_checkpoint(
    ckpt_path: Path, trainer: PPOTrainer, jsonl_path: Path, episodes_path: Path,
    early_stop_step: Optional[int], start_time: float,
) -> None:
    state = {
        "trainer": trainer.state_dict(),
        "rng": _capture_rng_state(),
        "jsonl_offset": jsonl_path.stat().st_size if jsonl_path.exists() else 0,
        "episodes_jsonl_offset": episodes_path.stat().st_size if episodes_path.exists() else 0,
        "early_stop_step": early_stop_step,
        "start_time": start_time,
    }
    _atomic_torch_save(state, ckpt_path)


def _write_manifest(
    manifest_path: Path, spec: RunSpec, trainer: PPOTrainer, start_time: float, end_time: float,
    early_stop_step: Optional[int], jsonl_path: Path, episodes_path: Path,
) -> None:
    n_params = sum(p.numel() for p in trainer.head.parameters())
    manifest = {
        "domain": spec.domain,
        "instance": spec.instance,
        "encoder": spec.encoder,
        "seed": spec.seed,
        "array_index": spec.index,
        "hyperparameters": dataclasses.asdict(trainer.cfg),
        "param_count": n_params,
        "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "start_time": start_time,
        "end_time": end_time,
        "total_env_steps": trainer.total_env_steps,
        "early_stop_step": early_stop_step,
        "jsonl_path": str(jsonl_path),
        "episodes_jsonl_path": str(episodes_path),
    }
    _atomic_json_dump(manifest, manifest_path)


# ----------------------------------------------------------------------
def _paths_for(spec: RunSpec) -> Dict[str, Path]:
    return {
        "ckpt": CKPT_DIR / f"{spec.run_id}.pt",
        "jsonl": RESULTS_DIR / f"{spec.run_id}.jsonl",
        "episodes": RESULTS_DIR / f"{spec.run_id}__episodes.jsonl",
        # A single shared results/manifest.json written concurrently by up
        # to 12 parallel array tasks (the --array=...%12 throttle) would
        # corrupt under concurrent writes with no locking -- one manifest
        # shard per run instead, same schema, trivially globbable by
        # analyze.py/check_status.py. Reported as a deliberate deviation
        # from the literal "results/manifest.json" path, not a silent one.
        "manifest": RESULTS_DIR / f"{spec.run_id}__manifest.json",
    }


def run_training(
    spec: RunSpec, num_envs: int = NUM_ENVS, ckpt_every_steps: int = CKPT_EVERY_STEPS,
    total_steps: int = TOTAL_STEPS, plateau_patience_steps: int = PLATEAU_PATIENCE_STEPS,
    plateau_enabled: bool = False,
) -> None:
    """IDEMPOTENT/RESUMABLE training loop for one (domain, instance,
    encoder, seed) run.

    Resume contract (checked by the local smoke test, kill-mid-run):
    if ``results/ckpt/{run_id}.pt`` exists, it is loaded and training
    continues from its ``trainer.state_dict()`` (weights, optimizer,
    env_steps, update_idx, plateau tracker) and its captured RNG state
    (python/numpy/torch) -- then BOTH jsonl files are truncated to the
    byte offsets recorded in that same checkpoint before any further
    writes, discarding any records appended after the last successful
    checkpoint but before this restart (an "orphan tail" -- the crash may
    have happened mid-update, after some jsonl writes but before the next
    checkpoint captured them). This keeps `update` strictly increasing
    with no gaps across a resume: the checkpoint and the truncated jsonl
    always agree on the last completed update.

    Checkpoint writes are atomic (write to `{path}.tmp`, fsync, rename)
    everywhere in this module -- a crash mid-write can never leave a
    corrupt checkpoint the next resume would load.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _paths_for(spec)
    ckpt_path, jsonl_path, episodes_path, manifest_path = (
        paths["ckpt"], paths["jsonl"], paths["episodes"], paths["manifest"],
    )

    trainer = build_trainer(
        spec, num_envs=num_envs, total_steps=total_steps, plateau_patience_steps=plateau_patience_steps,
        plateau_enabled=plateau_enabled,
    )

    early_stop_step: Optional[int] = None
    start_time = time.time()

    if ckpt_path.exists():
        # weights_only=False: PyTorch >=2.6 defaults torch.load to a strict
        # unpickler that rejects our RNG-state payload (numpy internals
        # among others). Checkpoints here are always self-written (never
        # loaded from an untrusted/third-party source), so the historical
        # (pre-2.6) unrestricted-unpickle behavior is safe to opt back into
        # explicitly rather than relying on a version-dependent default.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        trainer.load_state_dict(ckpt["trainer"])
        _restore_rng_state(ckpt["rng"])
        _truncate_to_offset(jsonl_path, ckpt["jsonl_offset"])
        _truncate_to_offset(episodes_path, ckpt["episodes_jsonl_offset"])
        early_stop_step = ckpt.get("early_stop_step")
        if not plateau_enabled and early_stop_step is not None:
            # The checkpoint may carry an early_stop_step recorded back
            # when plateau detection was enabled (the uniform-budget
            # extension, /tmp/PART1_FINAL.md item 1). Disabled here means
            # this run should continue toward total_steps regardless of
            # that stale marker -- otherwise the already_done check below
            # would treat a previously-plateaued run as finished and never
            # resume it.
            print(
                f"[{spec.run_id}] plateau disabled -- ignoring stale early_stop_step={early_stop_step} from checkpoint, continuing to {trainer.cfg.total_steps}",
                flush=True,
            )
            early_stop_step = None
        start_time = ckpt.get("start_time", start_time)
        print(
            f"[{spec.run_id}] resumed from checkpoint: env_steps={trainer.total_env_steps} "
            f"update={trainer.update_idx} early_stop_step={early_stop_step}",
            flush=True,
        )

    already_done = early_stop_step is not None or trainer.total_env_steps >= trainer.cfg.total_steps
    if already_done:
        print(f"[{spec.run_id}] already finished -- nothing to do", flush=True)
    else:
        # Dual-metric eval (/tmp/PART1_FINAL.md item 2, extended to trmgrid
        # in /tmp/TRMGRID_SWEEP.md): gridtimed and trmgrid only --
        # greedy_cycling_frac is defined in terms of grid position revisits
        # (SWEEP_FINAL.md B4), which has no modesel analogue. instance_obj
        # is rebuilt here (not threaded through from build_trainer) --
        # cheap and deterministic (INSTANCE_SEED fixed), not worth
        # plumbing through build_trainer's signature for one caller.
        eval_instance_obj = eval_ctor = eval_fn = None
        if spec.domain == "gridtimed":
            eval_instance_obj, eval_ctor = _build_instance(spec.domain, spec.instance)
            eval_fn = evaluate_gridtimed
        elif spec.domain == "trmgrid":
            eval_instance_obj, eval_ctor = _build_instance(spec.domain, spec.instance)
            eval_fn = evaluate_trmgrid

        steps_at_last_ckpt = trainer.total_env_steps
        while trainer.total_env_steps < trainer.cfg.total_steps:
            reset_vrep_fallback_stats()
            log, episodes, should_stop = trainer._do_one_update()
            vrep_stats = vrep_fallback_stats()

            greedy_success_rate = greedy_cycling_frac = stochastic_success_rate = None
            if eval_instance_obj is not None and log.update % EVAL_EVERY_K_UPDATES == 0:
                greedy_success_rate, greedy_cycling_frac, stochastic_success_rate = eval_fn(
                    trainer.head, eval_instance_obj, eval_ctor,
                )
            # Only meaningful for deepset_v (the only encoder that calls
            # cached_v_rep at all) -- null for the other five, matching
            # domain_metric's "null where undefined" convention, not a
            # spurious 0.0 that would misleadingly claim "zero fallbacks"
            # for encoders that never touch v_rep in the first place.
            vrep_fallback_frac = vrep_stats["fallback_frac"] if spec.encoder == "deepset_v" and vrep_stats["total_count"] > 0 else None

            record = {
                "update": log.update,
                "env_steps": log.env_steps,
                "wall_s": log.wall_clock_s,
                "ep_return_mean": log.mean_return,
                "ep_return_std": log.std_return,
                "ep_len_mean": log.mean_ep_length,
                "n_episodes_completed": log.n_episodes,
                "policy_entropy": log.entropy,
                "value_loss": log.value_loss,
                "policy_loss": log.policy_loss,
                "explained_variance": log.explained_variance,
                "approx_kl": log.approx_kl,
                "clip_frac": log.clip_frac,
                "grad_norm": log.grad_norm,
                "encoder_forward_s": log.encoder_forward_s,
                "raster_s": log.raster_s,
                "env_step_s": log.env_step_s,
                "domain_metric": None if np.isnan(log.mean_makespan) else log.mean_makespan,
                "vrep_fallback_frac": vrep_fallback_frac,
                # Dual-metric eval (item 2): null except every
                # EVAL_EVERY_K_UPDATES-th update on gridtimed runs -- same
                # "null where undefined" convention as vrep_fallback_frac,
                # and analyze.py must tolerate these being entirely absent
                # on older jsonl files written before this field existed.
                "greedy_success_rate": greedy_success_rate,
                "greedy_cycling_frac": greedy_cycling_frac,
                "stochastic_success_rate": stochastic_success_rate,
            }
            _append_jsonl(jsonl_path, record)

            for ep in episodes:
                _append_jsonl(
                    episodes_path,
                    {
                        "env_steps": log.env_steps,
                        "return": ep["return"],
                        "length": ep["length"],
                        "success": ep["success"],
                    },
                )

            if should_stop:
                early_stop_step = trainer.total_env_steps

            hit_ckpt_interval = trainer.total_env_steps - steps_at_last_ckpt >= ckpt_every_steps
            finished = trainer.total_env_steps >= trainer.cfg.total_steps
            if hit_ckpt_interval or should_stop or finished:
                _save_checkpoint(ckpt_path, trainer, jsonl_path, episodes_path, early_stop_step, start_time)
                steps_at_last_ckpt = trainer.total_env_steps

            if should_stop:
                print(f"[{spec.run_id}] early stop at env_steps={trainer.total_env_steps}", flush=True)
                break

    end_time = time.time()
    _write_manifest(manifest_path, spec, trainer, start_time, end_time, early_stop_step, jsonl_path, episodes_path)

    violations = verify_jsonl_monotonic(jsonl_path)
    if violations:
        print(f"[{spec.run_id}] LOG-INTEGRITY VIOLATIONS in {jsonl_path}:", flush=True)
        for v in violations:
            print(f"  {v}", flush=True)
    else:
        print(f"[{spec.run_id}] log integrity OK ({jsonl_path})", flush=True)


# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-index", type=int, default=None, help="overrides $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--dry-run", action="store_true", help="print resolved config and exit")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--ckpt-every-steps", type=int, default=CKPT_EVERY_STEPS)
    parser.add_argument(
        "--total-steps", type=int, default=TOTAL_STEPS,
        help="override, for local smoke-testing only -- SLURM runs should use the default",
    )
    parser.add_argument("--plateau-patience-steps", type=int, default=PLATEAU_PATIENCE_STEPS)
    parser.add_argument(
        "--enable-early-stop", action="store_true",
        help="opt IN to plateau early stopping (default OFF -- the uniform-budget requirement, "
             "/tmp/PART1_FINAL.md item 1: every run trains to exactly --total-steps unless this is passed)",
    )
    args = parser.parse_args(argv)

    array_index = args.array_index
    if array_index is None:
        env_val = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_val is None:
            parser.error("--array-index or $SLURM_ARRAY_TASK_ID is required")
        array_index = int(env_val)

    spec = resolve_spec(array_index)
    paths = _paths_for(spec)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "array_index": spec.index,
                    "domain": spec.domain,
                    "instance": spec.instance,
                    "encoder": spec.encoder,
                    "seed": spec.seed,
                    "run_id": spec.run_id,
                    "ckpt_path": str(paths["ckpt"]),
                    "jsonl_path": str(paths["jsonl"]),
                    "episodes_jsonl_path": str(paths["episodes"]),
                    "manifest_path": str(paths["manifest"]),
                    "total_steps": args.total_steps,
                    "plateau_patience_steps": args.plateau_patience_steps,
                    "plateau_enabled": args.enable_early_stop,
                    "num_envs": args.num_envs,
                },
                indent=2,
            )
        )
        return

    run_training(
        spec, num_envs=args.num_envs, ckpt_every_steps=args.ckpt_every_steps,
        total_steps=args.total_steps, plateau_patience_steps=args.plateau_patience_steps,
        plateau_enabled=args.enable_early_stop,
    )


if __name__ == "__main__":
    main()
