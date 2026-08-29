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
SEEDS = (0, 1, 2)

_ENCODER_KEY_MAP = {"mlp_point": "mlp", "deepset_h": "deepset", "cnn_dbm": "cnn_dbm", "cnn_proj": "cnn_proj"}

TOTAL_STEPS = 300_000
PLATEAU_PATIENCE_STEPS = 50_000
CKPT_EVERY_STEPS = 10_000
NUM_ENVS = 8


@dataclass(frozen=True)
class RunSpec:
    index: int
    domain: str  # "modesel" | "gridtimed"
    instance: str  # e.g. "M3"
    encoder: str  # e.g. "mlp_point"
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.domain}__{self.instance}__{self.encoder}__seed{self.seed}"


def _build_target_runs() -> List[RunSpec]:
    instances: List[Tuple[str, str]] = [("modesel", i) for i in MODESEL_INSTANCES] + [
        ("gridtimed", i) for i in GRIDTIMED_INSTANCES
    ]
    runs = []
    idx = 0
    for domain, instance in instances:
        for encoder in ENCODERS:
            for seed in SEEDS:
                runs.append(RunSpec(idx, domain, instance, encoder, seed))
                idx += 1
    return runs


#: THE authoritative array-index -> config mapping. len == 48
#: (2 instances/domain * 2 domains) * 4 encoders * 3 seeds.
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
    raise ValueError(f"unknown domain {domain!r}")


def build_trainer(
    spec: RunSpec, num_envs: int = NUM_ENVS,
    total_steps: int = TOTAL_STEPS, plateau_patience_steps: int = PLATEAU_PATIENCE_STEPS,
) -> PPOTrainer:
    inst, ctor = _build_instance(spec.domain, spec.instance)
    cfg = PPOConfig(
        total_steps=total_steps,
        plateau_patience_steps=plateau_patience_steps,
        seed=spec.seed,
        num_envs=num_envs,
    )
    vec_env = SyncVectorEnv(
        [(lambda s=cfg.seed + i, inst=inst, ctor=ctor: ctor(inst, seed=s)) for i in range(cfg.num_envs)]
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
    )

    early_stop_step: Optional[int] = None
    start_time = time.time()

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        trainer.load_state_dict(ckpt["trainer"])
        _restore_rng_state(ckpt["rng"])
        _truncate_to_offset(jsonl_path, ckpt["jsonl_offset"])
        _truncate_to_offset(episodes_path, ckpt["episodes_jsonl_offset"])
        early_stop_step = ckpt.get("early_stop_step")
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
        steps_at_last_ckpt = trainer.total_env_steps
        while trainer.total_env_steps < trainer.cfg.total_steps:
            log, episodes, should_stop = trainer._do_one_update()

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
                    "num_envs": args.num_envs,
                },
                indent=2,
            )
        )
        return

    run_training(
        spec, num_envs=args.num_envs, ckpt_every_steps=args.ckpt_every_steps,
        total_steps=args.total_steps, plateau_patience_steps=args.plateau_patience_steps,
    )


if __name__ == "__main__":
    main()
