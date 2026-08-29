"""Run the full 4-encoder x 3-instance x 3-seed PPO experiment matrix.

36 independent runs, one process each (``multiprocessing.Pool``). Each run
is fully self-contained -- separate pyudbm Federations per process, nothing
crosses the process boundary except job specs in and a plain-dict history
out (both fully picklable; no Federation object is ever pickled, satisfying
the same same-process-only constraint ``train.env.SyncVectorEnv`` already
enforces *within* one run). Jobs are scheduled longest-first (LPT
heuristic, using the throughput probe's per-update timings) across a fixed
worker pool so the wall-clock makespan approaches total_work / n_workers
rather than being dominated by however the 36 jobs happen to be listed.

Each worker sets ``torch.set_num_threads(1)`` before doing any torch work:
BLAS backends default to using every available core, which would silently
oversubscribe an 8-worker process pool on an 8-core machine and defeat the
whole point of process-level parallelism.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from multiprocessing import Pool
from typing import Dict, List, Tuple

INSTANCES = ["PJS_04", "PJS_05", "LJS_01"]
ENCODERS = ["mlp", "deepset", "cnn_dbm", "cnn_proj"]
SEEDS = [0, 1, 2]

RESULTS_DIR = "/tmp/slice_results"

# Rough per-update wall-clock (s) from the throughput probe, used only to
# order jobs longest-first for scheduling -- not used for anything
# quantitative in the report (the report uses each run's own measured
# wall-clock).
_APPROX_PER_UPDATE_S = {
    ("PJS_04", "mlp"): 0.41,
    ("PJS_04", "deepset"): 0.66,
    ("PJS_04", "cnn_dbm"): 2.54,
    ("PJS_04", "cnn_proj"): 4.86,
    ("PJS_05", "mlp"): 0.43,
    ("PJS_05", "deepset"): 0.91,
    ("PJS_05", "cnn_dbm"): 3.56,
    ("PJS_05", "cnn_proj"): 8.28,
    ("LJS_01", "mlp"): 0.56,
    ("LJS_01", "deepset"): 1.95,
    ("LJS_01", "cnn_dbm"): 7.73,
    ("LJS_01", "cnn_proj"): 20.45,
}


def _run_one(job: Tuple[str, str, int]) -> Dict:
    import torch

    torch.set_num_threads(1)

    import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
    from train.ppo import PPOConfig, PPOTrainer

    instance, encoder, seed = job
    tag = f"{instance}_{encoder}_seed{seed}"
    out_path = os.path.join(RESULTS_DIR, f"{tag}.json")
    ckpt_path = os.path.join(RESULTS_DIR, f"{tag}.pt")

    t0 = time.perf_counter()
    # Override hooks for smoke/dry-run testing only; unset in the real launch,
    # so real runs always use the spec defaults (300k steps, 50k plateau).
    total_steps = int(os.environ.get("SLICE_TOTAL_STEPS", 300_000))
    plateau = int(os.environ.get("SLICE_PLATEAU_STEPS", 50_000))
    cfg = PPOConfig(seed=seed, total_steps=total_steps, plateau_patience_steps=plateau)
    trainer = PPOTrainer(instance, encoder, cfg)
    history = trainer.train(log_every=10, verbose=True)
    wall = time.perf_counter() - t0

    record = {
        "instance": instance,
        "encoder": encoder,
        "seed": seed,
        "wall_clock_s": wall,
        "total_env_steps": trainer.total_env_steps,
        "n_updates": len(history),
        "param_count_total": sum(p.numel() for p in trainer.head.parameters()),
        "param_count_encoder_only": trainer.head.encoder.num_parameters(),
        "history": [asdict(h) for h in history],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f)
    # Saved separately from the JSON record (state_dict tensors aren't JSON
    # serializable) so the permutation ablation can load the actual trained
    # policy rather than re-deriving anything from the log alone.
    torch.save(trainer.head.state_dict(), ckpt_path)

    print(f"DONE {instance}/{encoder}/seed{seed}: wall={wall:.1f}s updates={len(history)} steps={trainer.total_env_steps}")
    return record


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs: List[Tuple[str, str, int]] = [
        (instance, encoder, seed) for instance in INSTANCES for encoder in ENCODERS for seed in SEEDS
    ]
    jobs.sort(key=lambda j: -_APPROX_PER_UPDATE_S[(j[0], j[1])])

    print(f"Launching {len(jobs)} jobs across a pool of {min(8, os.cpu_count() or 8)} workers...")
    n_workers = min(8, os.cpu_count() or 8)

    t0 = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        for _ in pool.imap_unordered(_run_one, jobs):
            pass
    wall = time.perf_counter() - t0
    print(f"ALL DONE. Total wall-clock: {wall / 3600:.2f} hours")


if __name__ == "__main__":
    main()
