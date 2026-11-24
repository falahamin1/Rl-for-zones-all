"""Scan results/ckpt/ and results/ and print per-config status: env_steps
done, finished y/n, early-stop step, last update timestamp, any
log-integrity violation. Read-only -- does not touch checkpoints or logs.

Usage: python3 check_status.py [--verbose]
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from run_one import CKPT_DIR, RESULTS_DIR, TARGET_RUNS, TOTAL_STEPS, verify_jsonl_monotonic


def _load_ckpt_summary(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    if not ckpt_path.exists():
        return None
    try:
        # weights_only=False: see the matching comment in run_one.py --
        # these checkpoints are always self-written, never from an
        # untrusted source, so PyTorch >=2.6's stricter default unpickler
        # (which rejects our RNG-state payload) isn't needed here.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": f"failed to load checkpoint: {e}"}
    trainer_state = ckpt["trainer"]
    return {
        "env_steps": trainer_state["total_env_steps"],
        "update_idx": trainer_state["update_idx"],
        "early_stop_step": ckpt.get("early_stop_step"),
        "mtime": ckpt_path.stat().st_mtime,
    }


def _last_update_timestamp(jsonl_path: Path) -> Optional[float]:
    if not jsonl_path.exists():
        return None
    return jsonl_path.stat().st_mtime


def collect_status() -> List[Dict[str, Any]]:
    rows = []
    for spec in TARGET_RUNS:
        ckpt_path = CKPT_DIR / f"{spec.run_id}.pt"
        jsonl_path = RESULTS_DIR / f"{spec.run_id}.jsonl"
        episodes_path = RESULTS_DIR / f"{spec.run_id}__episodes.jsonl"
        manifest_path = RESULTS_DIR / f"{spec.run_id}__manifest.json"

        ckpt_summary = _load_ckpt_summary(ckpt_path)
        env_steps = ckpt_summary["env_steps"] if ckpt_summary and "env_steps" in ckpt_summary else 0
        early_stop_step = ckpt_summary.get("early_stop_step") if ckpt_summary else None
        finished = (early_stop_step is not None) or (env_steps >= TOTAL_STEPS)
        last_update_ts = _last_update_timestamp(jsonl_path)

        violations = verify_jsonl_monotonic(jsonl_path)

        rows.append(
            {
                "run_id": spec.run_id,
                "domain": spec.domain,
                "instance": spec.instance,
                "encoder": spec.encoder,
                "seed": spec.seed,
                "started": ckpt_path.exists() or jsonl_path.exists(),
                "env_steps": env_steps,
                "total_steps": TOTAL_STEPS,
                "pct_done": round(100.0 * env_steps / TOTAL_STEPS, 1),
                "finished": finished,
                "early_stop_step": early_stop_step,
                "last_update_ts": last_update_ts,
                "last_update_iso": (
                    datetime.datetime.fromtimestamp(last_update_ts).isoformat() if last_update_ts else None
                ),
                "log_integrity_violations": violations,
                "manifest_exists": manifest_path.exists(),
                "ckpt_error": ckpt_summary.get("error") if ckpt_summary else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print full per-run detail, not just the summary table")
    parser.add_argument("--json", action="store_true", help="dump raw JSON instead of a table")
    args = parser.parse_args()

    rows = collect_status()

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    n_started = sum(1 for r in rows if r["started"])
    n_finished = sum(1 for r in rows if r["finished"])
    n_violations = sum(1 for r in rows if r["log_integrity_violations"])
    print(f"{len(rows)} configs total | {n_started} started | {n_finished} finished | {n_violations} with log-integrity violations\n")

    header = f"{'run_id':45s} {'steps':>10s} {'%':>6s} {'done':>5s} {'early_stop':>10s} {'last_update':>20s} {'integrity':>10s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if not r["started"]:
            status = "not started"
            print(f"{r['run_id']:45s} {'-':>10s} {'-':>6s} {'-':>5s} {'-':>10s} {'-':>20s} {status:>10s}")
            continue
        integrity = "OK" if not r["log_integrity_violations"] else f"{len(r['log_integrity_violations'])} BAD"
        early_stop = str(r["early_stop_step"]) if r["early_stop_step"] is not None else "-"
        print(
            f"{r['run_id']:45s} {r['env_steps']:10d} {r['pct_done']:5.1f}% {'yes' if r['finished'] else 'no':>5s} "
            f"{early_stop:>10s} {r['last_update_iso'] or '-':>20s} {integrity:>10s}"
        )
        if args.verbose and r["log_integrity_violations"]:
            for v in r["log_integrity_violations"]:
                print(f"    ! {v}")
        if r["ckpt_error"]:
            print(f"    ! checkpoint error: {r['ckpt_error']}")


if __name__ == "__main__":
    main()
