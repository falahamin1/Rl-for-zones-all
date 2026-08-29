"""Offline analysis, run locally after pulling results/ off Alpine (rsync
or similar -- this script only reads results/*.jsonl, no cluster access).

Produces, per (domain, instance):
  - learning curves: env_steps vs ep_return_mean, mean+/-1std band per
    encoder (across seeds), smoothed with a rolling window (--smooth-window).
  - the same curves vs wall-clock (cumulative wall_s).
  - steps-to-threshold: env_steps to first reach 50%/75%/90% of EACH
    encoder's OWN best smoothed return on that instance, mean+/-std across
    seeds -- the headline sample-efficiency number.
  - a summary table (CSV) as a fallback for when plotting isn't wanted/
    possible.

Handles partial runs: any (instance, encoder, seed) with data is plotted;
configs with zero data are listed as incomplete/missing rather than
silently skipped or crashing the whole run.

Usage: python3 analyze.py [--results-dir results] [--out-dir analysis]
                            [--smooth-window 5]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

DOMAINS_INSTANCES = {
    "modesel": ("M3", "M4"),
    "gridtimed": ("T2", "T3"),
}
ENCODERS = ("mlp_point", "deepset_h", "cnn_dbm", "cnn_proj")
SEEDS = (0, 1, 2)
THRESHOLDS = (0.5, 0.75, 0.9)


def _run_id(domain: str, instance: str, encoder: str, seed: int) -> str:
    return f"{domain}__{instance}__{encoder}__seed{seed}"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window) / window
    # 'same'-length rolling mean via padding with edge values, so the
    # smoothed curve has the same x-length as the input (no phase shift
    # from a shrinking valid-mode convolution).
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def steps_to_threshold(env_steps: np.ndarray, returns: np.ndarray, frac: float) -> Optional[float]:
    """First env_steps at which `returns` has closed `frac` of the way from
    its own worst (typically early-training) value to its own best value.
    Sign-agnostic by construction -- these domains log return=-cost, so
    returns are negative and "frac of best" measured from zero would run
    backwards (best*0.5 is a WORSE, more-negative value than best when
    best<0). Measuring from the run's own [worst, best] range instead is
    the standard, sign-independent way to express "X% of the way to
    convergence". None if the run never reaches that fraction (e.g.
    stopped early) or shows no learning signal (best <= worst)."""
    if len(returns) == 0:
        return None
    worst, best = np.min(returns), np.max(returns)
    if best <= worst:
        return None
    target = worst + frac * (best - worst)
    idx = np.argmax(returns >= target)
    if returns[idx] < target:
        return None
    return float(env_steps[idx])


def load_run(domain: str, instance: str, encoder: str, seed: int, results_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    path = results_dir / f"{_run_id(domain, instance, encoder, seed)}.jsonl"
    records = load_jsonl(path)
    if not records:
        return None
    records.sort(key=lambda r: r["update"])
    env_steps = np.array([r["env_steps"] for r in records], dtype=np.float64)
    ep_return_mean = np.array([r["ep_return_mean"] for r in records], dtype=np.float64)
    wall_s = np.array([r["wall_s"] for r in records], dtype=np.float64)
    cum_wall_s = np.cumsum(wall_s)
    # Drop NaN-return updates (no episode completed that update) for
    # curve-fitting purposes, but keep alignment by forward-filling from
    # the last valid value -- a flat segment is a more honest depiction of
    # "no new information" than a NaN gap or a misleading interpolation.
    valid = ~np.isnan(ep_return_mean)
    if valid.sum() == 0:
        return None
    ffilled = ep_return_mean.copy()
    last = ffilled[valid][0]
    for i in range(len(ffilled)):
        if valid[i]:
            last = ffilled[i]
        else:
            ffilled[i] = last
    return {"env_steps": env_steps, "ep_return_mean": ffilled, "cum_wall_s": cum_wall_s}


def collect(results_dir: Path) -> Tuple[Dict[Tuple[str, str, str], List[Dict[str, np.ndarray]]], List[str]]:
    """Returns ({(domain, instance, encoder): [per-seed run dict, ...]}, missing_run_ids)."""
    grouped: Dict[Tuple[str, str, str], List[Dict[str, np.ndarray]]] = defaultdict(list)
    missing: List[str] = []
    for domain, instances in DOMAINS_INSTANCES.items():
        for instance in instances:
            for encoder in ENCODERS:
                for seed in SEEDS:
                    run = load_run(domain, instance, encoder, seed, results_dir)
                    if run is None:
                        missing.append(_run_id(domain, instance, encoder, seed))
                    else:
                        grouped[(domain, instance, encoder)].append(run)
    return grouped, missing


def plot_curves(
    grouped: Dict[Tuple[str, str, str], List[Dict[str, np.ndarray]]],
    domain: str, instance: str, x_key: str, xlabel: str, out_path: Path, smooth_window: int,
) -> bool:
    if not HAVE_MPL:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    for encoder in ENCODERS:
        runs = grouped.get((domain, instance, encoder), [])
        if not runs:
            continue
        any_data = True
        # Interpolate every seed's curve onto a common x-grid (union range,
        # clipped to the shortest run's extent) so mean/std are computed
        # pointwise across seeds, not just concatenated.
        x_max = min(r[x_key][-1] for r in runs)
        x_min = max(r[x_key][0] for r in runs)
        if x_max <= x_min:
            continue
        grid = np.linspace(x_min, x_max, 200)
        curves = []
        for r in runs:
            y_smoothed = smooth(r["ep_return_mean"], smooth_window)
            curves.append(np.interp(grid, r[x_key], y_smoothed))
        curves = np.stack(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(grid, mean, label=f"{encoder} (n={len(runs)})")
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
    if not any_data:
        plt.close(fig)
        return False
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ep_return_mean (smoothed)")
    ax.set_title(f"{domain}/{instance}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def compute_steps_to_threshold_table(
    grouped: Dict[Tuple[str, str, str], List[Dict[str, np.ndarray]]], smooth_window: int,
) -> List[Dict[str, Any]]:
    rows = []
    for domain, instances in DOMAINS_INSTANCES.items():
        for instance in instances:
            for encoder in ENCODERS:
                runs = grouped.get((domain, instance, encoder), [])
                if not runs:
                    rows.append(
                        {"domain": domain, "instance": instance, "encoder": encoder, "n_seeds": 0,
                         **{f"steps_to_{int(t*100)}pct_mean": None for t in THRESHOLDS},
                         **{f"steps_to_{int(t*100)}pct_std": None for t in THRESHOLDS}}
                    )
                    continue
                per_threshold: Dict[float, List[float]] = {t: [] for t in THRESHOLDS}
                for r in runs:
                    y_smoothed = smooth(r["ep_return_mean"], smooth_window)
                    for t in THRESHOLDS:
                        v = steps_to_threshold(r["env_steps"], y_smoothed, t)
                        if v is not None:
                            per_threshold[t].append(v)
                row = {"domain": domain, "instance": instance, "encoder": encoder, "n_seeds": len(runs)}
                for t in THRESHOLDS:
                    vals = per_threshold[t]
                    row[f"steps_to_{int(t*100)}pct_mean"] = float(np.mean(vals)) if vals else None
                    row[f"steps_to_{int(t*100)}pct_std"] = float(np.std(vals)) if vals else None
                    row[f"steps_to_{int(t*100)}pct_n_reached"] = len(vals)
                rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_table(grouped, missing: List[str], out_path: Path) -> None:
    """Text fallback when matplotlib isn't available or for a quick read."""
    lines = []
    for domain, instances in DOMAINS_INSTANCES.items():
        for instance in instances:
            lines.append(f"\n=== {domain}/{instance} ===")
            for encoder in ENCODERS:
                runs = grouped.get((domain, instance, encoder), [])
                if not runs:
                    lines.append(f"  {encoder:12s}  NO DATA")
                    continue
                final_returns = [r["ep_return_mean"][-1] for r in runs]
                max_steps = [r["env_steps"][-1] for r in runs]
                lines.append(
                    f"  {encoder:12s}  n_seeds={len(runs)}  "
                    f"final_ep_return_mean={np.mean(final_returns):9.3f}+/-{np.std(final_returns):7.3f}  "
                    f"max_env_steps_seen={np.mean(max_steps):9.0f}"
                )
    if missing:
        lines.append(f"\n{len(missing)} run(s) with no data (not started, or crashed before first update):")
        for m in missing:
            lines.append(f"  {m}")
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()

    grouped, missing = collect(args.results_dir)

    if not HAVE_MPL:
        print("matplotlib not available -- skipping PNG curves, writing CSV/summary table only.")

    for domain, instances in DOMAINS_INSTANCES.items():
        for instance in instances:
            ok = plot_curves(
                grouped, domain, instance, "env_steps", "env_steps",
                args.out_dir / f"{domain}__{instance}__curves_steps.png", args.smooth_window,
            )
            if HAVE_MPL and not ok:
                print(f"[{domain}/{instance}] no data for env_steps curve -- skipped")
            ok = plot_curves(
                grouped, domain, instance, "cum_wall_s", "wall-clock seconds",
                args.out_dir / f"{domain}__{instance}__curves_wallclock.png", args.smooth_window,
            )
            if HAVE_MPL and not ok:
                print(f"[{domain}/{instance}] no data for wall-clock curve -- skipped")

    steps_rows = compute_steps_to_threshold_table(grouped, args.smooth_window)
    write_csv(steps_rows, args.out_dir / "steps_to_threshold.csv")

    write_summary_table(grouped, missing, args.out_dir / "summary.txt")

    print(f"\nWrote analysis to {args.out_dir}/ (steps_to_threshold.csv, summary.txt" + (", *.png" if HAVE_MPL else "") + ")")


if __name__ == "__main__":
    main()
