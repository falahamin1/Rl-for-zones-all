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
    #: trmgrid extension (artifacts/TRMGRID_SWEEP.md) -- run_one.TARGET_RUNS
    #: indices 168-287, a brand-new domain string appended after the
    #: shaping experiment. Folded in here (previously analyzed via a
    #: one-off script reusing this module's functions) so trmgrid gets the
    #: same curves/CSV treatment as everything else, not a separate path.
    "trmgrid": ("G_SMALL", "G_LARGE"),
}
#: Extended for the v_rep/GNN addition (/tmp/VREP_FEASIBILITY.md) -- six
#: encoders now, same DOMAINS_INSTANCES/SEEDS, run_one.TARGET_RUNS indices
#: 48-71 for the two new ones. This is the one intentional, explicitly-
#: requested modification to analyze.py (Step 3); everything else in this
#: file that loops over ENCODERS picks the six up automatically.
ENCODERS = ("mlp_point", "deepset_h", "cnn_dbm", "cnn_proj", "deepset_v", "gnn")
SEEDS = (0, 1, 2)
#: Gridtimed seed extension (/tmp/SWEEP_FINAL.md Task C): gridtimed grew
#: to 10 seeds (0-9) across all six encoders to characterize the bimodal
#: deepset_v/gnn outcomes; modesel stayed at 3. A single flat SEEDS tuple
#: can no longer describe both -- _seeds_for_domain below is the one other
#: intentional modification this extension requires (a hardcoded SEEDS=
#: (0,1,2) would silently under-report gridtimed the moment the new runs
#: land, which is a correctness bug, not just an incomplete manifest).
_GRIDTIMED_EXTRA_SEEDS = (3, 4, 5, 6, 7, 8, 9)
#: trmgrid was swept at 10 seeds from the start (TRMGRID_SEEDS in
#: run_one.py), not grown from a 3-seed base like gridtimed was -- same
#: resulting 0-9 range, kept as its own name rather than reusing
#: _GRIDTIMED_EXTRA_SEEDS so this doesn't imply a shared extension history
#: the two domains don't actually have.
_TRMGRID_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
THRESHOLDS = (0.5, 0.75, 0.9)


def _seeds_for_domain(domain: str) -> Tuple[int, ...]:
    if domain == "gridtimed":
        return SEEDS + _GRIDTIMED_EXTRA_SEEDS
    if domain == "trmgrid":
        return _TRMGRID_SEEDS
    return SEEDS


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
    encoder_forward_s = np.array([r.get("encoder_forward_s", np.nan) for r in records], dtype=np.float64)
    raster_s = np.array([r.get("raster_s", np.nan) for r in records], dtype=np.float64)
    vrep_fallback_frac = np.array(
        [r["vrep_fallback_frac"] if r.get("vrep_fallback_frac") is not None else np.nan for r in records],
        dtype=np.float64,
    )
    # Dual-metric eval (/tmp/PART1_FINAL.md item 2), gridtimed only, null
    # except every EVAL_EVERY_K_UPDATES-th update -- same ".get(...) is not
    # None else nan" tolerance as vrep_fallback_frac, so jsonl files written
    # before this field existed (every run prior to this extension) load
    # cleanly with all-nan columns rather than a KeyError.
    greedy_success_rate = np.array(
        [r["greedy_success_rate"] if r.get("greedy_success_rate") is not None else np.nan for r in records],
        dtype=np.float64,
    )
    greedy_cycling_frac = np.array(
        [r["greedy_cycling_frac"] if r.get("greedy_cycling_frac") is not None else np.nan for r in records],
        dtype=np.float64,
    )
    stochastic_success_rate = np.array(
        [r["stochastic_success_rate"] if r.get("stochastic_success_rate") is not None else np.nan for r in records],
        dtype=np.float64,
    )
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
    return {
        "env_steps": env_steps, "ep_return_mean": ffilled, "cum_wall_s": cum_wall_s,
        "wall_s": wall_s, "encoder_forward_s": encoder_forward_s, "raster_s": raster_s,
        "vrep_fallback_frac": vrep_fallback_frac,
        "greedy_success_rate": greedy_success_rate, "greedy_cycling_frac": greedy_cycling_frac,
        "stochastic_success_rate": stochastic_success_rate,
    }


def load_episodes(domain: str, instance: str, encoder: str, seed: int, results_dir: Path) -> List[Dict[str, Any]]:
    path = results_dir / f"{_run_id(domain, instance, encoder, seed)}__episodes.jsonl"
    return load_jsonl(path)


def collect(results_dir: Path) -> Tuple[Dict[Tuple[str, str, str], List[Dict[str, np.ndarray]]], List[str]]:
    """Returns ({(domain, instance, encoder): [per-seed run dict, ...]}, missing_run_ids)."""
    grouped: Dict[Tuple[str, str, str], List[Dict[str, np.ndarray]]] = defaultdict(list)
    missing: List[str] = []
    for domain, instances in DOMAINS_INSTANCES.items():
        for instance in instances:
            for encoder in ENCODERS:
                for seed in _seeds_for_domain(domain):
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


def compute_encoder_stats_table(results_dir: Path) -> List[Dict[str, Any]]:
    """Per (domain, instance, encoder), mean +/- std across seeds:
    success rate (overall and last-200 episodes, from the episodes.jsonl
    -- the "true" per-episode outcome, not derived from return sign, same
    convention used throughout this project's diagnostics), final return,
    encoder_forward_s as a fraction of wall_s (with raster_s -- which
    ALSO carries deepset_v's v_rep enumeration time, see
    zoneviews.raster.cached_v_rep -- broken out as its own fraction, same
    idea as cnn_proj's raster_s), and vrep_fallback_frac (deepset_v only,
    None for the other five encoders -- not a spurious 0.0)."""
    rows = []
    for domain, instances in DOMAINS_INSTANCES.items():
        for instance in instances:
            for encoder in ENCODERS:
                seed_rows = []
                for seed in _seeds_for_domain(domain):
                    run = load_run(domain, instance, encoder, seed, results_dir)
                    episodes = load_episodes(domain, instance, encoder, seed, results_dir)
                    if run is None or not episodes:
                        continue
                    last200 = episodes[-200:]
                    overall_sr = sum(1 for e in episodes if e["success"]) / len(episodes)
                    last200_sr = sum(1 for e in last200 if e["success"]) / len(last200)
                    wall = run["wall_s"]
                    enc_fwd = run["encoder_forward_s"]
                    raster = run["raster_s"]
                    valid_w = wall > 0
                    enc_frac = float(np.nanmean(enc_fwd[valid_w] / wall[valid_w])) if valid_w.any() else float("nan")
                    raster_frac = float(np.nanmean(raster[valid_w] / wall[valid_w])) if valid_w.any() else float("nan")
                    vff = run["vrep_fallback_frac"]
                    vff_valid = vff[~np.isnan(vff)]
                    vrep_frac = float(vff_valid.mean()) if len(vff_valid) else None
                    # Dual-metric eval (item 2): the LAST valid (non-nan)
                    # periodic eval, not a mean across training -- unlike
                    # vrep_fallback_frac (a stable property throughout),
                    # success rate improves over training, so averaging
                    # early-training evals in with the final one would
                    # understate the trained policy's actual quality.
                    # None (not nan) when this run has no eval data at all
                    # (non-gridtimed rows, or a jsonl written before this
                    # field existed) -- the "null where undefined"
                    # convention, tolerated by write_csv/DictWriter either way.
                    def _last_valid(arr: np.ndarray) -> Optional[float]:
                        valid_vals = arr[~np.isnan(arr)]
                        return float(valid_vals[-1]) if len(valid_vals) else None

                    seed_rows.append({
                        "final_return": run["ep_return_mean"][-1],
                        "overall_success_rate": overall_sr,
                        "last200_success_rate": last200_sr,
                        "encoder_forward_frac": enc_frac,
                        "raster_frac": raster_frac,
                        "vrep_fallback_frac": vrep_frac,
                        "greedy_success_rate": _last_valid(run["greedy_success_rate"]),
                        "greedy_cycling_frac": _last_valid(run["greedy_cycling_frac"]),
                        "stochastic_success_rate": _last_valid(run["stochastic_success_rate"]),
                        "n_episodes": len(episodes),
                    })
                if not seed_rows:
                    rows.append({
                        "domain": domain, "instance": instance, "encoder": encoder, "n_seeds": 0,
                        "final_return_mean": None, "final_return_std": None,
                        "overall_success_rate_mean": None, "overall_success_rate_std": None,
                        "last200_success_rate_mean": None, "last200_success_rate_std": None,
                        "encoder_forward_frac_mean": None, "raster_frac_mean": None,
                        "vrep_fallback_frac_mean": None,
                        "greedy_success_rate_mean": None, "greedy_cycling_frac_mean": None,
                        "stochastic_success_rate_mean": None,
                    })
                    continue
                vrep_vals = [r["vrep_fallback_frac"] for r in seed_rows if r["vrep_fallback_frac"] is not None]
                greedy_sr_vals = [r["greedy_success_rate"] for r in seed_rows if r["greedy_success_rate"] is not None]
                greedy_cyc_vals = [r["greedy_cycling_frac"] for r in seed_rows if r["greedy_cycling_frac"] is not None]
                stoch_sr_vals = [r["stochastic_success_rate"] for r in seed_rows if r["stochastic_success_rate"] is not None]
                rows.append({
                    "domain": domain, "instance": instance, "encoder": encoder, "n_seeds": len(seed_rows),
                    "final_return_mean": float(np.mean([r["final_return"] for r in seed_rows])),
                    "final_return_std": float(np.std([r["final_return"] for r in seed_rows])),
                    "overall_success_rate_mean": float(np.mean([r["overall_success_rate"] for r in seed_rows])),
                    "overall_success_rate_std": float(np.std([r["overall_success_rate"] for r in seed_rows])),
                    "last200_success_rate_mean": float(np.mean([r["last200_success_rate"] for r in seed_rows])),
                    "last200_success_rate_std": float(np.std([r["last200_success_rate"] for r in seed_rows])),
                    "encoder_forward_frac_mean": float(np.mean([r["encoder_forward_frac"] for r in seed_rows])),
                    "raster_frac_mean": float(np.mean([r["raster_frac"] for r in seed_rows])),
                    "vrep_fallback_frac_mean": float(np.mean(vrep_vals)) if vrep_vals else None,
                    "greedy_success_rate_mean": float(np.mean(greedy_sr_vals)) if greedy_sr_vals else None,
                    "greedy_cycling_frac_mean": float(np.mean(greedy_cyc_vals)) if greedy_cyc_vals else None,
                    "stochastic_success_rate_mean": float(np.mean(stoch_sr_vals)) if stoch_sr_vals else None,
                })
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

    encoder_stats_rows = compute_encoder_stats_table(args.results_dir)
    write_csv(encoder_stats_rows, args.out_dir / "encoder_stats.csv")

    write_summary_table(grouped, missing, args.out_dir / "summary.txt")

    print(f"\nWrote analysis to {args.out_dir}/ (steps_to_threshold.csv, encoder_stats.csv, summary.txt"
          + (", *.png" if HAVE_MPL else "") + ")")


if __name__ == "__main__":
    main()
