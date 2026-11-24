"""Part 1 learning-curve figure: success rate vs. env steps, one panel per
instance, one curve per encoder, mean +/- 1 std across seeds.

Reuses analyze.py's smooth() (a padded rolling-mean convolution) for the
episode-level moving average, per that file's own docstring convention
("smoothed with a rolling window") -- imported directly, not reimplemented,
since it's a generic 1D smoother with no dependency on what it's applied to
(analyze.py applies it to per-update returns; this script applies the same
function to a per-episode binary success sequence).

Reads only results/*__episodes.jsonl (never writes there). No network
access; deterministic given the same results/ contents and --window.

Usage (main-text, 3 panels):
    python3 scripts/make_part1_figure.py \
        --instances gridtimed/T2 trmgrid/G_SMALL modesel/M4 \
        --window 200 --out figures/part1_curves \
        --width 5.5 --height 1.9

Usage (appendix, 3x2 grid, all six instances):
    python3 scripts/make_part1_figure.py \
        --instances gridtimed/T2 gridtimed/T3 trmgrid/G_SMALL trmgrid/G_LARGE modesel/M3 modesel/M4 \
        --window 200 --out figures/part1_curves_appendix \
        --width 5.5 --height 3.6 --ncols 3
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze import smooth  # noqa: E402  -- reused, not reimplemented (see module docstring)

ENCODERS = ("mlp_point", "deepset_h", "deepset_v", "cnn_dbm", "cnn_proj", "gnn")

#: (color, linestyle, linewidth), mlp_point is the black baseline, the rest
#: are the colourblind-safe palette + distinct dash patterns specified.
ENCODER_STYLE: Dict[str, Dict[str, Any]] = {
    "mlp_point": dict(color="black", linestyle="-", linewidth=1.6),
    "deepset_h": dict(color="#0072B2", linestyle="-", linewidth=1.1),
    "deepset_v": dict(color="#E69F00", linestyle="--", linewidth=1.1),
    "cnn_dbm": dict(color="#009E73", linestyle="-.", linewidth=1.1),
    "cnn_proj": dict(color="#CC79A7", linestyle=":", linewidth=1.1),
    "gnn": dict(color="#56B4E9", linestyle=(0, (8, 4)), linewidth=1.1),  # long-dash
}
ENCODER_LABEL = {
    "mlp_point": "mlp_point", "deepset_h": "deepset_h", "deepset_v": "deepset_v",
    "cnn_dbm": "cnn_dbm", "cnn_proj": "cnn_proj", "gnn": "gnn",
}

#: Expected seed count per domain -- gridtimed/trmgrid were swept at 10
#: seeds, modesel at 3 (see run_one.py's SEEDS / EXTRA_SEEDS / TRMGRID_SEEDS).
EXPECTED_SEEDS: Dict[str, int] = {"gridtimed": 10, "trmgrid": 10, "modesel": 3}

#: All six sweep instances (domain, instance) -- used for the cross-check,
#: independent of whatever subset --instances plots.
ALL_INSTANCES: List[Tuple[str, str]] = [
    ("gridtimed", "T2"), ("gridtimed", "T3"),
    ("trmgrid", "G_SMALL"), ("trmgrid", "G_LARGE"),
    ("modesel", "M3"), ("modesel", "M4"),
]

PRETTY_LABEL: Dict[Tuple[str, str], str] = {
    ("gridtimed", "T2"): "GridTimed T2", ("gridtimed", "T3"): "GridTimed T3",
    ("trmgrid", "G_SMALL"): "TRMGrid G-small", ("trmgrid", "G_LARGE"): "TRMGrid G-large",
    ("modesel", "M3"): "ModeSel M3", ("modesel", "M4"): "ModeSel M4",
}

GRID_MAX_STEPS = 300_000
GRID_N_POINTS = 300

#: Cross-check table from the task -- last-200-episode success rate,
#: independently recomputed below and compared, not trusted.
EXPECTED_LAST200 = {
    "mlp_point": {"T2": 0.054, "T3": 0.039, "G_SMALL": 0.307, "G_LARGE": 0.282, "M3": 0.887, "M4": 0.797},
    "deepset_h": {"T2": 0.839, "T3": 0.900, "G_SMALL": 0.731, "G_LARGE": 0.671, "M3": 0.888, "M4": 0.873},
    "deepset_v": {"T2": 0.498, "T3": 0.200, "G_SMALL": 0.008, "G_LARGE": 0.001, "M3": 0.867, "M4": 0.673},
    "cnn_dbm": {"T2": 0.928, "T3": 0.761, "G_SMALL": 0.534, "G_LARGE": 0.716, "M3": 0.905, "M4": 0.855},
    "cnn_proj": {"T2": 1.000, "T3": 1.000, "G_SMALL": 0.375, "G_LARGE": 0.736, "M3": 0.820, "M4": 0.680},
    "gnn": {"T2": 1.000, "T3": 0.893, "G_SMALL": 0.733, "G_LARGE": 0.527, "M3": 0.837, "M4": 0.648},
}
DISAGREEMENT_TOL = 0.005


def run_id(domain: str, instance: str, encoder: str, seed: int) -> str:
    return f"{domain}__{instance}__{encoder}__seed{seed}"


def load_episodes(results_dir: Path, domain: str, instance: str, encoder: str, seed: int) -> Optional[List[Dict[str, Any]]]:
    path = results_dir / f"{run_id(domain, instance, encoder, seed)}__episodes.jsonl"
    if not path.exists():
        return None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return None
    return records


def smoothed_success_curve(episodes: List[Dict[str, Any]], window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Chronological per-episode success (0/1), smoothed with analyze.py's
    smooth() (padded rolling mean), paired with each episode's env_steps
    (the training env-step count at the end of the PPO update that episode
    completed in -- coarser than per-episode, shared across every episode
    finished within the same update; see the run notes for this figure)."""
    success = np.array([1.0 if e["success"] else 0.0 for e in episodes], dtype=np.float64)
    env_steps = np.array([e["env_steps"] for e in episodes], dtype=np.float64)
    smoothed = smooth(success, window)
    return env_steps, smoothed


def interp_no_extrapolate(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """np.interp onto `grid`, but grid points outside [x.min(), x.max()]
    are NaN rather than clamped to the nearest endpoint -- np.interp's
    default behavior pads with the first/last y-value past the data range,
    which is exactly the silent-padding-with-final-value this figure must
    not do for short/incomplete runs."""
    out = np.interp(grid, x, y)
    out[grid < x.min()] = np.nan
    out[grid > x.max()] = np.nan
    return out


def collect_instance(
    results_dir: Path, domain: str, instance: str, window: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    """Returns (mean_curve, std_curve, n_seeds_found) per encoder, on the
    shared GRID_N_POINTS-point grid over [0, GRID_MAX_STEPS]."""
    grid = np.linspace(0, GRID_MAX_STEPS, GRID_N_POINTS)
    means: Dict[str, np.ndarray] = {}
    stds: Dict[str, np.ndarray] = {}
    n_found: Dict[str, int] = {}
    expected = EXPECTED_SEEDS[domain]

    for encoder in ENCODERS:
        seed_curves = []
        seed = 0
        # Seeds are contiguous 0..N-1 in this sweep; scan until a gap
        # confirms we've found them all rather than assuming the count.
        misses_in_a_row = 0
        while misses_in_a_row < 3:
            episodes = load_episodes(results_dir, domain, instance, encoder, seed)
            if episodes is None:
                misses_in_a_row += 1
            else:
                misses_in_a_row = 0
                x, y = smoothed_success_curve(episodes, window)
                seed_curves.append(interp_no_extrapolate(grid, x, y))
            seed += 1
        n_found[encoder] = len(seed_curves)
        if len(seed_curves) != expected:
            print(
                f"WARNING: {domain}/{instance}/{encoder}: found {len(seed_curves)} seed(s), "
                f"expected {expected}. Proceeding with what was found -- this is not silent, "
                f"but check results/ for a missing/corrupt run.",
                file=sys.stderr,
            )
        if not seed_curves:
            means[encoder] = np.full(GRID_N_POINTS, np.nan)
            stds[encoder] = np.full(GRID_N_POINTS, np.nan)
            continue
        stacked = np.stack(seed_curves)  # (n_seeds, GRID_N_POINTS), NaN where a seed had no data yet
        # All-NaN columns are expected at the very first grid point(s) --
        # every seed's first episode lands after env_steps=0 (the first
        # PPO update completes at ~rollout_size steps), so grid=0 has no
        # data for any seed and correctly stays NaN. Suppress numpy's
        # "Mean/Std of empty slice" warnings for that known, harmless case
        # instead of letting them spam the console.
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
        means[encoder] = np.clip(mean, 0.0, 1.0)
        stds[encoder] = std  # band clipping happens at plot time (mean +/- std, then clip)
    return means, stds, n_found


def last200_success_rate(episodes: List[Dict[str, Any]]) -> float:
    last200 = episodes[-200:]
    return sum(1 for e in last200 if e["success"]) / len(last200)


def cross_check(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Recomputes last-200-episode success rate per (encoder, instance)
    across all six sweep instances, independent of --instances, and
    compares against the paper's stated table."""
    print("\n=== Cross-check: recomputed last-200-episode success rate vs. paper table ===")
    disagreements = []
    out: Dict[str, Dict[str, Any]] = {}
    for domain, instance in ALL_INSTANCES:
        out.setdefault(instance, {})
        for encoder in ENCODERS:
            per_seed = []
            seed = 0
            misses_in_a_row = 0
            while misses_in_a_row < 3:
                episodes = load_episodes(results_dir, domain, instance, encoder, seed)
                if episodes is None:
                    misses_in_a_row += 1
                else:
                    misses_in_a_row = 0
                    per_seed.append(last200_success_rate(episodes))
                seed += 1
            recomputed = float(np.mean(per_seed)) if per_seed else None
            expected = EXPECTED_LAST200.get(encoder, {}).get(instance)
            out[instance][encoder] = {"recomputed": recomputed, "n_seeds": len(per_seed), "per_seed": per_seed}
            if recomputed is not None and expected is not None:
                diff = abs(recomputed - expected)
                flag = "  <-- DISAGREES (> tol)" if diff > DISAGREEMENT_TOL else ""
                if diff > DISAGREEMENT_TOL:
                    disagreements.append((domain, instance, encoder, expected, recomputed, diff))
                print(f"  {domain}/{instance:8s} {encoder:10s}: paper={expected:.3f}  recomputed={recomputed:.3f}  diff={diff:.4f}{flag}")
            else:
                print(f"  {domain}/{instance:8s} {encoder:10s}: paper={expected}  recomputed={recomputed}  (missing data)")
    if disagreements:
        print(f"\n{len(disagreements)} cell(s) disagree by more than {DISAGREEMENT_TOL}:")
        for domain, instance, encoder, expected, recomputed, diff in disagreements:
            print(f"  {domain}/{instance}/{encoder}: paper={expected:.4f} recomputed={recomputed:.4f} diff={diff:.4f}")
    else:
        print(f"\nAll cells agree within tolerance ({DISAGREEMENT_TOL}).")
    return out


def format_steps(x: float, _pos=None) -> str:
    if x == 0:
        return "0"
    return f"{int(x / 1000)}k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--instances", nargs="+", default=["gridtimed/T2", "trmgrid/G_SMALL", "modesel/M4"],
        help="domain/instance pairs, e.g. gridtimed/T2",
    )
    parser.add_argument("--window", type=int, default=200, help="episode moving-average window")
    parser.add_argument("--out", type=Path, default=Path("figures/part1_curves"), help="output path prefix (no extension)")
    parser.add_argument("--width", type=float, default=5.5, help="figure width, inches")
    parser.add_argument("--height", type=float, default=1.9, help="figure height, inches")
    parser.add_argument("--ncols", type=int, default=None, help="panel grid columns (default: min(n_instances, 3))")
    parser.add_argument("--dpi", type=int, default=200, help="PNG output DPI")
    args = parser.parse_args()

    pairs: List[Tuple[str, str]] = []
    for spec in args.instances:
        domain, instance = spec.split("/", 1)
        if domain not in EXPECTED_SEEDS:
            parser.error(f"unknown domain {domain!r} in --instances (expected one of {list(EXPECTED_SEEDS)})")
        pairs.append((domain, instance))

    ncols = args.ncols or min(len(pairs), 3)
    nrows = -(-len(pairs) // ncols)  # ceil

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titlesize": 9,
    })

    fig, axes = plt.subplots(nrows, ncols, figsize=(args.width, args.height), sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    grid = np.linspace(0, GRID_MAX_STEPS, GRID_N_POINTS)
    data_out: Dict[str, Any] = {"smoothing_window": args.window, "instances": {}}

    letters = "abcdefghij"
    print("\n=== Seed counts found (per instance, per encoder) ===")
    for idx, (domain, instance) in enumerate(pairs):
        ax = axes_flat[idx]
        means, stds, n_found = collect_instance(args.results_dir, domain, instance, args.window)
        print(f"{domain}/{instance}: " + ", ".join(f"{enc}={n_found[enc]}" for enc in ENCODERS))

        for encoder in ENCODERS:
            style = ENCODER_STYLE[encoder]
            mean = means[encoder]
            std = stds[encoder]
            ax.plot(grid, mean, label=ENCODER_LABEL[encoder], **style, zorder=3)
            lo = np.clip(mean - std, 0.0, 1.0)
            hi = np.clip(mean + std, 0.0, 1.0)
            ax.fill_between(grid, lo, hi, color=style["color"], alpha=0.15, linewidth=0, zorder=2)

        label = PRETTY_LABEL.get((domain, instance), f"{domain} {instance}")
        ax.set_title(f"({letters[idx]}) {label}")
        ax.set_xlim(0, GRID_MAX_STEPS)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 100_000, 200_000, 300_000])
        ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
        # Adjacent panels in a row share no gap, so a panel's last label
        # ("300k") sits flush against the next panel's first ("0"),
        # rendering as "300k0". Suppress the last tick label on every
        # panel except the rightmost in its row (last column, or the
        # final panel overall if a row is short) -- the "0" stays on
        # every panel, only "300k" is dropped from non-final columns.
        is_row_last = (idx % ncols == ncols - 1) or (idx == len(pairs) - 1)
        if not is_row_last:
            ax.xaxis.get_major_ticks()[-1].label1.set_visible(False)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx % ncols == 0:
            ax.set_ylabel("Success rate")

        # Data file: seed count, window, and RAW (unsmoothed) final
        # success rate = mean of the last 200 episodes of training, same
        # convention as analyze.py's last200_success_rate -- distinct from
        # the smoothed curve's endpoint, which is a rolling mean, not a
        # window-of-exactly-200-episodes-at-the-end statistic.
        instance_entry: Dict[str, Any] = {}
        for encoder in ENCODERS:
            per_seed_final = []
            seed = 0
            misses_in_a_row = 0
            while misses_in_a_row < 3:
                episodes = load_episodes(args.results_dir, domain, instance, encoder, seed)
                if episodes is None:
                    misses_in_a_row += 1
                else:
                    misses_in_a_row = 0
                    per_seed_final.append(last200_success_rate(episodes))
                seed += 1
            instance_entry[encoder] = {
                "n_seeds": n_found[encoder],
                "final_success_rate_last200_mean": float(np.mean(per_seed_final)) if per_seed_final else None,
                "final_success_rate_last200_per_seed": per_seed_final,
            }
        data_out["instances"][f"{domain}/{instance}"] = instance_entry

    # Hide any unused trailing axes (e.g. a 6-instance request into a 3x2
    # grid with fewer than 6 pairs -- not expected here, but keep it safe).
    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].axis("off")

    # "Labelled once, centred under the row" -- an axis-level xlabel on the
    # bottom row's middle-column panel, rather than fig.supxlabel, so its
    # vertical position is tied to that panel's own tick labels (matplotlib
    # lays it out correctly relative to them automatically) instead of
    # fighting the legend below it for the same sliver of figure space.
    bottom_row_start = (nrows - 1) * ncols
    center_col = ncols // 2
    axes_flat[bottom_row_start + center_col].set_xlabel("Environment steps")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.subplots_adjust(
        bottom=0.50 if nrows == 1 else 0.24, top=0.86, left=0.09, right=0.99,
        wspace=0.08, hspace=0.55 if nrows > 1 else None,
    )
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.0 if nrows == 1 else 0.0),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out.with_suffix(".pdf")
    png_path = args.out.with_suffix(".png")
    json_path = Path(str(args.out) + "_data.json")

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    json_path.write_text(json.dumps(data_out, indent=2))
    print(f"\nWrote {pdf_path}, {png_path}, {json_path}")

    cross_check_result = cross_check(args.results_dir)
    data_out["cross_check"] = cross_check_result
    json_path.write_text(json.dumps(data_out, indent=2))


if __name__ == "__main__":
    main()
