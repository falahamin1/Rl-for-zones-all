"""D2: per-seed final-outcome strip plot -- one row of four panels
(GridTimed T2, GridTimed T3, TRMGrid G_SMALL, TRMGrid G_LARGE), x=encoder,
y=per-seed final success rate (last 200 training episodes), each seed a
single jittered point so bimodal splits (e.g. deepset_v on GridTimed)
stay visible as two clusters rather than being averaged into one bar.

Reuses make_part1_figure.py's run_id/load_episodes/last200_success_rate
and its ENCODER_STYLE color assignments (same palette, so a reader sees
one consistent color per encoder across every figure in the paper) --
imported, not reimplemented.

Reads only results/*__episodes.jsonl. No network access, deterministic
given --seed (jitter only) and the same results/ contents.

Usage:
    python3 scripts/make_part1_per_seed.py --out figures/part1_per_seed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_part1_figure import (  # noqa: E402
    ENCODER_STYLE, EXPECTED_SEEDS, PRETTY_LABEL, load_episodes, last200_success_rate,
)

#: D2's x-axis order is the shared cross-paper palette table's row order
#: (deepset_h, deepset_v, gnn, cnn_proj, cnn_dbm, mlp_point) -- deliberately
#: NOT make_part1_figure.py's ENCODERS tuple order (which is grouped for
#: that figure's own 2-row legend), per the brief's explicit instruction
#: to use the palette table's order here.
D2_ENCODER_ORDER = ("deepset_h", "deepset_v", "gnn", "cnn_proj", "cnn_dbm", "mlp_point")

PANELS: List[Tuple[str, str]] = [
    ("gridtimed", "T2"), ("gridtimed", "T3"),
    ("trmgrid", "G_SMALL"), ("trmgrid", "G_LARGE"),
]


def collect_per_seed(results_dir: Path, domain: str, instance: str, encoder: str) -> List[float]:
    values = []
    seed = 0
    misses_in_a_row = 0
    while misses_in_a_row < 3:
        episodes = load_episodes(results_dir, domain, instance, encoder, seed)
        if episodes is None:
            misses_in_a_row += 1
        else:
            misses_in_a_row = 0
            values.append(last200_success_rate(episodes))
        seed += 1
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures/part1_per_seed"))
    parser.add_argument("--width", type=float, default=5.5)
    parser.add_argument("--height", type=float, default=2.3)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--jitter", type=float, default=0.10, help="max horizontal jitter, in category units")
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument("--jitter-seed", type=int, default=0, help="RNG seed for horizontal jitter only")
    args = parser.parse_args()

    rng = np.random.RandomState(args.jitter_seed)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titlesize": 9,
    })

    fig, axes = plt.subplots(1, len(PANELS), figsize=(args.width, args.height), sharey=True)
    letters = "abcd"
    data_out: Dict[str, Dict] = {}

    print("=== Seed counts found (per instance, per encoder) ===")
    for idx, (domain, instance) in enumerate(PANELS):
        ax = axes[idx]
        expected = EXPECTED_SEEDS[domain]
        instance_entry = {}
        seed_counts = []
        for x, encoder in enumerate(D2_ENCODER_ORDER):
            values = collect_per_seed(args.results_dir, domain, instance, encoder)
            seed_counts.append(f"{encoder}={len(values)}")
            if len(values) != expected:
                print(
                    f"WARNING: {domain}/{instance}/{encoder}: found {len(values)} seed(s), "
                    f"expected {expected}. Plotting what was found -- reported, not padded.",
                    file=sys.stderr,
                )
            instance_entry[encoder] = {"n_seeds": len(values), "per_seed_final_success_rate": values}

            color = ENCODER_STYLE[encoder]["color"]
            if values:
                jitter = rng.uniform(-args.jitter, args.jitter, size=len(values))
                ax.scatter(
                    np.full(len(values), x) + jitter, values,
                    s=args.point_size, color=color, alpha=0.85, linewidths=0, zorder=3,
                )
                mean = float(np.mean(values))
                ax.plot([x - 0.18, x + 0.18], [mean, mean], color=color, linewidth=1.6, zorder=4, solid_capstyle="butt")
        print(f"{domain}/{instance}: " + ", ".join(seed_counts))
        data_out[f"{domain}/{instance}"] = instance_entry

        label = PRETTY_LABEL.get((domain, instance), f"{domain} {instance}")
        ax.set_title(f"({letters[idx]}) {label}")
        ax.set_xlim(-0.6, len(D2_ENCODER_ORDER) - 0.4)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(D2_ENCODER_ORDER)))
        ax.set_xticklabels(D2_ENCODER_ORDER, rotation=40, ha="right")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Final success rate\n(last 200 episodes)")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=ENCODER_STYLE[enc]["color"], label=enc, markersize=5)
        for enc in D2_ENCODER_ORDER
    ]
    fig.subplots_adjust(bottom=0.42, top=0.86, left=0.10, right=0.99, wspace=0.12)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out.with_suffix(".pdf")
    png_path = args.out.with_suffix(".png")
    json_path = Path(str(args.out) + "_data.json")

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    json_path.write_text(json.dumps(data_out, indent=2))
    print(f"\nWrote {pdf_path}, {png_path}, {json_path}")


if __name__ == "__main__":
    main()
