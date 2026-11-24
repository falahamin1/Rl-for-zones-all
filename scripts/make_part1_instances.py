"""A1: benchmark-instance diagram figure -- one row of three panels
((a) GridTimed T2, (b) TRMGrid G_SMALL, (c) ModeSel M3), each rendered
directly from the actual instance object run_one._build_instance()
resolves for the sweep (same construction path run_one.py itself uses,
same INSTANCE_SEED/TRMGRID_INSTANCE_SEED) -- not hand-drawn coordinates.

Neutral (grayscale) palette throughout -- the encoder color palette in
make_part1_figure.py is reserved for encoder data, not used here.

No network access, deterministic given the same repo contents.

Usage:
    python3 scripts/make_part1_instances.py --out figures/part1_instances
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_one import _build_instance  # noqa: E402

# Neutral, grayscale palette -- deliberately NOT the encoder palette.
C_GRID = "#BBBBBB"
C_WALL_TEXT = "#333333"
C_START_GOAL = "black"
C_SWITCH = "#666666"
C_GATE = "#999999"
C_PAIR_LINE = "#AAAAAA"
C_TASK_BOX = "#DDDDDD"
C_TASK_EDGE = "#666666"


def panel_gridtimed(ax, instance_name: str):
    inst, _ctor = _build_instance("gridtimed", instance_name)
    n = inst.grid_n
    # Plot convention: x=col, y=row, row 0 at the TOP (natural grid
    # reading order), via invert_yaxis() -- consistent with (b).
    for r in range(n + 1):
        ax.plot([0, n], [r, r], color=C_GRID, linewidth=0.6, zorder=1)
    for c in range(n + 1):
        ax.plot([c, c], [0, n], color=C_GRID, linewidth=0.6, zorder=1)

    def cell_center(pos):
        r, c = pos
        return c + 0.5, r + 0.5

    # Switch i <-> gate i pairing line, drawn first (beneath markers).
    for i, (sw, gt) in enumerate(zip(inst.switches, inst.gates)):
        sx, sy = cell_center(sw)
        gx, gy = cell_center(gt)
        ax.plot([sx, gx], [sy, gy], color=C_PAIR_LINE, linewidth=0.8, linestyle=":", zorder=2)

    for i, sw in enumerate(inst.switches):
        x, y = cell_center(sw)
        ax.scatter([x], [y], marker="s", s=95, color=C_SWITCH, zorder=3)
        ax.text(x, y - 0.30, f"S{i+1}", ha="center", va="center", fontsize=6.5, color=C_WALL_TEXT)

    for i, (gt, win) in enumerate(zip(inst.gates, inst.gate_windows)):
        x, y = cell_center(gt)
        ax.scatter([x], [y], marker="D", s=85, color=C_GATE, zorder=3)
        ax.text(x, y - 0.30, f"G{i+1}", ha="center", va="center", fontsize=6.5, color=C_WALL_TEXT)
        ax.text(x, y + 0.32, f"[{win[0]},{win[1]}]", ha="center", va="center", fontsize=5.5, color=C_WALL_TEXT)

    sx, sy = cell_center(inst.start)
    gx, gy = cell_center(inst.goal)
    ax.scatter([sx], [sy], marker="o", s=90, color=C_START_GOAL, zorder=4)
    ax.scatter([gx], [gy], marker="*", s=170, color=C_START_GOAL, zorder=4)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return inst.deadline


def panel_trmgrid(ax, instance_name: str):
    inst, _ctor = _build_instance("trmgrid", instance_name)
    n = inst.grid_n

    for r in range(n + 1):
        ax.plot([0, n], [r, r], color=C_GRID, linewidth=0.6, zorder=1)
    for c in range(n + 1):
        ax.plot([c, c], [0, n], color=C_GRID, linewidth=0.6, zorder=1)

    def cell_center(pos):
        r, c = pos
        return c + 0.5, r + 0.5

    for i, sg in enumerate(inst.sub_goals):
        x, y = cell_center(sg)
        ax.scatter([x], [y], marker="s", s=95, color=C_SWITCH, zorder=3)
        ax.text(x, y - 0.30, f"t{i}", ha="center", va="center", fontsize=6.5, color=C_WALL_TEXT)

    sx, sy = cell_center(inst.start)
    gx, gy = cell_center(inst.goal_pos)
    ax.scatter([sx], [sy], marker="o", s=90, color=C_START_GOAL, zorder=4)
    ax.scatter([gx], [gy], marker="*", s=170, color=C_START_GOAL, zorder=4)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Compact annotation: caps only (the 6 ordered-pair windows would
    # crowd a 4x4 panel this small -- omitted per the brief's own
    # "can go in a compact annotation or be omitted if they crowd" option).
    # Placed along the bottom edge, checked against this instance's actual
    # sub-goal/start/goal layout to confirm nothing sits there (unlike
    # gridtimed's deadline note, which collided with a gate for T2 and was
    # moved into the title instead -- title placement is layout-
    # independent, so used here too for the same robustness).
    cap_str = ", ".join(f"cap{i}={c}" for i, c in enumerate(inst.caps))
    return inst.deadline, cap_str


def panel_modesel(ax, instance_name: str):
    inst, _ctor = _build_instance("modesel", instance_name)
    n_jobs = len(inst.jobs)
    n_tasks = len(inst.jobs[0].tasks)

    box_w, box_h = 1.0, 0.6
    gap_x, gap_y = 0.35, 0.55
    row_h = box_h + gap_y

    for j, job in enumerate(inst.jobs):
        y = (n_jobs - 1 - j) * row_h
        ax.text(-0.5, y + box_h / 2, job.job_id, ha="right", va="center", fontsize=7.5, color=C_WALL_TEXT)
        for t, task in enumerate(job.tasks):
            x = t * (box_w + gap_x)
            ax.add_patch(Rectangle((x, y), box_w, box_h, facecolor=C_TASK_BOX, edgecolor=C_TASK_EDGE, linewidth=0.8, zorder=2))
            ax.text(
                x + box_w / 2, y + box_h * 0.62, task.machine, ha="center", va="center",
                fontsize=6.5, color=C_WALL_TEXT, zorder=3,
            )
            ax.text(
                x + box_w / 2, y + box_h * 0.25, f"{task.d_fast}/{task.d_slow}", ha="center", va="center",
                fontsize=6, color=C_WALL_TEXT, zorder=3,
            )
            if t < n_tasks - 1:
                arrow = FancyArrowPatch(
                    (x + box_w, y + box_h / 2), (x + box_w + gap_x, y + box_h / 2),
                    arrowstyle="-|>", mutation_scale=6, color=C_TASK_EDGE, linewidth=0.7, zorder=2,
                )
                ax.add_patch(arrow)
        end_x = n_tasks * (box_w + gap_x) - gap_x
        ax.text(end_x + 0.15, y + box_h / 2, f"D={job.deadline}", ha="left", va="center", fontsize=6.5, color=C_WALL_TEXT)

    ax.set_xlim(-0.9, n_tasks * (box_w + gap_x) + 0.9)
    ax.set_ylim(-0.2, n_jobs * row_h)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures/part1_instances"))
    parser.add_argument("--width", type=float, default=5.5)
    parser.add_argument("--height", type=float, default=2.1)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titlesize": 9,
    })

    fig, axes = plt.subplots(1, 3, figsize=(args.width, args.height))

    deadline_a = panel_gridtimed(axes[0], "T2")
    axes[0].set_title(f"(a) GridTimed T2 (D={deadline_a})")

    deadline_b, cap_str = panel_trmgrid(axes[1], "G_SMALL")
    axes[1].set_title(f"(b) TRMGrid G-small (D={deadline_b})")
    axes[1].text(
        0.5, -0.06, cap_str, transform=axes[1].transAxes,
        fontsize=6, ha="center", va="top", color=C_WALL_TEXT,
    )

    panel_modesel(axes[2], "M3")
    axes[2].set_title("(c) ModeSel M3")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=C_START_GOAL, label="start", markersize=6),
        Line2D([0], [0], marker="*", linestyle="", color=C_START_GOAL, label="goal", markersize=9),
        Line2D([0], [0], marker="s", linestyle="", color=C_SWITCH, label="switch / sub-goal", markersize=6.5),
        Line2D([0], [0], marker="D", linestyle="", color=C_GATE, label="gate", markersize=6),
    ]
    fig.subplots_adjust(bottom=0.30, top=0.86, left=0.06, right=0.98, wspace=0.28)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out.with_suffix(".pdf")
    png_path = args.out.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    print(f"Wrote {pdf_path}, {png_path}")


if __name__ == "__main__":
    main()
