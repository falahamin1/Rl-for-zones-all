"""
run_sg_comparison.py
--------------------
Single-Goal TRM Frozen Lake — zone-graph vs region Q-learning.

Guard: c_min <= y <= D  (time-window guard, single clock y)
  c_min : goal activates after c_min elapsed steps (lower bound)
  D     : global deadline — goal expires after D steps (upper bound)

Zone graph nodes (3 clock-state values, constant for all instances):
  y_node=0: y < c_min          (pre-window: too early)
  y_node=1: c_min <= y <= D    (in window: guard satisfied)
  y_node=2: y > D              (post-window: deadline exceeded)

Region clock states: D+1  (min(y, D) ∈ {0,...,D})  — grows linearly with D.

Instances fix c_min=10, vary D: 30, 60, 100, 200, 400.

Outputs
-------
  output/sg/SG_0X_convergence.png  — per-instance learning curve
  output/sg/sg_summary.csv         — full results table
  output/sg/sg_summary.png         — summary table figure
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from trm_comparison.sg_trm import INSTANCES, SGTRMInstance, SG_U_TERMINAL, SG_U1
from trm_comparison.sg_grid import ACTIONS, _DELTAS, HOLES, CHECKPOINTS, START, ROWS, COLS, GOAL_C
from trm_comparison.sg_zone_agent import SGZoneAgent
from trm_comparison.sg_region_agent import SGRegionAgent

# ─── Config ──────────────────────────────────────────────────────────────────

SEEDS         = [42, 123, 456, 789, 1337]
N_SEEDS       = len(SEEDS)
SLIP_PROB     = 0.2
EVAL_EPS      = 100
TRAIN_EPS     = 50_000
EVAL_INTERVAL = 2_000
EVAL_INTERIM  = 30
ALPHA         = 0.1
GAMMA         = 1.0
EPS_START     = 0.9
EPS_MIN       = 0.05
BFS_CAP       = 500_000

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output", "sg")

INSTANCE_NAMES = ["SG_01", "SG_02", "SG_03", "SG_04", "SG_05"]


# ─── BFS helpers ─────────────────────────────────────────────────────────────

def _all_reachable_positions(r: int, c: int, action: int) -> List[Tuple[int, int]]:
    """All grid cells reachable from (r,c) via action (including slip outcomes)."""
    result = set()
    dr, dc = _DELTAS[action]
    nr, nc = max(0, min(ROWS-1, r+dr)), max(0, min(COLS-1, c+dc))
    result.add((nr, nc))
    perp = [2, 3] if action in [0, 1] else [0, 1]
    for pa in perp:
        pdr, pdc = _DELTAS[pa]
        pnr, pnc = max(0, min(ROWS-1, r+pdr)), max(0, min(COLS-1, c+pdc))
        result.add((pnr, pnc))
    return list(result)


# ─── BFS — Zone graph ─────────────────────────────────────────────────────────

def count_zone_graph(trm: SGTRMInstance, max_states: int = BFS_CAP):
    """BFS over (pos, rm, y_node) zone-graph states.

    y_node = 0 if y < c_min
             1 if c_min <= y <= D
             2 if y > D

    Each zone key is explored from corner y values to ensure all successor
    zone keys are reachable.
    """
    def y_node(y: int) -> int:
        if y < trm.c_min:
            return 0
        elif y <= trm.D:
            return 1
        else:
            return 2

    def zone_key(pos, rm, y):
        return (pos, rm, y_node(y))

    def zone_corners(yn: int) -> List[int]:
        """Representative y values covering all transitions out of zone yn."""
        if yn == 0:
            return [0, max(0, trm.c_min - 1)]
        elif yn == 1:
            return [trm.c_min, trm.D]
        else:
            return [trm.D + 1]

    initial_key = zone_key(START, SG_U1, 0)
    visited_keys: set = {initial_key}
    queue = deque([initial_key])

    while queue and len(visited_keys) < max_states:
        zk = queue.popleft()
        pos, rm, yn = zk
        if rm == SG_U_TERMINAL:
            continue

        r, c = pos
        for y_c in zone_corners(yn):
            y_new = y_c + 1   # dt=1 per step (d_max=0)

            for action in ACTIONS:
                for npos in _all_reachable_positions(r, c, action):
                    if npos in HOLES:
                        nkey = (npos, SG_U_TERMINAL, 0)
                        if nkey not in visited_keys:
                            visited_keys.add(nkey)
                        continue

                    event = CHECKPOINTS.get(npos, "")
                    if event == "c":
                        guard_ok = trm.guard_satisfied(y_new)
                        nkey = (npos, SG_U_TERMINAL, y_node(y_new))
                        if nkey not in visited_keys:
                            visited_keys.add(nkey)
                    else:
                        nkey = zone_key(npos, rm, y_new)
                        if nkey not in visited_keys:
                            visited_keys.add(nkey)
                            queue.append(nkey)

    truncated = len(visited_keys) >= max_states
    return len(visited_keys), truncated


# ─── BFS — Region graph ───────────────────────────────────────────────────────

def count_region_graph(trm: SGTRMInstance, max_states: int = BFS_CAP):
    """BFS over (pos, rm, min(y, D)) region states.

    Tracks exact y for BFS transitions; state key caps at D.
    """
    initial_key = (START, SG_U1, 0)
    visited_keys: set = {initial_key}
    queue = deque([(START, SG_U1, 0)])   # (pos, rm, y_exact)

    while queue and len(visited_keys) < max_states:
        pos, rm, y_e = queue.popleft()
        if rm == SG_U_TERMINAL:
            continue

        y_new = y_e + 1   # dt=1 per step

        r, c = pos
        for action in ACTIONS:
            for npos in _all_reachable_positions(r, c, action):
                if npos in HOLES:
                    nkey = (npos, SG_U_TERMINAL, 0)
                    if nkey not in visited_keys:
                        visited_keys.add(nkey)
                    continue

                event = CHECKPOINTS.get(npos, "")
                if event == "c":
                    nkey = (npos, SG_U_TERMINAL, min(y_new, trm.D))
                    if nkey not in visited_keys:
                        visited_keys.add(nkey)
                else:
                    y_cap = min(y_new, trm.D)
                    nkey = (npos, rm, y_cap)
                    if nkey not in visited_keys:
                        visited_keys.add(nkey)
                        if y_new <= trm.D:
                            queue.append((npos, rm, y_new))
                        # once y_new > D, all further steps are still capped at D

    truncated = len(visited_keys) >= max_states
    return len(visited_keys), truncated


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_convergence(name, train_eps, zone_cps, region_cps, trm, save_path,
                     zone_stds=None, region_stds=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    if zone_cps:
        zx, zy = list(zip(*zone_cps))
        ax.plot(zx, zy, color="steelblue", linewidth=2.5, marker="o",
                markersize=4, label="Zone RL (3 clock nodes)")
        if zone_stds:
            _, zs = zip(*zone_stds)
            zs = np.array(zs)
            ax.fill_between(zx, np.array(zy) - zs, np.array(zy) + zs,
                            alpha=0.18, color="steelblue")

    if region_cps:
        rx, ry = list(zip(*region_cps))
        ax.plot(rx, ry, color="darkorange", linewidth=2.5, marker="s",
                markersize=4, label=f"Region RL ({trm.D+1} clock nodes)")
        if region_stds:
            _, rs = zip(*region_stds)
            rs = np.array(rs)
            ax.fill_between(rx, np.array(ry) - rs, np.array(ry) + rs,
                            alpha=0.18, color="darkorange")

    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Mean reward (greedy, {EVAL_INTERIM} eps, avg {N_SEEDS} seeds)")
    ax.set_title(
        f"{name} — Zone RL vs Region RL  ({train_eps:,} episodes × {N_SEEDS} seeds)\n"
        f"Guard: {trm.c_min} ≤ y ≤ {trm.D}"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def _fmt_bfs(n: int, trunc: bool) -> str:
    prefix = ">" if trunc else ""
    return f"{prefix}{n:,}"


def plot_summary_table(rows: list, save_path: str):
    headers = [
        "Instance",
        "c_min / D\n(guard)",
        "Region Clock\nNodes (D+1)",
        "Zone RL\nmean ± std",
        "Zone\nVisited States",
        "Region RL\nmean ± std",
        "Region\nVisited States",
        "Zone BFS\n(graph nodes)",
        "Region BFS\n(graph nodes)",
    ]

    cell_rows = []
    for r in rows:
        cell_rows.append([
            r["instance"],
            f"{r['c_min']} / {r['D']}",
            f"{r['D']+1}",
            f"{r['zone_mean']:.1f} ± {r['zone_std']:.1f}",
            f"{r['zone_visited']:,}",
            f"{r['region_mean']:.1f} ± {r['region_std']:.1f}",
            f"{r['region_visited']:,}",
            _fmt_bfs(r["zone_bfs"], r["zone_bfs_trunc"]),
            _fmt_bfs(r["region_bfs"], r["region_bfs_trunc"]),
        ])

    n_rows = len(cell_rows)
    fig_h  = max(3.5, n_rows * 0.7 + 1.8)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_rows, colLabels=headers,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(col=list(range(len(headers))))

    for ci in range(len(headers)):
        tbl[0, ci].set_facecolor("#e0e0e0")
        tbl[0, ci].set_text_props(weight="bold")
    for ri in range(1, n_rows + 1):
        for ci in [3, 4, 7]:
            tbl[ri, ci].set_facecolor("#ddeeff")
        for ci in [5, 6, 8]:
            tbl[ri, ci].set_facecolor("#fff0dd")
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_height(0.07)

    ax.set_title(
        f"Zone RL vs Region RL — Single-Goal TRM Frozen Lake  (slip=0.2, d_max=0)\n"
        f"Guard: c_min ≤ y ≤ D  |  Zone always has 3 clock nodes  |  "
        f"Training: {TRAIN_EPS:,} episodes  |  Final eval: {EVAL_EPS} eps  |  {N_SEEDS} seeds",
        fontsize=9, pad=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary table saved: {save_path}")


# ─── Multi-seed training ──────────────────────────────────────────────────────

def _train_multi_seed(AgentClass, trm, **agent_kwargs) -> dict:
    """Train N_SEEDS independent agents; return averaged checkpoint curves and final stats."""
    all_cps, final_means = [], []
    n_states_last = 0
    for seed in SEEDS:
        agent = AgentClass(trm, seed=seed, **agent_kwargs)
        res = agent.train(
            episodes=TRAIN_EPS,
            eval_interval=EVAL_INTERVAL,
            eval_episodes=EVAL_INTERIM,
            eval_seed=seed + 1000,
        )
        ev = agent.evaluate(n_episodes=EVAL_EPS, seed=seed + 2000)
        all_cps.append(res["checkpoint_evals"])
        final_means.append(ev["mean"])
        n_states_last = agent.n_states()

    episodes = [ep for ep, _ in all_cps[0]]
    matrix = np.array([[m for _, m in cps] for cps in all_cps])
    arr = np.array(final_means)
    return {
        "checkpoint_evals": list(zip(episodes, matrix.mean(axis=0).tolist())),
        "checkpoint_stds":  list(zip(episodes, matrix.std(axis=0).tolist())),
        "final_mean": float(arr.mean()),
        "final_std":  float(arr.std()),
        "n_states":   n_states_last,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []

    print("=" * 70)
    print("Zone RL vs Region RL — Single-Goal TRM Frozen Lake")
    print(f"Guard: c_min <= y <= D  |  d_max=0  slip={SLIP_PROB}")
    print(f"α={ALPHA}  γ={GAMMA}  ε: {EPS_START}→{EPS_MIN}")
    print(f"training: {TRAIN_EPS:,}  |  eval: {EVAL_EPS}")
    print("Zone clock nodes: 3 (constant)  |  Region clock nodes: D+1 (grows)")
    print("=" * 70 + "\n")

    for inst_name in INSTANCE_NAMES:
        trm = INSTANCES[inst_name]
        region_nodes = trm.D + 1

        print(f"=== {inst_name}: c_min={trm.c_min}, D={trm.D}  "
              f"(region clock nodes: {region_nodes}, zone clock nodes: 3) ===")

        # ── Zone RL ───────────────────────────────────────────────────────
        t0 = time.time()
        zone_ms = _train_multi_seed(
            SGZoneAgent, trm,
            alpha=ALPHA, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_min=EPS_MIN,
            slip_prob=SLIP_PROB,
        )
        print(f"  [Zone RL]   {time.time()-t0:.1f}s  "
              f"mean={zone_ms['final_mean']:.1f}±{zone_ms['final_std']:.1f}  "
              f"visited={zone_ms['n_states']:,}")

        # ── Region RL ─────────────────────────────────────────────────────
        t0 = time.time()
        region_ms = _train_multi_seed(
            SGRegionAgent, trm,
            alpha=ALPHA, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_min=EPS_MIN,
            slip_prob=SLIP_PROB,
        )
        print(f"  [Region RL] {time.time()-t0:.1f}s  "
              f"mean={region_ms['final_mean']:.1f}±{region_ms['final_std']:.1f}  "
              f"visited={region_ms['n_states']:,}")

        # ── BFS graph sizes ────────────────────────────────────────────────
        t0 = time.time()
        zbfs, ztrunc = count_zone_graph(trm, max_states=BFS_CAP)
        print(f"  BFS zone:   {zbfs:,}{'+ (cap)' if ztrunc else ''}  "
              f"({time.time()-t0:.3f}s)")

        t0 = time.time()
        rbfs, rtrunc = count_region_graph(trm, max_states=BFS_CAP)
        print(f"  BFS region: {rbfs:,}{'+ (cap)' if rtrunc else ''}  "
              f"({time.time()-t0:.3f}s)\n")

        # ── Convergence plot ───────────────────────────────────────────────
        plot_convergence(
            inst_name, TRAIN_EPS,
            zone_cps=zone_ms["checkpoint_evals"],
            region_cps=region_ms["checkpoint_evals"],
            zone_stds=zone_ms["checkpoint_stds"],
            region_stds=region_ms["checkpoint_stds"],
            trm=trm,
            save_path=os.path.join(OUTPUT_DIR, f"{inst_name}_convergence.png"),
        )

        rows.append({
            "instance":         inst_name,
            "c_min":            trm.c_min,
            "D":                trm.D,
            "region_nodes":     region_nodes,
            "zone_mean":        round(zone_ms["final_mean"], 2),
            "zone_std":         round(zone_ms["final_std"],  2),
            "zone_visited":     zone_ms["n_states"],
            "region_mean":      round(region_ms["final_mean"], 2),
            "region_std":       round(region_ms["final_std"],  2),
            "region_visited":   region_ms["n_states"],
            "zone_bfs":         zbfs,
            "zone_bfs_trunc":   ztrunc,
            "region_bfs":       rbfs,
            "region_bfs_trunc": rtrunc,
        })

    # ── Summary CSV ────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "sg_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved: {csv_path}")

    # ── Summary table figure ───────────────────────────────────────────────
    plot_summary_table(rows, save_path=os.path.join(OUTPUT_DIR, "sg_summary.png"))

    # ── Console summary ────────────────────────────────────────────────────
    print("\nResults:")
    print(f"{'Instance':<10} {'c/D':>8}  {'RegNodes':>9}  "
          f"{'ZoneRL':>8}  {'ZoneVis':>9}  "
          f"{'RegRL':>8}  {'RegVis':>9}  "
          f"{'ZoneBFS':>9}  {'RegBFS':>9}")
    print("-" * 98)
    for r in rows:
        print(
            f"{r['instance']:<10} "
            f"{r['c_min']}/{r['D']:>3}  "
            f"{r['region_nodes']:>9,}  "
            f"{r['zone_mean']:>8.1f}  "
            f"{r['zone_visited']:>9,}  "
            f"{r['region_mean']:>8.1f}  "
            f"{r['region_visited']:>9,}  "
            f"{_fmt_bfs(r['zone_bfs'], r['zone_bfs_trunc']):>9}  "
            f"{_fmt_bfs(r['region_bfs'], r['region_bfs_trunc']):>9}"
        )


if __name__ == "__main__":
    main()
