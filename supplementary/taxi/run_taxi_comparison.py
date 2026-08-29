"""
run_taxi_comparison.py
----------------------
Zone-graph vs Region Q-learning on six Taxi TRM instances.

Guard: x >= c_min  AND  y <= D
  x >= c_min : per-event minimum reaction time (x resets at each successful event)
  y <= D     : global deadline

Zone graph nodes (4 clock-state pairs, constant for all instances):
  x-node: 0 if x < c_min  else 1
  y-node: 0 if y <= D     else 1

Region clock-state pairs: (c_min+1) × (D+1) — grows linearly with c_min.

Task: pickup passenger at Red (0,0), visit Green (0,4), dropoff at Blue (4,3).
Starting state: taxi at (2,2), passenger at Red.
Stochastic movement: 80%/10%/10% (is_rainy). No holes.

Instances fix D=200, d_max=5; c_min varies 2 → 25.

Outputs
-------
  output/taxi/<NAME>_convergence.png  — per-instance learning curve
  output/taxi/taxi_summary.csv        — full results table
  output/taxi/taxi_summary.png        — summary table figure
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from taxi_comparison.taxi_trm import (
    INSTANCES, TaxiTRMInstance, U_TERMINAL, U1, U2, U3, RM_GOAL_EVENTS,
)
from taxi_comparison.taxi_env import (
    TaxiGridEnv, ACTIONS, MOVE_ACTIONS, BLOCKED_EAST, BLOCKED_WEST,
    TAXI_START, RED, GREEN, BLUE, ROWS, COLS,
    S, N, E, W, PICKUP, DROPOFF, _move,
)
from taxi_comparison.taxi_zone_agent import TaxiZoneAgent
from taxi_comparison.taxi_region_agent import TaxiRegionAgent

# ─── Config ──────────────────────────────────────────────────────────────────

SEEDS         = [42, 123, 456, 789, 1337]
N_SEEDS       = len(SEEDS)
SLIP_PROB     = 0.2
EVAL_EPS      = 100
TRAIN_EPS     = 150_000
EVAL_INTERVAL = 5_000
EVAL_INTERIM  = 30
ALPHA         = 0.1
GAMMA         = 1.0
EPS_START     = 0.9
EPS_MIN       = 0.05
BFS_CAP       = 500_000

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output", "taxi")

INSTANCE_NAMES = ["TAXI_01", "TAXI_02", "TAXI_03", "TAXI_04", "TAXI_05", "TAXI_06"]


# ─── Taxi movement helpers ────────────────────────────────────────────────────

def _all_reachable_taxi(r: int, c: int, action: int) -> List[Tuple[int, int]]:
    """All positions reachable from (r,c) via action (accounts for slip and walls).

    Movement actions: intended position + two perpendicular slip positions.
    PICKUP/DROPOFF: taxi stays in place.
    """
    pos = (r, c)
    if action == PICKUP or action == DROPOFF:
        return [pos]

    result = set()
    result.add(_move(pos, action))
    perp = [E, W] if action in (N, S) else [N, S]
    for pa in perp:
        result.add(_move(pos, pa))
    return list(result)


# ─── BFS — Region graph ───────────────────────────────────────────────────────

def count_region_graph(trm: TaxiTRMInstance, max_states: int = BFS_CAP) -> Tuple[int, bool]:
    """BFS over (pos, rm, x_capped, y_capped) states.

    x_capped = min(x, c_min),  y_capped = min(y, D).
    Exact clocks tracked for guard checking; capped pairs used as keys.
    pass_in_taxi is derived from rm state (U1 → False, U2/U3 → True).
    All slip outcomes expanded (conservative reachability).
    """
    initial_key = (TAXI_START, U1, 0, 0)
    visited_keys: set = {initial_key}
    queue = deque([(TAXI_START, U1, 0, 0)])   # (pos, rm, x_exact, y_exact)

    while queue and len(visited_keys) < max_states:
        pos, rm, x_e, y_e = queue.popleft()
        if rm == U_TERMINAL:
            continue

        goal_event = RM_GOAL_EVENTS.get(rm)
        pass_in_taxi = rm in (U2, U3)

        for delay in range(trm.d_max + 1):
            dt    = delay + 1
            x_new = x_e + dt
            y_new = y_e + dt

            r, c = pos
            for action in ACTIONS:
                for npos in _all_reachable_taxi(r, c, action):
                    # Determine event emitted by this (action, npos) combination
                    event = ""
                    if action == PICKUP and pos == RED and not pass_in_taxi:
                        event = "a"   # npos == pos for pickup
                    elif action in MOVE_ACTIONS and npos == GREEN and pass_in_taxi:
                        event = "b"
                    elif action == DROPOFF and pos == BLUE and pass_in_taxi:
                        event = "c"   # npos == pos for dropoff

                    if event == goal_event:
                        guard_ok = (x_new >= trm.c_min) and (y_new <= trm.D)
                        if guard_ok:
                            next_rm = U_TERMINAL if rm == U3 else rm + 1
                            nx_e    = 0   # x resets
                        else:
                            next_rm = U_TERMINAL
                            nx_e    = x_new
                        nx_cap = min(nx_e, trm.c_min)
                        ny_cap = min(y_new, trm.D)
                        nkey   = (npos, next_rm, nx_cap, ny_cap)
                        if nkey not in visited_keys:
                            visited_keys.add(nkey)
                            if next_rm != U_TERMINAL:
                                queue.append((npos, next_rm, nx_e, y_new))
                    else:
                        nx_cap = min(x_new, trm.c_min)
                        ny_cap = min(y_new, trm.D)
                        nkey   = (npos, rm, nx_cap, ny_cap)
                        if nkey not in visited_keys:
                            visited_keys.add(nkey)
                            queue.append((npos, rm, x_new, y_new))

    truncated = len(visited_keys) >= max_states
    return len(visited_keys), truncated


# ─── BFS — Zone graph ─────────────────────────────────────────────────────────

def count_zone_graph(trm: TaxiTRMInstance, max_states: int = BFS_CAP) -> Tuple[int, bool]:
    """BFS over (pos, rm, x_node, y_node) zone-graph states.

    x_node = 0 if x < c_min else 1
    y_node = 0 if y <= D    else 1

    Each zone key is explored from its corner exact values to ensure all
    reachable successor zone keys are discovered.
    """
    def zone_key(pos, rm, x, y):
        return (pos, rm, 0 if x < trm.c_min else 1, 0 if y <= trm.D else 1)

    def zone_corners(xn, yn):
        xs = [0, max(0, trm.c_min - 1)] if xn == 0 else [trm.c_min]
        ys = [0, trm.D]                  if yn == 0 else [trm.D + 1]
        return [(x, y) for x in xs for y in ys]

    initial_key = zone_key(TAXI_START, U1, 0, 0)
    visited_keys: set = {initial_key}
    queue = deque([initial_key])   # queue holds zone keys

    while queue and len(visited_keys) < max_states:
        zk = queue.popleft()
        pos, rm, xn, yn = zk
        if rm == U_TERMINAL:
            continue

        goal_event   = RM_GOAL_EVENTS.get(rm)
        pass_in_taxi = rm in (U2, U3)
        r, c = pos

        for delay in range(trm.d_max + 1):
            for x_c, y_c in zone_corners(xn, yn):
                x_new = x_c + delay + 1
                y_new = y_c + delay + 1

                for action in ACTIONS:
                    for npos in _all_reachable_taxi(r, c, action):
                        # Determine event
                        event = ""
                        if action == PICKUP and pos == RED and not pass_in_taxi:
                            event = "a"
                        elif action in MOVE_ACTIONS and npos == GREEN and pass_in_taxi:
                            event = "b"
                        elif action == DROPOFF and pos == BLUE and pass_in_taxi:
                            event = "c"

                        if event == goal_event:
                            guard_ok = (x_new >= trm.c_min) and (y_new <= trm.D)
                            if guard_ok:
                                next_rm = U_TERMINAL if rm == U3 else rm + 1
                                nkey = zone_key(npos, next_rm, 0, y_new)   # x resets
                            else:
                                next_rm = U_TERMINAL
                                nkey = zone_key(npos, U_TERMINAL, x_new, y_new)
                            if nkey not in visited_keys:
                                visited_keys.add(nkey)
                                if next_rm != U_TERMINAL:
                                    queue.append(nkey)
                        else:
                            nkey = zone_key(npos, rm, x_new, y_new)
                            if nkey not in visited_keys:
                                visited_keys.add(nkey)
                                queue.append(nkey)

    truncated = len(visited_keys) >= max_states
    return len(visited_keys), truncated


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_convergence(name, train_eps, zone_cps, region_cps, save_path,
                     zone_stds=None, region_stds=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    if zone_cps:
        zx, zy = list(zip(*zone_cps))
        ax.plot(zx, zy, color="steelblue", linewidth=2.5, marker="o",
                markersize=4, label="Zone RL (zone-graph)")
        if zone_stds:
            _, zs = zip(*zone_stds)
            zs = np.array(zs)
            ax.fill_between(zx, np.array(zy) - zs, np.array(zy) + zs,
                            alpha=0.18, color="steelblue")

    if region_cps:
        rx, ry = list(zip(*region_cps))
        ax.plot(rx, ry, color="darkorange", linewidth=2.5, marker="s",
                markersize=4, label="Region RL (integer-clock)")
        if region_stds:
            _, rs = zip(*region_stds)
            rs = np.array(rs)
            ax.fill_between(rx, np.array(ry) - rs, np.array(ry) + rs,
                            alpha=0.18, color="darkorange")

    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Mean reward (greedy, {EVAL_INTERIM} eps, avg {N_SEEDS} seeds)")
    ax.set_title(f"{name} — Zone RL vs Region RL  ({train_eps:,} episodes × {N_SEEDS} seeds)")
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
        "c_min / D\n(guards)",
        "Region State\nBound (c_min+1)×(D+1)",
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
            f"{(r['c_min']+1)*(r['D']+1):,}",
            f"{r['zone_mean']:.1f} ± {r['zone_std']:.1f}",
            f"{r['zone_visited']:,}",
            f"{r['region_mean']:.1f} ± {r['region_std']:.1f}",
            f"{r['region_visited']:,}",
            _fmt_bfs(r["zone_bfs"], r["zone_bfs_trunc"]),
            _fmt_bfs(r["region_bfs"], r["region_bfs_trunc"]),
        ])

    n_rows = len(cell_rows)
    fig_h  = max(3.5, n_rows * 0.7 + 1.6)
    fig, ax = plt.subplots(figsize=(16, fig_h))
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
        cell.set_height(0.065)

    ax.set_title(
        f"Zone RL vs Region RL — Taxi TRM (d_max=5, slip=0.2)\n"
        f"Guard: x ≥ c_min AND y ≤ D  |  "
        f"Training: {TRAIN_EPS:,} episodes  |  Final eval: {EVAL_EPS} episodes  |  {N_SEEDS} seeds",
        fontsize=9, pad=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


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
    summary_rows = []

    print("=" * 72)
    print("Taxi TRM — Zone RL vs Region RL")
    print(f"  Guard: x >= c_min AND y <= D  |  d_max=5, slip={SLIP_PROB}")
    print(f"  Train: {TRAIN_EPS:,} eps  |  Eval: {EVAL_EPS} eps  |  Seeds: {SEEDS}")
    print("=" * 72)

    for name in INSTANCE_NAMES:
        trm = INSTANCES[name]
        print(f"\n{'─'*60}")
        print(f"  {name}  c_min={trm.c_min}  D={trm.D}  d_max={trm.d_max}")
        print(f"  Region theoretical pairs: {(trm.c_min+1)*(trm.D+1):,}  |  Zone: 4 (constant)")
        print(f"{'─'*60}")

        # ── Zone RL ──────────────────────────────────────────────────────────
        t0 = time.time()
        zone_ms = _train_multi_seed(
            TaxiZoneAgent, trm,
            alpha=ALPHA, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_min=EPS_MIN,
            slip_prob=SLIP_PROB,
        )
        print(f"  Zone   RL: mean={zone_ms['final_mean']:+.1f}±{zone_ms['final_std']:.1f}"
              f"  states={zone_ms['n_states']:,}  ({time.time()-t0:.1f}s)")

        # ── Region RL ────────────────────────────────────────────────────────
        t0 = time.time()
        region_ms = _train_multi_seed(
            TaxiRegionAgent, trm,
            alpha=ALPHA, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_min=EPS_MIN,
            slip_prob=SLIP_PROB,
        )
        print(f"  Region RL: mean={region_ms['final_mean']:+.1f}±{region_ms['final_std']:.1f}"
              f"  states={region_ms['n_states']:,}  ({time.time()-t0:.1f}s)")

        # ── BFS ──────────────────────────────────────────────────────────────
        zone_bfs, zone_trunc     = count_zone_graph(trm)
        region_bfs, region_trunc = count_region_graph(trm)
        print(f"  Zone BFS: {_fmt_bfs(zone_bfs, zone_trunc)} nodes  |  "
              f"Region BFS: {_fmt_bfs(region_bfs, region_trunc)} nodes")

        # ── Convergence plot ─────────────────────────────────────────────────
        conv_path = os.path.join(OUTPUT_DIR, f"{name}_convergence.png")
        plot_convergence(
            name, TRAIN_EPS,
            zone_ms["checkpoint_evals"],
            region_ms["checkpoint_evals"],
            conv_path,
            zone_stds=zone_ms["checkpoint_stds"],
            region_stds=region_ms["checkpoint_stds"],
        )

        summary_rows.append({
            "instance":       name,
            "c_min":          trm.c_min,
            "D":              trm.D,
            "zone_mean":      zone_ms["final_mean"],
            "zone_std":       zone_ms["final_std"],
            "zone_visited":   zone_ms["n_states"],
            "region_mean":    region_ms["final_mean"],
            "region_std":     region_ms["final_std"],
            "region_visited": region_ms["n_states"],
            "zone_bfs":       zone_bfs,
            "zone_bfs_trunc": zone_trunc,
            "region_bfs":     region_bfs,
            "region_bfs_trunc": region_trunc,
        })

    # ── Summary CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, "taxi_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── Summary table figure ─────────────────────────────────────────────────
    png_path = os.path.join(OUTPUT_DIR, "taxi_summary.png")
    plot_summary_table(summary_rows, png_path)

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Instance':<10} {'c_min':>6} {'ZoneRL':>10} {'RegionRL':>10}"
          f"  {'ZoneBFS':>10} {'RegionBFS':>12}  {'Ratio':>7}")
    print("-" * 72)
    for r in summary_rows:
        ratio = r["region_bfs"] / r["zone_bfs"] if r["zone_bfs"] > 0 else float("inf")
        print(f"{r['instance']:<10} {r['c_min']:>6}  "
              f"{r['zone_mean']:>+9.1f}  {r['region_mean']:>+9.1f}  "
              f"{_fmt_bfs(r['zone_bfs'], r['zone_bfs_trunc']):>10}  "
              f"{_fmt_bfs(r['region_bfs'], r['region_bfs_trunc']):>12}  "
              f"{ratio:>7.1f}x")
    print("=" * 72)


if __name__ == "__main__":
    main()
