"""Zone RL vs Region RL — Hole-Disable TRM Frozen Lake

Holes are non-terminal: the agent bounces back and its intended direction is
disabled for x_h steps, tracked by clock h.

Three clocks (x, y, h) → 8 constant zone clock nodes regardless of c_min, D, x_h.
Region clock triples: (c_min+1) × (D+1) × (x_h+1) — grows with x_h.

Instances fix c_min=10, D=200, vary x_h: 5, 10, 20, 30, 50.
"""
import os
import time
from collections import deque
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trm_comparison.hole_grid import (
    ACTIONS, _DELTAS, HOLES, CHECKPOINTS, START, ROWS, COLS,
    UP, DOWN, LEFT, RIGHT,
)
from trm_comparison.hole_trm import (
    INSTANCES, HoleTRMInstance,
    U_TERMINAL, U1, U2, U3, NONTERMINAL_STATES, RM_GOAL_EVENTS,
)
from trm_comparison.hole_zone_agent import HoleZoneAgent
from trm_comparison.hole_region_agent import HoleRegionAgent

# ─── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR    = "output/hole"
SEEDS         = [42, 123, 456, 789, 1337]
N_SEEDS       = len(SEEDS)
TRAIN_EPS     = 50_000
EVAL_EPS      = 100
EVAL_INTERVAL = 2_000
EVAL_INTERIM  = 30
ALPHA         = 0.1
GAMMA         = 1.0
EPS_START     = 0.9
EPS_MIN       = 0.05
SLIP_PROB     = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── BFS helpers ─────────────────────────────────────────────────────────────

def _clamp(r: int, c: int) -> Tuple[int, int]:
    return max(0, min(ROWS - 1, r)), max(0, min(COLS - 1, c))


def _all_reachable(r: int, c: int, action: int):
    """All grid positions reachable from (r,c) with intended action (with slip)."""
    results = set()
    perp = [LEFT, RIGHT] if action in (UP, DOWN) else [UP, DOWN]
    for act in [action] + perp:
        dr, dc = _DELTAS[act]
        results.add((_clamp(r + dr, c + dc), act))
    return results  # set of (new_pos, actual_dir)


def count_zone_graph(trm: HoleTRMInstance) -> int:
    """BFS over zone keys: (pos, rm_u, x_node, y_node, h_node, disabled_action)."""
    def zone_key(pos, rm_u, x, y, h, da):
        xn = 0 if x < trm.c_min else 1
        yn = 0 if y <= trm.D    else 1
        hn = 0 if h < trm.x_h   else 1
        d  = da if hn == 0 else -1
        return (pos, rm_u, xn, yn, hn, d)

    def corners(xn, yn, hn):
        xs = [0, trm.c_min - 1] if xn == 0 else [trm.c_min]
        ys = [0, trm.D]         if yn == 0 else [trm.D + 1]
        hs = [0, trm.x_h - 1]  if hn == 0 else [trm.x_h]
        return [(x, y, h) for x in xs for y in ys for h in hs]

    init = zone_key(START, U1, 0, 0, trm.x_h, -1)
    visited = {init}
    queue = deque([init])

    while queue:
        key = queue.popleft()
        pos, rm_u, xn, yn, hn, da = key

        if rm_u == U_TERMINAL:
            continue

        for x, y, h in corners(xn, yn, hn):
            # Determine available intended directions
            avail = [a for a in ACTIONS if not (hn == 0 and a == da)]
            for intended in avail:
                for (npos, actual_dir) in _all_reachable(*pos, intended):
                    # Case 1: hole hit — bounce back, cooldown resets
                    if npos in HOLES:
                        nx, ny, nh, nda = x + 1, y + 1, 0, intended
                        nk = zone_key(pos, rm_u, nx, ny, nh, nda)
                        if nk not in visited:
                            visited.add(nk)
                            queue.append(nk)
                        continue

                    # Case 2: normal move
                    nx, ny = x + 1, y + 1
                    nh = h + 1
                    new_da = -1 if nh >= trm.x_h else da

                    event = CHECKPOINTS.get(npos, "")
                    goal_ev = RM_GOAL_EVENTS.get(rm_u)

                    if event == goal_ev:
                        if trm.guard_satisfied(nx, ny):
                            # Successful goal
                            nrm  = U_TERMINAL if rm_u == U3 else rm_u + 1
                            nnx  = 0          # x resets
                            nk   = zone_key(npos, nrm, nnx, ny, nh, new_da)
                        else:
                            nk = zone_key(npos, U_TERMINAL, nx, ny, nh, new_da)
                    else:
                        nk = zone_key(npos, rm_u, nx, ny, nh, new_da)

                    if nk not in visited:
                        visited.add(nk)
                        queue.append(nk)

    return len(visited)


def count_region_graph(trm: HoleTRMInstance) -> int:
    """BFS over region keys: (pos, rm_u, x_cap, y_cap, h_cap, disabled_action)."""
    def rkey(pos, rm_u, x, y, h, da):
        xc = min(x, trm.c_min)
        yc = min(y, trm.D)
        hc = min(h, trm.x_h)
        d  = da if hc < trm.x_h else -1
        return (pos, rm_u, xc, yc, hc, d)

    init = rkey(START, U1, 0, 0, trm.x_h, -1)
    visited = {init}
    queue = deque([init])

    while queue:
        key = queue.popleft()
        pos, rm_u, xc, yc, hc, da = key

        if rm_u == U_TERMINAL:
            continue

        x, y, h = xc, yc, hc   # exact values (capped = exact for region)
        avail = [a for a in ACTIONS if not (hc < trm.x_h and a == da)]

        for intended in avail:
            for (npos, _) in _all_reachable(*pos, intended):
                if npos in HOLES:
                    nx, ny, nh, nda = x + 1, y + 1, 0, intended
                    nk = rkey(pos, rm_u, nx, ny, nh, nda)
                    if nk not in visited:
                        visited.add(nk)
                        queue.append(nk)
                    continue

                nx, ny = x + 1, y + 1
                nh = h + 1
                new_da = -1 if nh >= trm.x_h else da

                event   = CHECKPOINTS.get(npos, "")
                goal_ev = RM_GOAL_EVENTS.get(rm_u)

                if event == goal_ev:
                    if trm.guard_satisfied(nx, ny):
                        nrm = U_TERMINAL if rm_u == U3 else rm_u + 1
                        nk  = rkey(npos, nrm, 0, ny, nh, new_da)
                    else:
                        nk = rkey(npos, U_TERMINAL, nx, ny, nh, new_da)
                else:
                    nk = rkey(npos, rm_u, nx, ny, nh, new_da)

                if nk not in visited:
                    visited.add(nk)
                    queue.append(nk)

    return len(visited)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_convergence(
    name: str,
    train_eps: int,
    zone_cps: List[Tuple[int, float]],
    region_cps: List[Tuple[int, float]],
    trm: HoleTRMInstance,
    save_path: str,
    zone_stds: List[Tuple[int, float]] = None,
    region_stds: List[Tuple[int, float]] = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    eps_z, mu_z = zip(*zone_cps)
    eps_r, mu_r = zip(*region_cps)

    ax.plot(eps_z, mu_z, label="Zone RL",   color="steelblue",  linewidth=1.8)
    ax.plot(eps_r, mu_r, label="Region RL", color="darkorange", linewidth=1.8)

    if zone_stds:
        _, sd_z = zip(*zone_stds)
        az, sz = np.array(mu_z), np.array(sd_z)
        ax.fill_between(eps_z, az - sz, az + sz, alpha=0.18, color="steelblue")
    if region_stds:
        _, sd_r = zip(*region_stds)
        ar, sr = np.array(mu_r), np.array(sd_r)
        ax.fill_between(eps_r, ar - sr, ar + sr, alpha=0.18, color="darkorange")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Mean reward (greedy, {EVAL_INTERIM} eps, avg {N_SEEDS} seeds)")
    ax.set_title(
        f"{name} — Zone RL vs Region RL  ({train_eps:,} episodes × {N_SEEDS} seeds)\n"
        f"Guard: x ≥ {trm.c_min} AND y ≤ {trm.D}  |  hole cooldown x_h={trm.x_h}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_summary_table(rows: list, save_path: str):
    cols = ["Instance", "x_h", "Region\nbound",
            "Zone RL\nmean±std", "Region RL\nmean±std",
            "Zone\nvisited", "Region\nvisited",
            "Zone\nBFS", "Region\nBFS", "Ratio"]
    table_data = []
    for r in rows:
        ratio = r["region_bfs"] / r["zone_bfs"] if r["zone_bfs"] > 0 else float("nan")
        table_data.append([
            r["instance"],
            r["x_h"],
            f"{r['region_bound']:,}",
            f"{r['zone_mean']:+.1f}±{r['zone_std']:.1f}",
            f"{r['region_mean']:+.1f}±{r['region_std']:.1f}",
            f"{r['zone_visited']:,}",
            f"{r['region_visited']:,}",
            f"{r['zone_bfs']:,}{'†' if r['zone_bfs_trunc'] else ''}",
            f"{r['region_bfs']:,}{'†' if r['region_bfs_trunc'] else ''}",
            f"{ratio:.1f}×",
        ])

    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.65 + 1.6))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=cols,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(cols))))

    for ci in range(len(cols)):
        tbl[0, ci].set_facecolor("#2c5f8a")
        tbl[0, ci].set_text_props(color="white", fontweight="bold")
    for ri in range(1, len(rows) + 1):
        bg = "#f5f9ff" if ri % 2 == 0 else "white"
        for ci in range(len(cols)):
            tbl[ri, ci].set_facecolor(bg)
        for ci in [3, 6]:
            tbl[ri, ci].set_facecolor("#ddeeff")
        for ci in [4, 7]:
            tbl[ri, ci].set_facecolor("#fff0dd")
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_height(0.065)

    ax.set_title(
        f"Zone RL vs Region RL — Hole-Disable TRM Frozen Lake  (slip=0.2, d_max=0)\n"
        f"Guard: x ≥ c_min AND y ≤ D  |  Zone always has 8 clock nodes  |  "
        f"Training: {TRAIN_EPS:,} episodes  |  Final eval: {EVAL_EPS} eps  |  {N_SEEDS} seeds",
        fontsize=9, pad=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary table saved: {save_path}")


# ─── Multi-seed training ──────────────────────────────────────────────────────

def _train_multi_seed(AgentClass, trm, **agent_kwargs) -> dict:
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
    matrix   = np.array([[m for _, m in cps] for cps in all_cps])
    arr      = np.array(final_means)
    return {
        "checkpoint_evals": list(zip(episodes, matrix.mean(axis=0).tolist())),
        "checkpoint_stds":  list(zip(episodes, matrix.std(axis=0).tolist())),
        "final_mean": float(arr.mean()),
        "final_std":  float(arr.std()),
        "n_states":   n_states_last,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

BFS_CAP = 2_000_000


def main():
    print("=" * 70)
    print("Zone RL vs Region RL — Hole-Disable TRM Frozen Lake")
    print("Guard: x >= c_min AND y <= D  |  hole cooldown clock h")
    print(f"α={ALPHA}  γ={GAMMA}  ε: {EPS_START}→{EPS_MIN}")
    print(f"training: {TRAIN_EPS:,}  |  eval: {EVAL_EPS}  |  seeds: {SEEDS}")
    print(f"Zone clock nodes: 8 (constant)  |  Region: (c_min+1)×(D+1)×(x_h+1) (grows)")
    print("=" * 70)
    print()

    rows = []
    for inst_name, trm in INSTANCES.items():
        print(f"=== {inst_name}: c_min={trm.c_min}, D={trm.D}, x_h={trm.x_h}"
              f"  (region bound: {trm.region_bound:,}, zone clock nodes: 8) ===")

        agent_kwargs = dict(
            alpha=ALPHA, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_min=EPS_MIN,
            slip_prob=SLIP_PROB,
        )

        t0 = time.time()
        zr = _train_multi_seed(HoleZoneAgent, trm, **agent_kwargs)
        zt = time.time() - t0

        t0 = time.time()
        rr = _train_multi_seed(HoleRegionAgent, trm, **agent_kwargs)
        rt = time.time() - t0

        print(f"  [Zone RL]   {zt:.1f}s  mean={zr['final_mean']:+.1f}±{zr['final_std']:.1f}"
              f"  visited={zr['n_states']:,}")
        print(f"  [Region RL] {rt:.1f}s  mean={rr['final_mean']:+.1f}±{rr['final_std']:.1f}"
              f"  visited={rr['n_states']:,}")

        t0 = time.time()
        zbfs_raw = count_zone_graph(trm)
        zbfs_t = time.time() - t0
        zbfs = min(zbfs_raw, BFS_CAP)
        zbfs_trunc = zbfs_raw > BFS_CAP

        t0 = time.time()
        rbfs_raw = count_region_graph(trm)
        rbfs_t = time.time() - t0
        rbfs = min(rbfs_raw, BFS_CAP)
        rbfs_trunc = rbfs_raw > BFS_CAP

        print(f"  BFS zone:   {zbfs:,}{'†' if zbfs_trunc else ''}  ({zbfs_t:.3f}s)")
        print(f"  BFS region: {rbfs:,}{'†' if rbfs_trunc else ''}  ({rbfs_t:.3f}s)")
        print()

        plot_convergence(
            inst_name, TRAIN_EPS,
            zr["checkpoint_evals"], rr["checkpoint_evals"],
            trm,
            save_path=os.path.join(OUTPUT_DIR, f"{inst_name}_convergence.png"),
            zone_stds=zr["checkpoint_stds"],
            region_stds=rr["checkpoint_stds"],
        )

        rows.append({
            "instance":        inst_name,
            "c_min":           trm.c_min,
            "D":               trm.D,
            "x_h":             trm.x_h,
            "region_bound":    trm.region_bound,
            "zone_mean":       zr["final_mean"],
            "zone_std":        zr["final_std"],
            "zone_visited":    zr["n_states"],
            "region_mean":     rr["final_mean"],
            "region_std":      rr["final_std"],
            "region_visited":  rr["n_states"],
            "zone_bfs":        zbfs,
            "zone_bfs_trunc":  zbfs_trunc,
            "region_bfs":      rbfs,
            "region_bfs_trunc": rbfs_trunc,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "hole_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved: {csv_path}")

    plot_summary_table(rows, save_path=os.path.join(OUTPUT_DIR, "hole_summary.png"))

    # Print compact summary
    print()
    print("=" * 70)
    print(f"{'Instance':12s}  {'x_h':>5}  {'ZoneRL':>8}  {'RegionRL':>10}  "
          f"{'ZoneBFS':>8}  {'RegionBFS':>10}  {'Ratio':>7}")
    print("-" * 70)
    for r in rows:
        ratio = r["region_bfs"] / r["zone_bfs"] if r["zone_bfs"] > 0 else float("nan")
        print(f"{r['instance']:12s}  {r['x_h']:>5}  "
              f"{r['zone_mean']:>+8.1f}  {r['region_mean']:>+10.1f}  "
              f"{r['zone_bfs']:>8,}  {r['region_bfs']:>10,}  {ratio:>6.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
