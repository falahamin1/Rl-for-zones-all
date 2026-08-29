"""Aggregate /tmp/slice_results/*.json + permutation_ablation.json into
/tmp/SLICE.md.

Pure aggregation/reporting -- no training happens here. Run after
``train.run_experiment`` and ``train.permutation_ablation`` have both
finished.
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

RESULTS_DIR = "/tmp/slice_results"
OUT_PATH = "/tmp/SLICE.md"

INSTANCES = ["PJS_04", "PJS_05", "LJS_01"]
ENCODERS = ["mlp", "deepset", "cnn_dbm", "cnn_proj"]
ENCODER_LABEL = {"mlp": "MLP-Point", "deepset": "DeepSet-H", "cnn_dbm": "CNN-DBM (CNN-M)", "cnn_proj": "CNN-Proj (CNN-P)"}


def load_runs() -> Dict:
    runs = defaultdict(list)  # (instance, encoder) -> list of run records (one per seed)
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        if os.path.basename(path) in ("permutation_ablation.json",):
            continue
        with open(path) as f:
            rec = json.load(f)
        runs[(rec["instance"], rec["encoder"])].append(rec)
    return runs


def load_ablation() -> List[Dict]:
    path = os.path.join(RESULTS_DIR, "permutation_ablation.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _interp_at_steps(history: List[Dict], target_steps: List[int]) -> List[float]:
    xs = [h["env_steps"] for h in history]
    ys = [h["mean_return"] for h in history]
    # forward-fill: value at step s = last logged mean_return with env_steps <= s
    out = []
    j = -1
    last = float("nan")
    for t in target_steps:
        while j + 1 < len(xs) and xs[j + 1] <= t:
            j += 1
            if not math.isnan(ys[j]):
                last = ys[j]
        out.append(last)
    return out


def _interp_at_wallclock(history: List[Dict], target_wall: List[float]) -> List[float]:
    cum_wall = np.cumsum([h["wall_clock_s"] for h in history])
    ys = [h["mean_return"] for h in history]
    out = []
    j = -1
    last = float("nan")
    for t in target_wall:
        while j + 1 < len(cum_wall) and cum_wall[j + 1] <= t:
            j += 1
            if not math.isnan(ys[j]):
                last = ys[j]
        out.append(last)
    return out


def check_anomalies(runs: Dict) -> List[str]:
    flags = []
    for (instance, encoder), recs in runs.items():
        for rec in recs:
            hist = rec["history"]
            if not hist:
                flags.append(f"{instance}/{encoder}/seed{rec['seed']}: zero updates logged")
                continue
            for h in hist:
                if any(math.isnan(h[k]) or math.isinf(h[k]) for k in ("value_loss", "policy_loss", "entropy")):
                    flags.append(f"{instance}/{encoder}/seed{rec['seed']}: NaN/Inf loss at update {h['update']}")
                    break
            final_entropy = hist[-1]["entropy"]
            first_entropy = hist[0]["entropy"]
            if first_entropy > 1e-6 and final_entropy < 0.02 * max(first_entropy, 1e-6):
                flags.append(
                    f"{instance}/{encoder}/seed{rec['seed']}: entropy collapse "
                    f"({first_entropy:.3f} -> {final_entropy:.3f})"
                )
            returns = [h["mean_return"] for h in hist if not math.isnan(h["mean_return"])]
            if len(returns) >= 5:
                first_third = np.mean(returns[: max(1, len(returns) // 3)])
                last_third = np.mean(returns[-max(1, len(returns) // 3) :])
                if last_third <= first_third + 1e-6:
                    flags.append(
                        f"{instance}/{encoder}/seed{rec['seed']}: no improvement over training "
                        f"(first-third mean {first_third:.2f} -> last-third mean {last_third:.2f})"
                    )
    return flags


def render_markdown(runs: Dict, ablation: List[Dict]) -> str:
    lines = []
    lines.append("# SLICE: shared PPO trainer + 4 encoders, preliminary de-risking slice\n")
    lines.append(
        "Scope: `train/env.py` (delta-free `SymbolicJobShopEnv` gym wrapper), `train/ppo.py` "
        "(PPO+GAE), `encoders/{mlp_point,deepset_h,cnn_dbm,cnn_proj}.py`, and the full "
        "3-instance x 4-encoder x 3-seed experiment matrix (36 runs). `prob_jobshop_ext` was not "
        "modified.\n"
    )

    # ---- parameter counts -------------------------------------------------
    lines.append("## Parameter counts (actual, per instance)\n")
    lines.append("| Instance | " + " | ".join(ENCODER_LABEL[e] for e in ENCODERS) + " | max/min ratio |")
    lines.append("|---|" + "---|" * (len(ENCODERS) + 1))
    for instance in INSTANCES:
        counts = []
        for e in ENCODERS:
            recs = runs.get((instance, e), [])
            counts.append(recs[0]["param_count_total"] if recs else None)
        if all(c is not None for c in counts):
            ratio = f"{max(counts) / min(counts):.2f}x"
        else:
            ratio = "n/a"
        row = " | ".join(str(c) if c is not None else "MISSING" for c in counts)
        lines.append(f"| {instance} | {row} | {ratio} |")
    lines.append("")
    lines.append(
        "Per-element action head: only **CNN-DBM** has one (per-clock/per-job rows of `dbm_tensor` "
        "broadcast to that job's task slot(s), since clock==job in this domain). MLP-Point, "
        "DeepSet-H, and CNN-Proj all read all N task logits off a single pooled embedding via a "
        "shared linear layer -- MLP by design (flat baseline), DeepSet-H because its elements are "
        "constraints with no fixed task correspondence, CNN-Proj because its channels are indexed "
        "by clock *pairs* rather than single clocks/jobs. See each encoder module's docstring for "
        "the full justification.\n"
    )

    # ---- per-run summary ----------------------------------------------------
    lines.append("## Final and best mean return per encoder per instance (mean +/- std across seeds)\n")
    lines.append("| Instance | Encoder | seeds | final return | best return | final makespan | total env steps (mean) | wall-clock (mean, s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for instance in INSTANCES:
        for e in ENCODERS:
            recs = runs.get((instance, e), [])
            if not recs:
                lines.append(f"| {instance} | {ENCODER_LABEL[e]} | 0 | MISSING | MISSING | | | |")
                continue
            finals, bests, makespans, steps, walls = [], [], [], [], []
            for r in recs:
                hist = r["history"]
                rets = [h["mean_return"] for h in hist if not math.isnan(h["mean_return"])]
                if rets:
                    finals.append(rets[-1])
                    bests.append(max(rets))
                mk = [h["mean_makespan"] for h in hist if not math.isnan(h["mean_makespan"])]
                if mk:
                    makespans.append(mk[-1])
                steps.append(r["total_env_steps"])
                walls.append(r["wall_clock_s"])
            f_mean, f_std = (np.mean(finals), np.std(finals)) if finals else (float("nan"), float("nan"))
            b_mean, b_std = (np.mean(bests), np.std(bests)) if bests else (float("nan"), float("nan"))
            mk_mean = np.mean(makespans) if makespans else float("nan")
            lines.append(
                f"| {instance} | {ENCODER_LABEL[e]} | {len(recs)} | {f_mean:.2f} +/- {f_std:.2f} | "
                f"{b_mean:.2f} +/- {b_std:.2f} | {mk_mean:.2f} | {np.mean(steps):.0f} | {np.mean(walls):.1f} |"
            )
    lines.append("")

    # ---- sample-matched and wall-clock-matched comparison -------------------
    lines.append("## Sample-matched comparison (return at a fixed env-step budget)\n")
    lines.append(
        "Mean return interpolated (forward-filled) at fixed env-step checkpoints, mean +/- std across seeds.\n"
    )
    checkpoints = [50_000, 100_000, 200_000, 300_000]
    for instance in INSTANCES:
        lines.append(f"### {instance}\n")
        lines.append("| Encoder | " + " | ".join(f"{c // 1000}k steps" for c in checkpoints) + " |")
        lines.append("|---|" + "---|" * len(checkpoints))
        for e in ENCODERS:
            recs = runs.get((instance, e), [])
            if not recs:
                lines.append(f"| {ENCODER_LABEL[e]} | " + " | ".join(["MISSING"] * len(checkpoints)) + " |")
                continue
            per_seed_vals = [_interp_at_steps(r["history"], checkpoints) for r in recs]
            cell = []
            for ci in range(len(checkpoints)):
                vals = [v[ci] for v in per_seed_vals if not math.isnan(v[ci])]
                cell.append(f"{np.mean(vals):.2f}+/-{np.std(vals):.2f}" if vals else "n/a")
            lines.append(f"| {ENCODER_LABEL[e]} | " + " | ".join(cell) + " |")
        lines.append("")

    lines.append("## Wall-clock-matched comparison (return at a fixed wall-clock budget)\n")
    lines.append(
        "Same idea, but the x-axis is cumulative wall-clock seconds rather than env steps -- this "
        "is where the raster encoders' per-step cost shows up directly: they get far fewer env "
        "steps in for the same time budget.\n"
    )
    wall_checkpoints = [60, 300, 900, 1800]
    for instance in INSTANCES:
        lines.append(f"### {instance}\n")
        lines.append("| Encoder | " + " | ".join(f"{c}s" for c in wall_checkpoints) + " |")
        lines.append("|---|" + "---|" * len(wall_checkpoints))
        for e in ENCODERS:
            recs = runs.get((instance, e), [])
            if not recs:
                lines.append(f"| {ENCODER_LABEL[e]} | " + " | ".join(["MISSING"] * len(wall_checkpoints)) + " |")
                continue
            per_seed_vals = [_interp_at_wallclock(r["history"], wall_checkpoints) for r in recs]
            cell = []
            for ci in range(len(wall_checkpoints)):
                vals = [v[ci] for v in per_seed_vals if not math.isnan(v[ci])]
                cell.append(f"{np.mean(vals):.2f}+/-{np.std(vals):.2f}" if vals else "n/a")
            lines.append(f"| {ENCODER_LABEL[e]} | " + " | ".join(cell) + " |")
        lines.append("")

    # ---- encoder forward time fraction --------------------------------------
    lines.append("## Encoder forward time as a fraction of total step time\n")
    lines.append("| Instance | Encoder | mean encoder_forward_frac | mean wall-clock/update (s) |")
    lines.append("|---|---|---|---|")
    for instance in INSTANCES:
        for e in ENCODERS:
            recs = runs.get((instance, e), [])
            if not recs:
                continue
            fracs = [h["encoder_forward_frac"] for r in recs for h in r["history"]]
            walls = [h["wall_clock_s"] for r in recs for h in r["history"]]
            lines.append(f"| {instance} | {ENCODER_LABEL[e]} | {np.mean(fracs) * 100:.1f}% | {np.mean(walls):.2f} |")
    lines.append("")

    # ---- permutation ablation ------------------------------------------------
    lines.append("## Permutation ablation (evaluation-time only)\n")
    n_ep_str = str(ablation[0]["n_episodes"]) if ablation else "N"
    lines.append(
        "Each encoder's own 'set of pieces' shuffled at eval time (see `train/permutation_ablation.py` "
        f"docstring for exactly what is shuffled per encoder); greedy policy, {n_ep_str} episodes/checkpoint, "
        "same env seeds for baseline and shuffled runs. Gap = baseline - shuffled mean return "
        "(positive gap = shuffling hurt).\n"
    )
    if not ablation:
        lines.append("**No ablation results found** -- `train/permutation_ablation.py` has not been run.\n")
    else:
        lines.append("| Instance | Encoder | seeds | baseline return | shuffled return | mean gap +/- std |")
        lines.append("|---|---|---|---|---|---|")
        grouped = defaultdict(list)
        for a in ablation:
            grouped[(a["instance"], a["encoder"])].append(a)
        for instance in INSTANCES:
            for e in ENCODERS:
                items = grouped.get((instance, e), [])
                if not items:
                    continue
                base = np.mean([a["baseline_mean_return"] for a in items])
                shuf = np.mean([a["shuffled_mean_return"] for a in items])
                gaps = [a["gap"] for a in items]
                lines.append(
                    f"| {instance} | {ENCODER_LABEL[e]} | {len(items)} | {base:.2f} | {shuf:.2f} | "
                    f"{np.mean(gaps):.2f} +/- {np.std(gaps):.2f} |"
                )
        lines.append("")
        lines.append(
            "Expected pattern: DeepSet-H's gap should be ~0 (sum-pool is invariant to element order "
            "by construction); MLP-Point and both CNNs should show a nonzero (likely positive) gap, "
            "quantifying how much of their return depends on position/order rather than content.\n"
        )

    # ---- anomalies ------------------------------------------------------------
    lines.append("## Anomalies\n")
    flags = check_anomalies(runs)
    if flags:
        for f in flags:
            lines.append(f"- {f}")
    else:
        lines.append("None detected (no NaN/Inf losses, no entropy collapse, every run showed some "
                      "improvement from first-third to last-third of training).")
    lines.append("")

    return "\n".join(lines)


def main():
    runs = load_runs()
    ablation = load_ablation()
    md = render_markdown(runs, ablation)
    with open(OUT_PATH, "w") as f:
        f.write(md)
    print(f"Wrote {OUT_PATH} ({len(runs)} (instance,encoder) groups, {len(ablation)} ablation records)")


if __name__ == "__main__":
    main()
