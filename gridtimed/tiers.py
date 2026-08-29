"""Task B: run the five tiers x 3 seeds each, collecting all required stats.

A single uniform state_limit is used across all tiers -- T0/T1 (and
usually T2) come in well under it (exact DP); T3/T4 do not (truncated,
reported as approximate/"trainable, not exact" tiers, not silently treated
as exact). DP and zone construction are each done ONCE per tier-seed and
shared between the geometry and decision-relevance passes.
"""
from __future__ import annotations

import sys
import time
from typing import Dict, List

from .decision_relevance import multi_zone_conflict_analysis
from .dp import explore_and_solve
from .generator import make_instance
from .geometry import analyze_geometry
from .zones import build_state_zones

W = 0.25  # Task A validated soundness at this w; used for all of Task B
STATE_LIMIT = 150_000

TIERS = [
    dict(name="T0", grid_n=3, n_switches=2, n_gates=2, window_width=2, D=17),
    dict(name="T1", grid_n=4, n_switches=2, n_gates=2, window_width=2, D=21),
    dict(name="T2", grid_n=5, n_switches=3, n_gates=3, window_width=2, D=27),
    dict(name="T3", grid_n=6, n_switches=4, n_gates=4, window_width=2, D=33),
    dict(name="T4", grid_n=8, n_switches=5, n_gates=5, window_width=2, D=43),
]
SEEDS = (0, 1, 2)


def run_tier_seed(cfg: dict, seed: int) -> dict:
    inst, info = make_instance(
        grid_n=cfg["grid_n"], n_switches=cfg["n_switches"], n_gates=cfg["n_gates"],
        window_width=cfg["window_width"], D=cfg["D"], w=W, seed=seed, label=cfg["name"],
    )

    t0 = time.perf_counter()
    dp_result = explore_and_solve(inst, state_limit=STATE_LIMIT)
    t_dp = time.perf_counter() - t0

    t0 = time.perf_counter()
    precomputed = build_state_zones(inst, dp_result)
    t_zone = time.perf_counter() - t0

    t0 = time.perf_counter()
    g = analyze_geometry(inst, precomputed=precomputed)
    t_geo = time.perf_counter() - t0

    t0 = time.perf_counter()
    d = multi_zone_conflict_analysis(inst, precomputed=precomputed)
    t_dec = time.perf_counter() - t0

    return {
        "tier": cfg["name"], "seed": seed,
        "grid_n": cfg["grid_n"], "n_switches": cfg["n_switches"], "n_gates": cfg["n_gates"],
        "D": cfg["D"], "w": W,
        "generation_attempts": info.attempts,
        "generation_rejected_unsolvable": info.rejected_unsolvable,
        "generation_rejected_inconclusive": info.rejected_inconclusive,
        "n_clocks_total": g["n_clocks_total"],
        "n_states": g["n_states"],
        "dp_truncated": g["dp_truncated"],
        "n_horizon_fallback_refs": dp_result.n_horizon_fallback_refs,
        "n_deadlock_states": dp_result.n_deadlock_states,
        "dp_tractable": not g["dp_truncated"],
        "n_zones_total": g["n_zones_total"],
        "n_zones_analyzed": g["n_zones"],
        "zones_sampled": g["zones_sampled"],
        "n_locations": g["n_locations"],
        "multi_zone_locations": d["multi_zone_locations"],
        "conflicting_locations": d["conflicting_locations"],
        "cross_zone_conflict_fraction": d["conflict_fraction"],
        "n_multistate_zones": d["n_multistate_zones"],
        "n_nonuniform_zones": d["n_nonuniform_zones"],
        "within_zone_nonuniform_fraction": d["within_zone_nonuniform_fraction"],
        "pinned_switch_pair_mean": g["pinned_switch_pair_mean"],
        "pinned_switch_pair_max": g["pinned_switch_pair_max"],
        "affine_dim_distribution": g["affine_dim_distribution"],
        "affine_dim_ge2_fraction": g["affine_dim_ge2_fraction"],
        "facet_count_mean": g["facet_count_mean"],
        "vertex_count_mean": g["vertex_count_mean"],
        "n_vertices_computed": g["n_vertices_computed"],
        "n_vertices_attempted": g["n_vertices_attempted"],
        "vertices_sampled": g["vertices_sampled"],
        "n_vertex_errors": g["n_vertex_errors"],
        "vertex_error_rate": g["vertex_error_rate"],
        "bounded_fraction": g["bounded_fraction"],
        "strict_fraction": g["strict_fraction"],
        "n_containment_ok": d["n_containment_ok"],
        "n_containment_bad": d["n_containment_bad"],
        "wall_dp_s": t_dp, "wall_zone_s": t_zone, "wall_geometry_s": t_geo, "wall_decision_s": t_dec,
        "wall_total_s": t_dp + t_zone + t_geo + t_dec,
    }


def run_all() -> Dict[str, List[dict]]:
    results: Dict[str, List[dict]] = {}
    for cfg in TIERS:
        results[cfg["name"]] = []
        for seed in SEEDS:
            r = run_tier_seed(cfg, seed)
            results[cfg["name"]].append(r)
            print(
                f"{r['tier']} seed={seed}: n_states={r['n_states']} truncated={r['dp_truncated']} "
                f"deadlocks={r['n_deadlock_states']} multi_zone_locations={r['multi_zone_locations']} "
                f"cross_zone_conflict={r['cross_zone_conflict_fraction']:.4f} "
                f"within_zone_nonuniform={r['within_zone_nonuniform_fraction']} "
                f"dim>=2_frac={r['affine_dim_ge2_fraction']:.4f} "
                f"wall={r['wall_total_s']:.2f}s (dp={r['wall_dp_s']:.2f} zone={r['wall_zone_s']:.2f} "
                f"geo={r['wall_geometry_s']:.2f} dec={r['wall_decision_s']:.2f})",
                flush=True,
            )
    return results


if __name__ == "__main__":
    import json

    import zoneviews  # noqa: F401

    t0 = time.perf_counter()
    results = run_all()
    total_wall = time.perf_counter() - t0
    print(f"\nTOTAL WALL CLOCK: {total_wall:.1f}s", flush=True)

    with open("/tmp/gridsound_results/tiers.json", "w") as f:
        json.dump({"results": results, "total_wall_s": total_wall}, f, indent=2, default=str)
    print("Wrote /tmp/gridsound_results/tiers.json", flush=True)
