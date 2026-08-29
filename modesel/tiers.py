"""Orchestrates Steps 3 and 4 across all five tiers x 3 seeds: exact DP,
the partition diagnostic (Step 3, w-independent), and geometry +
decision-relevance (Step 4, at w in {0.0, 0.25}).
"""
from __future__ import annotations

import time
from typing import Dict, List

from .decision_relevance import multi_zone_conflict_analysis
from .diagnostics import run_diagnostic
from .dp import explore_and_solve
from .generator import make_instance
from .geometry import analyze_geometry
from .zones import build_state_zones

STATE_LIMIT = 200_000

TIER_SPECS = {
    "M0": dict(n_jobs=2, n_tasks=1, n_machines=1, tightness=0.25),
    "M1": dict(n_jobs=2, n_tasks=2, n_machines=2, tightness=0.25),
    "M2": dict(n_jobs=3, n_tasks=2, n_machines=2, tightness=0.5),
    "M3": dict(n_jobs=3, n_tasks=3, n_machines=3, tightness=0.25),
    "M4": dict(n_jobs=4, n_tasks=3, n_machines=3, tightness=0.5),
}
SEEDS = (0, 1, 2)
FAST_SLOW_RATIO = 2.0
COST_PENALTY = 6.0


def run_tier_seed(name: str, seed: int) -> dict:
    spec = TIER_SPECS[name]

    inst25 = make_instance(
        n_jobs=spec["n_jobs"], n_tasks=spec["n_tasks"], n_machines=spec["n_machines"],
        fast_slow_ratio=FAST_SLOW_RATIO, cost_penalty=COST_PENALTY,
        tightness=spec["tightness"], w=0.25, seed=seed, label=name,
    )

    t0 = time.perf_counter()
    dp_result = explore_and_solve(inst25, state_limit=STATE_LIMIT)
    t_dp = time.perf_counter() - t0

    t0 = time.perf_counter()
    diag = run_diagnostic(inst25, dp_result)
    t_diag = time.perf_counter() - t0

    t0 = time.perf_counter()
    precomputed25 = build_state_zones(inst25, dp_result)
    g25 = analyze_geometry(inst25, precomputed=precomputed25)
    d25 = multi_zone_conflict_analysis(inst25, precomputed=precomputed25)
    t_geo25 = time.perf_counter() - t0

    inst0 = make_instance(
        n_jobs=spec["n_jobs"], n_tasks=spec["n_tasks"], n_machines=spec["n_machines"],
        fast_slow_ratio=FAST_SLOW_RATIO, cost_penalty=COST_PENALTY,
        tightness=spec["tightness"], w=0.0, seed=seed, label=name,
    )
    t0 = time.perf_counter()
    dp_result0 = explore_and_solve(inst0, state_limit=STATE_LIMIT)
    precomputed0 = build_state_zones(inst0, dp_result0)
    g0 = analyze_geometry(inst0, precomputed=precomputed0)
    d0 = multi_zone_conflict_analysis(inst0, precomputed=precomputed0)
    t_geo0 = time.perf_counter() - t0

    return {
        "tier": name, "seed": seed,
        "n_jobs": spec["n_jobs"], "n_tasks": spec["n_tasks"], "n_machines": spec["n_machines"],
        "tightness": spec["tightness"],
        "n_states": len(dp_result.states),
        "dp_truncated": dp_result.truncated,
        "n_horizon_fallback_refs": dp_result.n_horizon_fallback_refs,
        "diagnostic": diag,
        "geometry_w0": g0,
        "geometry_w25": g25,
        "decision_relevance_w0": {k: v for k, v in d0.items() if k not in ("conflict_examples", "within_zone_examples")},
        "decision_relevance_w25": {k: v for k, v in d25.items() if k not in ("conflict_examples", "within_zone_examples")},
        "conflict_examples_w25": d25["conflict_examples"][:5],
        "within_zone_examples_w25": d25["within_zone_examples"][:5],
        "wall_dp_s": t_dp, "wall_diag_s": t_diag, "wall_geo_w25_s": t_geo25, "wall_geo_w0_s": t_geo0,
        "wall_total_s": t_dp + t_diag + t_geo25 + t_geo0,
    }


def run_all() -> Dict[str, List[dict]]:
    results: Dict[str, List[dict]] = {}
    for name in TIER_SPECS:
        results[name] = []
        for seed in SEEDS:
            r = run_tier_seed(name, seed)
            results[name].append(r)
            diag = r["diagnostic"]
            drw0 = r["decision_relevance_w0"]
            drw25 = r["decision_relevance_w25"]
            print(
                f"{name} seed={seed}: n_states={r['n_states']} truncated={r['dp_truncated']} "
                f"ceiling_classes={diag['measurement1']['n_classes']} "
                f"m3_clean={diag['measurement3']['n_clean_of_top']}/{diag['measurement3']['top_n']} "
                f"conflict_w0={drw0['conflict_fraction']:.3f} within_nonuniform_w0={drw0['within_zone_nonuniform_fraction']} "
                f"conflict_w25={drw25['conflict_fraction']:.3f} within_nonuniform_w25={drw25['within_zone_nonuniform_fraction']} "
                f"mean_states_per_zone_w25={drw25['mean_states_per_zone']:.2f} "
                f"wall={r['wall_total_s']:.2f}s",
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

    with open("/tmp/modesel_results/tiers.json", "w") as f:
        json.dump({"results": results, "total_wall_s": total_wall}, f, indent=2, default=str)
    print("Wrote /tmp/modesel_results/tiers.json", flush=True)
