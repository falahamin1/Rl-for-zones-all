"""Measurement 1 -- THE GATE. For w in {0.0, 0.1, 0.25} and the
per-instance t calibrated in the previous (deadline_track) report, on
PJS_01..05 and LJS_01: affine-dimension distribution (fraction at dim 0
first and prominently -- w=0.0 is the control and must reproduce
deadline_track's dim-0-everywhere result exactly), strict-inequality
fraction, facet count, exact vertex count, bounded fraction.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from typing import Dict, Optional

from zoneviews.zone import Zone, UnboundedZone, VertexEnumerationLimit

from .collect import collect_zones
from .instance import SCALE, GateJobShopInstance

INSTANCES = ["PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05", "LJS_01"]
W_VALUES = [0.0, 0.1, 0.25]


def analyze_zones(
    instance: GateJobShopInstance,
    n_episodes: int = 250,
    seed: int = 0,
    max_zones_sampled: int = 2000,
    max_zones_for_vertices: int = 200,
) -> dict:
    m_bound = int(instance.worst_case_makespan_upper_bound() * SCALE) + SCALE

    t0 = time.perf_counter()
    raw = collect_zones(instance, n_episodes=n_episodes, seed=seed, m_bound=m_bound)
    collect_wall = time.perf_counter() - t0

    if len(raw) > max_zones_sampled:
        random.Random(seed).shuffle(raw)
        raw = raw[:max_zones_sampled]

    n_clocks = len(instance.jobs) + 1  # n_jobs c_J's + 1 global g
    k_vec = tuple([m_bound] * n_clocks)

    affine_dim_counts: Counter = Counter()
    strict_flags = []
    facet_counts = []
    bounded_flags = []
    vertex_counts = []
    n_vertex_errors = 0
    vertex_error_kinds: Counter = Counter()

    for idx, (loc, Z, zctx) in enumerate(raw):
        zone = Zone(Z, zctx.clock_names, k_vec).clip_to_k_box()

        dim = zone.affine_dim()
        affine_dim_counts[dim] += 1

        has_strict = any(hs.strict for hs in zone.h_rep())
        strict_flags.append(has_strict)

        facet_counts.append(zone.facet_count())
        bounded_flags.append(zone.is_bounded())

        if idx < max_zones_for_vertices:
            try:
                vertex_counts.append(len(zone.v_rep()))
            except (UnboundedZone, VertexEnumerationLimit) as e:
                n_vertex_errors += 1
                vertex_error_kinds[type(e).__name__] += 1

    n = len(raw)
    total_dim_count = sum(affine_dim_counts.values())
    return {
        "n_zones_sampled": n,
        "collect_wall_s": collect_wall,
        "affine_dim_distribution": dict(sorted(affine_dim_counts.items())),
        "affine_dim_0_fraction": affine_dim_counts.get(0, 0) / total_dim_count if total_dim_count else float("nan"),
        "affine_dim_ge2_fraction": sum(c for d, c in affine_dim_counts.items() if d >= 2) / total_dim_count if total_dim_count else float("nan"),
        "affine_dim_mean": sum(d * c for d, c in affine_dim_counts.items()) / total_dim_count if total_dim_count else float("nan"),
        "strict_fraction": sum(strict_flags) / n if n else float("nan"),
        "facet_count_mean": sum(facet_counts) / n if n else float("nan"),
        "facet_count_min": min(facet_counts) if facet_counts else None,
        "facet_count_max": max(facet_counts) if facet_counts else None,
        "bounded_fraction": sum(bounded_flags) / n if n else float("nan"),
        "vertex_count_mean": (sum(vertex_counts) / len(vertex_counts)) if vertex_counts else float("nan"),
        "vertex_count_min": min(vertex_counts) if vertex_counts else None,
        "vertex_count_max": max(vertex_counts) if vertex_counts else None,
        "n_vertices_sampled": len(vertex_counts),
        "n_vertex_errors": n_vertex_errors,
        "vertex_error_kinds": dict(vertex_error_kinds),
        "total_clocks": n_clocks,
    }


def load_calibrated_t() -> Dict[str, float]:
    import json

    with open("/tmp/deadline_results/calibration.json") as f:
        calib = json.load(f)
    return {k: float(v) for k, v in calib["calibrated_t"].items()}


if __name__ == "__main__":
    import json

    import zoneviews  # noqa: F401
    from deadline_track.instance import load_base_instance
    from .instance import make_gate_variant

    t0_total = time.perf_counter()
    calibrated_t = load_calibrated_t()
    print("Using calibrated t from deadline_track's report:", calibrated_t)

    results: Dict[str, Dict[float, dict]] = {}
    for name in INSTANCES:
        t = calibrated_t[name]
        base = load_base_instance(name)
        results[name] = {}
        for w in W_VALUES:
            inst = make_gate_variant(base, t=t, w=w)
            r = analyze_zones(inst, n_episodes=250, seed=7)
            results[name][w] = r
            print(
                f"{name} t={t} w={w}: dim0_frac={r['affine_dim_0_fraction']:.3f} "
                f"dim>=2_frac={r['affine_dim_ge2_fraction']:.3f} "
                f"dim_dist={r['affine_dim_distribution']} "
                f"strict_frac={r['strict_fraction']:.3f} "
                f"facets_mean={r['facet_count_mean']:.2f} "
                f"bounded_frac={r['bounded_fraction']:.3f} "
                f"vertices_mean={r['vertex_count_mean']:.2f} (n={r['n_vertices_sampled']}, errors={r['n_vertex_errors']})"
            )

    total_wall = time.perf_counter() - t0_total
    print(f"\nTotal wall-clock for Measurement 1: {total_wall:.1f}s")

    out = {
        "calibrated_t": calibrated_t,
        "results": {name: {str(w): r for w, r in by_w.items()} for name, by_w in results.items()},
        "total_wall_s": total_wall,
    }
    with open("/tmp/gate_results/geometry.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("Wrote /tmp/gate_results/geometry.json")
