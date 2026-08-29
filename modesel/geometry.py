"""Step 4 -- zone geometry: affine dimension, facets, exact vertices,
boundedness, strictness. Analyzed over every DISTINCT (location, zone)
pair (deduped via zones.distinct_zones), exhaustively -- these tiers are
small enough (max ~8.5k states) that no sampling is needed anywhere.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Optional

from zoneviews.zone import UnboundedZone, VertexEnumerationLimit, Zone

from .dp import DPResult, explore_and_solve
from .instance import ModeSelInstance
from .zones import build_state_zones, distinct_zones

#: Uniform clock bound (real units) for k-normalization/clip_to_k_box --
#: MUST scale with this instance's own deadlines, not be a fixed global
#: constant (the exact bug class that hit gridtimed's K_BOUND twice; see
#: /tmp/GRIDSOUND.md and the task's own "watch for these" list).
_K_MARGIN = 4


def k_bound_for(instance: ModeSelInstance) -> float:
    return max(j.deadline for j in instance.jobs) + _K_MARGIN


def analyze_geometry(instance: ModeSelInstance, dp_result: Optional[DPResult] = None, precomputed=None) -> dict:
    t0 = time.perf_counter()
    if precomputed is not None:
        dp_result, zctx, zones = precomputed
    else:
        dp_result, zctx, zones = build_state_zones(instance, dp_result)
    build_wall = time.perf_counter() - t0

    zone_list = distinct_zones(dp_result, zones, instance)
    k_bound_real = k_bound_for(instance)
    k_vec = tuple([int(round(k_bound_real * 20))] * len(zctx.clock_names))

    n_zones = 0
    affine_dim_counts: Counter = Counter()
    n_strict = 0
    facet_counts = []
    bounded_flags = []
    vertex_counts = []
    n_vertex_errors = 0
    vertex_error_kinds: Counter = Counter()

    for loc, fed in zone_list:
        n_zones += 1
        zone = Zone(fed, zctx.clock_names, k_vec).clip_to_k_box()
        dim = zone.affine_dim()
        affine_dim_counts[dim] += 1

        h = zone.h_rep()
        if any(hs.strict for hs in h):
            n_strict += 1
        facet_counts.append(len(h))
        bounded_flags.append(zone.is_bounded())
        try:
            vertex_counts.append(len(zone.v_rep()))
        except (UnboundedZone, VertexEnumerationLimit) as e:
            n_vertex_errors += 1
            vertex_error_kinds[type(e).__name__] += 1

    total_dim = sum(affine_dim_counts.values())
    return {
        "label": instance.label,
        "w": instance.w,
        "n_clocks_total": len(zctx.clock_names),
        "n_states": len(dp_result.states),
        "dp_truncated": dp_result.truncated,
        "n_zones": n_zones,
        "n_locations": len({loc for loc, _ in zone_list}),
        "build_wall_s": build_wall,
        "affine_dim_distribution": dict(sorted(affine_dim_counts.items())),
        "affine_dim_0_fraction": affine_dim_counts.get(0, 0) / total_dim if total_dim else float("nan"),
        "affine_dim_ge2_fraction": sum(c for d, c in affine_dim_counts.items() if d >= 2) / total_dim if total_dim else float("nan"),
        "strict_fraction": n_strict / n_zones if n_zones else float("nan"),
        "facet_count_mean": sum(facet_counts) / len(facet_counts) if facet_counts else float("nan"),
        "facet_count_min": min(facet_counts) if facet_counts else None,
        "facet_count_max": max(facet_counts) if facet_counts else None,
        "vertex_count_mean": sum(vertex_counts) / len(vertex_counts) if vertex_counts else float("nan"),
        "vertex_count_min": min(vertex_counts) if vertex_counts else None,
        "vertex_count_max": max(vertex_counts) if vertex_counts else None,
        "n_vertices_computed": len(vertex_counts),
        "n_vertex_errors": n_vertex_errors,
        "vertex_error_kinds": dict(vertex_error_kinds),
        "bounded_fraction": sum(bounded_flags) / len(bounded_flags) if bounded_flags else float("nan"),
    }
