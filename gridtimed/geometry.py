"""Measurement 2 -- pinning and zone geometry, over the SOUND per-state
zone construction (zones.py). Analyzed exhaustively over every DISTINCT
(location, zone) pair (deduped -- see zones.distinct_zones -- so a zone
with many concrete member points isn't over-weighted) where tractable;
for large tiers (Task B), the zone list and the (expensive) exact-vertex
computation are each capped and randomly sampled, with the sampling
documented in the returned dict rather than silently applied.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from itertools import combinations
from typing import Dict, Optional

from zoneviews.zone import UnboundedZone, VertexEnumerationLimit, Zone

from .dp import DPResult, explore_and_solve
from .instance import K_BOUND, SCALE, GridInstance
from .zones import build_state_zones, distinct_zones


def _pinned(dbm, i: int, j: int) -> bool:
    if dbm.is_infinity(i, j) or dbm.is_infinity(j, i):
        return False
    if dbm.is_strict(i, j) or dbm.is_strict(j, i):
        return False
    return dbm.bound(i, j) + dbm.bound(j, i) == 0


def analyze_geometry(
    instance: GridInstance,
    dp_result: Optional[DPResult] = None,
    max_zones_sampled: int = 4000,
    max_zones_for_vertices: int = 400,
    seed: int = 0,
    precomputed=None,
) -> dict:
    """``precomputed``, if given, is ``(dp_result, zctx, zones)`` already
    built by ``build_state_zones`` -- avoids rebuilding (zone construction
    is cheap per-state but adds up at Task B's larger tiers, and
    geometry.py/decision_relevance.py would otherwise each build it
    independently for the same instance)."""
    t0 = time.perf_counter()
    if precomputed is not None:
        dp_result, zctx, zones = precomputed
    else:
        dp_result, zctx, zones = build_state_zones(instance, dp_result)
    dpz_wall = time.perf_counter() - t0

    zone_list_full = distinct_zones(dp_result, zones)
    n_zones_total = len(zone_list_full)
    zones_sampled = n_zones_total > max_zones_sampled
    zone_list = zone_list_full
    if zones_sampled:
        zone_list = list(zone_list_full)
        random.Random(seed).shuffle(zone_list)
        zone_list = zone_list[:max_zones_sampled]

    idx = {name: zctx.clock_names.index(name) + 1 for name in zctx.clock_names}
    switch_pairs = list(combinations(instance.clock_names, 2))
    # K_BOUND must scale with THIS instance's own deadline, not be a fixed
    # global constant -- exactly the clip-bound bug class flagged
    # repeatedly elsewhere in this repo (deadline_track's m_c note,
    # gate_track's uniform-bound fix): a bound tuned for T0's D=16 silently
    # emptied zones for larger generated tiers with D up to 43. Caught by
    # running the code (a real ValueError), not by inspection.
    k_bound_real = max(K_BOUND, instance.deadline + 4)
    k_vec = tuple([k_bound_real * SCALE] * len(zctx.clock_names))

    n_zones = 0
    n_pinned_switch_pair = Counter()   # per pair
    n_pinned_y_switch = Counter()      # per switch clock
    affine_dim_counts: Counter = Counter()
    n_strict = 0
    facet_counts = []
    bounded_flags = []
    vertex_counts = []
    n_vertex_errors = 0
    vertex_error_kinds: Counter = Counter()

    for loc, fed in zone_list:
        n_zones += 1
        dbm = fed.to_dbm_list()[0]
        for a, b in switch_pairs:
            if _pinned(dbm, idx[a], idx[b]):
                n_pinned_switch_pair[(a, b)] += 1
        for a in instance.clock_names:
            if _pinned(dbm, idx["y"], idx[a]):
                n_pinned_y_switch[a] += 1

        zone = Zone(fed, zctx.clock_names, k_vec).clip_to_k_box()
        dim = zone.affine_dim()
        affine_dim_counts[dim] += 1

        h = zone.h_rep()
        if any(hs.strict for hs in h):
            n_strict += 1
        facet_counts.append(len(h))
        bounded_flags.append(zone.is_bounded())
        if n_zones <= max_zones_for_vertices:
            try:
                vertex_counts.append(len(zone.v_rep()))
            except (UnboundedZone, VertexEnumerationLimit) as e:
                n_vertex_errors += 1
                vertex_error_kinds[type(e).__name__] += 1

    total_dim = sum(affine_dim_counts.values())
    pinned_pair_fracs = {f"{a}_{b}": (n_pinned_switch_pair[(a, b)] / n_zones if n_zones else float("nan"))
                          for a, b in switch_pairs}
    pinned_y_fracs = {a: (n_pinned_y_switch[a] / n_zones if n_zones else float("nan")) for a in instance.clock_names}

    return {
        "label": instance.label,
        "w": instance.w,
        "n_switches": instance.n_switches,
        "n_clocks_total": len(zctx.clock_names),
        "n_states": len(dp_result.states),
        "dp_truncated": dp_result.truncated,
        "n_zones_total": n_zones_total,
        "n_zones": n_zones,
        "zones_sampled": zones_sampled,
        "n_locations": len({loc for loc, _ in zone_list}),
        "dp_plus_zone_wall_s": dpz_wall,
        # pinning, reported first/prominently by the caller
        "pinned_switch_pair_fraction": pinned_pair_fracs,
        "pinned_switch_pair_mean": (sum(pinned_pair_fracs.values()) / len(pinned_pair_fracs)) if pinned_pair_fracs else float("nan"),
        "pinned_switch_pair_max": max(pinned_pair_fracs.values()) if pinned_pair_fracs else float("nan"),
        "pinned_y_switch_fraction": pinned_y_fracs,
        "affine_dim_distribution": dict(sorted(affine_dim_counts.items())),
        "affine_dim_0_fraction": affine_dim_counts.get(0, 0) / total_dim if total_dim else float("nan"),
        "affine_dim_ge2_fraction": sum(c for d, c in affine_dim_counts.items() if d >= 2) / total_dim if total_dim else float("nan"),
        "facet_count_mean": sum(facet_counts) / len(facet_counts) if facet_counts else float("nan"),
        "facet_count_min": min(facet_counts) if facet_counts else None,
        "facet_count_max": max(facet_counts) if facet_counts else None,
        "vertex_count_mean": sum(vertex_counts) / len(vertex_counts) if vertex_counts else float("nan"),
        "vertex_count_min": min(vertex_counts) if vertex_counts else None,
        "vertex_count_max": max(vertex_counts) if vertex_counts else None,
        "n_vertices_computed": len(vertex_counts),
        "n_vertices_attempted": len(vertex_counts) + n_vertex_errors,
        "vertices_sampled": n_zones > max_zones_for_vertices,
        "n_vertex_errors": n_vertex_errors,
        "vertex_error_rate": (n_vertex_errors / (len(vertex_counts) + n_vertex_errors)) if (len(vertex_counts) + n_vertex_errors) else float("nan"),
        "vertex_error_kinds": dict(vertex_error_kinds),
        "bounded_fraction": sum(bounded_flags) / len(bounded_flags) if bounded_flags else float("nan"),
        "strict_fraction": n_strict / n_zones if n_zones else float("nan"),
        "allowed_delays": list(instance.allowed_delays),
    }


if __name__ == "__main__":
    import json

    import zoneviews  # noqa: F401
    from .instance import DEFAULT_DELAYS, RESTRICTED_DELAYS

    results = {}
    for label, delays in [("default_delays", DEFAULT_DELAYS), ("restricted_delay0", RESTRICTED_DELAYS)]:
        inst = GridInstance(allowed_delays=delays)
        r = analyze_geometry(inst)
        print(f"\n=== {label} (allowed_delays={delays}) ===")
        print(f"pinned switch pairs: {r['pinned_switch_pair_fraction']}")
        print(f"affine_dim dist: {r['affine_dim_distribution']} (dim>=2={r['affine_dim_ge2_fraction']:.4f})")
        print(f"strict_fraction={r['strict_fraction']:.4f} facets_mean={r['facet_count_mean']:.2f} "
              f"bounded_frac={r['bounded_fraction']:.4f} vertices_mean={r['vertex_count_mean']:.2f} "
              f"(n={r['n_vertices_computed']}, errors={r['n_vertex_errors']})")
        results[label] = r

    with open("/tmp/gridgate_results/geometry.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
