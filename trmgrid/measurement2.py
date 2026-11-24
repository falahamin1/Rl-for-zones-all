"""Measurement 2 -- CLOCK INDEPENDENCE. This is where the design can die.

Over every DISTINCT (location, phase, zone) triple (deduped -- see
zones.distinct_zones -- so a zone with many concrete member points isn't
over-weighted), reports FIRST and prominently:
  - fraction of zones where each pair of sub-goal clocks (t_A,t_B),
    (t_A,t_C), (t_B,t_C) is mutually pinned (bound(i,j)+bound(j,i)==0,
    both non-strict)
  - fraction where g is pinned to each t_i
  - affine dimension distribution over CLIPPED zones, fraction at dim>=2
  - facets, exact vertex count (Fraction-based enumeration via
    zoneviews.zone.Zone.v_rep -- affine-dim-first short-circuit, no
    scipy.spatial.qhull anywhere), bounded fraction, strict fraction

w=0 is run as a control: geometry should collapse to dim 0 everywhere at
w=0, since delay is exact-per-edge (not widened) and w is the ONLY other
source of width -- if dim 0 does NOT dominate at w=0, something else is
producing width and that needs to be found before trusting w=0.25's
numbers at all.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from itertools import combinations
from typing import Optional

from zoneviews.zone import UnboundedZone, VertexEnumerationLimit, Zone

from .dp import DPResult, explore_and_solve
from .instance import K_BOUND, SCALE, TRMGridInstance
from .zones import build_state_zones, distinct_zones


def _pinned(dbm, i: int, j: int) -> bool:
    if dbm.is_infinity(i, j) or dbm.is_infinity(j, i):
        return False
    if dbm.is_strict(i, j) or dbm.is_strict(j, i):
        return False
    return dbm.bound(i, j) + dbm.bound(j, i) == 0


def analyze_geometry(
    instance: TRMGridInstance,
    dp_result: Optional[DPResult] = None,
    max_zones_sampled: int = 4000,
    max_zones_for_vertices: int = 400,
    seed: int = 0,
    precomputed=None,
) -> dict:
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
    subgoal_pairs = list(combinations(instance.clock_names, 2))  # (t_A,t_B), (t_A,t_C), (t_B,t_C)
    # K_BOUND must scale with THIS instance's own deadline, not be reused
    # as a fixed global constant across differently-scaled instances (the
    # watch-list's clip-bound bug class) -- max() with a per-instance
    # floor derived from deadline, exactly as gridtimed's own geometry.py
    # does (read as a reference).
    k_bound_real = max(K_BOUND, instance.deadline + 4)
    k_vec = tuple([k_bound_real * SCALE] * len(zctx.clock_names))

    n_zones = 0
    n_pinned_subgoal_pair = Counter()
    n_pinned_g_subgoal = Counter()
    affine_dim_counts: Counter = Counter()
    n_strict = 0
    facet_counts = []
    bounded_flags = []
    vertex_counts = []
    n_vertex_errors = 0
    vertex_error_kinds: Counter = Counter()

    for loc, phase, fed in zone_list:
        n_zones += 1
        dbm = fed.to_dbm_list()[0]
        for a, b in subgoal_pairs:
            if _pinned(dbm, idx[a], idx[b]):
                n_pinned_subgoal_pair[(a, b)] += 1
        for a in instance.clock_names:
            if _pinned(dbm, idx["g"], idx[a]):
                n_pinned_g_subgoal[a] += 1

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
    pinned_pair_fracs = {f"{a}_{b}": (n_pinned_subgoal_pair[(a, b)] / n_zones if n_zones else float("nan"))
                          for a, b in subgoal_pairs}
    pinned_g_fracs = {a: (n_pinned_g_subgoal[a] / n_zones if n_zones else float("nan")) for a in instance.clock_names}

    return {
        "label": instance.label,
        "w": instance.w,
        "n_states": len(dp_result.states),
        "dp_truncated": dp_result.truncated,
        "n_zones_total": n_zones_total,
        "n_zones": n_zones,
        "zones_sampled": zones_sampled,
        "n_locations": len({loc for loc, _phase, _fed in zone_list}),
        "dp_plus_zone_wall_s": dpz_wall,
        # pinning, reported first/prominently by the caller
        "pinned_subgoal_pair_fraction": pinned_pair_fracs,
        "pinned_subgoal_pair_mean": (sum(pinned_pair_fracs.values()) / len(pinned_pair_fracs)) if pinned_pair_fracs else float("nan"),
        "pinned_subgoal_pair_max": max(pinned_pair_fracs.values()) if pinned_pair_fracs else float("nan"),
        "pinned_g_subgoal_fraction": pinned_g_fracs,
        "affine_dim_distribution": dict(sorted(affine_dim_counts.items())),
        "affine_dim_0_fraction": affine_dim_counts.get(0, 0) / total_dim if total_dim else float("nan"),
        "affine_dim_0_or_1_fraction": sum(c for d, c in affine_dim_counts.items() if d <= 1) / total_dim if total_dim else float("nan"),
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


def gate_check(result_w25: dict, result_w0: dict) -> dict:
    """The GATE: sub-goal clocks mutually pinned in most zones, or affine
    dim concentrated at 0-1 at w=0.25 -> STOP, same failure as the Taxi
    TRM. Returns {"passed": bool, "reasons": [...]}."""
    reasons = []
    pinned_mean = result_w25["pinned_subgoal_pair_mean"]
    if pinned_mean > 0.5:
        reasons.append(f"sub-goal clocks mutually pinned in {pinned_mean:.1%} of zones (>50%) at w=0.25")
    dim01 = result_w25["affine_dim_0_or_1_fraction"]
    if dim01 > 0.5:
        reasons.append(f"affine dimension concentrated at 0-1 in {dim01:.1%} of zones (>50%) at w=0.25")
    passed = len(reasons) == 0
    return {"passed": passed, "reasons": reasons}


if __name__ == "__main__":
    import json

    from .instance import W_VALUES

    results = {}
    for w in W_VALUES:
        inst = TRMGridInstance(w=w)
        r = analyze_geometry(inst)
        print(f"\n=== w={w} ===")
        print(f"pinned subgoal pairs: {r['pinned_subgoal_pair_fraction']}")
        print(f"pinned g-subgoal: {r['pinned_g_subgoal_fraction']}")
        print(f"affine_dim dist: {r['affine_dim_distribution']} (dim>=2={r['affine_dim_ge2_fraction']:.4f})")
        print(f"strict_fraction={r['strict_fraction']:.4f} facets_mean={r['facet_count_mean']:.2f} "
              f"bounded_frac={r['bounded_fraction']:.4f} vertices_mean={r['vertex_count_mean']:.2f} "
              f"(n={r['n_vertices_computed']}, errors={r['n_vertex_errors']})")
        results[f"w{w}"] = r

    gate = gate_check(results["w0.25"], results["w0.0"])
    print("\n=== GATE (Measurement 2) ===")
    print(json.dumps(gate, indent=2))

    with open("/tmp/trmgrid_measurement2.json", "w") as f:
        json.dump({"results": results, "gate": gate}, f, indent=2, default=str)
    print("\nWrote /tmp/trmgrid_measurement2.json")
