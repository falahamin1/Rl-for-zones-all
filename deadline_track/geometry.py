"""Section 4: zone geometry, mutual-pinning check first and prominently.

Over up to 2000 sampled clipped zones per (instance, release-variant): the
fraction where ALL g_J are mutually pinned, then the same affine-dim /
facet-count / vertex-count / bounded-fraction metrics used in the earlier
(no-deadline) audits, for direct comparability.

Reuses ``zoneviews.zone.Zone`` unchanged -- it's generic over whatever
clocks a Federation has, so no fork was needed there.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

from zoneviews.zone import Zone, UnboundedZone, VertexEnumerationLimit

from .collect import collect_zones
from .instance import SCALE, DeadlineJobShopInstance


def union_find_pinned_groups(zone: Zone, clock_indices: List[int]) -> int:
    """Number of distinct mutual-pinning equivalence classes among the
    given 1-based DBM clock indices (pinned iff bound(i,j)+bound(j,i)==0
    with both non-strict). Returns 1 if all given clocks are mutually
    pinned into a single class."""
    parent = {i: i for i in clock_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    dbm = zone.dbm
    for a_idx, i in enumerate(clock_indices):
        for j in clock_indices[a_idx + 1:]:
            if dbm.bound(i, j) + dbm.bound(j, i) == 0 and not dbm.is_strict(i, j) and not dbm.is_strict(j, i):
                union(i, j)

    return len({find(i) for i in clock_indices})


def analyze_zones(
    instance: DeadlineJobShopInstance,
    n_episodes: int = 250,
    seed: int = 0,
    max_zones_sampled: int = 2000,
    max_zones_for_vertices: int = 200,
) -> dict:
    zctx_clock_names = None
    # Bound used for BOTH clock families. A "waiting" job's own c_J (not
    # yet dispatched) is *not* bounded by a single task's max duration --
    # like g_J, it just tracks elapsed time since episode start until its
    # first reset, so it can be as large as the whole episode's worst-case
    # elapsed time. Using a small task-duration-scale bound for c_J here
    # (this module's first version did) makes clip_to_k_box's hard
    # intersection empty the zone the moment a waiting job's c_J genuinely
    # exceeds it -- a real crash (PJS_03), not a hypothetical, caught by
    # actually running this rather than assuming the Track-1 bound choice
    # would still apply. The instance's own worst-case-makespan bound (sum
    # of every task's max duration) is the safe ceiling for both.
    m_bound = int(instance.worst_case_makespan_upper_bound() * SCALE) + SCALE

    t0 = time.perf_counter()
    raw = collect_zones(instance, n_episodes=n_episodes, seed=seed, m_c=m_bound, m_g=m_bound)
    collect_wall = time.perf_counter() - t0

    if len(raw) > max_zones_sampled:
        random.Random(seed).shuffle(raw)
        raw = raw[:max_zones_sampled]

    n_g = len(instance.jobs)
    n_c = len(instance.jobs)
    k_vec = tuple([m_bound] * (n_c + n_g))

    n_all_pinned = 0
    affine_dims = []
    facet_counts = []
    bounded_flags = []
    vertex_counts = []
    n_vertex_errors = 0

    for idx, (loc, Z, zctx) in enumerate(raw):
        if zctx_clock_names is None:
            zctx_clock_names = zctx.clock_names
        zone = Zone(Z, zctx.clock_names, k_vec).clip_to_k_box()

        g_indices = [zone.clock_names.index(gn) + 1 for gn in zctx.g_names]
        n_groups = union_find_pinned_groups(zone, g_indices)
        if n_groups == 1:
            n_all_pinned += 1

        affine_dims.append(zone.affine_dim())
        facet_counts.append(zone.facet_count())
        bounded_flags.append(zone.is_bounded())

        if idx < max_zones_for_vertices:
            try:
                vertex_counts.append(len(zone.v_rep()))
            except (UnboundedZone, VertexEnumerationLimit) as e:
                n_vertex_errors += 1

    n = len(raw)
    return {
        "n_zones_sampled": n,
        "collect_wall_s": collect_wall,
        "mutual_pinning_fraction": n_all_pinned / n if n else float("nan"),
        "affine_dim_mean": sum(affine_dims) / n if n else float("nan"),
        "affine_dim_min": min(affine_dims) if affine_dims else None,
        "affine_dim_max": max(affine_dims) if affine_dims else None,
        "facet_count_mean": sum(facet_counts) / n if n else float("nan"),
        "facet_count_min": min(facet_counts) if facet_counts else None,
        "facet_count_max": max(facet_counts) if facet_counts else None,
        "bounded_fraction": sum(bounded_flags) / n if n else float("nan"),
        "vertex_count_mean": (sum(vertex_counts) / len(vertex_counts)) if vertex_counts else float("nan"),
        "vertex_count_min": min(vertex_counts) if vertex_counts else None,
        "vertex_count_max": max(vertex_counts) if vertex_counts else None,
        "n_vertices_sampled": len(vertex_counts),
        "n_vertex_errors": n_vertex_errors,
        "total_clocks": n_c + n_g,
    }


if __name__ == "__main__":
    import json

    import zoneviews  # noqa: F401
    from .instance import load_base_instance, make_deadline_variant

    with open("/tmp/deadline_results/calibration.json") as f:
        calib = json.load(f)
    chosen_t = calib["calibrated_t"]

    results = {}
    for name, t in chosen_t.items():
        base = load_base_instance(name)
        t = float(t)

        # Uniform-release (default, r_J=0) variant.
        inst_uniform = make_deadline_variant(base, t=t)
        r_uniform = analyze_zones(inst_uniform, n_episodes=250, seed=7)
        print(f"{name} t={t} [uniform release]: pinning={r_uniform['mutual_pinning_fraction']:.3f} "
              f"affine_dim_mean={r_uniform['affine_dim_mean']:.2f} "
              f"facets_mean={r_uniform['facet_count_mean']:.2f} "
              f"bounded_frac={r_uniform['bounded_fraction']:.3f} "
              f"vertices_mean={r_uniform['vertex_count_mean']:.2f} (n={r_uniform['n_vertices_sampled']})")

        # Staggered-release variant, run regardless (per task: "report both
        # variants regardless").
        n_jobs = len(base.jobs)
        spread = list(range(n_jobs))
        inst_staggered = make_deadline_variant(base, t=t, release_spread=spread)
        r_staggered = analyze_zones(inst_staggered, n_episodes=250, seed=7)
        print(f"{name} t={t} [staggered release {spread}]: pinning={r_staggered['mutual_pinning_fraction']:.3f} "
              f"affine_dim_mean={r_staggered['affine_dim_mean']:.2f} "
              f"facets_mean={r_staggered['facet_count_mean']:.2f} "
              f"bounded_frac={r_staggered['bounded_fraction']:.3f} "
              f"vertices_mean={r_staggered['vertex_count_mean']:.2f} (n={r_staggered['n_vertices_sampled']})")

        results[name] = {"t": t, "uniform_release": r_uniform, "staggered_release": r_staggered}

    with open("/tmp/deadline_results/geometry.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nWrote /tmp/deadline_results/geometry.json")
