"""SPIKE2 Task A: delta removal + boundedness + clip_to_k_box + collapse check
+ vertex/facet counts on clipped zones. THROWAWAY, /tmp only.
"""
from __future__ import annotations
import sys
import json
import random

import pyudbm as _warmup_pyudbm  # noqa: F401
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")
sys.path.append("/tmp")

import numpy as np

from zone_common import affine_dim, explore, exact_vertices_and_facets, qhull_vertices_and_facets

import prob_jobshop_spike.symbolic_env as sym_env
import prob_jobshop_spike.zone_ops as zone_ops
from prob_jobshop_ext.benchmarks.all_instances import get_instance_by_name
from prob_jobshop_ext.benchmarks.all_large_instances import get_large_instance_by_name

INSTANCES = ["PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05", "LJS_01"]
CAP = 20_000
VSAMPLE = 1000
RNG = random.Random(0)


def get_inst(name):
    return get_large_instance_by_name(name) if name.startswith("LJS") else get_instance_by_name(name)


def run_one(name):
    inst = get_inst(name)
    res = explore(sym_env, zone_ops, inst, state_cap=CAP)
    items = list(res["zone_by_key"].items())
    n = len(items)

    # SPIKE2: MUST reuse explore()'s own ZoneContext — pyudbm rejects
    # Federation ops across two different (even structurally-identical)
    # Context objects with ValueError("...same context..."). Encountered
    # this for real on the first run; see /tmp/SPIKE2.md.
    zctx = res["zctx"]
    M = res["M"]
    bounds = {cn: M for cn in zctx.clock_names}

    bounded_count = 0
    n_clip_empty = 0
    n_clip_unbounded = 0
    terminal_bounded = 0
    n_terminal = 0

    extrap_purestr = set()
    clipped_purestr = set()
    clipped_lockeyed = set()
    group_lockeyed: dict = {}
    group_purestr: dict = {}

    for key, (Z, loc) in items:
        dbm = Z.to_dbm_list()[0]
        b_ok = zone_ops.is_bounded(dbm)
        if b_ok:
            bounded_count += 1
        if all(s == "done" for s in loc):
            n_terminal += 1
            if b_ok:
                terminal_bounded += 1

        s_extrap = str(Z)
        Z_clip = zone_ops.clip_to_k_box(Z, zctx, bounds)
        s_clip = str(Z_clip)

        extrap_purestr.add(s_extrap)
        clipped_purestr.add(s_clip)
        clipped_lockeyed.add((loc, s_clip))
        group_lockeyed.setdefault((loc, s_clip), set()).add((loc, s_extrap))
        group_purestr.setdefault(s_clip, set()).add(s_extrap)

        if Z_clip.is_empty():
            n_clip_empty += 1
        else:
            dbm_clip = Z_clip.to_dbm_list()[0]
            if not zone_ops.is_bounded(dbm_clip):
                n_clip_unbounded += 1

    merge_groups_lockeyed = {k: v for k, v in group_lockeyed.items() if len(v) >= 2}
    merge_groups_purestr = {k: v for k, v in group_purestr.items() if len(v) >= 2}

    sample = items if n <= VSAMPLE else RNG.sample(items, VSAMPLE)
    dims = []
    ev_list, ef_list, qv_list, qf_list = [], [], [], []
    disagreements = []
    ev_unknown = 0
    qv_unknown = 0

    for key, (Z, loc) in sample:
        Z_clip = zone_ops.clip_to_k_box(Z, zctx, bounds)
        if Z_clip.is_empty():
            continue
        dbm_clip = Z_clip.to_dbm_list()[0]
        dims.append(affine_dim(dbm_clip))
        ev, ef = exact_vertices_and_facets(dbm_clip)
        qv, qf = qhull_vertices_and_facets(dbm_clip)
        if ev is None:
            ev_unknown += 1
        else:
            ev_list.append(ev)
            if ef is not None:
                ef_list.append(ef)
        if qv is None:
            qv_unknown += 1
        else:
            qv_list.append(qv)
            if qf is not None:
                qf_list.append(qf)
        if ev is not None and qv is not None and ev != qv:
            disagreements.append((loc, str(Z), ev, qv))

    row = {
        "instance": name,
        "n_states": n,
        "truncated": res["truncated"],
        "max_depth": res["max_depth"],
        "wall_s": res["wall_s"],
        "bounded_frac": bounded_count / n,
        "n_terminal": n_terminal,
        "terminal_bounded_frac": (terminal_bounded / n_terminal) if n_terminal else None,
        "n_clip_empty": n_clip_empty,
        "n_clip_unbounded": n_clip_unbounded,
        "distinct_extrap_purestr": len(extrap_purestr),
        "distinct_clipped_purestr": len(clipped_purestr),
        "distinct_clipped_lockeyed": len(clipped_lockeyed),
        "n_states_lockeyed_baseline": n,
        "n_merge_groups_lockeyed": len(merge_groups_lockeyed),
        "max_merge_group_lockeyed": max((len(v) for v in merge_groups_lockeyed.values()), default=0),
        "n_merge_groups_purestr": len(merge_groups_purestr),
        "max_merge_group_purestr": max((len(v) for v in merge_groups_purestr.values()), default=0),
        "vertex_sample_n": len(sample),
        "affine_dim_mean_clipped": float(np.mean(dims)) if dims else None,
        "affine_dim_max_clipped": int(max(dims)) if dims else None,
        "exact_vertices_mean": float(np.mean(ev_list)) if ev_list else None,
        "exact_vertices_unknown_frac": ev_unknown / len(sample) if sample else None,
        "exact_facets_mean": float(np.mean(ef_list)) if ef_list else None,
        "qhull_vertices_mean": float(np.mean(qv_list)) if qv_list else None,
        "qhull_vertices_unknown_frac": qv_unknown / len(sample) if sample else None,
        "qhull_facets_mean": float(np.mean(qf_list)) if qf_list else None,
        "n_vertex_disagreements": len(disagreements),
        "disagreement_examples": disagreements[:5],
    }
    return row


if __name__ == "__main__":
    all_rows = []
    for name in INSTANCES:
        row = run_one(name)
        all_rows.append(row)
        print(
            f"{name:8s} n={row['n_states']:6d} trunc={row['truncated']!s:5s} "
            f"bounded_frac={row['bounded_frac']:.4f} term_bounded_frac={row['terminal_bounded_frac']} "
            f"clip_empty={row['n_clip_empty']} clip_unbounded={row['n_clip_unbounded']} "
            f"distinct(extrap/clip_purestr/clip_lockeyed)={row['distinct_extrap_purestr']}/{row['distinct_clipped_purestr']}/{row['distinct_clipped_lockeyed']} "
            f"merge_groups(lockeyed,max)={row['n_merge_groups_lockeyed']},{row['max_merge_group_lockeyed']} "
            f"merge_groups(purestr,max)={row['n_merge_groups_purestr']},{row['max_merge_group_purestr']} "
            f"dim(mean/max)={row['affine_dim_mean_clipped']}/{row['affine_dim_max_clipped']} "
            f"exact_vert_mean={row['exact_vertices_mean']} (unk={row['exact_vertices_unknown_frac']}) "
            f"qhull_vert_mean={row['qhull_vertices_mean']} (unk={row['qhull_vertices_unknown_frac']}) "
            f"disagreements={row['n_vertex_disagreements']}",
            flush=True,
        )

    with open("/tmp/spike2_taskA_results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print("Saved /tmp/spike2_taskA_results.json")
