"""Step 3: w-sweep comparison, current vs interval-duration semantics.
THROWAWAY — /tmp only.
"""
from __future__ import annotations
import sys
import json
import random

import pyudbm as _warmup_pyudbm  # noqa: F401
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")
sys.path.append("/tmp")

import numpy as np

from zone_common import affine_dim, exact_vertices, explore, n_concurrent_active

import prob_jobshop_spike.symbolic_env as spike_env
import prob_jobshop_spike.zone_ops as spike_zone_ops
from prob_jobshop_ext.benchmarks.all_instances import get_instance_by_name
from prob_jobshop_ext.benchmarks.all_large_instances import get_large_instance_by_name

STATE_CAP = 20_000
VERTEX_SAMPLE = 3000
RNG = random.Random(0)

INSTANCES = ["PJS_01", "PJS_02", "PJS_03", "PJS_04", "LJS_01"]
WIDTHS = [0.0, 0.1, 0.25, 0.4]


def get_inst(name):
    return get_large_instance_by_name(name) if name.startswith("LJS") else get_instance_by_name(name)


def run_one(name, w):
    spike_env.INTERVAL_WIDTH = w
    inst = get_inst(name)
    res = explore(spike_env, spike_zone_ops, inst, state_cap=STATE_CAP)

    items = list(res["zone_by_key"].items())
    dims = []
    concur = []
    for key, (Z, loc) in items:
        dbm = Z.to_dbm_list()[0]
        dims.append(affine_dim(dbm))
        concur.append(n_concurrent_active(loc))
    dims = np.array(dims)
    concur = np.array(concur)

    v_sample = items if len(items) <= VERTEX_SAMPLE else RNG.sample(items, VERTEX_SAMPLE)
    verts = []
    n_unknown = 0
    for key, (Z, loc) in v_sample:
        v = exact_vertices(Z.to_dbm_list()[0])
        if v is None:
            n_unknown += 1
        else:
            verts.append(v)

    corr = None
    if w == 0.25:
        if dims.std() > 0 and concur.std() > 0:
            corr = float(np.corrcoef(dims, concur)[0, 1])
        else:
            corr = float("nan")

    row = {
        "instance": name,
        "w": w,
        "n_states": res["n_states"],
        "truncated": res["truncated"],
        "max_depth": res["max_depth"],
        "wall_s": res["wall_s"],
        "affine_dim_mean": float(dims.mean()),
        "affine_dim_max": int(dims.max()),
        "frac_dim_ge2": float((dims >= 2).mean()),
        "vertex_sample_n": len(v_sample),
        "vertex_unknown_frac": n_unknown / len(v_sample) if v_sample else None,
        "vertex_mean_known": float(np.mean(verts)) if verts else None,
        "mean_concurrent_active": float(concur.mean()),
        "corr_dim_vs_concurrent_w025": corr,
    }
    return row


if __name__ == "__main__":
    all_rows = []
    for name in INSTANCES:
        for w in WIDTHS:
            row = run_one(name, w)
            all_rows.append(row)
            print(
                f"{name:8s} w={w:.2f} n_states={row['n_states']:6d} "
                f"trunc={row['truncated']!s:5s} depth={row['max_depth']:2d} "
                f"wall={row['wall_s']:6.2f}s dim(mean/max)={row['affine_dim_mean']:.2f}/{row['affine_dim_max']} "
                f"frac_dim>=2={row['frac_dim_ge2']:.3f} "
                f"vert(mean,known;unk_frac)={row['vertex_mean_known']},{row['vertex_unknown_frac']:.3f} "
                f"concur_mean={row['mean_concurrent_active']:.3f}"
                + (f" corr={row['corr_dim_vs_concurrent_w025']:.3f}" if row["corr_dim_vs_concurrent_w025"] is not None else ""),
                flush=True,
            )

    with open("/tmp/zone_dim_interval_results.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print("Saved /tmp/zone_dim_interval_results.json")
