"""Step 1: affine_dim sanity checks + current-semantics measurement.
THROWAWAY — /tmp only. Run from a cwd without a literal `pyudbm/` dir
(e.g. `cd scheduling && python3 /tmp/zone_dim.py`), see import-collision
note in AUDIT.md §1.
"""
from __future__ import annotations
import sys

import pyudbm as _warmup_pyudbm  # noqa: F401 — must import before repo root hits sys.path
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")
sys.path.append("/tmp")

import numpy as np

from zone_common import affine_dim, exact_vertices, explore, n_concurrent_active

from prob_jobshop_ext import symbolic_env as stock_symbolic_env
from prob_jobshop_ext import zone_ops as stock_zone_ops
from prob_jobshop_ext.benchmarks.all_instances import get_instance_by_name
from prob_jobshop_ext.benchmarks.all_large_instances import get_large_instance_by_name

ALL_LARGE_NAMES = {"LJS_01", "LJS_02", "LJS_03", "LJS_04", "LJS_05",
                    "LJS_06", "LJS_07", "LJS_08", "LJS_09", "LJS_10"}


def _get_instance(name):
    return get_large_instance_by_name(name) if name in ALL_LARGE_NAMES else get_instance_by_name(name)


def sanity_checks():
    from prob_jobshop_ext.zone_ops import ZoneContext
    from prob_jobshop_ext.benchmarks.all_instances import get_instance_by_name

    inst = get_instance_by_name("PJS_01")
    zctx = ZoneContext(inst)
    Z0 = zctx.ctx.get_zero_federation()
    dbm0 = Z0.to_dbm_list()[0]
    d0 = affine_dim(dbm0)

    Zu = Z0.up()
    dbmu = Zu.to_dbm_list()[0]
    du = affine_dim(dbmu)

    print(f"[sanity] zero zone affine_dim = {d0}  (expected 0)  -> {'PASS' if d0 == 0 else 'FAIL'}")
    print(f"[sanity] up()'d zero zone affine_dim = {du}  (expected 1) -> {'PASS' if du == 1 else 'FAIL'}")

    # exact_vertices sanity: zero zone (a single point) -> 1 vertex
    v0 = exact_vertices(dbm0)
    print(f"[sanity] zero zone exact_vertices = {v0} (expected 1) -> {'PASS' if v0 == 1 else 'FAIL'}")
    return d0 == 0 and du == 1 and v0 == 1


def measure_current(instance_names, state_cap=20_000):
    rows = []
    for name in instance_names:
        inst = _get_instance(name)
        res = explore(stock_symbolic_env, stock_zone_ops, inst, state_cap=state_cap)
        dims = []
        concur = []
        for key, (Z, loc) in res["zone_by_key"].items():
            dbm = Z.to_dbm_list()[0]
            dims.append(affine_dim(dbm))
            concur.append(n_concurrent_active(loc))
        dims = np.array(dims)
        concur = np.array(concur)
        hist = {int(k): int(v) for k, v in zip(*np.unique(dims, return_counts=True))}
        row = {
            "instance": name,
            "n_states": res["n_states"],
            "truncated": res["truncated"],
            "max_depth": res["max_depth"],
            "wall_s": res["wall_s"],
            "affine_dim_hist": hist,
            "affine_dim_mean": float(dims.mean()),
            "affine_dim_max": int(dims.max()),
            "mean_concurrent_active": float(concur.mean()),
        }
        rows.append(row)
        print(f"{name}: n_states={res['n_states']} truncated={res['truncated']} "
              f"max_depth={res['max_depth']} wall={res['wall_s']:.2f}s "
              f"affine_dim hist={hist} mean={row['affine_dim_mean']:.3f} "
              f"mean_concurrent_active={row['mean_concurrent_active']:.3f}", flush=True)
    return rows


if __name__ == "__main__":
    ok = sanity_checks()
    print(f"[sanity] overall: {'ALL PASS' if ok else 'SOME FAILED'}")
    print()
    print("=== Current (exact-duration) semantics, cap=20000 ===")
    rows = measure_current(["PJS_01", "PJS_02", "PJS_03", "PJS_04", "LJS_01"])
    import json
    with open("/tmp/zone_dim_current_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("Saved /tmp/zone_dim_current_results.json")
