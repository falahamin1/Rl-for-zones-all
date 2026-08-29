"""
THROWAWAY audit script — empirical zone-graph structure measurement.
Not part of the repo; not to be committed. Run from anywhere with:
    python3 /tmp/zone_audit.py

Requires `import pyudbm` to be warmed up BEFORE the repo root (which
contains a literal `pyudbm/` source directory) is appended to sys.path,
otherwise the namespace-package shadowing breaks `from pyudbm import Context`.
"""
from __future__ import annotations
import sys
import time
import random
import json
from collections import deque

import pyudbm as _warmup_pyudbm  # noqa: F401  (must import before repo root on path)

REPO_ROOT = "/Users/falah/Desktop/git-projects/RL-for-zones"
sys.path.append(REPO_ROOT)

import numpy as np
from scipy.optimize import linprog

from prob_jobshop_ext.symbolic_env import SymbolicJobShopEnv
from prob_jobshop_ext.zone_ops import (
    ZoneContext, init_zone, apply_guard, reset_clocks, apply_delay, normalize,
)
from prob_jobshop_ext.benchmarks.all_instances import get_all_instances
from prob_jobshop_ext.benchmarks.all_large_instances import get_all_large_instances

STATE_CAP = 200_000
SAMPLE_FOR_AB = 2000     # (a) off-diag count, (b) box fraction
SAMPLE_FOR_C = 200       # (c) vertex count (expensive LP-based) — reduced for tractability
N_RANDOM_DIRS = 24
RNG = random.Random(0)
NP_RNG = np.random.default_rng(0)


def propagate(env, zctx, Z, M, action):
    """Same logic as SymbolicRLAgent._propagate_zone / OptimalSymbolicSolver._propagate_zone."""
    Z_g = apply_guard(Z, zctx, action["guards"])
    if Z_g is None:
        return []
    Z_r = reset_clocks(Z_g, zctx, action["resets"])
    out = []
    for branch in action["branches"]:
        Z_curr = apply_delay(Z_r, zctx, env.invariant(branch["next_loc"]), 1)
        valid = True
        for step in branch["path"]:
            for cname, exact_d in step["completion_guards"]:
                Z_curr = Z_curr & (zctx.clocks[cname] >= exact_d)
                if Z_curr.is_empty():
                    valid = False
                    break
            if not valid:
                break
            Z_curr = apply_delay(Z_curr, zctx, env.invariant(step["next_loc"]), 1)
        if not valid or Z_curr.is_empty():
            continue
        Z_b = normalize(Z_curr, zctx, M)
        next_key = (branch["decision_loc"], str(Z_b))
        out.append((Z_b, next_key, branch["decision_loc"]))
    return out


def explore(instance, state_cap=STATE_CAP):
    env = SymbolicJobShopEnv(instance)
    zctx = ZoneContext(instance)
    M = instance.worst_case_makespan_upper_bound()

    loc0, actions0 = env.reset()
    Z0 = init_zone(zctx, env.invariant(loc0))
    Z0 = apply_delay(Z0, zctx, env.invariant(loc0), 1)
    Z0 = normalize(Z0, zctx, M)
    key0 = (loc0, str(Z0))

    visited = {key0: 0}          # key -> depth
    zone_by_key = {key0: Z0}
    queue = deque([(loc0, Z0, actions0, 0)])
    truncated = False
    max_depth = 0
    t0 = time.time()

    while queue:
        if len(visited) >= state_cap:
            truncated = True
            break
        loc, Z, actions, depth = queue.popleft()
        max_depth = max(max_depth, depth)

        for action in actions:
            for Z_b, next_key, dec_loc in propagate(env, zctx, Z, M, action):
                if next_key not in visited:
                    visited[next_key] = depth + 1
                    zone_by_key[next_key] = Z_b
                    if len(visited) >= state_cap:
                        truncated = True
                        break
                    done = all(s == "done" for s in dec_loc)
                    if not done:
                        child_actions = env._compute_actions(dec_loc)
                        queue.append((dec_loc, Z_b, child_actions, depth + 1))
            if truncated:
                break
        if truncated:
            break
    wall = time.time() - t0
    return {
        "n_states": len(visited),
        "truncated": truncated,
        "max_depth": max_depth,
        "wall_s": wall,
        "zone_by_key": zone_by_key,
        "zctx": zctx,
    }


def zone_stats(zone_str, dbm, zctx, clock_names_no_delta_incl_delta):
    """(a) genuine off-diagonal finite entries between two real (non-zero) clocks.
    (b) whether the zone is a box (a == 0).
    """
    n_clocks = len(zctx.clock_names)  # includes delta; dims 1..n_clocks map to these
    off_diag = 0
    for i in range(1, n_clocks + 1):
        for j in range(1, n_clocks + 1):
            if i == j:
                continue
            if not dbm.is_infinity(i, j):
                off_diag += 1
    return off_diag


def free_dims_and_bounds(dbm, zctx):
    """Return (list of clock indices with lower<upper, per-clock (lo,hi))."""
    n = len(zctx.clock_names)
    bounds = []
    for idx in range(1, n + 1):
        hi = None if dbm.is_infinity(idx, 0) else dbm.bound(idx, 0)
        lo = None if dbm.is_infinity(0, idx) else -dbm.bound(0, idx)
        bounds.append((lo, hi))
    return bounds


def count_vertices_lp(dbm, zctx, n_random=N_RANDOM_DIRS):
    """Approximate vertex count via extreme-point LP search along axis + random dirs.
    Exact for boxes; a documented lower-bound approximation otherwise.
    """
    n = len(zctx.clock_names)
    A_ub = []
    b_ub = []
    for i in range(0, n + 1):
        for j in range(0, n + 1):
            if i == j:
                continue
            if dbm.is_infinity(i, j):
                continue
            b = dbm.bound(i, j)
            strict = dbm.is_strict(i, j)
            # constraint: x_i - x_j <= b   (x_0 fixed to 0, so drop var 0)
            row = [0.0] * n
            bb = float(b) - (1e-6 if strict else 0.0)
            if i == 0 and j == 0:
                continue
            if i == 0:
                row[j - 1] += 1.0   # -x_j <= b  ->  x_j >= -b, handled as -x_j <= b
                A_ub.append(row); b_ub.append(bb)
                continue
            if j == 0:
                row[i - 1] += 1.0
                A_ub.append(row); b_ub.append(bb)
                continue
            row[i - 1] += 1.0
            row[j - 1] -= 1.0
            A_ub.append(row); b_ub.append(bb)

    A_ub = np.array(A_ub) if A_ub else np.zeros((0, n))
    b_ub = np.array(b_ub) if b_ub else np.zeros((0,))

    dirs = []
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0
        dirs.append(e.copy()); dirs.append(-e)
    for _ in range(n_random):
        v = NP_RNG.normal(size=n)
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            dirs.append(v / norm)

    verts = set()
    for d in dirs:
        res = linprog(c=d, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * n,
                       method="highs")
        if res.status == 0:
            key = tuple(np.round(res.x, 4))
            verts.add(key)
    return len(verts)


def run_instance(name, instance, do_vertices=True):
    res = explore(instance)
    zone_by_key = res["zone_by_key"]
    n_states = res["n_states"]

    keys = list(zone_by_key.keys())
    sample_ab = keys if len(keys) <= SAMPLE_FOR_AB else RNG.sample(keys, SAMPLE_FOR_AB)

    off_diag_counts = []
    box_flags = []
    for k in sample_ab:
        Z = zone_by_key[k]
        dbms = Z.to_dbm_list()
        # federation here is always a single convex DBM in this codebase (no unions ever taken)
        d0 = dbms[0]
        od = zone_stats(k[1], d0, res["zctx"], None)
        off_diag_counts.append(od)
        box_flags.append(od == 0)

    vertex_counts = []
    if do_vertices:
        sample_c = sample_ab if len(sample_ab) <= SAMPLE_FOR_C else RNG.sample(sample_ab, SAMPLE_FOR_C)
        for k in sample_c:
            Z = zone_by_key[k]
            d0 = Z.to_dbm_list()[0]
            vertex_counts.append(count_vertices_lp(d0, res["zctx"]))

    return {
        "name": name,
        "n_states": n_states,
        "truncated": res["truncated"],
        "max_depth": res["max_depth"],
        "wall_s": res["wall_s"],
        "n_fed_dbms_max": max(len(z.to_dbm_list()) for z in list(zone_by_key.values())[:200]),
        "off_diag_counts": off_diag_counts,
        "box_fraction": sum(box_flags) / len(box_flags) if box_flags else None,
        "vertex_counts": vertex_counts,
    }


if __name__ == "__main__":
    all_results = {}
    small = get_all_instances()
    large = get_all_large_instances()
    for inst in small + large:
        print(f"=== {inst.name} ===", flush=True)
        t0 = time.time()
        r = run_instance(inst.name, inst, do_vertices=True)
        print(f"  n_states={r['n_states']} truncated={r['truncated']} "
              f"max_depth={r['max_depth']} wall={r['wall_s']:.2f}s "
              f"box_frac={r['box_fraction']:.3f} "
              f"off_diag(mean/max)={np.mean(r['off_diag_counts']):.2f}/{max(r['off_diag_counts'])} "
              f"vertices(mean/max)={np.mean(r['vertex_counts']):.2f}/{max(r['vertex_counts'])} "
              f"[took {time.time()-t0:.1f}s total]", flush=True)
        all_results[inst.name] = r

    with open("/tmp/zone_audit_results.json", "w") as f:
        json.dump(
            {k: {kk: (vv if kk not in ("off_diag_counts", "vertex_counts") else vv)
                 for kk, vv in v.items()} for k, v in all_results.items()},
            f, default=str, indent=2,
        )
    print("Saved /tmp/zone_audit_results.json")
