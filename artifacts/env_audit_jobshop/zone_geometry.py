"""Section C: BFS-explore real pyudbm zone graph for PJS_01..PJS_05,
sample zones, compute affine dimension / boundedness / facets / strictness /
exact vertex counts via fractions.Fraction linear algebra.

Run from /tmp/env_audit_jobshop/ (avoid pyudbm import-path collision).
"""
import pyudbm  # import BEFORE sys.path is polluted with the repo root
import sys, os, time, random, json
from collections import deque
from fractions import Fraction

sys.path.insert(0, '/Users/falah/Desktop/git-projects/RL-for-zones')

from prob_jobshop_ext.benchmarks.all_instances import get_instance_by_name
from prob_jobshop_ext.zone_ops import (
    ZoneContext, init_zone, apply_guard, reset_clocks, apply_delay, normalize, min_cost
)
from prob_jobshop_ext.symbolic_env import SymbolicJobShopEnv

INSTANCES = ["PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05"]
STATE_CAP = 50_000
SAMPLE_CAP = 2000
VERTEX_SAMPLE_CAP = 200
VERTEX_TIME_BUDGET_S = 90  # per-instance wall time budget for vertex counting

_UDBM_INF = 1073741823


def bfs_zone_graph(instance, zctx, M, state_cap=STATE_CAP):
    env = SymbolicJobShopEnv(instance, seed=0)
    loc0, actions0 = env.reset()
    Z0 = init_zone(zctx, env.invariant(loc0))
    Z0 = apply_delay(Z0, zctx, env.invariant(loc0), 1)
    Z0 = normalize(Z0, zctx, M)
    key0 = (loc0, str(Z0))

    states = {key0: Z0}
    snap0 = env.save_state()
    queue = deque([(loc0, Z0, actions0, key0, snap0)])
    visited_trans = set()
    truncated = False
    max_fed_dbms = 1

    while queue:
        if len(states) >= state_cap:
            truncated = True
            break
        loc, Z, actions, key, snap = queue.popleft()
        if key in visited_trans:
            continue
        visited_trans.add(key)

        for action in actions:
            Zg = apply_guard(Z, zctx, action["guards"])
            if Zg is None:
                continue
            Zr = reset_clocks(Zg, zctx, action["resets"])

            for branch in action["branches"]:
                Zcurr = apply_delay(Zr, zctx, env.invariant(branch["next_loc"]), 1)
                valid = True
                for step in branch["path"]:
                    for cname, exact_d in step["completion_guards"]:
                        Zcurr = Zcurr & (zctx.clocks[cname] >= exact_d)
                        if Zcurr.is_empty():
                            valid = False
                            break
                    if not valid:
                        break
                    Zcurr = apply_delay(Zcurr, zctx, env.invariant(step["next_loc"]), 1)
                if not valid or Zcurr.is_empty():
                    continue

                Zb = normalize(Zcurr, zctx, M)
                dec_loc = branch["decision_loc"]
                next_key = (dec_loc, str(Zb))

                if next_key not in states and len(states) < state_cap:
                    states[next_key] = Zb
                    ndbms = len(Zb.to_dbm_list())
                    if ndbms > max_fed_dbms:
                        max_fed_dbms = ndbms
                    env.restore_state(snap)
                    next_loc, next_actions, done = env.step_deterministic(
                        action["action_id"], branch["d"]
                    )
                    next_snap = env.save_state()
                    if not done:
                        queue.append((dec_loc, Zb, next_actions, next_key, next_snap))

    return states, truncated, max_fed_dbms


def dbm_matrix(dbm, n):
    """Return (bound, strict, inf) as n+1 x n+1 arrays, indices 0..n (0=reference)."""
    bnd = [[0] * (n + 1) for _ in range(n + 1)]
    strict = [[False] * (n + 1) for _ in range(n + 1)]
    inf = [[False] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                continue
            bnd[i][j] = dbm.bound(i, j)
            strict[i][j] = dbm.is_strict(i, j)
            inf[i][j] = dbm.is_infinity(i, j)
    return bnd, strict, inf


def affine_dimension(bnd, strict, inf, n):
    """Union-find over 0..n; union i,j iff bound(i,j)+bound(j,i)==0 and both non-strict
    (and neither infinite). dimension = components - 1."""
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if inf[i][j] or inf[j][i]:
                continue
            if bnd[i][j] + bnd[j][i] == 0 and not strict[i][j] and not strict[j][i]:
                union(i, j)

    n_components = len({find(x) for x in range(n + 1)})
    return n_components - 1


def is_bounded(bnd, inf, n):
    """Every clock i in 1..n has finite bound(0,i) [x_i>=... lower] and bound(i,0) [x_i<=... upper]."""
    for i in range(1, n + 1):
        if inf[0][i] or inf[i][0]:
            return False
    return True


def facet_constraints(bnd, strict, inf, n):
    """Return list of non-redundant (i,j) among off-diagonal finite cells.

    Redundant iff exists k (k != i,j) with bnd[i][k]+bnd[k][j]==bnd[i][j] and
    matching strictness: composed_strict = strict[i][k] or strict[k][j]; matches
    iff composed_strict == strict[i][j].
    """
    cells = [(i, j) for i in range(n + 1) for j in range(n + 1) if i != j and not inf[i][j]]
    non_redundant = []
    for (i, j) in cells:
        redundant = False
        for k in range(n + 1):
            if k == i or k == j:
                continue
            if inf[i][k] or inf[k][j]:
                continue
            if bnd[i][k] + bnd[k][j] == bnd[i][j]:
                composed_strict = strict[i][k] or strict[k][j]
                if composed_strict == strict[i][j]:
                    redundant = True
                    break
        if not redundant:
            non_redundant.append((i, j))
    return non_redundant


def has_strict(strict, inf, n):
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j and not inf[i][j] and strict[i][j]:
                return True
    return False


def constraint_to_equation(i, j, bound, n):
    """Constraint x_i - x_j <= bound (x_0 == 0 fixed).
    Returns (coeffs[1..n] as Fractions, rhs) for the EQUALITY x_i - x_j == bound.
    """
    coeffs = [Fraction(0)] * (n + 1)  # index 0 unused (x_0 fixed at 0)
    if i != 0:
        coeffs[i] += 1
    if j != 0:
        coeffs[j] -= 1
    rhs = Fraction(bound)
    return coeffs, rhs


def solve_exact(rows, n):
    """Gauss-Jordan elimination with Fraction. rows: list of (coeffs[1..n], rhs).
    Returns solution dict {idx: Fraction} or None if singular/inconsistent/underdetermined.
    """
    # Build augmented matrix restricted to columns 1..n
    A = [[c[k] for k in range(1, n + 1)] + [r] for c, r in rows]
    m = len(A)
    if m != n:
        return None
    # Gauss-Jordan
    for col in range(n):
        pivot_row = None
        for row in range(col, m):
            if A[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            return None  # singular
        A[col], A[pivot_row] = A[pivot_row], A[col]
        pivot_val = A[col][col]
        A[col] = [v / pivot_val for v in A[col]]
        for row in range(m):
            if row != col and A[row][col] != 0:
                factor = A[row][col]
                A[row] = [A[row][k] - factor * A[col][k] for k in range(n + 1)]
    solution = {}
    for row in range(n):
        solution[row + 1] = A[row][n]
    return solution


def check_solution(solution, bnd, strict, inf, n):
    """Check the solution (dict idx->Fraction, idx in 1..n) against ALL constraints."""
    x = {0: Fraction(0)}
    x.update(solution)
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j or inf[i][j]:
                continue
            lhs = x[i] - x[j]
            b = Fraction(bnd[i][j])
            if strict[i][j]:
                if not (lhs < b):
                    return False
            else:
                if not (lhs <= b):
                    return False
    return True


from itertools import combinations


def exact_vertex_count(bnd, strict, inf, n, non_redundant, time_budget=None, t_start=None):
    """Enumerate n-subsets of non_redundant constraints, solve exactly, dedupe."""
    candidates = non_redundant
    seen = set()
    count = 0
    for subset in combinations(candidates, n):
        if time_budget is not None and t_start is not None:
            if time.time() - t_start > time_budget:
                return count, True  # truncated
        rows = []
        for (i, j) in subset:
            coeffs, rhs = constraint_to_equation(i, j, bnd[i][j], n)
            rows.append((coeffs, rhs))
        sol = solve_exact(rows, n)
        if sol is None:
            continue
        if check_solution(sol, bnd, strict, inf, n):
            key = tuple(sol[k] for k in range(1, n + 1))
            if key not in seen:
                seen.add(key)
                count += 1
    return count, False


def analyze_instance(name):
    print(f"=== {name} ===", flush=True)
    inst = get_instance_by_name(name)
    zctx = ZoneContext(inst)
    M = inst.worst_case_makespan_upper_bound()
    n_clocks = len(zctx.clock_names)

    t0 = time.time()
    states, truncated, max_fed_dbms = bfs_zone_graph(inst, zctx, M, STATE_CAP)
    bfs_time = time.time() - t0
    n_states = len(states)
    print(f"  BFS: {n_states} states, truncated={truncated}, max_fed_dbms={max_fed_dbms}, "
          f"wall={bfs_time:.2f}s", flush=True)

    keys = list(states.keys())
    rng = random.Random(12345)
    sample_keys = keys if len(keys) <= SAMPLE_CAP else rng.sample(keys, SAMPLE_CAP)

    affine_dims = []
    bounded_flags = []
    strict_flags = []
    facet_counts = []
    single_concrete_flags = []  # zone contains exactly one concrete point (affine dim 0)
    fed_dbm_counts = []

    for k in sample_keys:
        Z = states[k]
        dbms = Z.to_dbm_list()
        fed_dbm_counts.append(len(dbms))
        dbm = dbms[0]
        bnd, strict, inf = dbm_matrix(dbm, n_clocks)
        ad = affine_dimension(bnd, strict, inf, n_clocks)
        affine_dims.append(ad)
        bounded_flags.append(is_bounded(bnd, inf, n_clocks))
        strict_flags.append(has_strict(strict, inf, n_clocks))
        nr = facet_constraints(bnd, strict, inf, n_clocks)
        facet_counts.append(len(nr))
        single_concrete_flags.append(ad == 0)

    max_fed_dbms_sample = max(fed_dbm_counts) if fed_dbm_counts else 1

    # ---- vertex counting on a sub-sample ----
    vertex_keys = sample_keys if len(sample_keys) <= VERTEX_SAMPLE_CAP else rng.sample(sample_keys, VERTEX_SAMPLE_CAP)
    vertex_counts = []
    t_vstart = time.time()
    n_truncated_zones = 0
    per_zone_times = []
    for k in vertex_keys:
        if time.time() - t_vstart > VERTEX_TIME_BUDGET_S:
            break
        Z = states[k]
        dbm = Z.to_dbm_list()[0]
        bnd, strict, inf = dbm_matrix(dbm, n_clocks)
        nr = facet_constraints(bnd, strict, inf, n_clocks)
        tz0 = time.time()
        vc, trunc = exact_vertex_count(bnd, strict, inf, n_clocks, nr,
                                        time_budget=5.0, t_start=time.time())
        per_zone_times.append(time.time() - tz0)
        if trunc:
            n_truncated_zones += 1
        else:
            vertex_counts.append(vc)
    v_wall = time.time() - t_vstart

    def stats(lst):
        if not lst:
            return {"mean": None, "min": None, "max": None}
        return {"mean": sum(lst) / len(lst), "min": min(lst), "max": max(lst)}

    result = {
        "instance": name,
        "n_clocks": n_clocks,
        "clock_names": zctx.clock_names,
        "n_states_bfs": n_states,
        "bfs_truncated": truncated,
        "bfs_wall_s": bfs_time,
        "max_fed_dbms_ever": max_fed_dbms,
        "max_fed_dbms_sample": max_fed_dbms_sample,
        "sample_size": len(sample_keys),
        "affine_dim_stats": stats(affine_dims),
        "affine_dim_hist": {str(d): affine_dims.count(d) for d in sorted(set(affine_dims))},
        "frac_bounded": sum(bounded_flags) / len(bounded_flags) if bounded_flags else None,
        "frac_strict": sum(strict_flags) / len(strict_flags) if strict_flags else None,
        "facet_count_stats": stats(facet_counts),
        "facet_count_hist": {str(c): facet_counts.count(c) for c in sorted(set(facet_counts))},
        "frac_single_concrete": sum(single_concrete_flags) / len(single_concrete_flags) if single_concrete_flags else None,
        "vertex_sample_size": len(vertex_keys),
        "vertex_counts_computed": len(vertex_counts),
        "vertex_truncated_zones": n_truncated_zones,
        "vertex_count_stats": stats(vertex_counts),
        "vertex_count_hist": {str(c): vertex_counts.count(c) for c in sorted(set(vertex_counts))},
        "vertex_wall_s": v_wall,
        "vertex_mean_per_zone_s": (sum(per_zone_times) / len(per_zone_times)) if per_zone_times else None,
    }
    print(json.dumps(result, indent=2, default=str), flush=True)
    return result


if __name__ == "__main__":
    all_results = []
    for name in INSTANCES:
        r = analyze_instance(name)
        all_results.append(r)
    with open("/tmp/env_audit_jobshop/zone_geometry_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("DONE")
