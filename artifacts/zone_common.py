"""Shared helpers for the interval-duration zone-dimension spike.
THROWAWAY — /tmp only, not part of any package, not committed anywhere.
"""
from __future__ import annotations
import time
from collections import deque
from fractions import Fraction

import numpy as np


# ---------------------------------------------------------------------------
# affine_dim: exact zone-dimension metric
# ---------------------------------------------------------------------------

def affine_dim(dbm) -> int:
    """Number of independent free directions in the zone, excluding the
    reference-clock component.

    Two indices i,j (0 = reference clock) are PINNED (same affine value up to
    a known constant offset) iff:
      - neither (i,j) nor (j,i) is the infinity sentinel,
      - neither is strict,
      - bound(i,j) + bound(j,i) == 0   (x_i - x_j <= b and x_j - x_i <= -b
        together force x_i - x_j == b exactly).
    Union-find over {0, ..., dim-1} under 'pinned'; affine_dim =
    (number of connected components) - 1 (the component containing the
    reference clock does not count as a free dimension).
    """
    dim = dbm.dimension
    parent = list(range(dim))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(dim):
        for j in range(i + 1, dim):
            if dbm.is_infinity(i, j) or dbm.is_infinity(j, i):
                continue
            if dbm.is_strict(i, j) or dbm.is_strict(j, i):
                continue
            if dbm.bound(i, j) + dbm.bound(j, i) == 0:
                union(i, j)

    roots = {find(i) for i in range(dim)}
    return len(roots) - 1


def pinned_groups(dbm):
    """Return (group_of[i] for i in range(dim), offset_from_group_repr[i]).

    group_of[i] is the union-find root; offset[i] is the exact value of
    (x_i - x_group_repr) implied by the pinning chain (0 if i is its own
    representative). Only meaningful for indices in a pinned group of size
    > 1; representative chosen as min index in the group.
    """
    dim = dbm.dimension
    parent = list(range(dim))
    offset = [0] * dim   # offset[i] = x_i - x_parent[i], accumulated on find with path compression care

    def find(x):
        # iterative find with offset accumulation
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        root = x
        # recompute offsets along path relative to root
        acc = 0
        for node in reversed(path):
            acc += offset[node]
            offset[node] = acc
            parent[node] = root
        return root

    def union(a, b, delta_ab):
        # delta_ab = x_a - x_b (exact)
        ra, oa = find(a), offset[a]
        rb, ob = find(b), offset[b]
        if ra == rb:
            return
        # x_a - x_ra = oa ; x_b - x_rb = ob ; x_a - x_b = delta_ab
        # attach ra under rb: x_ra = x_a - oa = (x_b + delta_ab) - oa = x_rb + ob + delta_ab - oa
        parent[ra] = rb
        offset[ra] = ob + delta_ab - oa

    for i in range(dim):
        for j in range(i + 1, dim):
            if dbm.is_infinity(i, j) or dbm.is_infinity(j, i):
                continue
            if dbm.is_strict(i, j) or dbm.is_strict(j, i):
                continue
            b = dbm.bound(i, j)
            if b + dbm.bound(j, i) == 0:
                union(i, j, b)   # x_i - x_j == b

    groups = [find(i) for i in range(dim)]
    offs = [offset[i] for i in range(dim)]
    return groups, offs


def reduced_constraints(dbm):
    """Collapse pinned groups to representatives; return
    (representatives (sorted list of group roots), reduced list of
    (repr_i, repr_j, bound, strict) meaning x_repr_i - x_repr_j <= bound
    for the collapsed system, dropping intra-group and reference-only
    trivial constraints). Representative for reference clock's group is
    fixed at value 0.
    """
    dim = dbm.dimension
    groups, offs = pinned_groups(dbm)
    reps = sorted(set(groups))
    reduced = []
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            if dbm.is_infinity(i, j):
                continue
            gi, gj = groups[i], groups[j]
            if gi == gj:
                continue  # collapses to a trivial/redundant constraint within one group
            b = dbm.bound(i, j)
            strict = dbm.is_strict(i, j)
            # x_i - x_j <= b  =>  (x_gi + offs[i]) - (x_gj + offs[j]) <= b
            # => x_gi - x_gj <= b - offs[i] + offs[j]
            b2 = b - offs[i] + offs[j]
            reduced.append((gi, gj, b2, strict))
    return reps, reduced, groups, offs


def exact_vertices_fraction(dbm, cap_m=40, cap_k=6):
    """SPIKE3: exact rational vertex enumeration via Python `fractions.Fraction`
    arithmetic — NO scipy/qhull anywhere, no floating point at all. Enumerates
    k-subsets of the deduplicated (tightest-bound-per-pair — this is the
    "non-redundant" reduction used throughout this file) reduced constraint
    system in the affine_dim-collapsed free space, solves each subset as an
    exact linear system via Fraction Gaussian elimination, keeps solutions
    that satisfy every constraint exactly, dedupes.

    Returns (n_vertices, any_strict_seen). n_vertices is None (UNKNOWN,
    not a guess) if the subset count exceeds the tractability cap.
    any_strict_seen flags whether any constraint in the system was a strict
    inequality (none has been observed in this codebase across every prior
    spike — worth flagging loudly if it ever turns up here, since it would
    invalidate the "treat as <=" simplification this function relies on for
    solving; see /tmp/SPIKE3.md).
    """
    reps, reduced, groups, offs = reduced_constraints(dbm)
    ref_group = groups[0]
    free_reps = [r for r in reps if r != ref_group]
    k = len(free_reps)
    if k == 0:
        return 1, False
    idx_of = {r: i for i, r in enumerate(free_reps)}

    any_strict = False
    best = {}
    for (gi, gj, b2, strict) in reduced:
        if gi == ref_group and gj == ref_group:
            continue
        if strict:
            any_strict = True
        row = [Fraction(0)] * k
        if gi == ref_group:
            row[idx_of[gj]] = Fraction(-1)
            key = ("ref", idx_of[gj], -1)
        elif gj == ref_group:
            row[idx_of[gi]] = Fraction(1)
            key = ("ref", idx_of[gi], 1)
        else:
            row[idx_of[gi]] = Fraction(1)
            row[idx_of[gj]] = Fraction(-1)
            key = (idx_of[gi], idx_of[gj])
        bb = Fraction(int(b2))
        if key not in best or bb < best[key][1]:
            best[key] = (row, bb)

    if not best:
        return None, any_strict
    rows = list(best.values())
    m = len(rows)
    if m > cap_m or k > cap_k:
        return None, any_strict

    from itertools import combinations
    verts = set()
    for combo in combinations(range(m), k):
        A = [rows[i][0] for i in combo]
        b = [rows[i][1] for i in combo]
        x = _solve_fraction_system(A, b, k)
        if x is None:
            continue
        ok = True
        for row, bnd in rows:
            lhs = sum(row[i] * x[i] for i in range(k))
            if lhs > bnd:
                ok = False
                break
        if ok:
            verts.add(tuple(x))
    return len(verts), any_strict


def _solve_fraction_system(A, b, k):
    """Exact Gaussian-Jordan elimination with Fraction pivoting.
    A: k rows of k Fractions; b: k Fractions. Returns a list of k Fractions,
    or None if the system is singular (no unique solution — degenerate
    subset, not a vertex).
    """
    Maug = [list(A[i]) + [b[i]] for i in range(k)]
    for col in range(k):
        pivot = None
        for r in range(col, k):
            if Maug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            return None
        Maug[col], Maug[pivot] = Maug[pivot], Maug[col]
        pv = Maug[col][col]
        Maug[col] = [v / pv for v in Maug[col]]
        for r in range(k):
            if r != col and Maug[r][col] != 0:
                factor = Maug[r][col]
                Maug[r] = [Maug[r][c2] - factor * Maug[col][c2] for c2 in range(k + 1)]
    return [Maug[i][k] for i in range(k)]


def _reduced_halfspace_system(dbm):
    """Build the deduplicated (tightest-bound-per-pair) halfspace system in
    the affine_dim-reduced free space. Returns (A, b, k) or (None, None, 0)
    for a single point (k=0).
    """
    reps, reduced, groups, offs = reduced_constraints(dbm)
    ref_group = groups[0]
    free_reps = [r for r in reps if r != ref_group]
    k = len(free_reps)
    if k == 0:
        return None, None, 0
    idx_of = {r: i for i, r in enumerate(free_reps)}

    # Keep only the tightest bound per (row-key) pair — collapsing duplicate
    # original (i,j) pairs that landed on the same reduced pair after
    # grouping keeps the constraint count (and hence LP/qhull cost) minimal.
    best: dict = {}
    for (gi, gj, b2, strict) in reduced:
        if gi == ref_group and gj == ref_group:
            continue
        row = np.zeros(k)
        if gi == ref_group:
            row[idx_of[gj]] -= 1.0
            key = ("ref", idx_of[gj], -1)
        elif gj == ref_group:
            row[idx_of[gi]] += 1.0
            key = ("ref", idx_of[gi], +1)
        else:
            row[idx_of[gi]] += 1.0
            row[idx_of[gj]] -= 1.0
            key = (idx_of[gi], idx_of[gj])
        bb = float(b2) - (1e-9 if strict else 0.0)
        if key not in best or bb < best[key][1]:
            best[key] = (row, bb)

    if not best:
        return None, None, k
    A = np.array([v[0] for v in best.values()])
    b = np.array([v[1] for v in best.values()])
    return A, b, k


def exact_vertices(dbm):
    """Exact FINITE-vertex count in the affine_dim-reduced free subspace.

    Pinned (zero-slack) dimensions are collapsed first (same union-find as
    affine_dim), so this only ever enumerates vertices of a genuinely
    k = affine_dim(dbm)-dimensional polytope in the reduced space — avoiding
    combinatorial blow-up from the *original* (unreduced) dimension.

    NOTE (spike finding, see /tmp/SPIKE.md): an earlier version of this
    function used scipy's qhull-backed HalfspaceIntersection. That was
    WRONG and has been removed: these reduced polyhedra are routinely
    UNBOUNDED (the `delta` cost clock is deliberately never given an upper
    bound by zone_ops.normalize() — see AUDIT.md §1 — so any zone where
    `delta`'s group is a free dimension is an unbounded polyhedron in that
    direction). scipy's HalfspaceIntersection assumes/requires a bounded
    region and was empirically observed to fabricate a spurious extra vertex
    on unbounded inputs (verified on a concrete PJS_03 zone: qhull reported 3,
    correct answer is 2 finite vertices — (0,0) and (1,1) of the wedge
    {x>=0, y>=0, x<=1, x<=y}, which is unbounded as y->inf). The
    combinatorial tight-constraint-subset method below has no such issue: it
    only ever reports FINITE extreme points, which is the correct notion of
    "vertices" for a possibly-unbounded polyhedron, and was cross-validated
    against a hand-checked synthetic box (correctly gives 4/4).
    """
    A, b, k = _reduced_halfspace_system(dbm)
    if k == 0:
        return 1  # single point, no free dimensions
    if A is None:
        return None  # no constraints at all in a nonzero-dim free space — degenerate/unbounded
    return _exact_vertices_bruteforce(A, b, k)


def _exact_vertices_bruteforce(A, b, k):
    """Exact combinatorial tight-constraint-subset enumeration: for every
    combination of k constraint rows, solve as equalities and keep the
    solution if it satisfies every constraint. Correct for unbounded
    polyhedra (only ever finds finite extreme points, never invents a vertex
    at infinity). Caps out for tractability on the rare pathological case.
    """
    from itertools import combinations
    m = len(A)
    # SPIKE finding: affine_dim is often large (Step 1 refuted the low-dim
    # hypothesis), so C(m, k) blows up well before m=40/k=8 — empirically,
    # k=6,m=36 alone is a ~2M-combination search per zone and does not
    # finish in reasonable time across a several-hundred-zone sample. Capped
    # much lower; zones beyond this are reported as UNKNOWN for this metric
    # (see /tmp/SPIKE.md) rather than left to hang.
    if m > 18 or k > 4:
        return None  # UNKNOWN — combinatorially intractable at this budget
    verts = set()
    for combo in combinations(range(m), k):
        Asub = A[list(combo)]
        bsub = b[list(combo)]
        try:
            if abs(np.linalg.det(Asub)) < 1e-9:
                continue
            x = np.linalg.solve(Asub, bsub)
        except np.linalg.LinAlgError:
            continue
        # feasibility check against all constraints (small tolerance)
        if np.all(A @ x <= b + 1e-6):
            verts.add(tuple(np.round(x, 6)))
    return len(verts)


# ---------------------------------------------------------------------------
# SPIKE 2 additions: vertex+facet counting on CLIPPED (bounded) zones.
# Unlike Spike 1's zones (routinely unbounded because of `delta`), Task A's
# clip_to_k_box() zones are bounded by construction, so qhull's
# HalfspaceIntersection is a legitimate cross-check here (it was NOT valid in
# Spike 1 — see /tmp/SPIKE.md's qhull-on-unbounded-polyhedra finding).
# ---------------------------------------------------------------------------

def exact_vertices_points(dbm):
    """Like exact_vertices, but returns the actual point list in the
    affine_dim-reduced free space (or None if intractable/UNKNOWN), plus k.
    """
    A, b, k = _reduced_halfspace_system(dbm)
    if k == 0:
        return [], 0
    if A is None:
        return None, k
    from itertools import combinations
    m = len(A)
    if m > 18 or k > 4:
        return None, k
    pts = []
    seen = set()
    for combo in combinations(range(m), k):
        Asub = A[list(combo)]
        bsub = b[list(combo)]
        try:
            if abs(np.linalg.det(Asub)) < 1e-9:
                continue
            x = np.linalg.solve(Asub, bsub)
        except np.linalg.LinAlgError:
            continue
        if np.all(A @ x <= b + 1e-6):
            key = tuple(np.round(x, 6))
            if key not in seen:
                seen.add(key)
                pts.append(x)
    return pts, k


def _facets_from_points(pts, k):
    if k == 0:
        return 0
    if k == 1:
        return len(pts)  # bounded 1-polytope: 2 endpoints == 2 vertices == 2 facets
    if pts is None or len(pts) < k + 1:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(np.array(pts))
        return len(hull.equations)
    except Exception:
        return None


def exact_vertices_and_facets(dbm):
    """Finite-extreme-point method: (n_vertices, n_facets), None for UNKNOWN."""
    pts, k = exact_vertices_points(dbm)
    if k == 0:
        return 1, 0
    if pts is None:
        return None, None
    return len(pts), _facets_from_points(pts, k)


def qhull_vertices_and_facets(dbm):
    """qhull-based method (HalfspaceIntersection + ConvexHull), valid here
    because clip_to_k_box() zones are bounded by construction (unlike Spike
    1's routinely-unbounded zones, where this method was empirically wrong
    — see /tmp/SPIKE.md). Returns (n_vertices, n_facets), None for UNKNOWN.
    """
    A, b, k = _reduced_halfspace_system(dbm)
    if k == 0:
        return 1, 0
    if A is None:
        return None, None

    from scipy.optimize import linprog
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0
    c = np.zeros(k + 1)
    c[-1] = -1.0
    A_ub = np.hstack([A, norms.reshape(-1, 1)])
    lp = linprog(c, A_ub=A_ub, b_ub=b, bounds=[(None, None)] * k + [(1e-9, None)], method="highs")
    if lp.status != 0 or lp.x[-1] < 1e-7:
        return None, None

    interior = lp.x[:k]
    from scipy.spatial import HalfspaceIntersection
    try:
        from scipy.spatial import QhullError
    except ImportError:
        from scipy.spatial.qhull import QhullError
    halfspaces = np.hstack([A, -b.reshape(-1, 1)])
    try:
        hs = HalfspaceIntersection(halfspaces, interior)
    except QhullError:
        return None, None

    n_vert = len(hs.intersections)
    n_facet = _facets_from_points(list(hs.intersections), k)
    return n_vert, n_facet


# ---------------------------------------------------------------------------
# BFS zone-graph explorer (mirrors symbolic_rl.py / symbolic_optimal.py's
# _propagate_zone, generalised over an injected env/zone_ops module so the
# interval-duration variant can reuse it against a different symbolic_env).
# ---------------------------------------------------------------------------

def make_propagate(zone_ops_mod):
    apply_guard = zone_ops_mod.apply_guard
    reset_clocks = zone_ops_mod.reset_clocks
    apply_delay = zone_ops_mod.apply_delay
    normalize = zone_ops_mod.normalize

    def propagate(env, zctx, Z, M, action):
        Z_g = apply_guard(Z, zctx, action["guards"])
        if Z_g is None:
            return []
        Z_r = reset_clocks(Z_g, zctx, action["resets"])
        out = []
        for branch in action["branches"]:
            Z_curr = apply_delay(Z_r, zctx, env.invariant(branch["next_loc"]), 1)
            valid = True
            for step in branch["path"]:
                for cname, lo in step["completion_guards"]:
                    Z_curr = Z_curr & (zctx.clocks[cname] >= lo)
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

    return propagate


def make_propagate_one(zone_ops_mod):
    """SPIKE2 (Task B): like make_propagate, but propagates only the ONE
    branch matching a given sampled duration `d` — for tracking a symbolic
    zone alongside a concrete random rollout (mirrors symbolic_rl.py's
    `matching = [b for b in chosen_branches if b[0] == sampled_d]` pattern,
    factored out so it doesn't require re-deriving chosen_branches first).
    Returns (Z_b, decision_loc) or None if the branch leads to an empty zone
    or isn't found (shouldn't happen if the concrete env actually sampled d).
    """
    apply_guard = zone_ops_mod.apply_guard
    reset_clocks = zone_ops_mod.reset_clocks
    apply_delay = zone_ops_mod.apply_delay
    normalize = zone_ops_mod.normalize

    def propagate_one(env, zctx, Z, M, action, sampled_d):
        Z_g = apply_guard(Z, zctx, action["guards"])
        if Z_g is None:
            return None
        Z_r = reset_clocks(Z_g, zctx, action["resets"])
        branch = next((b for b in action["branches"] if b["d"] == sampled_d), None)
        if branch is None:
            return None
        Z_curr = apply_delay(Z_r, zctx, env.invariant(branch["next_loc"]), 1)
        valid = True
        for step in branch["path"]:
            for cname, lo in step["completion_guards"]:
                Z_curr = Z_curr & (zctx.clocks[cname] >= lo)
                if Z_curr.is_empty():
                    valid = False
                    break
            if not valid:
                break
            Z_curr = apply_delay(Z_curr, zctx, env.invariant(step["next_loc"]), 1)
        if not valid or Z_curr.is_empty():
            return None
        Z_b = normalize(Z_curr, zctx, M)
        return Z_b, branch["decision_loc"]

    return propagate_one


def explore(env_mod, zone_ops_mod, instance, state_cap=20_000):
    """BFS-explore the zone graph of `instance` using the given symbolic_env
    module and zone_ops module (either the stock ones or the interval-variant
    scratch-copy ones). Returns per-state records for downstream analysis.
    """
    SymbolicJobShopEnv = env_mod.SymbolicJobShopEnv
    ZoneContext = zone_ops_mod.ZoneContext
    init_zone = zone_ops_mod.init_zone
    apply_delay = zone_ops_mod.apply_delay
    normalize = zone_ops_mod.normalize
    propagate = make_propagate(zone_ops_mod)

    env = SymbolicJobShopEnv(instance)
    zctx = ZoneContext(instance)
    M = instance.worst_case_makespan_upper_bound()

    loc0, actions0 = env.reset()
    Z0 = init_zone(zctx, env.invariant(loc0))
    Z0 = apply_delay(Z0, zctx, env.invariant(loc0), 1)
    Z0 = normalize(Z0, zctx, M)
    key0 = (loc0, str(Z0))

    visited = {key0: 0}
    zone_by_key = {key0: (Z0, loc0)}
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
                    zone_by_key[next_key] = (Z_b, dec_loc)
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
        "zctx": zctx,   # SPIKE2: callers MUST reuse this exact ZoneContext for
                        # any further Federation op on these zones — pyudbm
                        # raises ValueError("...same context...") otherwise
                        # (a real rejection, encountered and fixed; see SPIKE2.md)
        "M": M,
        "zone_by_key": zone_by_key,
    }


def n_concurrent_active(loc):
    return sum(1 for s in loc if s.startswith("active_d"))
