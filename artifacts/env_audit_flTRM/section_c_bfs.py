"""
Independent zone-graph BFS verification for all 3 TRM Frozen-Lake variants.

Re-implements the "corner expansion" symbolic BFS technique independently
from run_*.py (not by importing their count_zone_graph functions) to
cross-check the README's claimed constant zone-BFS counts (385 / 100 / 1905).

All clock domains here are INTEGER-valued (no continuous/real-valued time),
so ordinary Python ints give exact arithmetic; Fraction is not needed for
this environment (there is no real-valued/rational clock geometry to get
wrong the way qhull could get wrong on continuous zones).
"""
import sys, os, json
from collections import deque

FL_DIR = "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/frozen_lake"
sys.path.insert(0, FL_DIR)

from trm_comparison.grid import ACTIONS as G_ACTIONS, _DELTAS as G_DELTAS, HOLES as G_HOLES, \
    CHECKPOINTS as G_CP, START as G_START, ROWS as G_ROWS, COLS as G_COLS
from trm_comparison.trm import INSTANCES as TRM_INSTANCES, U_TERMINAL as T_TERM, U1 as T_U1, \
    RM_GOAL_EVENTS as T_GOALEV

from trm_comparison.sg_grid import ACTIONS as SG_ACTIONS, _DELTAS as SG_DELTAS, HOLES as SG_HOLES, \
    CHECKPOINTS as SG_CP, START as SG_START, ROWS as SG_ROWS, COLS as SG_COLS
from trm_comparison.sg_trm import INSTANCES as SG_INSTANCES, SG_U_TERMINAL, SG_U1

from trm_comparison.hole_grid import ACTIONS as H_ACTIONS, _DELTAS as H_DELTAS, HOLES as H_HOLES, \
    CHECKPOINTS as H_CP, START as H_START, ROWS as H_ROWS, COLS as H_COLS, \
    UP as H_UP, DOWN as H_DOWN, LEFT as H_LEFT, RIGHT as H_RIGHT
from trm_comparison.hole_trm import INSTANCES as HOLE_INSTANCES, U_TERMINAL as H_TERM, U1 as H_U1, \
    U3 as H_U3, RM_GOAL_EVENTS as H_GOALEV

CAP = 50_000


def _reachable_positions(rows, cols, deltas, r, c, action):
    """All grid cells reachable from (r,c) via `action`, including the two
    perpendicular slip outcomes (slip_prob>0 => both are possible)."""
    result = set()
    dr, dc = deltas[action]
    nr, nc = max(0, min(rows - 1, r + dr)), max(0, min(cols - 1, c + dc))
    result.add((nr, nc))
    perp = [2, 3] if action in (0, 1) else [0, 1]
    for pa in perp:
        pdr, pdc = deltas[pa]
        pnr, pnc = max(0, min(rows - 1, r + pdr)), max(0, min(cols - 1, c + pdc))
        result.add((pnr, pnc))
    return result


# ───────────────────────── three-goal variant ─────────────────────────
def bfs_zone_threegoal(trm, cap=CAP):
    def zone_key(pos, rm, x, y):
        return (pos, rm, 0 if x < trm.c_min else 1, 0 if y <= trm.D else 1)

    def corners(xn, yn):
        xs = [0, max(0, trm.c_min - 1)] if xn == 0 else [trm.c_min]
        ys = [0, trm.D] if yn == 0 else [trm.D + 1]
        return [(x, y) for x in xs for y in ys]

    init = zone_key(G_START, T_U1, 0, 0)
    visited = {init}
    q = deque([init])
    truncated = False
    while q:
        if len(visited) >= cap:
            truncated = True
            break
        zk = q.popleft()
        pos, rm, xn, yn = zk
        if rm == T_TERM:
            continue
        r, c = pos
        for delay in range(trm.d_max + 1):
            for x_c, y_c in corners(xn, yn):
                x_new, y_new = x_c + delay + 1, y_c + delay + 1
                for action in G_ACTIONS:
                    for npos in _reachable_positions(G_ROWS, G_COLS, G_DELTAS, r, c, action):
                        if npos in G_HOLES:
                            nk = zone_key(npos, T_TERM, 0, 0)
                            if nk not in visited:
                                visited.add(nk)
                            continue
                        event = G_CP.get(npos, "")
                        goal_event = T_GOALEV.get(rm)
                        if event == goal_event:
                            guard_ok = (x_new >= trm.c_min) and (y_new <= trm.D)
                            if guard_ok:
                                next_rm = T_TERM if rm == 3 else rm + 1
                                nk = zone_key(npos, next_rm, 0, y_new)
                            else:
                                next_rm = T_TERM
                                nk = zone_key(npos, T_TERM, x_new, y_new)
                            if nk not in visited:
                                visited.add(nk)
                                if next_rm != T_TERM:
                                    q.append(nk)
                        else:
                            nk = zone_key(npos, rm, x_new, y_new)
                            if nk not in visited:
                                visited.add(nk)
                                q.append(nk)
    return len(visited), truncated


# ───────────────────────── single-goal variant ─────────────────────────
def bfs_zone_singlegoal(trm, cap=CAP):
    def y_node(y):
        if y < trm.c_min:
            return 0
        elif y <= trm.D:
            return 1
        return 2

    def zone_key(pos, rm, y):
        return (pos, rm, y_node(y))

    def corners(yn):
        if yn == 0:
            return [0, max(0, trm.c_min - 1)]
        elif yn == 1:
            return [trm.c_min, trm.D]
        return [trm.D + 1]

    init = zone_key(SG_START, SG_U1, 0)
    visited = {init}
    q = deque([init])
    truncated = False
    while q:
        if len(visited) >= cap:
            truncated = True
            break
        zk = q.popleft()
        pos, rm, yn = zk
        if rm == SG_U_TERMINAL:
            continue
        r, c = pos
        for y_c in corners(yn):
            y_new = y_c + 1
            for action in SG_ACTIONS:
                for npos in _reachable_positions(SG_ROWS, SG_COLS, SG_DELTAS, r, c, action):
                    if npos in SG_HOLES:
                        nk = (npos, SG_U_TERMINAL, 0)
                        if nk not in visited:
                            visited.add(nk)
                        continue
                    event = SG_CP.get(npos, "")
                    if event == "c":
                        nk = (npos, SG_U_TERMINAL, y_node(y_new))
                        if nk not in visited:
                            visited.add(nk)
                    else:
                        nk = zone_key(npos, rm, y_new)
                        if nk not in visited:
                            visited.add(nk)
                            q.append(nk)
    return len(visited), truncated


# ───────────────────────── hole-disable variant ─────────────────────────
def bfs_zone_hole(trm, cap=CAP):
    def zone_key(pos, rm_u, x, y, h, da):
        xn = 0 if x < trm.c_min else 1
        yn = 0 if y <= trm.D else 1
        hn = 0 if h < trm.x_h else 1
        d = da if hn == 0 else -1
        return (pos, rm_u, xn, yn, hn, d)

    def corners(xn, yn, hn):
        xs = [0, max(0, trm.c_min - 1)] if xn == 0 else [trm.c_min]
        ys = [0, trm.D] if yn == 0 else [trm.D + 1]
        hs = [0, max(0, trm.x_h - 1)] if hn == 0 else [trm.x_h]
        return [(x, y, h) for x in xs for y in ys for h in hs]

    def all_reachable(r, c, action):
        results = set()
        perp = [H_LEFT, H_RIGHT] if action in (H_UP, H_DOWN) else [H_UP, H_DOWN]
        for act in [action] + perp:
            dr, dc = H_DELTAS[act]
            nr, nc = max(0, min(H_ROWS - 1, r + dr)), max(0, min(H_COLS - 1, c + dc))
            results.add(((nr, nc), act))
        return results

    init = zone_key(H_START, H_U1, 0, 0, trm.x_h, -1)
    visited = {init}
    q = deque([init])
    truncated = False
    while q:
        if len(visited) >= cap:
            truncated = True
            break
        key = q.popleft()
        pos, rm_u, xn, yn, hn, da = key
        if rm_u == H_TERM:
            continue
        for x, y, h in corners(xn, yn, hn):
            avail = [a for a in H_ACTIONS if not (hn == 0 and a == da)]
            for intended in avail:
                for (npos, actual_dir) in all_reachable(*pos, intended):
                    if npos in H_HOLES:
                        nx, ny, nh, nda = x + 1, y + 1, 0, intended
                        nk = zone_key(pos, rm_u, nx, ny, nh, nda)
                        if nk not in visited:
                            visited.add(nk)
                            q.append(nk)
                        continue
                    nx, ny = x + 1, y + 1
                    nh = h + 1
                    new_da = -1 if nh >= trm.x_h else da
                    event = H_CP.get(npos, "")
                    goal_ev = H_GOALEV.get(rm_u)
                    if event == goal_ev:
                        if trm.guard_satisfied(nx, ny):
                            nrm = H_TERM if rm_u == H_U3 else rm_u + 1
                            nk = zone_key(npos, nrm, 0, ny, nh, new_da)
                        else:
                            nk = zone_key(npos, H_TERM, nx, ny, nh, new_da)
                    else:
                        nk = zone_key(npos, rm_u, nx, ny, nh, new_da)
                    if nk not in visited:
                        visited.add(nk)
                        if not (event == goal_ev and trm.guard_satisfied(nx, ny) and rm_u == H_U3):
                            q.append(nk)
    return len(visited), truncated


results = {}

print("=== three-goal (TRM) ===")
results["three_goal"] = {}
for name in ["TRM_01", "TRM_03", "TRM_06"]:
    trm = TRM_INSTANCES[name]
    n, trunc = bfs_zone_threegoal(trm)
    print(f"  {name}: c_min={trm.c_min} D={trm.D} d_max={trm.d_max} -> zone BFS nodes = {n}{' (TRUNCATED)' if trunc else ''}")
    results["three_goal"][name] = dict(c_min=trm.c_min, D=trm.D, d_max=trm.d_max, zone_bfs=n, truncated=trunc)

print("=== single-goal (SG) ===")
results["single_goal"] = {}
for name in ["SG_01", "SG_03", "SG_05"]:
    trm = SG_INSTANCES[name]
    n, trunc = bfs_zone_singlegoal(trm)
    print(f"  {name}: c_min={trm.c_min} D={trm.D} -> zone BFS nodes = {n}{' (TRUNCATED)' if trunc else ''}")
    results["single_goal"][name] = dict(c_min=trm.c_min, D=trm.D, zone_bfs=n, truncated=trunc)

print("=== hole-disable (HOLE) ===")
results["hole"] = {}
for name in ["HOLE_01", "HOLE_03", "HOLE_05"]:
    trm = HOLE_INSTANCES[name]
    n, trunc = bfs_zone_hole(trm)
    print(f"  {name}: c_min={trm.c_min} D={trm.D} x_h={trm.x_h} -> zone BFS nodes = {n}{' (TRUNCATED)' if trunc else ''}")
    results["hole"][name] = dict(c_min=trm.c_min, D=trm.D, x_h=trm.x_h, zone_bfs=n, truncated=trunc)

with open("/tmp/env_audit_flTRM/section_c_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved /tmp/env_audit_flTRM/section_c_results.json")
