"""
Section E — exact DP over the concrete (region-capped) state space, and
multi-zone / conflicting-optimal-action analysis.

Key structural fact exploited (verified analytically from the code):
  dt = delay + 1 >= 1 always (delay in {0,...,d_max}), and y (resp. the
  hole-cooldown/x clocks) never decreases within a phase, so a transition
  from clock-layer y_cap=k (k<D) always lands in a STRICTLY higher layer
  (k+dt, clipped to D). Only the top layer y_cap=D is self-referential
  (all transitions from it stay at D). This makes the DP acyclic across
  y_cap layers except for one small self-consistent top layer, which we
  solve by value iteration; every lower layer is then a single backward
  substitution (no iteration needed) referencing already-solved layers.
  This is what makes exact DP tractable in pure Python for these sizes.
"""
import sys, os, json
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/frozen_lake")

from trm_comparison.grid import (
    ACTIONS, _DELTAS as DELTAS, HOLES, CHECKPOINTS, START, ROWS, COLS, UP, DOWN, LEFT, RIGHT,
)
from trm_comparison.trm import INSTANCES as TRM_INSTANCES, U1, U2, U3, RM_GOAL_EVENTS, \
    R_STEP, R_GOAL, R_FINAL, R_LATE, R_HOLE

from trm_comparison.sg_grid import CHECKPOINTS as SG_CP, HOLES as SG_HOLES, START as SG_START, \
    ACTIONS as SG_ACTIONS, GOAL_C
from trm_comparison.sg_trm import INSTANCES as SG_INSTANCES, R_STEP as SG_R_STEP, \
    R_GOAL as SG_R_GOAL, R_LATE as SG_R_LATE, R_HOLE as SG_R_HOLE

SLIP = 0.2
GAMMA_TRM = 1.0    # matches run_trm_comparison.py GAMMA=1.0 (NOT the 0.99 in class default/README prose)
GAMMA_SG = 1.0      # matches run_sg_comparison.py GAMMA=1.0

ALL_POS = [(r, c) for r in range(ROWS) for c in range(COLS)]


def reach_outcomes(r, c, action, rows, cols, deltas, slip):
    """(pos', prob) outcomes for intended `action` from (r,c) with slip."""
    outcomes = {}

    def clip(nr, nc):
        return max(0, min(rows - 1, nr)), max(0, min(cols - 1, nc))

    dr, dc = deltas[action]
    outcomes[clip(r + dr, c + dc)] = outcomes.get(clip(r + dr, c + dc), 0.0) + (1 - slip)
    perp = [LEFT, RIGHT] if action in (UP, DOWN) else [UP, DOWN]
    for pa in perp:
        pdr, pdc = deltas[pa]
        key = clip(r + pdr, c + pdc)
        outcomes[key] = outcomes.get(key, 0.0) + slip / 2.0
    return list(outcomes.items())


# Precompute transition outcomes for the 6x6 grid (shared shape for all variants).
TRANS = {(r, c): {a: reach_outcomes(r, c, a, ROWS, COLS, DELTAS, SLIP) for a in ACTIONS}
         for (r, c) in ALL_POS}


# ───────────────────────── three-goal exact DP ─────────────────────────
def dp_three_goal(trm, gamma=GAMMA_TRM, tol=1e-7, max_iter=20000):
    c_min, D, d_max = trm.c_min, trm.D, trm.d_max
    goal_ev = RM_GOAL_EVENTS  # {U1:'a',U2:'b',U3:'c'}

    # V[rm][x_cap][y_cap][pos] ; rm in (1,2,3)
    V = {rm: {x: [0.0] * ((D + 1)) for x in range(c_min + 1)} for rm in (1, 2, 3)}
    # we need per-pos too; restructure as dict keyed by pos
    V = {rm: {x: {y: {p: 0.0 for p in ALL_POS} for y in range(D + 1)} for x in range(c_min + 1)}
         for rm in (1, 2, 3)}
    POLICY = {rm: {x: {y: {} for y in range(D + 1)} for x in range(c_min + 1)} for rm in (1, 2, 3)}

    def bellman(rm, x_cap, y_cap, pos, on_layer_solve_for_D):
        r, c = pos
        best_val, best_act = None, None
        for delay in range(d_max + 1):
            dt = delay + 1
            for ga in ACTIONS:
                total = 0.0
                for (npos, prob) in TRANS[pos][ga]:
                    if npos in HOLES:
                        total += prob * R_HOLE
                        continue
                    event = CHECKPOINTS.get(npos, "")
                    if event == goal_ev.get(rm):
                        y_new = y_cap + dt  # exact (sound even from capped y_cap, see module docstring)
                        guard_ok = (x_cap + dt >= c_min) and (y_new <= D)
                        if guard_ok:
                            if rm == U3:
                                total += prob * R_FINAL   # terminal, no continuation
                            else:
                                ny_cap = min(y_new, D)
                                cont = V[rm + 1][0][ny_cap][npos]
                                total += prob * (R_GOAL + (gamma ** dt) * cont)
                        else:
                            total += prob * R_LATE  # terminal
                    else:
                        nx_cap = min(x_cap + dt, c_min)
                        ny_cap = min(y_cap + dt, D)
                        if on_layer_solve_for_D and ny_cap == y_cap:
                            cont = V[rm][nx_cap][ny_cap][npos]  # same (top) layer, current estimate
                        else:
                            cont = V[rm][nx_cap][ny_cap][npos]
                        total += prob * (R_STEP + (gamma ** dt) * cont)
                if best_val is None or total > best_val + 1e-12:
                    best_val, best_act = total, (delay, ga)
        return best_val, best_act

    # --- Step 1: solve top layer y_cap = D iteratively (self-referential) ---
    valid_pos = [p for p in ALL_POS if p not in HOLES]
    for it in range(max_iter):
        delta = 0.0
        for rm in (1, 2, 3):
            for x_cap in range(c_min + 1):
                for pos in valid_pos:
                    old = V[rm][x_cap][D][pos]
                    val, act = bellman(rm, x_cap, D, pos, on_layer_solve_for_D=True)
                    V[rm][x_cap][D][pos] = val
                    POLICY[rm][x_cap][D][pos] = act
                    delta = max(delta, abs(val - old))
        if delta < tol:
            break

    # --- Step 2: backward substitution for y_cap = D-1 .. 0 ---
    for y_cap in range(D - 1, -1, -1):
        for rm in (1, 2, 3):
            for x_cap in range(c_min + 1):
                for pos in valid_pos:
                    val, act = bellman(rm, x_cap, y_cap, pos, on_layer_solve_for_D=False)
                    V[rm][x_cap][y_cap][pos] = val
                    POLICY[rm][x_cap][y_cap][pos] = act

    return V, POLICY, valid_pos


def zone_tuple_threegoal(trm, x_cap, y_exact_flag):
    """y_exact_flag: 0 means the true y==D corner (label 0), 1 means true y>D (label 1)."""
    x_node = 0 if x_cap < trm.c_min else 1
    y_node = y_exact_flag
    return (x_node, y_node)


def analyze_three_goal(name):
    trm = TRM_INSTANCES[name]
    print(f"--- exact DP: {name} (c_min={trm.c_min}, D={trm.D}, d_max={trm.d_max}) ---")
    V, POLICY, valid_pos = dp_three_goal(trm)

    # Build concrete-state -> (zone tuple, optimal action) table.
    # zone tuple for y_cap < D is unambiguous: y_node=0. For y_cap==D, the true
    # zone key formula (y<=D else >D) further distinguishes true_y==D (y_node=0)
    # from true_y>D (y_node=1); both map to the SAME DP bucket y_cap==D by
    # construction, so we evaluate both labels using the SAME solved policy
    # (this is exactly the "redundant split" fact discussed in the report).
    from collections import defaultdict
    loc_zone_actions = defaultdict(lambda: defaultdict(set))  # (pos,rm) -> zone_tuple -> {actions}

    for rm in (1, 2, 3):
        for x_cap in range(trm.c_min + 1):
            for y_cap in range(trm.D + 1):
                for pos in valid_pos:
                    act = POLICY[rm][x_cap][y_cap][pos]
                    x_node = 0 if x_cap < trm.c_min else 1
                    if y_cap < trm.D:
                        zt = (x_node, 0)
                        loc_zone_actions[(pos, rm)][zt].add(act)
                    else:
                        # y_cap == D: represents true_y in {D, D+1, D+2, ...}; label 0 = true_y==D,
                        # label 1 = true_y>D. Same DP state/action for both (see note above).
                        loc_zone_actions[(pos, rm)][(x_node, 0)].add(act)
                        loc_zone_actions[(pos, rm)][(x_node, 1)].add(act)

    multi_zone_locs = {loc: zt for loc, zt in loc_zone_actions.items() if len(zt) >= 2}
    conflicts = []
    for loc, zt_map in multi_zone_locs.items():
        zts = list(zt_map.keys())
        # "conflict" = no common optimal action shared between at least one pair of zones
        found_conflict = False
        example = None
        for i in range(len(zts)):
            for j in range(i + 1, len(zts)):
                a_set, b_set = zt_map[zts[i]], zt_map[zts[j]]
                if a_set.isdisjoint(b_set):
                    found_conflict = True
                    example = (zts[i], sorted(a_set), zts[j], sorted(b_set))
        if found_conflict:
            conflicts.append((loc, example))

    n_multi = len(multi_zone_locs)
    n_conflict = len(conflicts)
    frac = n_conflict / n_multi if n_multi else float("nan")
    print(f"  multi-zone locations: {n_multi}  conflicts: {n_conflict}  frac={frac:.3f}")
    example_out = None
    if conflicts:
        loc, (zt_a, act_a, zt_b, act_b) = conflicts[0]
        example_out = dict(location=list(loc), zone_a=zt_a, actions_a=act_a, zone_b=zt_b, actions_b=act_b)
        print(f"  EXAMPLE conflict: loc={loc}  zoneA={zt_a} actsA={act_a}  zoneB={zt_b} actsB={act_b}")
    else:
        # report a representative non-conflict multi-zone location for illustration
        if multi_zone_locs:
            loc, zt_map = next(iter(multi_zone_locs.items()))
            example_out = {"no_conflict_example": {"location": list(loc),
                            "zones": {str(k): sorted(v) for k, v in zt_map.items()}}}
    return dict(instance=name, n_states=len(valid_pos) * 3 * (trm.c_min + 1) * (trm.D + 1),
                n_multi_zone_locations=n_multi, n_conflicts=n_conflict, conflict_fraction=frac,
                example=example_out)


# ───────────────────────── single-goal exact DP ─────────────────────────
def dp_single_goal(trm, gamma=GAMMA_SG, tol=1e-9, max_iter=20000):
    D, c_min = trm.D, trm.c_min
    valid_pos = [p for p in ALL_POS if p not in SG_HOLES]
    V = {y: {p: 0.0 for p in ALL_POS} for y in range(D + 1)}
    POLICY = {y: {} for y in range(D + 1)}

    def bellman(y, pos):
        r, c = pos
        best_val, best_act = None, None
        for ga in SG_ACTIONS:
            total = 0.0
            for (npos, prob) in TRANS[pos][ga]:
                if npos in SG_HOLES:
                    total += prob * SG_R_HOLE
                    continue
                event = SG_CP.get(npos, "")
                if event == "c":
                    y_new = y + 1
                    guard_ok = (c_min <= y_new <= D)
                    if guard_ok:
                        total += prob * SG_R_GOAL  # terminal
                    else:
                        total += prob * SG_R_LATE  # terminal
                else:
                    ny = min(y + 1, D)
                    cont = V[ny][npos]
                    total += prob * (SG_R_STEP + gamma * cont)
            if best_val is None or total > best_val + 1e-12:
                best_val, best_act = total, ga
        return best_val, best_act

    # top layer y=D is self-referential (dt fixed = 1)
    for it in range(max_iter):
        delta = 0.0
        for pos in valid_pos:
            old = V[D][pos]
            val, act = bellman(D, pos)
            V[D][pos] = val
            POLICY[D][pos] = act
            delta = max(delta, abs(val - old))
        if delta < tol:
            break

    for y in range(D - 1, -1, -1):
        for pos in valid_pos:
            val, act = bellman(y, pos)
            V[y][pos] = val
            POLICY[y][pos] = act

    return V, POLICY, valid_pos


def analyze_single_goal(name):
    trm = SG_INSTANCES[name]
    print(f"--- exact DP: {name} (c_min={trm.c_min}, D={trm.D}) ---")
    V, POLICY, valid_pos = dp_single_goal(trm)

    from collections import defaultdict
    loc_zone_actions = defaultdict(lambda: defaultdict(set))  # pos -> zone label -> {actions}
    # rm is a single nonterminal state (SG_U1) throughout; loc = pos only
    for y in range(trm.D + 1):
        for pos in valid_pos:
            act = POLICY[y][pos]
            if y < trm.c_min:
                zn = 0
            elif y <= trm.D:
                zn = 1
            else:
                zn = 2
            loc_zone_actions[pos][zn].add(act)
            if y == trm.D:
                # true y could also be > D at this DP bucket (self-referential top layer);
                # label that case zn=2 as well, sharing the same solved action set.
                loc_zone_actions[pos][2].add(act)

    multi_zone_locs = {loc: zt for loc, zt in loc_zone_actions.items() if len(zt) >= 2}
    conflicts = []
    for loc, zt_map in multi_zone_locs.items():
        zts = list(zt_map.keys())
        for i in range(len(zts)):
            for j in range(i + 1, len(zts)):
                a_set, b_set = zt_map[zts[i]], zt_map[zts[j]]
                if a_set.isdisjoint(b_set):
                    conflicts.append((loc, zts[i], sorted(a_set), zts[j], sorted(b_set)))

    n_multi = len(multi_zone_locs)
    n_conflict = len(set(c[0] for c in conflicts))
    frac = n_conflict / n_multi if n_multi else float("nan")
    print(f"  multi-zone locations: {n_multi}  conflicting locations: {n_conflict}  frac={frac:.3f}")
    example_out = None
    if conflicts:
        loc, zt_a, act_a, zt_b, act_b = conflicts[0]
        example_out = dict(location=list(loc), zone_a=zt_a, actions_a=act_a, zone_b=zt_b, actions_b=act_b)
        print(f"  EXAMPLE conflict: loc={loc} zoneA={zt_a} actsA={act_a} zoneB={zt_b} actsB={act_b}")
    return dict(instance=name, n_states=len(valid_pos) * (trm.D + 1),
                n_multi_zone_locations=n_multi, n_conflicts=n_conflict, conflict_fraction=frac,
                example=example_out)


if __name__ == "__main__":
    results = {"three_goal": [], "single_goal": []}

    # Sizes check (report only, no run) for all TRM / SG instances
    print("=== concrete state-space sizes (three-goal): 36 pos * 3 rm * (c_min+1) * (D+1) ===")
    for name, trm in TRM_INSTANCES.items():
        n = 36 * 3 * (trm.c_min + 1) * (trm.D + 1)
        print(f"  {name}: c_min={trm.c_min} D={trm.D} -> {n:,} states  {'[QUALIFIES]' if n < 1e5 else '[TOO LARGE]'}")

    print("\n=== concrete state-space sizes (single-goal): 36 pos * 1 rm * (D+1) ===")
    for name, trm in SG_INSTANCES.items():
        n = 36 * (trm.D + 1)
        print(f"  {name}: D={trm.D} -> {n:,} states  {'[QUALIFIES]' if n < 1e5 else '[TOO LARGE]'}")

    print("\n=== concrete state-space sizes (hole): 36 pos * 3 rm * (c_min+1)*(D+1)*(x_h+1) ===")
    from trm_comparison.hole_trm import INSTANCES as HOLE_INSTANCES
    for name, trm in HOLE_INSTANCES.items():
        n = 36 * 3 * trm.region_bound
        print(f"  {name}: x_h={trm.x_h} region_bound={trm.region_bound:,} -> {n:,} states  "
              f"{'[QUALIFIES]' if n < 1e5 else '[TOO LARGE]'}")

    print()
    results["three_goal"].append(analyze_three_goal("TRM_01"))

    for nm in ["SG_01", "SG_02", "SG_03", "SG_04", "SG_05"]:
        results["single_goal"].append(analyze_single_goal(nm))

    with open("/tmp/env_audit_flTRM/section_e_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved /tmp/env_audit_flTRM/section_e_results.json")
