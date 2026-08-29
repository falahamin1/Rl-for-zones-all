"""Section E, optimized: exact DP over (taxi_pos, rm_state, x_capped, y_capped)
for TAXI_01 (the only instance with exact size formula < 1e5).

Key structural fact exploited for speed: y is monotonically non-decreasing
(never resets, dt=delay+1>=1 every step), so the transition graph is a DAG
in the y dimension except for self-loops at the saturated y_capped=D+1
"sentinel" layer. So:
  1. Solve the y_capped=D+1 layer (small self-contained sub-MDP: any event
     there always fails the guard since y_new_raw>D unconditionally) with a
     short value iteration.
  2. Sweep y_capped = D, D-1, ..., 0 in strictly decreasing order; every
     transition out of layer yc goes to some yc' > yc (either yc' in
     (yc, D] already finalized by an earlier sweep step, or the yc=D+1
     sentinel already finalized in step 1) -- so each layer is solved in a
     single O(1)-per-state,action pass, no iteration required. This is
     exact, not an approximation.
"""
import sys, time
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/taxi")

from taxi_comparison.taxi_trm import (
    INSTANCES, U1, U2, U3, RM_GOAL_EVENTS, R_STEP, R_GOAL, R_FINAL, R_LATE,
)
from taxi_comparison.taxi_env import (
    ROWS, COLS, ACTIONS, MOVE_ACTIONS, RED, GREEN, BLUE,
    PICKUP, DROPOFF, S, N, E, W, _move,
)

SLIP_PROB = 0.2
GAMMA = 1.0
NONTERM_RM = (U1, U2, U3)
ALL_POS = [(r, c) for r in range(ROWS) for c in range(COLS)]


def movement_dist(pos, action):
    if action in (PICKUP, DROPOFF):
        return [(pos, 1.0)]
    intended = _move(pos, action)
    perp = [E, W] if action in (N, S) else [N, S]
    dist = {}
    dist[intended] = dist.get(intended, 0.0) + (1.0 - SLIP_PROB)
    for pa in perp:
        npos = _move(pos, pa)
        dist[npos] = dist.get(npos, 0.0) + SLIP_PROB / 2.0
    return list(dist.items())


def build_template(trm):
    """template[(pos,rm,taxi_action)] = list of (npos, prob, event)"""
    template = {}
    for pos in ALL_POS:
        for rm in NONTERM_RM:
            goal_event = RM_GOAL_EVENTS.get(rm)
            pass_in_taxi = rm in (U2, U3)
            for taxi_action in ACTIONS:
                entries = []
                for npos, prob in movement_dist(pos, taxi_action):
                    event = ""
                    if taxi_action == PICKUP and pos == RED and not pass_in_taxi:
                        event = "a"
                    elif taxi_action in MOVE_ACTIONS and npos == GREEN and pass_in_taxi:
                        event = "b"
                    elif taxi_action == DROPOFF and pos == BLUE and pass_in_taxi:
                        event = "c"
                    entries.append((npos, prob, event))
                template[(pos, rm, taxi_action)] = (entries, goal_event)
    return template


def solve(trm):
    c_min, D, d_max = trm.c_min, trm.D, trm.d_max
    actions = [(d, a) for d in range(d_max + 1) for a in ACTIONS]
    template = build_template(trm)

    V = {}       # (pos,rm,xc,yc) -> value
    POLICY = {}  # (pos,rm,xc,yc) -> best (delay,taxi_action)

    # ---- Step 1: solve the yc = D+1 sentinel layer ----
    YC_TOP = D + 1
    top_states = [(pos, rm, xc) for pos in ALL_POS for rm in NONTERM_RM for xc in range(c_min + 1)]
    Vtop = {s: 0.0 for s in top_states}
    for _ in range(2000):
        delta = 0.0
        newV = {}
        for (pos, rm, xc) in top_states:
            entries, goal_event = None, None
            best_q = -float("inf")
            best_a = None
            for (delay, taxi_action) in actions:
                dt = delay + 1
                x_new_raw = xc + dt
                entries, goal_event = template[(pos, rm, taxi_action)]
                q = 0.0
                for npos, prob, event in entries:
                    if event == goal_event:
                        # y_new_raw = D+1+dt > D always -> guard always fails here
                        reward = R_LATE
                        nv = 0.0  # terminal
                    else:
                        reward = R_STEP
                        nxc = min(x_new_raw, c_min)
                        nv = Vtop[(npos, rm, nxc)]
                    q += prob * (reward + GAMMA * nv)
                if q > best_q:
                    best_q = q
                    best_a = (delay, taxi_action)
            newV[(pos, rm, xc)] = best_q
            POLICY[(pos, rm, xc, YC_TOP)] = best_a
            delta = max(delta, abs(best_q - Vtop[(pos, rm, xc)]))
        Vtop = newV
        if delta < 1e-7:
            break
    for s, v in Vtop.items():
        pos, rm, xc = s
        V[(pos, rm, xc, YC_TOP)] = v

    # ---- Step 2: single backward sweep over yc = D, D-1, ..., 0 ----
    for yc in range(D, -1, -1):
        for pos in ALL_POS:
            for rm in NONTERM_RM:
                goal_event = RM_GOAL_EVENTS.get(rm)
                for xc in range(c_min + 1):
                    x_new_raw = xc + 0  # placeholder, computed per delay below
                    best_q = -float("inf")
                    best_a = None
                    for (delay, taxi_action) in actions:
                        dt = delay + 1
                        x_new_raw = xc + dt
                        y_new_raw = yc + dt
                        entries, ge = template[(pos, rm, taxi_action)]
                        q = 0.0
                        for npos, prob, event in entries:
                            if event == ge:
                                guard_ok = (x_new_raw >= c_min) and (y_new_raw <= D)
                                if guard_ok:
                                    if rm == U3:
                                        reward = R_FINAL
                                        nv = 0.0  # success terminal
                                    else:
                                        reward = R_GOAL
                                        nxc = 0
                                        nyc = y_new_raw if y_new_raw <= D else YC_TOP
                                        nv = V[(npos, rm + 1, nxc, nyc)]
                                else:
                                    reward = R_LATE
                                    nv = 0.0  # fail terminal
                            else:
                                reward = R_STEP
                                nxc = min(x_new_raw, c_min)
                                nyc = y_new_raw if y_new_raw <= D else YC_TOP
                                nv = V[(npos, rm, nxc, nyc)]
                            q += prob * (reward + GAMMA * nv)
                        if q > best_q:
                            best_q = q
                            best_a = (delay, taxi_action)
                    V[(pos, rm, xc, yc)] = best_q
                    POLICY[(pos, rm, xc, yc)] = best_a
    return V, POLICY


def zone_of(state, trm):
    pos, rm, xc, yc = state
    xn = 0 if xc < trm.c_min else 1
    yn = 0 if yc <= trm.D else 1
    return (pos, rm, xn, yn)


def analyze(name):
    trm = INSTANCES[name]
    formula_size = 25 * 4 * (trm.c_min + 1) * (trm.D + 1)
    print(f"\n=== {name}  c_min={trm.c_min} D={trm.D}  formula_size={formula_size:,} ===")
    t0 = time.time()
    V, policy = solve(trm)
    t1 = time.time()
    print(f"DP table states solved (excl. terminal): {len(V):,}   time={t1-t0:.1f}s")

    from collections import defaultdict
    loc_zone_states = defaultdict(lambda: defaultdict(list))
    for s in V:
        pos, rm, xc, yc = s
        loc_zone_states[(pos, rm)][zone_of(s, trm)].append(s)

    multi_zone_locations = 0
    conflict_buckets = 0
    total_buckets = 0
    example = None
    for loc, zones in loc_zone_states.items():
        if len(zones) < 2:
            continue
        multi_zone_locations += 1
        for zk, sts in zones.items():
            total_buckets += 1
            acts = set(policy[s] for s in sts)
            if len(acts) > 1:
                conflict_buckets += 1
                if example is None:
                    a0 = policy[sts[0]]
                    for s2 in sts[1:]:
                        if policy[s2] != a0:
                            example = (zk, sts[0], a0, s2, policy[s2])
                            break

    conflict_fraction = conflict_buckets / total_buckets if total_buckets else float("nan")
    print(f"multi-zone locations (>=2 zones reachable): {multi_zone_locations}")
    print(f"zone buckets at those locations: {total_buckets}")
    print(f"within-zone optimal-action-conflicted buckets: {conflict_buckets}")
    print(f"conflict fraction: {conflict_fraction:.4f}")
    if example:
        zk, s1, a1, s2, a2 = example
        print(f"EXAMPLE conflict: zone_key={zk}")
        print(f"  concrete state {s1} -> optimal action {a1}")
        print(f"  concrete state {s2} -> optimal action {a2}")
    else:
        print("No within-zone optimal-action conflicts found.")
    return dict(instance=name, formula_size=formula_size, dp_states=len(V),
                multi_zone_locations=multi_zone_locations, total_buckets=total_buckets,
                conflict_buckets=conflict_buckets, conflict_fraction=conflict_fraction,
                example=example)


if __name__ == "__main__":
    analyze("TAXI_01")
