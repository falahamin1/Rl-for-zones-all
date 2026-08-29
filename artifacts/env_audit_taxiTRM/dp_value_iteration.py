"""Section E: exact value iteration over (taxi_pos, rm_state, x_capped, y_capped)
accounting for the taxi's stochastic slip dynamics (0.8 intended / 0.1 / 0.1
perpendicular), for instances whose exact state-space size
25 (taxi positions) * 4 (rm states) * (c_min+1) * (D+1) is tractable.

x_capped = min(x_exact, c_min)   -- once x>=c_min guard is permanently true
                                     for this phase regardless of magnitude
y_capped = min(y_exact, D+1)     -- once y>D guard is permanently false
                                     forever (y never resets); D+1 is a
                                     sentinel "over-deadline" bucket

Both cappings are exact (lossless) abstractions of the underlying MDP: once
x>=c_min, further growth of x cannot change any future guard outcome; once
y>D, y only grows further and the y<=D check can never again succeed. So
value iteration on the capped state space gives EXACT optimal values/actions
identical to the uncapped (fully exact clock) MDP -- this is not an
approximation, unlike the corner-based zone-graph BFS.

Then: for every (pos, rm) location with >=2 distinct reachable zone-keys
(x_node,y_node), check whether the single-optimal-action is consistent
WITHIN each zone bucket across the concrete (x_capped,y_capped) states that
fall in it. A within-zone disagreement is a genuine "zone loses
decision-relevant information" conflict, since a Zone-RL agent's Q-table has
exactly one entry per (pos,rm,x_node,y_node) and cannot represent two
different optimal actions for the same zone key.
"""
import sys
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/taxi")

from taxi_comparison.taxi_trm import (
    INSTANCES, U_TERMINAL, U1, U2, U3, RM_GOAL_EVENTS, R_STEP, R_GOAL, R_FINAL, R_LATE,
)
from taxi_comparison.taxi_env import (
    ROWS, COLS, ACTIONS, MOVE_ACTIONS, RED, GREEN, BLUE, TAXI_START,
    PICKUP, DROPOFF, S, N, E, W, _move,
)

SLIP_PROB = 0.2
GAMMA = 1.0
TOL = 1e-6
MAX_ITERS = 3000

ALL_POS = [(r, c) for r in range(ROWS) for c in range(COLS)]
NONTERM_RM = (U1, U2, U3)


def movement_dist(pos, action):
    """Return list of (next_pos, prob) for a movement action under slip."""
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


def build_and_solve(trm):
    c_min, D, d_max = trm.c_min, trm.D, trm.d_max
    actions = [(d, a) for d in range(d_max + 1) for a in ACTIONS]

    states = []
    for pos in ALL_POS:
        for rm in NONTERM_RM:
            for xc in range(c_min + 1):        # 0..c_min
                for yc in range(D + 2):         # 0..D+1
                    states.append((pos, rm, xc, yc))

    V = {s: 0.0 for s in states}

    # Precompute event lookup and transitions per state,action -> list of (next_state_or_TERM, prob, reward)
    def transitions(state, act):
        pos, rm, xc, yc = state
        delay, taxi_action = act
        dt = delay + 1
        goal_event = RM_GOAL_EVENTS.get(rm)
        pass_in_taxi = rm in (U2, U3)

        x_new_raw = xc + dt   # exact, since xc==c_min already implies x>=c_min
        y_new_raw = yc + dt

        results = []
        for npos, prob in movement_dist(pos, taxi_action):
            event = ""
            if taxi_action == PICKUP and pos == RED and not pass_in_taxi:
                event = "a"
            elif taxi_action in MOVE_ACTIONS and npos == GREEN and pass_in_taxi:
                event = "b"
            elif taxi_action == DROPOFF and pos == BLUE and pass_in_taxi:
                event = "c"

            if event == goal_event:
                guard_ok = (x_new_raw >= c_min) and (y_new_raw <= D)
                if guard_ok:
                    reward = R_FINAL if rm == U3 else R_GOAL
                    if rm == U3:
                        results.append((None, prob, reward))  # terminal, V=0
                    else:
                        nxc = 0
                        nyc = min(y_new_raw, D + 1)
                        results.append(((npos, rm + 1, nxc, nyc), prob, reward))
                else:
                    reward = R_LATE
                    results.append((None, prob, reward))  # terminal
            else:
                reward = R_STEP
                nxc = min(x_new_raw, c_min)
                nyc = min(y_new_raw, D + 1)
                results.append(((npos, rm, nxc, nyc), prob, reward))
        return results

    policy = {}
    for it in range(MAX_ITERS):
        delta = 0.0
        new_V = {}
        for s in states:
            best_q = -float("inf")
            best_a = None
            for act in actions:
                q = 0.0
                for nstate, prob, reward in transitions(s, act):
                    nv = 0.0 if nstate is None else V[nstate]
                    q += prob * (reward + GAMMA * nv)
                if q > best_q:
                    best_q = q
                    best_a = act
            new_V[s] = best_q
            policy[s] = best_a
            delta = max(delta, abs(best_q - V[s]))
        V = new_V
        if delta < TOL:
            break
    return V, policy, it + 1, delta


def zone_of(state, trm):
    pos, rm, xc, yc = state
    xn = 0 if xc < trm.c_min else 1
    yn = 0 if yc <= trm.D else 1
    return (pos, rm, xn, yn)


def analyze(name):
    trm = INSTANCES[name]
    exact_size = 25 * 4 * (trm.c_min + 1) * (trm.D + 1)
    print(f"\n=== {name}  c_min={trm.c_min} D={trm.D} exact_size(formula)={exact_size:,} ===")
    import time
    t0 = time.time()
    V, policy, n_iters, delta = build_and_solve(trm)
    t1 = time.time()
    print(f"  DP states solved: {len(V):,}  iters={n_iters}  final_delta={delta:.2e}  time={t1-t0:.1f}s")

    # group by location
    from collections import defaultdict
    loc_zone_states = defaultdict(lambda: defaultdict(list))  # loc -> zone -> [state]
    for s in V:
        pos, rm, xc, yc = s
        loc = (pos, rm)
        zk = zone_of(s, trm)
        loc_zone_states[loc][zk].append(s)

    multi_zone_locations = 0
    conflict_zone_bucket_count = 0
    total_zone_buckets_at_multizone_locs = 0
    example = None

    for loc, zones in loc_zone_states.items():
        if len(zones) < 2:
            continue
        multi_zone_locations += 1
        for zk, sts in zones.items():
            total_zone_buckets_at_multizone_locs += 1
            acts = set(policy[s] for s in sts)
            if len(acts) > 1:
                conflict_zone_bucket_count += 1
                if example is None:
                    # find two concrete states with differing optimal action
                    a0 = policy[sts[0]]
                    for s2 in sts[1:]:
                        if policy[s2] != a0:
                            example = (zk, sts[0], a0, s2, policy[s2])
                            break

    conflict_fraction = (conflict_zone_bucket_count / total_zone_buckets_at_multizone_locs
                          if total_zone_buckets_at_multizone_locs else float("nan"))

    print(f"  multi-zone locations (>=2 zones reachable): {multi_zone_locations}")
    print(f"  zone buckets at those locations: {total_zone_buckets_at_multizone_locs}")
    print(f"  within-zone optimal-action-conflicted buckets: {conflict_zone_bucket_count}")
    print(f"  conflict fraction: {conflict_fraction:.4f}")
    if example:
        zk, s1, a1, s2, a2 = example
        print(f"  EXAMPLE conflict: zone_key={zk}")
        print(f"    concrete state {s1} -> optimal action {a1} (delay,taxi_action)")
        print(f"    concrete state {s2} -> optimal action {a2} (delay,taxi_action)")
    else:
        print("  No within-zone optimal-action conflicts found.")

    return {
        "instance": name, "exact_size": exact_size,
        "multi_zone_locations": multi_zone_locations,
        "zone_buckets_at_multizone": total_zone_buckets_at_multizone_locs,
        "conflicted_buckets": conflict_zone_bucket_count,
        "conflict_fraction": conflict_fraction,
        "example": example,
    }


if __name__ == "__main__":
    for name in ["TAXI_01"]:
        analyze(name)
