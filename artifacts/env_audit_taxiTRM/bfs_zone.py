"""BFS over reachable (taxi_pos, rm_state, x_node, y_node) zone-key states.
Independent implementation (not copy-pasted from run_taxi_comparison.py's
count_zone_graph) to sanity-check the '4 clock-nodes, constant' claim.
We do NOT special-case zone corners; instead we track exact (x,y) during BFS
expansion and only bucket into (x_node,y_node) at the point of recording,
using every distinct exact (x,y) pair reachable up to a cutoff, so we don't
silently under/over-approximate the zone graph via corner-only expansion.
This is more expensive but exact for these small clock domains since the
regions cap at c_min and D respectively (real clock magnitude eventually
stops mattering once >= c_min or > D).
"""
import sys, os
from collections import deque
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/taxi")

from taxi_comparison.taxi_trm import INSTANCES, U_TERMINAL, U1, U2, U3, RM_GOAL_EVENTS
from taxi_comparison.taxi_env import (
    ACTIONS, MOVE_ACTIONS, TAXI_START, RED, GREEN, BLUE, PICKUP, DROPOFF, _move,
)

CAP = 50_000

def all_reachable(pos, action):
    if action in (PICKUP, DROPOFF):
        return [pos]
    result = {_move(pos, action)}
    perp = [2, 3] if action in (1, 0) else [1, 0]  # E,W if N/S else N,S  (using taxi_env S=0,N=1,E=2,W=3)
    for pa in perp:
        result.add(_move(pos, pa))
    return list(result)


def bfs_exact_clocks(trm, cap=CAP):
    """BFS over exact (pos, rm, x, y) but with x capped at c_min and y capped
    at D+1 (since clock magnitude beyond those thresholds is indistinguishable
    for guard purposes and for x/y_node computation -- capping only affects
    which further increments we simulate, not the zone bucket assigned)."""
    x_cap_val = trm.c_min       # once x >= c_min, further increments don't change x_node
    y_cap_val = trm.D + 1       # once y > D, further increments don't change y_node

    start = (TAXI_START, U1, 0, 0)
    visited_exact = {start}
    zone_keys = set()

    def zk(pos, rm, x, y):
        return (pos, rm, 0 if x < trm.c_min else 1, 0 if y <= trm.D else 1)

    zone_keys.add(zk(*start))
    q = deque([start])
    truncated = False

    while q:
        if len(visited_exact) >= cap:
            truncated = True
            break
        pos, rm, x, y = q.popleft()
        if rm == U_TERMINAL:
            continue
        goal_event = RM_GOAL_EVENTS.get(rm)
        pass_in_taxi = rm in (U2, U3)

        for delay in range(trm.d_max + 1):
            dt = delay + 1
            x_new_raw = x + dt
            y_new_raw = y + dt
            x_new = min(x_new_raw, x_cap_val)
            y_new = min(y_new_raw, y_cap_val)

            for action in ACTIONS:
                for npos in all_reachable(pos, action):
                    event = ""
                    if action == PICKUP and pos == RED and not pass_in_taxi:
                        event = "a"
                    elif action in MOVE_ACTIONS and npos == GREEN and pass_in_taxi:
                        event = "b"
                    elif action == DROPOFF and pos == BLUE and pass_in_taxi:
                        event = "c"

                    if event == goal_event:
                        guard_ok = (x_new_raw >= trm.c_min) and (y_new_raw <= trm.D)
                        if guard_ok:
                            next_rm = U_TERMINAL if rm == U3 else rm + 1
                            nx = 0  # x resets
                        else:
                            next_rm = U_TERMINAL
                            nx = x_new
                        nstate = (npos, next_rm, nx, y_new)
                    else:
                        nstate = (npos, rm, x_new, y_new)

                    zone_keys.add(zk(*nstate))
                    if nstate not in visited_exact:
                        visited_exact.add(nstate)
                        if nstate[1] != U_TERMINAL:
                            q.append(nstate)

    return len(zone_keys), len(visited_exact), truncated


if __name__ == "__main__":
    for name in ["TAXI_01", "TAXI_02", "TAXI_03", "TAXI_06"]:
        trm = INSTANCES[name]
        n_zone, n_exact_capped, trunc = bfs_exact_clocks(trm)
        print(f"{name}: c_min={trm.c_min:3d}  zone_nodes={n_zone}  "
              f"capped_exact_states_explored={n_exact_capped}  truncated={trunc}")
