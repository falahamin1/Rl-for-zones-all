import sys, os, time
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/Timed-RM")
os.chdir("/Users/falah/Desktop/git-projects/RL-for-zones/Timed-RM")

from env.Taxi.taxi import TaxiEnv
from reward_machines.timed_reward_machine import TimedRewardMachine

gamma = 0.999
discretization_param = 1.0
TRM_FILE = "env/Taxi/disc_vs_cont.txt"

taxi = TaxiEnv(is_rainy=False, fickle_passenger=False)  # matches main.py's actual registration (defaults)
rm = TimedRewardMachine(TRM_FILE)

clock_names = rm.clock_names  # ['x','y']
max_x = rm.max_constant_dict['x']
max_y = rm.max_constant_dict['y']
rm_states_nonterm = [u for u in rm.states if u != rm.terminal_state]
delay_n = int(rm.max_delay * (1 / discretization_param)) + 1
n_env_actions = 6

print(f"Base taxi states: 500, RM states(nonterm): {rm_states_nonterm}, max_x={max_x}, max_y={max_y}, delay_n={delay_n}")

def get_events(base_state):
    taxi_row, taxi_col, pass_idx, dest_idx = taxi.decode(base_state)
    props = ()
    taxi_loc = (taxi_row, taxi_col)
    if pass_idx == 4:
        props += ("in_taxi",)
    if taxi_loc == taxi.locs[0]:
        props += ("at_red",)
    if taxi_loc == taxi.locs[1]:
        props += ("at_green",)
    if taxi_loc == taxi.locs[2]:
        props += ("at_yellow",)
    if taxi_loc == taxi.locs[3]:
        props += ("at_blue",)
    if taxi_loc == taxi.locs[dest_idx] and pass_idx == 4:
        props += ("at_dest",)
    if pass_idx == dest_idx:
        props += ("drop_off",)
    return props

states = []
for base in range(500):
    for u in rm_states_nonterm:
        for x in range(max_x + 1):
            for y in range(max_y + 1):
                states.append((base, u, x, y))
print("Total concrete states:", len(states))

actions = [(d, a) for a in range(n_env_actions) for d in range(delay_n)]
print("Num actions:", len(actions))

t0 = time.time()
model = {}
for s in states:
    base, u, x, y = s
    model[s] = {}
    for (d, act) in actions:
        delay = d * discretization_param
        time_el = delay + 1
        outs = []
        for (p, new_base, base_r, base_term) in taxi.P[base][act]:
            true_props = get_events(new_base)
            next_config, rew_pair, rm_done = rm.step((u, {'x': x, 'y': y}), true_props, time_el, None)
            u2, cv2 = next_config
            x2, y2 = cv2['x'], cv2['y']
            if x2 > max_x: x2 = max_x
            if y2 > max_y: y2 = max_y
            state_rew, transition_rew = rew_pair
            reward = (((gamma ** delay - 1) / (gamma - 1)) * state_rew) + transition_rew
            done = rm_done or base_term
            next_state = None if done else (new_base, u2, x2, y2)
            outs.append((p, next_state, reward))
        model[s][(d, act)] = outs
print(f"Model built in {time.time()-t0:.2f}s")

V = {s: 0.0 for s in states}
t0 = time.time()
max_iters = 1500
for it in range(max_iters):
    delta = 0.0
    newV = {}
    for s in states:
        best = -1e18
        for (d, act) in actions:
            delay = d * discretization_param
            time_el = delay + 1
            q = 0.0
            for (p, ns, r) in model[s][(d, act)]:
                cont = 0.0 if ns is None else (gamma ** time_el) * V[ns]
                q += p * (r + cont)
            if q > best:
                best = q
        newV[s] = best
        d2 = abs(best - V[s])
        if d2 > delta:
            delta = d2
    V = newV
    if delta < 1e-6:
        print(f"Converged at iter {it}, delta={delta:.2e}, time={time.time()-t0:.2f}s")
        break
    if it % 100 == 0:
        print(f"iter {it}, delta={delta:.4f}, elapsed={time.time()-t0:.1f}s")
else:
    print(f"Did NOT converge in {max_iters} iters, delta={delta:.2e}, time={time.time()-t0:.2f}s")

policy = {}
for s in states:
    qvals = {}
    for (d, act) in actions:
        delay = d * discretization_param
        time_el = delay + 1
        q = 0.0
        for (p, ns, r) in model[s][(d, act)]:
            cont = 0.0 if ns is None else (gamma ** time_el) * V[ns]
            q += p * (r + cont)
        qvals[(d, act)] = q
    qmax = max(qvals.values())
    best_actions = [a for a, v in qvals.items() if abs(v - qmax) < 1e-6]
    policy[s] = (best_actions, qmax)

from collections import defaultdict
loc_to_zones = defaultdict(set)
loc_zone_to_action = {}
for s in states:
    base, u, x, y = s
    zone = (u, x, y)
    loc_to_zones[base].add(zone)
    loc_zone_to_action[(base, zone)] = tuple(sorted(policy[s][0]))

multi_zone_locations = [loc for loc, zs in loc_to_zones.items() if len(zs) > 1]
conflict_locations = 0
for loc in multi_zone_locations:
    acts = set()
    for zone in loc_to_zones[loc]:
        acts.add(loc_zone_to_action[(loc, zone)])
    if len(acts) > 1:
        conflict_locations += 1

print(f"\nTotal base locations: 500")
print(f"Multi-zone locations: {len(multi_zone_locations)}")
print(f"Conflict locations: {conflict_locations}")
if multi_zone_locations:
    print(f"Conflict fraction: {conflict_locations/len(multi_zone_locations):.3f}")
