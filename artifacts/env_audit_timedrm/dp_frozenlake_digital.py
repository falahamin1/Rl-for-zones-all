import sys, os, time
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/Timed-RM")
os.chdir("/Users/falah/Desktop/git-projects/RL-for-zones/Timed-RM")

from env.Frozen_Lake.frozen_lake import FrozenLakeEnv
from reward_machines.timed_reward_machine import TimedRewardMachine
import itertools

gamma = 0.999
discretization_param = 1.0
TRM_FILE = "env/Frozen_Lake/disc_vs_cont.txt"

fl = FrozenLakeEnv(map_name="8x8_3Goals", is_slippery=True, success_rate=0.8, reward_schedule=(0, 0, 0))
rm = TimedRewardMachine(TRM_FILE)

nrow, ncol = fl.nrow, fl.ncol
clock_names = rm.clock_names  # ['y']
assert clock_names == ['y']
max_c = rm.max_constant_dict['y']  # 2
rm_states_nonterm = [u for u in rm.states if u != rm.terminal_state]
delay_n = int(rm.max_delay * (1 / discretization_param)) + 1
n_env_actions = 4

print(f"Grid: {nrow}x{ncol}, RM states (nonterm): {rm_states_nonterm}, max clock y: {max_c}, delay_space.n={delay_n}")

def get_events(row, col):
    tile = fl.desc[row, col]
    if tile == b"G": return ("g",)
    if tile == b"H": return ("h",)
    if tile == b"A": return ("a",)
    if tile == b"B": return ("b",)
    if tile == b"C": return ("c",)
    return tuple()

# Enumerate full concrete state space: (row, col, u, y)
states = []
for row in range(nrow):
    for col in range(ncol):
        for u in rm_states_nonterm:
            for y in range(max_c + 1):
                states.append((row, col, u, y))
print("Total concrete states:", len(states))

actions = [(d, a) for a in range(n_env_actions) for d in range(delay_n)]
print("Actions:", actions)

# Precompute transition model: state -> action -> list of (prob, next_state_or_None, reward)
t0 = time.time()
model = {}
for s in states:
    row, col, u, y = s
    model[s] = {}
    for (d, act) in actions:
        delay = d * discretization_param
        time_el = delay + 1
        outs = []
        for (p, new_idx, base_r, base_term) in fl.P[(row, col)][act]:
            new_row, new_col = new_idx // ncol, new_idx % ncol
            true_props = get_events(new_row, new_col)
            next_config, rew_pair, rm_done = rm.step((u, {'y': y}), true_props, time_el, None)
            u2, cv2 = next_config
            y2 = cv2['y']
            if y2 > max_c:
                y2 = max_c
            state_rew, transition_rew = rew_pair
            reward = (((gamma ** delay - 1) / (gamma - 1)) * state_rew) + transition_rew
            done = rm_done or base_term
            next_state = None if done else (new_row, new_col, u2, y2)
            outs.append((p, next_state, reward))
        model[s][(d, act)] = outs
print(f"Model built in {time.time()-t0:.2f}s, transitions per state*action ~ {sum(len(v) for m in model.values() for v in m.values())}")

# Value iteration
V = {s: 0.0 for s in states}
t0 = time.time()
for it in range(3000):
    delta = 0.0
    newV = {}
    for s in states:
        row, col, u, y = s
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
        delta = max(delta, abs(best - V[s]))
    V = newV
    if delta < 1e-6:
        print(f"Converged at iter {it}, delta={delta:.2e}, time={time.time()-t0:.2f}s")
        break
else:
    print(f"Did NOT converge in {it+1} iters, delta={delta:.2e}")

# Extract optimal action(s) per state
policy = {}
for s in states:
    row, col, u, y = s
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

# ---- location-only conflict check ----
# zone := (u, y) ; location := (row, col)
from collections import defaultdict
loc_to_zones = defaultdict(set)
loc_zone_to_action = {}
for s in states:
    row, col, u, y = s
    zone = (u, y)
    loc_to_zones[(row, col)].add(zone)
    loc_zone_to_action[(row, col, zone)] = tuple(sorted(policy[s][0]))

multi_zone_locations = [loc for loc, zs in loc_to_zones.items() if len(zs) > 1]
conflict_locations = 0
for loc in multi_zone_locations:
    acts = set()
    for zone in loc_to_zones[loc]:
        acts.add(loc_zone_to_action[(loc[0], loc[1], zone)])
    if len(acts) > 1:
        conflict_locations += 1

print(f"\nTotal locations: {nrow*ncol}")
print(f"Multi-zone locations (>1 distinct (u,y) zone reachable there): {len(multi_zone_locations)}")
print(f"Of those, locations where optimal action SET differs across zones (conflict): {conflict_locations}")
if multi_zone_locations:
    print(f"Conflict fraction: {conflict_locations/len(multi_zone_locations):.3f}")

# Sample a few example states/policies
print("\nSample state -> best actions (d,act), Qmax:")
for s in states[:5] + [states[len(states)//2]]:
    print(s, "->", policy[s])
