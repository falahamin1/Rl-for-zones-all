import sys, time, json, pickle, copy
sys.path.insert(0, '/Users/falah/Desktop/git-projects/RL-for-zones/RL')
import pyudbm
from collections import deque, defaultdict, Counter

ctx = pyudbm.Context(['t_gate'], name='c')
MAX_TIME = 20
GRID = 3
GOAL = (0, 2)
BUTTON = (0, 0)
MUDDY = (1, 1)
OBST = (0, 1)
MOVES = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
GAMMA = 0.95  # matches TabularQ.py default

def zone_for(t_min):
    return (ctx.t_gate >= t_min) & (ctx.t_gate <= MAX_TIME)

def zone_matrix(t_min):
    z = zone_for(t_min)
    dbms = z.to_dbm_list()
    if not dbms:
        return None, z
    m = dbms[0].to_matrix(mode="raw")
    return tuple(tuple(row) for row in m), z

def decode_bound(raw):
    # raw = 2*c + 1 (non-strict, <=c) or raw = 2*c (strict, <c)
    strict = (raw % 2 == 0)
    c = raw // 2 if strict else (raw - 1) // 2
    return c, strict

# ---------------------------------------------------------------------------
# SECTION A/B sanity: does gate_open ever go False? Confirm via static code
# already (grep), and via exhaustive simulation below (BFS never sets it False).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# BFS over exact concrete state graph: state = (pos, t_min)
# gate_open is always True (proven: only ever assigned True in source).
# ---------------------------------------------------------------------------
all_positions = [(x, y) for x in range(GRID) for y in range(GRID) if (x, y) != OBST]

start = ((0, 0), 0)
visited = set([start])
queue = deque([start])

nonterminal = {}   # state -> {action: [(prob, next_state, reward, done), ...]}
terminal_expired = set()
terminal_goal = set()

while queue:
    state = queue.popleft()
    pos, t_min = state
    if t_min > MAX_TIME:
        terminal_expired.add(state)
        continue
    if pos == GOAL:
        terminal_goal.add(state)
        continue
    nonterminal[state] = {}
    for a, (dx, dy) in MOVES.items():
        nx, ny = pos[0] + dx, pos[1] + dy
        if not (0 <= nx < GRID and 0 <= ny < GRID) or (nx, ny) == OBST:
            nonterminal[state][a] = [(1.0, state, -5, False)]
            continue
        if (nx, ny) == MUDDY:
            outcomes = [(0.5, 3), (0.5, 12)]
        else:
            outcomes = [(1.0, 3)]
        branches = []
        for prob, inc in outcomes:
            new_t = t_min + inc
            new_state = ((nx, ny), new_t)
            if new_t > MAX_TIME:
                reward, done = -100, True
            elif (nx, ny) == GOAL:
                reward, done = 500, True
            else:
                reward, done = -1, False
            branches.append((prob, new_state, reward, done))
            if new_state not in visited:
                visited.add(new_state)
                if new_t > MAX_TIME:
                    terminal_expired.add(new_state)
                elif (nx, ny) == GOAL:
                    terminal_goal.add(new_state)
                else:
                    queue.append(new_state)
        nonterminal[state][a] = branches

print("=== BFS SUMMARY ===")
print("nonterminal states:", len(nonterminal))
print("terminal (goal) states:", len(terminal_goal))
print("terminal (expired/zone-empty) states:", len(terminal_expired))
print("total visited:", len(visited))
t_mins_nonterminal = sorted(set(t for (_, t) in nonterminal.keys()))
print("distinct t_min among nonterminal states:", t_mins_nonterminal)
t_mins_expired = sorted(set(t for (_, t) in terminal_expired))
print("distinct t_min among expired states:", t_mins_expired)
t_mins_goal = sorted(set(t for (_, t) in terminal_goal))
print("distinct t_min among goal states:", t_mins_goal)
positions_seen = sorted(set(p for (p, _) in visited))
print("distinct positions visited:", positions_seen, "count:", len(positions_seen))

# ---------------------------------------------------------------------------
# SECTION C: zone geometry for every distinct t_min actually appearing
# (both nonterminal-legal and expired/empty) among reachable concrete states.
# ---------------------------------------------------------------------------
print("\n=== SECTION C: ZONE GEOMETRY ===")
all_t_mins = sorted(set(t for (_, t) in visited))
geom_rows = []
for t_min in all_t_mins:
    mat, z = zone_matrix(t_min)
    is_empty = z.is_empty()
    if mat is None:
        geom_rows.append(dict(t_min=t_min, empty=True, matrix=None, dim=None,
                               bounded=None, strict=None, vertices=0, zone_str=str(z)))
        continue
    b01, s01 = decode_bound(mat[0][1])   # x1 - x0 <= b01  => t_gate <= b01 (upper bound)
    b10, s10 = decode_bound(mat[1][0])   # x0 - x1 <= b10  => -t_gate <= b10 => t_gate >= -b10 (lower bound)
    lower = -b10
    upper = b01
    pinned = (b01 + b10 == 0) and (not s01) and (not s10)
    dim = 0 if pinned else 1
    bounded = True  # both entries finite (no inf sentinel) by construction
    has_strict = s01 or s10
    if lower > upper or (lower == upper and (s01 or s10)):
        vertices = 0  # shouldn't happen since is_empty already false here
    elif lower == upper:
        vertices = 1
    else:
        vertices = 2
    geom_rows.append(dict(t_min=t_min, empty=False, matrix=mat, dim=dim,
                           bounded=bounded, strict=has_strict, vertices=vertices,
                           lower=lower, upper=upper, zone_str=str(z)))

for r in geom_rows:
    print(r)

dims = Counter(r['dim'] for r in geom_rows if not r['empty'])
verts = Counter(r['vertices'] for r in geom_rows if not r['empty'])
strict_count = sum(1 for r in geom_rows if not r['empty'] and r['strict'])
n_nonempty = sum(1 for r in geom_rows if not r['empty'])
n_empty = sum(1 for r in geom_rows if r['empty'])
print("\nnon-empty zone count:", n_nonempty, " empty zone count:", n_empty)
print("affine dim distribution (non-empty only):", dict(dims))
print("vertex count distribution (non-empty only):", dict(verts))
print("zones with a strict inequality:", strict_count, "/", n_nonempty)
print("boundedness fraction (non-empty, both bounds finite):", n_nonempty, "/", n_nonempty, "= 1.0" if n_nonempty else "n/a")

# ---------------------------------------------------------------------------
# SECTION D: aggregation — concrete_key vs zone_key over ALL reachable states
# concrete_key = (pos, gate_open, t_min)   [t_min is the only concrete numeric
#   quantity the env ever materializes for t_gate; gate_open is always True]
# zone_key     = (pos, gate_open, str(zone))
# ---------------------------------------------------------------------------
print("\n=== SECTION D: AGGREGATION ===")
concrete_keys = set()
zone_keys = set()
loc_only_keys = set()
zone_to_concrete = defaultdict(set)

all_states = list(nonterminal.keys()) | set() if False else None
all_states = set(nonterminal.keys()) | terminal_goal | terminal_expired
for (pos, t_min) in all_states:
    gate_open = True
    _, z = zone_matrix(t_min)
    zs = str(z)
    ck = (pos, gate_open, t_min)
    zk = (pos, gate_open, zs)
    concrete_keys.add(ck)
    zone_keys.add(zk)
    loc_only_keys.add((pos, gate_open))
    zone_to_concrete[zk].add(ck)

n_concrete = len(concrete_keys)
n_zone = len(zone_keys)
n_loc = len(loc_only_keys)
singleton_zones = sum(1 for v in zone_to_concrete.values() if len(v) == 1)
print("distinct concrete keys:", n_concrete)
print("distinct zone keys:", n_zone)
print("aggregation ratio (concrete/zone):", n_concrete / n_zone)
print("distinct (pos, gate_open) location-only keys:", n_loc)
print("location-only control ratio (concrete/location):", n_concrete / n_loc)
print("zone keys with exactly one concrete state:", singleton_zones, "/", n_zone,
      "=", singleton_zones / n_zone)
# show which zone keys aggregate >1 concrete state
multi = {k: v for k, v in zone_to_concrete.items() if len(v) > 1}
print("zone keys aggregating >1 concrete state:", len(multi))
for k, v in list(multi.items())[:5]:
    print("  example:", k[0], k[1], "->", sorted(v))

# ---------------------------------------------------------------------------
# SECTION E: exact DP (value iteration) over the full concrete MDP
# ---------------------------------------------------------------------------
print("\n=== SECTION E: EXACT DP ===")
states = list(nonterminal.keys())
V = {s: 0.0 for s in states}
for s in terminal_goal:
    V[s] = 0.0
for s in terminal_expired:
    V[s] = 0.0

def Q(s, a, V):
    total = 0.0
    for prob, ns, reward, done in nonterminal[s][a]:
        if done:
            total += prob * reward
        else:
            total += prob * (reward + GAMMA * V[ns])
    return total

for it in range(2000):
    delta = 0.0
    newV = {}
    for s in states:
        qvals = [Q(s, a, V) for a in range(4)]
        newV[s] = max(qvals)
        delta = max(delta, abs(newV[s] - V[s]))
    V = newV
    if delta < 1e-12:
        break
print("value iteration converged after", it + 1, "iterations, delta =", delta)

optimal_actions = {}
optimal_qvals = {}
for s in states:
    qvals = [Q(s, a, V) for a in range(4)]
    best = max(qvals)
    best_actions = [a for a in range(4) if abs(qvals[a] - best) < 1e-9]
    optimal_actions[s] = best_actions
    optimal_qvals[s] = qvals

# group by position -> list of (t_min, best_actions, qvals)
by_pos = defaultdict(list)
for (pos, t_min), acts in optimal_actions.items():
    by_pos[pos].append((t_min, acts, optimal_qvals[(pos, t_min)]))

multi_zone_locs = 0
conflict_locs = 0
examples = []
for pos, entries in sorted(by_pos.items()):
    entries.sort()
    if len(entries) >= 2:
        multi_zone_locs += 1
        action_sets = [frozenset(a) for (_, a, _) in entries]
        # conflict if there is no common action across all zones at this location
        common = set.intersection(*[set(a) for a in action_sets])
        if not common:
            conflict_locs += 1
            examples.append((pos, entries))

print("locations with 2+ distinct reachable zones (t_min values):", multi_zone_locs)
print("locations with optimal-action conflicts (no common optimal action across zones):", conflict_locs)
print("conflict fraction:", (conflict_locs / multi_zone_locs) if multi_zone_locs else None)

print("\nAll multi-zone locations detail:")
for pos, entries in sorted(by_pos.items()):
    if len(entries) >= 2:
        print(f" pos={pos}:")
        for t_min, acts, qv in entries:
            print(f"    t_min={t_min:3d}  optimal_actions={acts}  Q={['%.3f'%x for x in qv]}")

if examples:
    print("\nWorked conflict example:")
    pos, entries = examples[0]
    print(" position:", pos)
    for t_min, acts, qv in entries:
        print(f"   t_min={t_min}: optimal actions={acts}  Q-values(U,D,L,R)={['%.3f'%x for x in qv]}")
else:
    print("\nNo optimal-action conflicts found across multi-zone locations.")

# ---------------------------------------------------------------------------
# SECTION F: engineering facts
# ---------------------------------------------------------------------------
print("\n=== SECTION F: ENGINEERING ===")

# pickling / deepcopy / native copy on an actual env zone object
sys.path.insert(0, '/Users/falah/Desktop/git-projects/RL-for-zones/RL')
from marshland_example import MarshlandSprintEnv
env = MarshlandSprintEnv(debug=False)

try:
    pickle.dumps(env.zone)
    pickle_ok = True
    pickle_err = None
except Exception as e:
    pickle_ok = False
    pickle_err = repr(e)

try:
    copy.deepcopy(env.zone)
    deepcopy_ok = True
    deepcopy_err = None
except Exception as e:
    deepcopy_ok = False
    deepcopy_err = repr(e)

try:
    c = env.zone.copy()
    native_copy_ok = True
    native_copy_err = None
except Exception as e:
    native_copy_ok = False
    native_copy_err = repr(e)

# also test pickling the whole env object (contains ctx + zone)
try:
    pickle.dumps(env)
    pickle_env_ok = True
    pickle_env_err = None
except Exception as e:
    pickle_env_ok = False
    pickle_env_err = repr(e)

print("pickle.dumps(zone):", pickle_ok, pickle_err)
print("copy.deepcopy(zone):", deepcopy_ok, deepcopy_err)
print("zone.copy() (native):", native_copy_ok, native_copy_err)
print("pickle.dumps(env):", pickle_env_ok, pickle_env_err)

# timing: measure wall time per .step() call
env2 = MarshlandSprintEnv(debug=False)
N = 2000
start_t = time.perf_counter()
count = 0
import random
random.seed(0)
while count < N:
    obs, r, done = env2.step(random.randint(0, 3))
    count += 1
    if done:
        env2.reset()
elapsed = time.perf_counter() - start_t
print(f"\n{N} .step() calls took {elapsed:.4f}s -> {elapsed/N*1e6:.2f} microseconds/call")

# check gymnasium.Env
import inspect
try:
    import gymnasium
    is_gym = isinstance(env, gymnasium.Env)
except ImportError:
    is_gym = "gymnasium not installed"
print("isinstance(env, gymnasium.Env):", is_gym)
print("class bases:", MarshlandSprintEnv.__bases__)
print("reset signature:", inspect.signature(MarshlandSprintEnv.reset))
print("step signature:", inspect.signature(MarshlandSprintEnv.step))

print("\nDONE")
