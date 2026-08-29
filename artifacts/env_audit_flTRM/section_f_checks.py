import sys, os, time, pickle, inspect
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/frozen_lake")

# ── gym.Env subclass check ──────────────────────────────────────────────
from trm_comparison.zone_agent import TRMZoneEnv, TRMZoneAgent
from trm_comparison.region_agent import TRMRegionEnv, TRMRegionAgent
from trm_comparison.sg_zone_agent import SGZoneEnv, SGZoneAgent
from trm_comparison.hole_zone_agent import HoleZoneEnv, HoleZoneAgent

print("=== base classes (MRO) ===")
for cls in [TRMZoneEnv, TRMZoneAgent, TRMRegionEnv, SGZoneEnv, HoleZoneEnv]:
    print(f"  {cls.__name__}: MRO = {[c.__name__ for c in cls.__mro__]}")

# ── picklability of zone-state keys ─────────────────────────────────────
print("\n=== picklability ===")
from trm_comparison.trm import INSTANCES as TRM_INSTANCES
trm = TRM_INSTANCES["TRM_01"]
env = TRMZoneEnv(trm, d_max=trm.d_max, slip_prob=0.2, seed=1)
env.reset()
env.step(0, 3)
zk = env.state_key()
print("  zone state key:", zk, "type:", type(zk))
blob = pickle.dumps(zk)
zk2 = pickle.loads(blob)
print("  pickle round-trip OK:", zk == zk2, " nbytes:", len(blob))

# ── wall-clock per environment step ─────────────────────────────────────
print("\n=== wall-clock per step (TRMZoneEnv.step, 20000 calls) ===")
env = TRMZoneEnv(trm, d_max=trm.d_max, slip_prob=0.2, seed=1)
pos, rm = env.reset()
N = 20000
t0 = time.perf_counter()
for i in range(N):
    if rm == 0:
        pos, rm = env.reset()
    pos, rm, r, done = env.step(2, i % 4)
    if done:
        pos, rm = env.reset()
t1 = time.perf_counter()
print(f"  {N} steps in {t1-t0:.4f}s -> {(t1-t0)/N*1e6:.2f} us/step")

from trm_comparison.sg_trm import INSTANCES as SG_INSTANCES
sg = SG_INSTANCES["SG_01"]
env = SGZoneEnv(sg, slip_prob=0.2, seed=1)
pos, rm = env.reset()
t0 = time.perf_counter()
for i in range(N):
    if rm == 0:
        pos, rm = env.reset()
    pos, rm, r, done = env.step(i % 4)
    if done:
        pos, rm = env.reset()
t1 = time.perf_counter()
print(f"  SGZoneEnv: {N} steps in {t1-t0:.4f}s -> {(t1-t0)/N*1e6:.2f} us/step")

from trm_comparison.hole_trm import INSTANCES as HOLE_INSTANCES
hi = HOLE_INSTANCES["HOLE_01"]
env = HoleZoneEnv(hi, slip_prob=0.2, seed=1)
pos, rm = env.reset()
t0 = time.perf_counter()
for i in range(N):
    if rm == 0:
        pos, rm = env.reset()
    valid = env.valid_actions()
    a = valid[i % len(valid)]
    pos, rm, r, done = env.step(a)
    if done:
        pos, rm = env.reset()
t1 = time.perf_counter()
print(f"  HoleZoneEnv: {N} steps in {t1-t0:.4f}s -> {(t1-t0)/N*1e6:.2f} us/step")

# ── general instance generator vs hardcoded list ────────────────────────
print("\n=== generate_instance search ===")
import trm_comparison.trm as trm_mod
import trm_comparison.sg_trm as sg_mod
import trm_comparison.hole_trm as hole_mod
for mod in (trm_mod, sg_mod, hole_mod):
    names = [n for n in dir(mod) if "generate" in n.lower() or "make_instance" in n.lower()]
    print(f"  {mod.__name__}: generator functions found = {names}")

# ── hole-variant region BFS (capped, to test the '>1M for x_h=5' README claim) ──
print("\n=== hole region BFS (capped at 150,000, x_h=5) ===")
from collections import deque
from trm_comparison.hole_grid import (
    ACTIONS as H_ACTIONS, _DELTAS as H_DELTAS, HOLES as H_HOLES, CHECKPOINTS as H_CP,
    START as H_START, ROWS as H_ROWS, COLS as H_COLS, UP, DOWN, LEFT, RIGHT,
)
from trm_comparison.hole_trm import U_TERMINAL as H_TERM, U1 as H_U1, U3 as H_U3, RM_GOAL_EVENTS as H_GOALEV

def all_reachable(r, c, action):
    results = set()
    perp = [LEFT, RIGHT] if action in (UP, DOWN) else [UP, DOWN]
    for act in [action] + perp:
        dr, dc = H_DELTAS[act]
        nr, nc = max(0, min(H_ROWS - 1, r + dr)), max(0, min(H_COLS - 1, c + dc))
        results.add(((nr, nc), act))
    return results

def count_region_graph_capped(trm, cap):
    def rkey(pos, rm_u, x, y, h, da):
        xc = min(x, trm.c_min); yc = min(y, trm.D); hc = min(h, trm.x_h)
        d = da if hc < trm.x_h else -1
        return (pos, rm_u, xc, yc, hc, d)
    init = rkey(H_START, H_U1, 0, 0, trm.x_h, -1)
    visited = {init}
    q = deque([init])
    truncated = False
    while q:
        if len(visited) >= cap:
            truncated = True
            break
        pos, rm_u, xc, yc, hc, da = q.popleft()
        if rm_u == H_TERM:
            continue
        x, y, h = xc, yc, hc
        avail = [a for a in H_ACTIONS if not (hc < trm.x_h and a == da)]
        for intended in avail:
            for (npos, _) in all_reachable(*pos, intended):
                if npos in H_HOLES:
                    nk = rkey(pos, rm_u, x + 1, y + 1, 0, intended)
                    if nk not in visited:
                        visited.add(nk); q.append(nk)
                    continue
                nx, ny = x + 1, y + 1
                nh = h + 1
                new_da = -1 if nh >= trm.x_h else da
                event = H_CP.get(npos, "")
                goal_ev = H_GOALEV.get(rm_u)
                if event == goal_ev:
                    if trm.guard_satisfied(nx, ny):
                        nrm = H_TERM if rm_u == H_U3 else rm_u + 1
                        nk = rkey(npos, nrm, 0, ny, nh, new_da)
                    else:
                        nk = rkey(npos, H_TERM, nx, ny, nh, new_da)
                else:
                    nk = rkey(npos, rm_u, nx, ny, nh, new_da)
                if nk not in visited:
                    visited.add(nk); q.append(nk)
                    if len(visited) >= cap:
                        truncated = True
                        break
            if truncated:
                break
        if truncated:
            break
    return len(visited), truncated

t0 = time.time()
n, trunc = count_region_graph_capped(HOLE_INSTANCES["HOLE_01"], cap=150_000)
print(f"  HOLE_01 (x_h=5) region BFS: reached {n:,} states in {time.time()-t0:.2f}s, "
      f"cap-truncated={trunc} (README claims final count exceeds 1,000,000)")
