"""
Section D — aggregation ratio under a uniform-random policy.

For each variant, pick benchmark instances spanning small/large parameters.
Run 200 seeded episodes with BOTH grid_action and delay (where applicable)
chosen uniformly at random among legal choices. At every decision point,
record:
  concrete_key = (grid_pos, rm_state, exact clock values)
  zone_key     = (grid_pos, rm_state, zone-node-tuple)   [env.state_key()]

Report per instance: distinct concrete keys, distinct zone keys, ratio,
location-only control ratio, and fraction of zones with exactly 1 concrete
state.
"""
import sys, os, json, random

FL_DIR = "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/frozen_lake"
sys.path.insert(0, FL_DIR)

from trm_comparison.trm import INSTANCES as TRM_INSTANCES, U_TERMINAL as T_TERM
from trm_comparison.zone_agent import TRMZoneEnv
from trm_comparison.grid import ACTIONS as G_ACTIONS

from trm_comparison.sg_trm import INSTANCES as SG_INSTANCES, SG_U_TERMINAL
from trm_comparison.sg_zone_agent import SGZoneEnv
from trm_comparison.sg_grid import ACTIONS as SG_ACTIONS

from trm_comparison.hole_trm import INSTANCES as HOLE_INSTANCES, U_TERMINAL as H_TERM
from trm_comparison.hole_zone_agent import HoleZoneEnv

N_EPISODES = 200
MAX_STEPS = 400
SLIP_PROB = 0.2


def run_threegoal(trm, seed0=0):
    concrete_counts = {}
    zone_counts = {}
    loc_only = set()
    rng = random.Random(seed0)
    for ep in range(N_EPISODES):
        env = TRMZoneEnv(trm, d_max=trm.d_max, slip_prob=SLIP_PROB, seed=rng.randint(0, 2**31))
        pos, rm = env.reset()
        for _ in range(MAX_STEPS):
            if rm == T_TERM:
                break
            ck = (pos, rm, env._x, env._y)
            zk = env.state_key()
            concrete_counts[ck] = concrete_counts.get(ck, 0) + 1
            zone_counts.setdefault(zk, set()).add(ck)
            loc_only.add((pos, rm))
            delay = rng.randint(0, trm.d_max)
            grid_action = rng.choice(G_ACTIONS)
            pos, rm, reward, done = env.step(delay, grid_action)
            if done:
                break
    return concrete_counts, zone_counts, loc_only


def run_singlegoal(trm, seed0=0):
    concrete_counts = {}
    zone_counts = {}
    loc_only = set()
    rng = random.Random(seed0)
    for ep in range(N_EPISODES):
        env = SGZoneEnv(trm, slip_prob=SLIP_PROB, seed=rng.randint(0, 2**31))
        pos, rm = env.reset()
        for _ in range(MAX_STEPS):
            if rm == SG_U_TERMINAL:
                break
            ck = (pos, rm, env._y)
            zk = env.state_key()
            concrete_counts[ck] = concrete_counts.get(ck, 0) + 1
            zone_counts.setdefault(zk, set()).add(ck)
            loc_only.add((pos, rm))
            grid_action = rng.choice(SG_ACTIONS)
            pos, rm, reward, done = env.step(grid_action)
            if done:
                break
    return concrete_counts, zone_counts, loc_only


def run_hole(trm, seed0=0):
    concrete_counts = {}
    zone_counts = {}
    loc_only = set()
    rng = random.Random(seed0)
    for ep in range(N_EPISODES):
        env = HoleZoneEnv(trm, slip_prob=SLIP_PROB, seed=rng.randint(0, 2**31))
        pos, rm = env.reset()
        for _ in range(MAX_STEPS):
            if rm == H_TERM:
                break
            t = env._trm_s
            # concrete key includes disabled_action since it's needed for legal
            # action masking (not a clock, but part of exact/concrete state).
            ck = (pos, rm, t.x, t.y, t.h, t.disabled_action)
            zk = env.state_key()
            concrete_counts[ck] = concrete_counts.get(ck, 0) + 1
            zone_counts.setdefault(zk, set()).add(ck)
            loc_only.add((pos, rm))
            valid = env.valid_actions()
            grid_action = rng.choice(valid)
            pos, rm, reward, done = env.step(grid_action)
            if done:
                break
    return concrete_counts, zone_counts, loc_only


def summarize(name, concrete_counts, zone_counts, loc_only):
    n_concrete = len(concrete_counts)
    n_zone = len(zone_counts)
    ratio = n_concrete / n_zone if n_zone else float("nan")
    n_loc = len(loc_only)
    loc_ratio = n_concrete / n_loc if n_loc else float("nan")
    singleton_zones = sum(1 for s in zone_counts.values() if len(s) == 1)
    frac_singleton = singleton_zones / n_zone if n_zone else float("nan")
    print(f"  {name}: distinct concrete={n_concrete}  distinct zone={n_zone}  "
          f"agg_ratio={ratio:.2f}  loc_only_pairs={n_loc}  loc_ratio={loc_ratio:.2f}  "
          f"singleton_zone_frac={frac_singleton:.3f}")
    return dict(instance=name, n_concrete=n_concrete, n_zone=n_zone, agg_ratio=ratio,
                n_loc_only=n_loc, loc_ratio=loc_ratio, frac_singleton_zone=frac_singleton)


results = {"three_goal": [], "single_goal": [], "hole": []}

print("=== three-goal ===")
for i, name in enumerate(["TRM_01", "TRM_03", "TRM_06"]):
    trm = TRM_INSTANCES[name]
    cc, zc, lo = run_threegoal(trm, seed0=1000 + i)
    results["three_goal"].append(summarize(name, cc, zc, lo))

print("=== single-goal ===")
for i, name in enumerate(["SG_01", "SG_03", "SG_05"]):
    trm = SG_INSTANCES[name]
    cc, zc, lo = run_singlegoal(trm, seed0=2000 + i)
    results["single_goal"].append(summarize(name, cc, zc, lo))

print("=== hole-disable ===")
for i, name in enumerate(["HOLE_01", "HOLE_03", "HOLE_05"]):
    trm = HOLE_INSTANCES[name]
    cc, zc, lo = run_hole(trm, seed0=3000 + i)
    results["hole"].append(summarize(name, cc, zc, lo))

with open("/tmp/env_audit_flTRM/section_d_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved /tmp/env_audit_flTRM/section_d_results.json")
