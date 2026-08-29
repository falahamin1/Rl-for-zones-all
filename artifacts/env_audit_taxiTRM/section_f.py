"""Section F: scaling/engineering facts.
- gymnasium.Env or bespoke loop? (inspect base classes)
- picklability of zone-state keys/objects
- wall-clock per environment step
"""
import sys, time, pickle, inspect
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/taxi")

from taxi_comparison.taxi_env import TaxiGridEnv
from taxi_comparison.taxi_trm import INSTANCES
from taxi_comparison.taxi_zone_agent import TaxiZoneEnv

print("=== Base classes ===")
print("TaxiGridEnv MRO:", [c.__name__ for c in TaxiGridEnv.__mro__])
print("TaxiZoneEnv MRO:", [c.__name__ for c in TaxiZoneEnv.__mro__])
try:
    import gymnasium
    print("gymnasium is importable in this env:", gymnasium.__version__)
    print("TaxiGridEnv is gymnasium.Env subclass:", issubclass(TaxiGridEnv, gymnasium.Env))
except ImportError as e:
    print("gymnasium NOT installed / not imported by this package:", e)

print("\n=== Picklability ===")
trm = INSTANCES["TAXI_01"]
env = TaxiZoneEnv(trm, d_max=trm.d_max, slip_prob=0.2, seed=1)
env.reset()
key = env.state_key()
print("zone state_key:", key, "type:", type(key))
try:
    blob = pickle.dumps(key)
    back = pickle.loads(blob)
    print("pickle.dumps/loads zone key OK, round-trip equal:", back == key, "bytes:", len(blob))
except Exception as e:
    print("PICKLE FAILED:", e)

# Also test a whole Q-table dict (as used by TaxiZoneAgent.Q) for picklability
q_sample = {key: {(0, 0): 1.23, (1, 4): -5.0}}
try:
    blob = pickle.dumps(q_sample)
    print("pickle Q-table-shaped dict OK, bytes:", len(blob))
except Exception as e:
    print("PICKLE Q-table FAILED:", e)

print("\n=== Wall-clock per step ===")
N = 200_000
env = TaxiZoneEnv(trm, d_max=trm.d_max, slip_prob=0.2, seed=123)
env.reset()
actions_cycle = [(d, a) for d in range(trm.d_max + 1) for a in range(6)]
t0 = time.perf_counter()
for i in range(N):
    d, a = actions_cycle[i % len(actions_cycle)]
    _, rm, r, done = env.step(d, a)
    if done:
        env.reset()
t1 = time.perf_counter()
per_step_us = (t1 - t0) / N * 1e6
print(f"{N} TaxiZoneEnv.step() calls in {t1-t0:.3f}s -> {per_step_us:.2f} microseconds/step")

# Also raw TaxiGridEnv.step for comparison
genv = TaxiGridEnv(slip_prob=0.2, seed=123)
genv.reset()
t0 = time.perf_counter()
for i in range(N):
    a = actions_cycle[i % len(actions_cycle)][1]
    genv.step(a)
t1 = time.perf_counter()
per_step_us2 = (t1 - t0) / N * 1e6
print(f"{N} raw TaxiGridEnv.step() calls in {t1-t0:.3f}s -> {per_step_us2:.2f} microseconds/step")
