"""Concrete worked example for Section B: same (taxi_pos, rm_state, x_node, y_node)
decision point, two actions differing ONLY in delay, with opposite outcomes
(guard satisfied vs guard failed) because of the x >= c_min guard.

We manually walk TAXI_01 (c_min=2, D=200, d_max=5) to reach: passenger
picked up (rm=u2, x freshly reset to 0), taxi adjacent to GREEN (0,4), about
to move north into GREEN. From that exact state we branch into two
one-step futures with different `delay`.
"""
import sys, copy
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/taxi")

from taxi_comparison.taxi_trm import INSTANCES, U1, U2, U3
from taxi_comparison.taxi_zone_agent import TaxiZoneEnv
from taxi_comparison.taxi_env import S, N, E, W, PICKUP, DROPOFF, RED, GREEN

trm = INSTANCES["TAXI_03"]  # c_min=10, D=200, d_max=5

# slip_prob=0 for a fully deterministic walk to reach the decision point reliably
env = TaxiZoneEnv(trm, d_max=trm.d_max, slip_prob=0.0, seed=1)
pos, rm = env.reset()
print("start:", pos, rm)

# TAXI_START=(2,2). Move to RED=(0,0): W, W, N, N  (avoids the (0,2)/(1,2) west wall)
for a in (W, W, N, N):
    pos, rm, r, done = env.step(0, a)
    print("  step move-to-red:", pos, rm, "reward", r, "done", done)

# Pickup at RED -- use delay=5 (dt=6) so x_new = 4+6 = 10 >= c_min=10, guard OK
pos, rm, r, done = env.step(5, PICKUP)
print("  PICKUP:", pos, rm, "reward", r, "done", done, "x=",env._x,"y=",env._y)
assert rm == U2 and env._x == 0, "expected rm=U2 with x freshly reset to 0"

# Walk to a cell adjacent to GREEN=(0,4) via the row-2 corridor (avoids the
# row0/row1 col1-col2 wall): S,S,E,E,E,E,N -> lands at (1,4), 7 unit steps (x=7)
for a in (S, S, E, E, E, E, N):
    pos, rm, r, done = env.step(0, a)
    print("  step move-to-green-area:", pos, rm, "reward", r, "done", done, "x=",env._x,"y=",env._y)

print("\nDecision point BEFORE branching:")
print("  taxi_pos =", pos, " rm_state =", rm, " exact x =", env._x, " exact y =", env._y)
zk = env.state_key()
print("  zone_key =", zk, "  (x_node=0 means x < c_min=%d)" % trm.c_min)

# Branch A: delay=0, move N into GREEN immediately (dt=1)
envA = copy.deepcopy(env)
posA, rmA, rA, doneA = envA.step(0, N)
print("\nBranch A: action=(delay=0, taxi_action=N)  dt=1  x_new = %d+1 = %d (< c_min=%d -> guard FAILS)"
      % (env._x, env._x+1, trm.c_min))
print("  result: pos=%s rm=%s reward=%s done=%s" % (posA, rmA, rA, doneA))

# Branch B: delay=2, move N into GREEN (dt=3) -- waits before acting
envB = copy.deepcopy(env)
posB, rmB, rB, doneB = envB.step(2, N)
print("\nBranch B: action=(delay=2, taxi_action=N)  dt=3  x_new = %d+3 = %d (>= c_min=%d -> guard OK)"
      % (env._x, env._x+3, trm.c_min))
print("  result: pos=%s rm=%s reward=%s done=%s" % (posB, rmB, rB, doneB))
