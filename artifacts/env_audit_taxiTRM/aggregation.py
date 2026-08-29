"""Section D: run 200 seeded episodes under a uniform-random policy over the
full (delay, taxi_action) action set (no masking present in taxi_zone_agent.py
-- all d_max+1 * 6 actions always in self._actions), recording at each
decision point:
  concrete_key = (taxi_pos, rm_state, x_exact, y_exact)
  zone_key     = (taxi_pos, rm_state, x_node, y_node)
  loc_key      = (taxi_pos, rm_state)
"""
import sys, random
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/taxi")

from taxi_comparison.taxi_trm import INSTANCES, U_TERMINAL
from taxi_comparison.taxi_env import ACTIONS
from taxi_comparison.taxi_zone_agent import TaxiZoneEnv

N_EPISODES = 200
MAX_STEPS = 400
SLIP_PROB = 0.2
MASTER_SEED = 2026


def run_instance(name):
    trm = INSTANCES[name]
    actions = [(d, a) for d in range(trm.d_max + 1) for a in ACTIONS]
    master_rng = random.Random(MASTER_SEED)

    concrete_keys = set()
    zone_keys = set()
    loc_keys = set()
    zone_to_concrete = {}
    total_decisions = 0

    for ep in range(N_EPISODES):
        env_seed = master_rng.randint(0, 2**31)
        act_rng = random.Random(master_rng.randint(0, 2**31))
        env = TaxiZoneEnv(trm, d_max=trm.d_max, slip_prob=SLIP_PROB, seed=env_seed)
        pos, rm = env.reset()

        for _ in range(MAX_STEPS):
            if rm == U_TERMINAL:
                break
            zk = env.state_key()  # (pos, rm, x_node, y_node)
            ck = (pos, rm, env._x, env._y)
            lk = (pos, rm)

            concrete_keys.add(ck)
            zone_keys.add(zk)
            loc_keys.add(lk)
            zone_to_concrete.setdefault(zk, set()).add(ck)
            total_decisions += 1

            delay, taxi_action = act_rng.choice(actions)
            pos, rm, reward, done = env.step(delay, taxi_action)
            if done:
                break

    n_concrete = len(concrete_keys)
    n_zone = len(zone_keys)
    n_loc = len(loc_keys)
    ratio = n_concrete / n_zone if n_zone else float("nan")
    loc_ratio = n_concrete / n_loc if n_loc else float("nan")
    singleton_zones = sum(1 for v in zone_to_concrete.values() if len(v) == 1)
    frac_singleton = singleton_zones / n_zone if n_zone else float("nan")

    return {
        "instance": name,
        "c_min": trm.c_min,
        "D": trm.D,
        "decisions": total_decisions,
        "n_concrete": n_concrete,
        "n_zone": n_zone,
        "n_loc": n_loc,
        "agg_ratio": ratio,
        "loc_ratio": loc_ratio,
        "frac_singleton_zones": frac_singleton,
    }


if __name__ == "__main__":
    for name in ["TAXI_01", "TAXI_03", "TAXI_06"]:
        r = run_instance(name)
        print(r)
