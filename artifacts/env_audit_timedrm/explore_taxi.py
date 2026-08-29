import sys, os, time, pickle, random
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/Timed-RM")
os.chdir("/Users/falah/Desktop/git-projects/RL-for-zones/Timed-RM")

from env.Taxi.taxi import TaxiEnv
from reward_machines.trm_environment_gym import TimedRewardMachineEnvGym
from reward_machines.trm_environment_gym_corner_abstraction import CornerAbstractionTimedRewardMachineEnvGym

TRM_FILE = "env/Taxi/disc_vs_cont.txt"
GAMMA = 0.999
N_EPISODES = 200

random.seed(0)

def make_base_env():
    return TaxiEnv(render_mode=None, is_rainy=False, fickle_passenger=False)

print("=" * 60)
print("DIGITAL MODE (Taxi)")
env = TimedRewardMachineEnvGym(make_base_env(), [TRM_FILE], GAMMA, discretization_param=1.0, seed=0)

concrete_keys = set()
digital_keys = set()
location_keys = set()
n_steps = 0
t0 = time.time()
for ep in range(N_EPISODES):
    obs, info = env.reset(seed=ep)
    done = False
    step_count = 0
    while not done and step_count < 200:
        action = env.action_space.sample()
        u1, clock_dict = env.current_config
        loc = env.obs
        concrete_key = (loc, u1, tuple(sorted((k, round(v, 3)) for k, v in clock_dict.items())))
        digital_key = tuple(env.get_observation(env.obs, env.current_rm_id, env.current_config, False).tolist())
        concrete_keys.add(concrete_key)
        digital_keys.add(digital_key)
        location_keys.add(loc)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        step_count += 1
        n_steps += 1
elapsed = time.time() - t0
print(f"Episodes: {N_EPISODES}, total steps: {n_steps}, wall time: {elapsed:.3f}s, per-step: {elapsed/n_steps*1000:.4f} ms")
print(f"Distinct concrete keys: {len(concrete_keys)}")
print(f"Distinct digital observation keys: {len(digital_keys)}")
print(f"Distinct location-only keys: {len(location_keys)}")
print(f"Aggregation ratio: {len(concrete_keys)/max(1,len(digital_keys)):.4f}")

print("=" * 60)
print("REGION MODE (Taxi)")
env2 = CornerAbstractionTimedRewardMachineEnvGym(make_base_env(), [TRM_FILE], GAMMA, seed=0)

concrete_keys2 = set()
abstraction_keys2 = set()
location_keys2 = set()
abscell_to_concrete = {}
n_steps2 = 0
t0 = time.time()
for ep in range(N_EPISODES):
    obs, info = env2.reset(seed=ep)
    done = False
    step_count = 0
    while not done and step_count < 200:
        action = env2.action_space.sample()
        u1, region1, corner1 = env2.current_config
        loc = env2.obs
        concrete_key = (loc, u1, tuple(sorted(corner1.items())))
        abs_key = (loc, u1, env2.region_to_index[region1])
        concrete_keys2.add(concrete_key)
        abstraction_keys2.add(abs_key)
        location_keys2.add(loc)
        abscell_to_concrete.setdefault(abs_key, set()).add(concrete_key)
        obs, reward, term, trunc, info = env2.step(action)
        done = term or trunc
        step_count += 1
        n_steps2 += 1
elapsed2 = time.time() - t0
print(f"Episodes: {N_EPISODES}, total steps: {n_steps2}, wall time: {elapsed2:.3f}s, per-step: {elapsed2/n_steps2*1000:.4f} ms")
print(f"Distinct concrete keys (loc,u,corner): {len(concrete_keys2)}")
print(f"Distinct abstraction keys (loc,u,region_index): {len(abstraction_keys2)}")
print(f"Distinct location-only keys: {len(location_keys2)}")
print(f"Aggregation ratio: {len(concrete_keys2)/max(1,len(abstraction_keys2)):.4f}")
singleton_cells = sum(1 for k, v in abscell_to_concrete.items() if len(v) == 1)
print(f"Fraction of abstraction-cells with exactly 1 concrete state: {singleton_cells}/{len(abscell_to_concrete)} = {singleton_cells/len(abscell_to_concrete):.4f}")
