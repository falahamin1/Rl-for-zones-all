import sys, os
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/Timed-RM")
os.chdir("/Users/falah/Desktop/git-projects/RL-for-zones/Timed-RM")

from reward_machines.timed_reward_machine import TimedRewardMachine
from reward_machines.timed_reward_machine_corner_abstraction import CornerAbstractionTimedRewardMachine, create_region_space

for label, path in [
    ("FrozenLake disc_vs_cont", "env/Frozen_Lake/disc_vs_cont.txt"),
    ("Taxi disc_vs_cont", "env/Taxi/disc_vs_cont.txt"),
]:
    rm = TimedRewardMachine(path)
    print(f"=== {label} (base TRM) ===")
    print("clocks:", rm.clock_names)
    print("states:", rm.states, "init:", rm.init_state, "terminal:", rm.terminal_state)
    print("max_constant:", rm.max_constant, "max_constant_dict:", rm.max_constant_dict)
    print("max_delay:", rm.max_delay)
    print("guard_space:", rm.guard_space)

    rm2 = CornerAbstractionTimedRewardMachine(path)
    print(f"--- {label} (corner/region TRM) ---")
    print("max_constant_dict:", rm2.max_constant_dict, "max_delay:", rm2.max_delay)
    regions = create_region_space(rm2.clock_names, rm2.max_constant_dict)
    print("num regions:", len(regions))
    print()
