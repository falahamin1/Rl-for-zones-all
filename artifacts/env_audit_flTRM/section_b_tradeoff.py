"""Concrete worked example: does optimal delay choice vary with clock state
(verifying / correcting the README's 'small delay vs jump past threshold'
claim), and does gamma actually matter (run_trm_comparison.py uses GAMMA=1.0,
not the class-default/README-documented 0.99)?
"""
import sys
sys.path.insert(0, "/Users/falah/Desktop/git-projects/RL-for-zones/supplementary/frozen_lake")
sys.path.insert(0, "/tmp/env_audit_flTRM")
from section_e_dp import dp_three_goal
from trm_comparison.trm import INSTANCES as TRM_INSTANCES

trm = TRM_INSTANCES["TRM_01"]  # c_min=2, D=200, d_max=5
V, POLICY, valid_pos = dp_three_goal(trm)

pos = (0, 1)  # a frozen cell near start, rm=1 (seeking A)
print(f"pos={pos}, rm=1, c_min={trm.c_min}, D={trm.D}, d_max={trm.d_max}")
print("x_cap  y_cap  optimal(delay,grid_action)  V")
for y_cap in [0, 50, 100, 150, 195, 198, 199, 200]:
    for x_cap in range(trm.c_min + 1):
        act = POLICY[1][x_cap][y_cap][pos]
        val = V[1][x_cap][y_cap][pos]
        print(f"  {x_cap:2d}     {y_cap:3d}    {act}   {val:.2f}")
    print()
