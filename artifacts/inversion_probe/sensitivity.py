"""Sensitivity sweep: measurements 2 (decision relevance) and 3 (aggregation
ratio) for deadline in {10, 12, 16, 24}. At D=24 SLOW is always feasible
(tau+8<=24 for all tau in 1..9) so it should always be optimal -- decision
relevance should fall to 0 (negative control). THROWAWAY, /tmp only.
"""
import pyudbm as _warmup_pyudbm  # noqa: F401
import sys
sys.path.append("/tmp/inversion_probe")
sys.path.append("/tmp")
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")

import json
from measure1_threshold import run as run_threshold
from measure2_decision_relevance import run as run_dr
from measure3_aggregation import run as run_agg

DEADLINES = [10.0, 12.0, 16.0, 24.0]

if __name__ == "__main__":
    all_rows = []
    for D in DEADLINES:
        rows, _ = run_threshold(deadline=D)
        slow = sorted(r["tau_bucket"] for r in rows if r["optimal_action"] == ["dispatch_SLOW"])
        fast = sorted(r["tau_bucket"] for r in rows if r["optimal_action"] == ["dispatch_FAST"])
        threshold_tau = max(slow) if slow else 0

        dr = run_dr(deadline=D)
        agg = run_agg(deadline=D)

        row = {
            "deadline": D,
            "slow_buckets": slow,
            "fast_buckets": fast,
            "threshold_tau": threshold_tau,
            "n_multi_zone_locations": dr["n_multi_zone_locations"],
            "n_decision_relevant_locations": dr["n_decision_relevant_locations"],
            "frac_decision_relevant": dr["frac_decision_relevant"],
            "distinct_concrete": agg["distinct_concrete"],
            "distinct_zone": agg["distinct_zone"],
            "distinct_loc": agg["distinct_loc"],
            "aggregation_ratio_zone": agg["aggregation_ratio_zone"],
            "aggregation_ratio_loc_control": agg["aggregation_ratio_loc_control"],
            "per_location": {str(k): v for k, v in agg["per_location"].items()},
        }
        all_rows.append(row)
        print(f"D={D:5.1f} slow={slow} fast={fast} threshold_tau={threshold_tau} | "
              f"decision_relevant={dr['n_decision_relevant_locations']}/{dr['n_multi_zone_locations']} "
              f"frac={dr['frac_decision_relevant']} | "
              f"agg_ratio_zone={agg['aggregation_ratio_zone']:.3f} "
              f"agg_ratio_loc={agg['aggregation_ratio_loc_control']:.3f}",
              flush=True)

    with open("/tmp/inversion_probe/sensitivity_results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print("Saved /tmp/inversion_probe/sensitivity_results.json")

    d24 = next(r for r in all_rows if r["deadline"] == 24.0)
    if d24["fast_buckets"] == [] and d24["frac_decision_relevant"] == 0.0:
        print("NEGATIVE CONTROL CONFIRMED at D=24: SLOW always optimal, decision relevance = 0.")
    else:
        print("!!! NEGATIVE CONTROL FAILED at D=24 -- expected SLOW always optimal and "
              "decision relevance 0. Flagging per instructions.")
