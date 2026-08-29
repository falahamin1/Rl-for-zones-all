"""Measurement 1: THRESHOLD RECOVERY. Enumerate reachable zones at the
post-task-1 decision location ('done','waiting'). For each, report the
interval of g it permits (raw DBM bound(0,g_idx), bound(g_idx,0)) and the
optimal action from the exact DP. Confirm the split occurs at g=4 (i.e.
between bucket tau=4 and tau=5). THROWAWAY, /tmp only.
"""
import pyudbm as _warmup_pyudbm  # noqa: F401
import sys
sys.path.append("/tmp/inversion_probe")
sys.path.append("/tmp")
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")

from explore import explore_zone_graph, backward_dp, optimal_action_set
from model import SCALE

G_IDX = 2  # clock order ["c","g"] -> reference=0, c=1, g=2


def run(deadline=12.0):
    res = explore_zone_graph(deadline=deadline)
    V, Q = backward_dp(res["states"], res["transitions"])

    rows = []
    for key, (loc, Z, actions) in res["states"].items():
        if loc != ("done", "waiting"):
            continue
        dbm = Z.to_dbm_list()[0]
        g_lo_raw = dbm.bound(0, G_IDX)   # x_0 - g <= g_lo_raw  =>  g >= -g_lo_raw
        g_hi_raw = dbm.bound(G_IDX, 0)   # g - x_0 <= g_hi_raw  =>  g <= g_hi_raw
        g_lo = -g_lo_raw / SCALE
        g_hi = g_hi_raw / SCALE
        c_lo_raw = dbm.bound(0, 1)
        tau_bucket = round(-c_lo_raw / SCALE)
        opt = optimal_action_set(Q.get(key, {}))
        rows.append({
            "tau_bucket": tau_bucket,
            "zone_str": str(Z),
            "g_interval_real": (g_lo, g_hi),
            "optimal_action": sorted(opt),
        })

    rows.sort(key=lambda r: r["tau_bucket"])
    return rows, res


if __name__ == "__main__":
    rows, res = run(deadline=12.0)
    print(f"n zones at ('done','waiting'): {len(rows)} (expect 9)")
    print(f"{'tau':>4} {'g interval (real units)':>26} {'optimal action':>18}   zone string")
    for r in rows:
        lo, hi = r["g_interval_real"]
        print(f"{r['tau_bucket']:>4} {f'[{lo:.2f}, {hi:.2f}]':>26} {','.join(r['optimal_action']):>18}   {r['zone_str']}")

    slow_taus = [r["tau_bucket"] for r in rows if r["optimal_action"] == ["dispatch_SLOW"]]
    fast_taus = [r["tau_bucket"] for r in rows if r["optimal_action"] == ["dispatch_FAST"]]
    print()
    print("SLOW-optimal buckets:", sorted(slow_taus))
    print("FAST-optimal buckets:", sorted(fast_taus))
    if sorted(slow_taus) == [1, 2, 3, 4] and sorted(fast_taus) == [5, 6, 7, 8, 9]:
        print("CONFIRMED: split occurs exactly at tau=4/5 (g=4 boundary).")
    else:
        print("!!! ANOMALY: split does NOT occur at the expected tau=4/5 boundary — reported loudly per instructions.")

    n_distinguishing = len({r["g_interval_real"] for r in rows})
    if n_distinguishing <= 1:
        print("!!! ANOMALY: the zone graph does NOT distinguish g<=4 from g>4 at this "
              "location -- all zones report the same g-interval. The abstraction "
              "cannot represent the threshold. Reported loudly per instructions.")
    else:
        print(f"Zone graph DOES distinguish states at this location: {n_distinguishing} distinct g-intervals observed.")
