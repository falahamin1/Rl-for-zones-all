"""Assert the hand-verified ground truth: optimal action is SLOW iff tau<=4,
FAST otherwise, at deadline=12. THROWAWAY, /tmp only.
"""
import pyudbm as _warmup  # noqa: F401 -- MUST import before repo root hits sys.path (shadowing bug, see PROBE.md)
import sys
sys.path.append("/tmp/inversion_probe")
sys.path.append("/tmp")
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")

from explore import explore_zone_graph, backward_dp, optimal_action_set


def bucket_of_zone_key(states, key):
    """Recover which tau bucket a ('done','waiting') zone corresponds to by
    reading its lower bound on c directly off the DBM (c's scaled lower
    bound / SCALE == tau, since c>=scaled(tau) is the tightest lower bound
    present after the completion guard)."""
    from model import SCALE
    loc, Z, actions = states[key]
    dbm = Z.to_dbm_list()[0]
    # c is clock index 1 in this 2-clock context (c, g); reference is 0.
    lo = -dbm.bound(0, 1)  # infimum of c
    return round(lo / SCALE)


def main():
    res = explore_zone_graph(deadline=12.0)
    V, Q = backward_dp(res["states"], res["transitions"])

    checked = 0
    failures = []
    for key, (loc, Z, actions) in res["states"].items():
        if loc != ("done", "waiting"):
            continue
        tau = bucket_of_zone_key(res["states"], key)
        opt = optimal_action_set(Q.get(key, {}))
        expected = {"dispatch_SLOW"} if tau <= 4 else {"dispatch_FAST"}
        checked += 1
        if opt != expected:
            failures.append((tau, opt, expected))

    print(f"checked {checked} buckets (expect 9)")
    if failures:
        print("FAIL:", failures)
        raise SystemExit(1)
    print("PASS: optimal action is SLOW iff tau<=4, FAST otherwise (deadline=12)")


if __name__ == "__main__":
    main()
