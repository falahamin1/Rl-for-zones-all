"""Measurement 2: DECISION RELEVANCE, exact Spike 3 metric. For each
location, collect the zones sharing it; a zone's "optimal action set" is the
DP's tied-optimal action set (within tol). A location is decision-relevant
if at least two of its zones have DISJOINT optimal-action-sets. Report the
fraction of multi-zone locations that are decision-relevant.

IF THIS COMES BACK 0: report loudly and stop (per instructions) rather than
proceeding to measurement 3. THROWAWAY, /tmp only.
"""
import pyudbm as _warmup_pyudbm  # noqa: F401
import sys
sys.path.append("/tmp/inversion_probe")
sys.path.append("/tmp")
sys.path.append("/Users/falah/Desktop/git-projects/RL-for-zones")

from explore import explore_zone_graph, backward_dp, optimal_action_set


def run(deadline=12.0):
    res = explore_zone_graph(deadline=deadline)
    V, Q = backward_dp(res["states"], res["transitions"])

    loc_zones: dict = {}
    for key, (loc, Z, actions) in res["states"].items():
        loc_zones.setdefault(loc, set()).add(key)

    n_multi_zone_locs = 0
    n_relevant = 0
    relevant_locs = []
    examples = []

    for loc, zks in loc_zones.items():
        # only zones with defined outgoing actions (non-terminal) have a
        # meaningful "optimal action" to compare
        defined = [(zk, optimal_action_set(Q.get(zk, {}))) for zk in zks]
        defined = [(zk, s) for zk, s in defined if s]
        if len(defined) < 2:
            continue
        n_multi_zone_locs += 1
        found = False
        for i in range(len(defined)):
            for j in range(i + 1, len(defined)):
                if not (defined[i][1] & defined[j][1]):
                    found = True
                    if len(examples) < 10:
                        examples.append((loc, defined[i], defined[j]))
                    break
            if found:
                break
        if found:
            n_relevant += 1
            relevant_locs.append(loc)

    frac = (n_relevant / n_multi_zone_locs) if n_multi_zone_locs else None
    return {
        "deadline": deadline,
        "n_multi_zone_locations": n_multi_zone_locs,
        "n_decision_relevant_locations": n_relevant,
        "frac_decision_relevant": frac,
        "relevant_locs": relevant_locs,
        "examples": examples,
        "n_states": len(res["states"]),
    }


if __name__ == "__main__":
    r = run(deadline=12.0)
    print(f"deadline=12.0: multi_zone_locations={r['n_multi_zone_locations']} "
          f"decision_relevant={r['n_decision_relevant_locations']} "
          f"frac={r['frac_decision_relevant']}")
    print("relevant locations:", r["relevant_locs"])
    print("examples:", r["examples"][:3])

    if r["frac_decision_relevant"] == 0.0 or r["n_multi_zone_locations"] == 0:
        print()
        print("!!!!! DECISION RELEVANCE CAME BACK 0 -- THE METRIC IS BROKEN. "
              "STOPPING per instructions, NOT proceeding to measurement 3. !!!!!")
        raise SystemExit(1)

    if ("done", "waiting") not in r["relevant_locs"]:
        print()
        print("!!! ANOMALY: decision relevance is positive somewhere, but NOT at "
              "the post-task-1 location ('done','waiting') -- that location was "
              "expected to be among the relevant ones. Flagging per instructions.")
    else:
        print()
        print("OK: decision relevance is strictly positive, and ('done','waiting') "
              "is among the relevant locations, as expected. Proceeding to measurement 3.")
