"""Consolidated Section 5 run: PJS_01-04 at calibrated t, uniform-release
variant (staggered release gave *identical* mutual-pinning fractions in
Section 4 -- see geometry.json, every instance 1.000 either way -- so
there is no variant with "lower" pinning to prefer; uniform is used as the
simpler default and this is stated explicitly rather than silently picked).
"""
from __future__ import annotations

import json

import zoneviews  # noqa: F401

from .decision_relevance import multi_zone_conflict_analysis, slack
from .instance import load_base_instance, make_deadline_variant, nominal_duration
from .symbolic_env import DeadlineSymbolicJobShopEnv

NO_DEADLINE_BASELINE = {"PJS_01": 0.0, "PJS_02": 0.0, "PJS_03": 0.0, "PJS_04": 1.4}


def run():
    with open("/tmp/deadline_results/calibration.json") as f:
        calib = json.load(f)

    out = {}
    for name in ["PJS_01", "PJS_02", "PJS_03", "PJS_04"]:
        t = float(calib["calibrated_t"][name])
        base = load_base_instance(name)
        inst = make_deadline_variant(base, t=t)
        r = multi_zone_conflict_analysis(inst)

        env = DeadlineSymbolicJobShopEnv(inst, seed=0)
        env.reset()

        entry = {
            "t": t,
            "penalty": inst.penalty,
            "deadlines": {k: round(v, 3) for k, v in inst.deadlines.items()},
            "n_states": r["n_states"],
            "truncated": r["truncated"],
            "multi_zone_locations": r["multi_zone_locations"],
            "conflicting_locations": r["conflicting_locations"],
            "conflict_fraction": r["conflict_fraction"],
            "no_deadline_baseline_pct": NO_DEADLINE_BASELINE[name],
            "conflict_examples": [],
        }

        for ex in r["conflict_examples"]:
            loc = ex["loc"]
            example = {"loc": loc, "actions_i": ex["actions_i"], "actions_j": ex["actions_j"], "jobs": {}}
            for label, sk in [("I", ex["state_i"]), ("J", ex["state_j"])]:
                _loc, vc, gv, ad, fj = sk
                gv = dict(gv)
                jobs = {}
                for j in inst.jobs:
                    jid = j.job_id
                    remaining = [tk.task_id for tk in j.tasks if loc[env._task_to_idx[tk.task_id]] != "done"]
                    g = gv[jid]
                    w = sum(nominal_duration(inst.task_by_id(tid2)) for tid2 in remaining)
                    sl = slack(inst, jid, g, remaining)
                    jobs[jid] = {"g": g, "D": round(inst.deadline_of(jid), 3), "remaining": remaining,
                                 "W": round(w, 3), "slack": round(sl, 3)}
                example["jobs"][label] = jobs
            example["slack_ranking_I"] = sorted(example["jobs"]["I"].keys(), key=lambda j: example["jobs"]["I"][j]["slack"])
            example["slack_ranking_J"] = sorted(example["jobs"]["J"].keys(), key=lambda j: example["jobs"]["J"][j]["slack"])
            example["ranking_inverted"] = example["slack_ranking_I"] != example["slack_ranking_J"]
            entry["conflict_examples"].append(example)

        out[name] = entry
        print(f"{name} t={t}: states={r['n_states']} truncated={r['truncated']} "
              f"multi_zone={r['multi_zone_locations']} conflicts={r['conflicting_locations']} "
              f"frac={r['conflict_fraction']:.4f} (baseline {NO_DEADLINE_BASELINE[name]}%)")

    with open("/tmp/deadline_results/section5.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nWrote /tmp/deadline_results/section5.json")
    return out


if __name__ == "__main__":
    run()
