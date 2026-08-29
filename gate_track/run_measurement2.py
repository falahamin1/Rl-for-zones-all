"""Measurement 2: exact DP decision relevance at w=0.25, recalibrated t,
on PJS_01-04, plus worked examples with the interval-aware slack formula.
"""
from __future__ import annotations

import json

import zoneviews  # noqa: F401

from .decision_relevance import multi_zone_conflict_analysis, slack
from .instance import load_base_instance, make_gate_variant, nominal_duration
from .symbolic_env import GateSymbolicJobShopEnv

NO_DEADLINE_BASELINE = {"PJS_01": 0.0, "PJS_02": 0.0, "PJS_03": 0.0, "PJS_04": 1.4}
DEADLINE_NO_INTERVAL_BASELINE = {"PJS_01": 0.0, "PJS_02": 0.0, "PJS_03": 1.85, "PJS_04": 0.0}
W = 0.25


def run():
    with open("/tmp/gate_results/calibration.json") as f:
        calib = json.load(f)

    out = {}
    for name in ["PJS_01", "PJS_02", "PJS_03", "PJS_04"]:
        t = float(calib["calibrated_t"][name])
        base = load_base_instance(name)
        inst = make_gate_variant(base, t=t, w=W)
        r = multi_zone_conflict_analysis(inst)

        env = GateSymbolicJobShopEnv(inst, seed=0)
        env.reset()

        entry = {
            "t": t,
            "w": W,
            "penalty": inst.penalty,
            "deadlines": {k: round(v, 3) for k, v in inst.deadlines.items()},
            "n_states": r["n_states"],
            "truncated": r["truncated"],
            "multi_zone_locations": r["multi_zone_locations"],
            "conflicting_locations": r["conflicting_locations"],
            "conflict_fraction": r["conflict_fraction"],
            "no_deadline_baseline_pct": NO_DEADLINE_BASELINE[name],
            "deadline_no_interval_baseline_pct": DEADLINE_NO_INTERVAL_BASELINE[name],
            "conflict_examples": [],
        }

        for ex in r["conflict_examples"][:5]:
            loc = ex["loc"]
            example = {"loc": loc, "actions_i": ex["actions_i"], "actions_j": ex["actions_j"], "jobs": {}}
            for label, sk in [("I", ex["state_i"]), ("J", ex["state_j"])]:
                _loc, vc, g, ad, fj = sk
                vc = dict(vc)
                ad = dict(ad)
                jobs = {}
                for j in inst.jobs:
                    jid = j.job_id
                    job_task_ids = [tk.task_id for tk in j.tasks]
                    active_task = next((tid2 for tid2 in job_task_ids if loc[env._task_to_idx[tid2]].startswith("active_d")), None)
                    remaining = [tid2 for tid2 in job_task_ids if loc[env._task_to_idx[tid2]] != "done"]
                    c_val = vc.get(jid, 0)
                    active_nom = ad.get(jid) if active_task else None
                    sl = slack(inst, jid, g, c_val, active_nom, remaining)
                    w_j = None
                    jobs[jid] = {
                        "g": g, "c": c_val, "D": round(inst.deadline_of(jid), 3),
                        "active_task": active_task, "active_nominal_d": active_nom,
                        "remaining": remaining, "slack": round(sl, 3),
                    }
                example["jobs"][label] = jobs
            example["slack_ranking_I"] = sorted(example["jobs"]["I"].keys(), key=lambda j: example["jobs"]["I"][j]["slack"])
            example["slack_ranking_J"] = sorted(example["jobs"]["J"].keys(), key=lambda j: example["jobs"]["J"][j]["slack"])
            example["ranking_inverted"] = example["slack_ranking_I"] != example["slack_ranking_J"]
            entry["conflict_examples"].append(example)

        out[name] = entry
        print(f"{name} t={t} w={W}: states={r['n_states']} truncated={r['truncated']} "
              f"multi_zone={r['multi_zone_locations']} conflicts={r['conflicting_locations']} "
              f"frac={r['conflict_fraction']:.4f} "
              f"(no-deadline baseline {NO_DEADLINE_BASELINE[name]}%, "
              f"deadline-no-interval baseline {DEADLINE_NO_INTERVAL_BASELINE[name]}%)")

    with open("/tmp/gate_results/section5.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nWrote /tmp/gate_results/section5.json")
    return out


if __name__ == "__main__":
    run()
