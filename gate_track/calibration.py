"""Re-calibration note: interval durations change realized makespans, so
the deadline tightness t calibrated on exact durations may no longer land
in the 5-50% violation band. This sweeps t at w=0.25 fresh, rather than
reusing deadline_track's t values, and reports what actually happens.
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np

from .instance import GateJobShopInstance, load_base_instance, make_gate_variant
from .symbolic_env import GateSymbolicJobShopEnv

INSTANCES = ["PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05", "LJS_01"]
TIGHTNESS_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0]
N_EPISODES = 500
W = 0.25


def run_one(instance: GateJobShopInstance, n_episodes: int = N_EPISODES, seed_base: int = 0) -> dict:
    n_violated_episodes = 0
    total_violated_jobs = 0
    n_deadlock_episodes = 0

    for ep in range(n_episodes):
        env = GateSymbolicJobShopEnv(instance, seed=seed_base + ep)
        loc, actions, info = env.reset()
        n_violated_this_ep = len(info["newly_failed"])
        deadlocked = False
        done = all(s in ("done", "failed") for s in loc)
        rng = np.random.default_rng(seed_base + ep + 500_000)

        while not done:
            if not actions:
                deadlocked = True
                break
            a = actions[rng.integers(len(actions))]
            loc, actions, done, info = env.step(a["action_id"])
            n_violated_this_ep += len(info["newly_failed"])

        if n_violated_this_ep > 0:
            n_violated_episodes += 1
        total_violated_jobs += n_violated_this_ep
        if deadlocked:
            n_deadlock_episodes += 1

    return {
        "n_episodes": n_episodes,
        "violation_fraction": n_violated_episodes / n_episodes,
        "mean_violated_jobs_per_episode": total_violated_jobs / n_episodes,
        "deadlock_fraction": n_deadlock_episodes / n_episodes,
    }


def run_sweep(w: float = W) -> Dict[str, Dict[float, dict]]:
    results: Dict[str, Dict[float, dict]] = {}
    for name in INSTANCES:
        base = load_base_instance(name)
        results[name] = {}
        seed_base = hash(name) % 1_000_000
        for t in TIGHTNESS_VALUES:
            inst = make_gate_variant(base, t=t, w=w)
            r = run_one(inst, seed_base=seed_base)
            r["penalty"] = inst.penalty
            results[name][t] = r
            print(
                f"{name:8s} t={t:<5g} w={w}: violation_frac={r['violation_fraction']:.3f} "
                f"mean_violated_jobs={r['mean_violated_jobs_per_episode']:.3f} "
                f"deadlock_frac={r['deadlock_fraction']:.4f}"
            )
    return results


def pick_calibrated_t(results: Dict[str, Dict[float, dict]], target: float = 0.25) -> Dict[str, float]:
    chosen = {}
    for name, by_t in results.items():
        best_t = min(by_t.keys(), key=lambda t: abs(by_t[t]["violation_fraction"] - target))
        chosen[name] = best_t
    return chosen


if __name__ == "__main__":
    import zoneviews  # noqa: F401

    old_calib_t = json.load(open("/tmp/deadline_results/calibration.json"))["calibrated_t"]

    results = run_sweep()
    chosen = pick_calibrated_t(results)
    print("\nRecalibrated t (closest to 25% violation) at w=0.25, vs. deadline_track's old (exact-duration) t:")
    for name, t in chosen.items():
        vf = results[name][t]["violation_fraction"]
        old_t = float(old_calib_t[name])
        changed = "CHANGED" if abs(t - old_t) > 1e-9 else "same"
        print(f"  {name}: new t={t} (violation_fraction={vf:.3f})  |  old t={old_t}  [{changed}]")

    out = {
        "w": W,
        "sweep": {name: {str(t): r for t, r in by_t.items()} for name, by_t in results.items()},
        "calibrated_t": chosen,
        "old_calibrated_t": old_calib_t,
    }
    with open("/tmp/gate_results/calibration.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nWrote /tmp/gate_results/calibration.json")
