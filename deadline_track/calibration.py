"""Section 3: tightness (t) calibration sweep.

For each instance and each t in {0.0, 0.1, 0.25, 0.5, 1.0}: 500 seeded
episodes, uniform-random policy over currently-available actions. Reports,
per (instance, t): fraction of episodes with >=1 deadline violation, mean
violated-jobs-per-episode, and fraction of episodes where the environment
ever reported zero available actions at a non-terminal decision point (a
correctness bug if nonzero -- see ``symbolic_env.py``'s
``_advance_and_resolve`` docstring for why this should never happen; if it
does, this script flags it rather than working around it, per the task).
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from .instance import DeadlineJobShopInstance, load_base_instance, make_deadline_variant
from .symbolic_env import DeadlineSymbolicJobShopEnv

INSTANCES = ["PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05", "LJS_01"]
TIGHTNESS_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0]
N_EPISODES = 500


def run_one(instance: DeadlineJobShopInstance, n_episodes: int = N_EPISODES, seed_base: int = 0) -> dict:
    n_violated_episodes = 0
    total_violated_jobs = 0
    n_deadlock_episodes = 0
    n_jobs = len(instance.jobs)

    for ep in range(n_episodes):
        env = DeadlineSymbolicJobShopEnv(instance, seed=seed_base + ep)
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
        "n_jobs": n_jobs,
        "violation_fraction": n_violated_episodes / n_episodes,
        "mean_violated_jobs_per_episode": total_violated_jobs / n_episodes,
        "deadlock_fraction": n_deadlock_episodes / n_episodes,
    }


def run_sweep() -> Dict[str, Dict[float, dict]]:
    results: Dict[str, Dict[float, dict]] = {}
    for name in INSTANCES:
        base = load_base_instance(name)
        results[name] = {}
        seed_base = hash(name) % 1_000_000  # shared across t for a paired comparison
        for t in TIGHTNESS_VALUES:
            inst = make_deadline_variant(base, t=t)
            r = run_one(inst, seed_base=seed_base)
            r["penalty"] = inst.penalty
            r["deadlines"] = {k: round(v, 2) for k, v in inst.deadlines.items()}
            results[name][t] = r
            print(
                f"{name:8s} t={t:<5g} violation_frac={r['violation_fraction']:.3f} "
                f"mean_violated_jobs={r['mean_violated_jobs_per_episode']:.3f} "
                f"deadlock_frac={r['deadlock_fraction']:.4f}"
            )
    return results


def pick_calibrated_t(results: Dict[str, Dict[float, dict]], target: float = 0.25) -> Dict[str, float]:
    """Per instance, the t whose violation_fraction is closest to `target`."""
    chosen = {}
    for name, by_t in results.items():
        best_t = min(by_t.keys(), key=lambda t: abs(by_t[t]["violation_fraction"] - target))
        chosen[name] = best_t
    return chosen


if __name__ == "__main__":
    import zoneviews  # noqa: F401

    results = run_sweep()
    chosen = pick_calibrated_t(results)
    print("\nCalibrated t (closest to 25% violation) per instance:")
    for name, t in chosen.items():
        vf = results[name][t]["violation_fraction"]
        print(f"  {name}: t={t} (violation_fraction={vf:.3f})")

    out = {
        "sweep": {name: {str(t): r for t, r in by_t.items()} for name, by_t in results.items()},
        "calibrated_t": chosen,
    }
    with open("/tmp/deadline_results/calibration.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nWrote /tmp/deadline_results/calibration.json")
