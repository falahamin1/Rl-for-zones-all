"""Step 2 -- calibrate tightness t per tier via a uniform-random policy
over 500 seeded episodes, real-valued duration sampling (env.step, not
step_deterministic -- genuine uncertainty must be exercised here, not the
DP's nominal shortcut).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .env import action_mask, initial_state, is_terminal, step
from .generator import make_instance
from .instance import MODES, ModeSelInstance, parse_action_id

TIGHTNESS_VALUES = (0.0, 0.1, 0.25, 0.5, 1.0)
N_EPISODES = 500


def run_episode(instance: ModeSelInstance, seed: int) -> Dict[str, bool]:
    rng = np.random.default_rng(seed)
    state = initial_state(instance)
    violated = False
    no_action_available = False

    while not is_terminal(instance, state):
        mask = action_mask(instance, state)
        legal = [aid for aid, ok in mask.items() if ok]
        if not legal:
            no_action_available = True
            break
        aid = legal[rng.integers(len(legal))]
        task_id, mode = parse_action_id(aid)
        res = step(instance, state, task_id, mode, rng)
        if res.newly_failed:
            violated = True
        state = res.next_state

    return {"violated": violated, "no_action_available": no_action_available}


def run_sweep_for_tier(gen_kwargs: dict, n_episodes: int = N_EPISODES) -> Dict[float, dict]:
    results = {}
    for t in TIGHTNESS_VALUES:
        seed_base = hash((gen_kwargs["label"], t)) % 1_000_000
        n_violated = 0
        n_no_action = 0
        for ep in range(n_episodes):
            inst = make_instance(**{**gen_kwargs, "tightness": t, "seed": seed_base})
            r = run_episode(inst, seed=seed_base + ep + 500_000)
            if r["violated"]:
                n_violated += 1
            if r["no_action_available"]:
                n_no_action += 1
        results[t] = {
            "violation_fraction": n_violated / n_episodes,
            "no_action_available_fraction": n_no_action / n_episodes,
            "n_episodes": n_episodes,
        }
    return results


def pick_calibrated_t(sweep: Dict[float, dict], target: float = 0.25, lo: float = 0.05, hi: float = 0.5):
    in_band = [t for t, r in sweep.items() if lo <= r["violation_fraction"] <= hi]
    if in_band:
        return min(in_band, key=lambda t: abs(sweep[t]["violation_fraction"] - target))
    # nothing landed in band -- report the closest anyway, caller must flag this
    return min(sweep.keys(), key=lambda t: abs(sweep[t]["violation_fraction"] - target))


if __name__ == "__main__":
    import json

    TIER_GEN_KWARGS = {
        "M0": dict(n_jobs=2, n_tasks=1, n_machines=1, fast_slow_ratio=2.0, cost_penalty=6.0, w=0.25, label="M0_gen"),
        "M1": dict(n_jobs=2, n_tasks=2, n_machines=2, fast_slow_ratio=2.0, cost_penalty=6.0, w=0.25, label="M1"),
        "M2": dict(n_jobs=3, n_tasks=2, n_machines=2, fast_slow_ratio=2.0, cost_penalty=6.0, w=0.25, label="M2"),
        "M3": dict(n_jobs=3, n_tasks=3, n_machines=3, fast_slow_ratio=2.0, cost_penalty=6.0, w=0.25, label="M3"),
        "M4": dict(n_jobs=4, n_tasks=3, n_machines=3, fast_slow_ratio=2.0, cost_penalty=6.0, w=0.25, label="M4"),
    }

    out = {}
    for name, kwargs in TIER_GEN_KWARGS.items():
        sweep = run_sweep_for_tier(kwargs)
        t_star = pick_calibrated_t(sweep)
        in_band = 0.05 <= sweep[t_star]["violation_fraction"] <= 0.5
        print(f"=== {name} ===")
        for t, r in sweep.items():
            print(f"  t={t}: violation_frac={r['violation_fraction']:.3f} no_action_frac={r['no_action_available_fraction']:.4f}")
        print(f"  chosen t={t_star} (in [5%,50%] band: {in_band})")
        out[name] = {"sweep": {str(t): r for t, r in sweep.items()}, "calibrated_t": t_star, "in_band": in_band}

    with open("/tmp/modesel_results_calibration.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nWrote /tmp/modesel_results_calibration.json")
