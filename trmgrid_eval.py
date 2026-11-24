"""Dual-metric periodic evaluation for trmgrid training runs, reusing
gridtimed_eval.py's approach (read as the reference, adapted here rather
than imported -- trmgrid is a separate package).

DETERMINISM (see trmgrid/gym_env.py's module docstring): trmgrid's
concrete dynamics are fully deterministic given the policy (elapsed=
delay+1 throughout; the real [1,1+w] duration jitter only ever affects
the zone's symbolic geometry, never the concrete transition) -- checked
directly (two rollouts of the same action sequence produce bit-identical
trajectories), not assumed by analogy to gridtimed. A greedy (argmax)
rollout is therefore fully deterministic given the policy, exactly as it
was for gridtimed: EVAL_N_GREEDY_EPISODES=1 loses no information relative
to running more (they would all be bit-identical). The stochastic pass
genuinely samples, so it uses more episodes with a fixed-seeded RNG for a
controlled, reproducible comparison.

Same cycling-fraction definition as gridtimed_eval.py: fraction of
non-terminal steps that revisit an already-visited grid cell earlier in
the same episode.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from trmgrid.env import is_terminal

EVAL_SEED = 999_983
EVAL_N_GREEDY_EPISODES = 1
EVAL_N_STOCHASTIC_EPISODES = 10


def _max_steps(instance) -> int:
    """trmgrid episodes are bounded by instance.deadline+1 in practice (g
    increases by >=1 every step and must not exceed D) -- generous margin
    added as a guard against a logic bug, not expected to ever bind."""
    return instance.deadline + 10


def _rollout_episode(head, env, greedy: bool, rng: Optional[np.random.RandomState]) -> Tuple[bool, float]:
    obs, _ = env.reset(seed=EVAL_SEED)
    visited = set()
    n_revisit = 0
    n_steps = 0
    success = False
    max_steps = _max_steps(env.instance)
    for _ in range(max_steps):
        obs_batch = {"point": obs["point"][None, :], "zone": [obs["zone"]], "mask": obs["mask"][None, :]}
        with torch.no_grad():
            logits, _value = head(obs_batch)
        logits = logits.masked_fill(~torch.as_tensor(obs_batch["mask"]), float("-inf"))
        if greedy:
            action = int(torch.argmax(logits[0]).item())
        else:
            probs = torch.softmax(logits[0], dim=-1).numpy().astype(np.float64)
            probs = probs / probs.sum()
            action = int(rng.choice(len(probs), p=probs))
        obs, _reward, done, _trunc, info = env.step(action)
        n_steps += 1
        if not is_terminal(env._state):
            pos = env._state.pos
            if pos in visited:
                n_revisit += 1
            visited.add(pos)
        if done:
            success = bool(info.get("success"))
            break
    cycling_frac = n_revisit / n_steps if n_steps > 0 else 0.0
    return success, cycling_frac


def evaluate_trmgrid(head, instance, ctor) -> Tuple[float, float, float]:
    """Returns (greedy_success_rate, greedy_cycling_frac, stochastic_success_rate)."""
    env = ctor(instance, seed=EVAL_SEED)
    was_training = head.training
    head.eval()
    try:
        greedy_successes, greedy_cyc = [], []
        for _ in range(EVAL_N_GREEDY_EPISODES):
            s, c = _rollout_episode(head, env, greedy=True, rng=None)
            greedy_successes.append(s)
            greedy_cyc.append(c)

        rng = np.random.RandomState(EVAL_SEED)
        stoch_successes = []
        for _ in range(EVAL_N_STOCHASTIC_EPISODES):
            s, _c = _rollout_episode(head, env, greedy=False, rng=rng)
            stoch_successes.append(s)
    finally:
        if was_training:
            head.train()

    return (
        float(np.mean(greedy_successes)),
        float(np.mean(greedy_cyc)),
        float(np.mean(stoch_successes)),
    )
