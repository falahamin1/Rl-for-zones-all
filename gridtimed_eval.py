"""Dual-metric periodic evaluation for gridtimed training runs
(/tmp/PART1_FINAL.md item 2): ``greedy_success_rate``/``greedy_cycling_frac``
and a separate ``stochastic_success_rate``, using a FIXED eval seed so the
episode stream is controlled across encoders/seeds/updates -- isolating
policy differences from evaluation randomness. Same cycling-fraction
definition as SWEEP_FINAL.md's B4 rollout (fraction of non-terminal steps
that revisit an already-visited grid cell earlier in that same episode).

Gridtimed has NO environment stochasticity (fixed initial state, fixed
transition dynamics -- confirmed in SWEEP_FINAL.md's B4) -- a greedy
(argmax) rollout is therefore fully deterministic given the policy, so
``EVAL_N_GREEDY_EPISODES = 1`` loses no information relative to running
more (they would all be bit-identical); the stochastic pass genuinely
samples, so it uses more episodes to reduce variance in the estimate.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from gridtimed.env import is_terminal

#: Fixed, arbitrary -- same every call, every encoder, every seed, every
#: eval point in training. Not derived from spec.seed: the whole point is
#: an IDENTICAL evaluation condition across runs, not a per-run variant.
EVAL_SEED = 999_983
EVAL_N_GREEDY_EPISODES = 1
EVAL_N_STOCHASTIC_EPISODES = 10
#: Generous safety cap -- gridtimed episodes are bounded by instance.deadline+1
#: in practice (~28-34 steps for T2/T3; y increases by >=1 every step so
#: FAILED always eventually triggers), this is just a guard against a logic
#: bug, not expected to ever bind.
EVAL_MAX_STEPS = 60


def _rollout_episode(head, env, greedy: bool, rng: Optional[np.random.RandomState]) -> Tuple[bool, float]:
    obs, _ = env.reset(seed=EVAL_SEED)
    visited = set()
    n_revisit = 0
    n_steps = 0
    success = False
    for _ in range(EVAL_MAX_STEPS):
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


def evaluate_gridtimed(head, instance_obj: Any, ctor) -> Tuple[float, float, float]:
    """Returns (greedy_success_rate, greedy_cycling_frac, stochastic_success_rate).
    Temporarily switches ``head`` to eval mode (no dropout/BN in this
    project's encoders, but harmless and correct regardless); restores
    ``.train()`` afterward if it was training, so this is safe to call
    mid-training without disturbing the training loop's own state."""
    env = ctor(instance_obj, seed=EVAL_SEED)
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
