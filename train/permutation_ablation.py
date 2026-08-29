"""Permutation ablation: at evaluation only, shuffle each encoder's own
"set of pieces" and re-measure return with the (already-trained, frozen)
policy.

* ``deepset`` -- shuffles the order ``h_rep()`` returns constraints in.
  Sum-pool is invariant to element order by construction, so this is the
  expected-invariant control the other three are compared against.
* ``cnn_dbm`` -- relabels which grid row/column each real clock occupies in
  ``dbm_tensor`` before rasterizing (reference clock 0 fixed).
* ``cnn_proj`` -- randomly permutes ``projection_stack``'s channel order.
* ``mlp`` -- permutes which task's status/elapsed-fraction features occupy
  which slot of the ``point`` vector.

In every case only the encoder's *input representation* is shuffled; the
mask and the fixed slot-means-task convention used to pick the action are
untouched (see each encoder's ``forward`` docstring for why that is the
right, and simplest, thing to permute). This isolates exactly what the
architecture had to memorize about *position* rather than content.

Evaluation uses a greedy (argmax over masked logits) policy -- deterministic
given the trained weights, appropriate for measuring degradation rather
than exploration behavior.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from train.env import SymbolicJobShopGymEnv
from train.ppo import build_head

RESULTS_DIR = "/tmp/slice_results"

SHUFFLE_KWARG = {
    "mlp": "shuffle_tasks",
    "deepset": "shuffle_elements",
    "cnn_dbm": "shuffle_clocks",
    "cnn_proj": "shuffle_channels",
}


def _greedy_episode_return(head, env: SymbolicJobShopGymEnv, shuffle_kwargs: Dict) -> float:
    obs, _ = env.reset()
    done = False
    total = 0.0
    while not done:
        batch = {"point": obs["point"][None], "mask": obs["mask"][None], "zone": [obs["zone"]]}
        with torch.no_grad():
            logits, _ = head(batch, **shuffle_kwargs)
        mask_t = torch.as_tensor(obs["mask"])[None]
        masked = logits.masked_fill(~mask_t, float("-inf"))
        action = int(torch.argmax(masked, dim=-1).item())
        obs, r, term, trunc, _ = env.step(action)
        total += r
        done = term or trunc
    return total


def evaluate_checkpoint(instance: str, encoder: str, seed: int, n_episodes: int = 20) -> Optional[Dict]:
    tag = f"{instance}_{encoder}_seed{seed}"
    ckpt_path = os.path.join(RESULTS_DIR, f"{tag}.pt")
    if not os.path.exists(ckpt_path):
        return None

    env = SymbolicJobShopGymEnv(instance, seed=10_000 + seed)
    point_dim = env._point_dim

    class _FakeVecEnv:  # build_head only reads .n_tasks/.n_clocks/.task_job_idx
        n_tasks = env.n_tasks
        n_clocks = env.n_clocks
        task_job_idx = env.task_job_idx

    head = build_head(encoder, _FakeVecEnv(), point_dim)
    head.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    head.eval()

    shuffle_kwarg_name = SHUFFLE_KWARG[encoder]

    baseline_returns = []
    shuffled_returns = []
    for ep in range(n_episodes):
        np.random.seed(20_000 + ep)
        torch.manual_seed(20_000 + ep)
        r = _greedy_episode_return(head, SymbolicJobShopGymEnv(instance, seed=30_000 + ep), {})
        baseline_returns.append(r)

    for ep in range(n_episodes):
        np.random.seed(20_000 + ep)
        torch.manual_seed(20_000 + ep)
        r = _greedy_episode_return(
            head, SymbolicJobShopGymEnv(instance, seed=30_000 + ep), {shuffle_kwarg_name: True}
        )
        shuffled_returns.append(r)

    return {
        "instance": instance,
        "encoder": encoder,
        "seed": seed,
        "n_episodes": n_episodes,
        "baseline_mean_return": float(np.mean(baseline_returns)),
        "baseline_std_return": float(np.std(baseline_returns)),
        "shuffled_mean_return": float(np.mean(shuffled_returns)),
        "shuffled_std_return": float(np.std(shuffled_returns)),
        "gap": float(np.mean(baseline_returns) - np.mean(shuffled_returns)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "permutation_ablation.json"))
    args = parser.parse_args()

    ckpts = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.pt")))
    print(f"Found {len(ckpts)} checkpoints.")

    results: List[Dict] = []
    for ckpt in ckpts:
        tag = os.path.splitext(os.path.basename(ckpt))[0]
        parts = tag.split("_")
        # tags look like "PJS_04_mlp_seed0" or "LJS_01_cnn_dbm_seed1" -- the
        # encoder name may itself contain underscores, and so can the
        # instance name, so split off the known suffix/prefix pieces instead
        # of a fixed-position split.
        seed = int(parts[-1].replace("seed", ""))
        rest = "_".join(parts[:-1])
        instance = None
        for cand in ("PJS_01", "PJS_04", "PJS_05", "LJS_01"):
            if rest.startswith(cand + "_"):
                instance = cand
                encoder = rest[len(cand) + 1 :]
                break
        if instance is None:
            print(f"  skip (unrecognized tag) {tag}")
            continue

        print(f"Evaluating {instance}/{encoder}/seed{seed} ...", flush=True)
        result = evaluate_checkpoint(instance, encoder, seed, n_episodes=args.n_episodes)
        if result is not None:
            results.append(result)
            print(f"  baseline={result['baseline_mean_return']:.2f} shuffled={result['shuffled_mean_return']:.2f} gap={result['gap']:.2f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} ablation records to {args.out}")


if __name__ == "__main__":
    main()
