"""Difference-feature probe (/tmp/ENCODERS_SETUP.md, hypothesis-specific
probe): on each FROZEN trained encoder, fit a small supervised head to
predict pairwise clock differences (c_i - c_j) present in the concrete
state, from the embedding alone. Report R^2 per encoder per instance.

Claim under test: deepset_h and cnn_dbm receive clock identity/pairing
explicitly (job-indexed / clock-pair-indexed structure baked into the
architecture), so a linear-ish probe should recover pairwise differences
well; mlp_point must infer any such structure purely from the raw
concatenated point vector, with no architectural help.

Frozen encoder source: results/ckpt/{run_id}.pt if it exists (a real
run_one.py training checkpoint); falls back to a short in-script training
run via --quick-train-steps for local smoke-testing when no checkpoint is
available yet (clearly labeled as such in the printed report, not
conflated with a properly trained encoder).

The supervised head is closed-form ridge regression (numpy, no extra
dependency, no local optimum/convergence concerns) -- appropriate for a
literal "small head", and its R^2 is a lower bound on how linearly
decodable the target is from the embedding, which is exactly what this
probe wants to measure (not squeeze out every last decodable bit via a
deep nonlinear probe, which would confound "the embedding contains info"
with "the probe is powerful enough to extract it").
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Local-environment compatibility only: older numpy (<2.0) lacks the
# numpy._core submodule path newer numpy uses internally, which the
# checkpoints (written on Alpine, numpy>=2) were pickled under. No-op
# wherever numpy._core already exists for real or numpy.core doesn't exist
# at all. Same shim as RL-for-Polytopes/tangram-intermediate/train_single.py.
try:
    import numpy._core  # noqa: F401  -- already real, nothing to shim
except ImportError:
    try:
        import numpy.core as _np_core
        import numpy.core.multiarray as _np_multiarray
        sys.modules.setdefault("numpy._core", _np_core)
        sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
    except ImportError:
        pass

import torch

from run_one import CKPT_DIR, ENCODERS, NEW_ENCODERS, RunSpec, build_trainer

DOMAINS_INSTANCES = {"modesel": ("M3", "M4"), "gridtimed": ("T2", "T3"), "trmgrid": ("G_SMALL", "G_LARGE")}
#: Extended to cover trmgrid and the v_rep/GNN pair (deepset_v, gnn) -- the
#: original probe only ever covered modesel/gridtimed x the first 4
#: encoders (ENCODERS alone), leaving deepset_v and gnn's R^2 unmeasured
#: on every domain, and trmgrid unmeasured on every encoder. Appendix data
#: report needs both, per the "R^2 for all six encoders" ask.
ALL_ENCODERS = ENCODERS + NEW_ENCODERS


# ----------------------------------------------------------------------
def concrete_clocks(domain: str, env: Any) -> np.ndarray:
    """The exact real-valued clock vector, in the SAME order as
    zone.clock_names (verified against modesel/zone_ops.py and
    gridtimed/zone_ops.py, not assumed)."""
    if domain == "modesel":
        # ModeSelZoneContext._clock_names = job_clock_names + [GLOBAL_CLOCK]
        # (job clocks first, g LAST -- not the other way around).
        state = env._state
        return np.array([js.c for js in state.jobs] + [state.g], dtype=np.float64)
    if domain == "gridtimed":
        # GridZoneContext._clock_names = switches + [y] + [e]; e is always
        # 0 by the time any concrete state is read out (see gym_env.py).
        state = env._state
        return np.concatenate([np.asarray(state.clocks, dtype=np.float64), [state.y, 0.0]])
    if domain == "trmgrid":
        # TRMGridZoneContext._clock_names = sub-goal clocks + [g] + [e],
        # identical pattern to gridtimed's (trmgrid/zone_ops.py:42) -- e is
        # likewise always 0 by observation time.
        state = env._state
        return np.concatenate([np.asarray(state.clocks, dtype=np.float64), [state.g, 0.0]])
    raise ValueError(domain)


def collect_dataset(domain: str, instance_obj: Any, ctor, n_samples: int, seed: int):
    env = ctor(instance_obj, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.RandomState(seed)

    points: List[np.ndarray] = []
    zones: List[Any] = []
    masks: List[np.ndarray] = []
    clocks: List[np.ndarray] = []

    tries = 0
    while len(points) < n_samples and tries < n_samples * 20:
        tries += 1
        legal = np.nonzero(obs["mask"])[0]
        if len(legal) == 0:
            obs, _ = env.reset()
            continue
        points.append(obs["point"])
        zones.append(obs["zone"])
        masks.append(obs["mask"])
        clocks.append(concrete_clocks(domain, env))

        a = rng.choice(legal)
        obs, _r, term, trunc, _info = env.step(int(a))
        if term or trunc:
            obs, _ = env.reset()

    obs_batch = {"point": np.stack(points), "zone": zones, "mask": np.stack(masks)}
    clock_matrix = np.stack(clocks)
    return obs_batch, clock_matrix


def pairwise_diffs(clock_matrix: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    d = clock_matrix.shape[1]
    pairs = list(combinations(range(d), 2))
    diffs = np.stack([clock_matrix[:, i] - clock_matrix[:, j] for i, j in pairs], axis=1)
    return diffs, pairs


# ----------------------------------------------------------------------
def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    n_feat = Xb.shape[1]
    A = Xb.T @ Xb + alpha * np.eye(n_feat)
    W = np.linalg.solve(A, Xb.T @ Y)
    return W


def predict_ridge(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ W


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ----------------------------------------------------------------------
def load_frozen_head(domain: str, instance: str, encoder: str, seed: int, quick_train_steps: int = 0):
    spec = RunSpec(index=-1, domain=domain, instance=instance, encoder=encoder, seed=seed)
    trainer = build_trainer(spec)
    ckpt_path = CKPT_DIR / f"{spec.run_id}.pt"
    if ckpt_path.exists():
        # weights_only=False: checkpoints are always self-written (see the
        # matching fix/comment in run_one.py and check_status.py -- PyTorch
        # >=2.6's stricter default unpickler rejects the RNG-state payload
        # otherwise).
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        trainer.load_state_dict(ckpt["trainer"])
        source = f"checkpoint ({trainer.total_env_steps} env_steps trained)"
    elif quick_train_steps > 0:
        trainer.cfg.total_steps = quick_train_steps
        trainer.cfg.plateau_patience_steps = quick_train_steps  # never trip early stop
        trainer.train(verbose=False)
        source = f"QUICK in-script training ({quick_train_steps} env_steps -- NOT a real trained checkpoint, smoke-test fallback only)"
    else:
        raise FileNotFoundError(
            f"no checkpoint at {ckpt_path} and --quick-train-steps not given -- run run_one.py "
            f"for this config first, or pass --quick-train-steps for a smoke-test-only probe"
        )
    trainer.head.eval()
    for p in trainer.head.parameters():
        p.requires_grad_(False)
    return trainer, source


def run_probe_for_encoder(
    domain: str, instance: str, encoder: str, instance_obj: Any, ctor, seed: int,
    n_samples: int, quick_train_steps: int, test_frac: float = 0.3,
) -> Dict[str, Any]:
    trainer, source = load_frozen_head(domain, instance, encoder, seed, quick_train_steps)

    obs_batch, clock_matrix = collect_dataset(domain, instance_obj, ctor, n_samples, seed=1000 + seed)
    diffs, pairs = pairwise_diffs(clock_matrix)

    with torch.no_grad():
        out = trainer.head.encoder(obs_batch)
        emb = out["pooled"].numpy()

    n = emb.shape[0]
    n_test = max(1, int(n * test_frac))
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    W = fit_ridge(emb[train_idx], diffs[train_idx])
    pred_test = predict_ridge(W, emb[test_idx])

    overall_r2 = r2_score(diffs[test_idx], pred_test)
    per_pair_r2 = [
        r2_score(diffs[test_idx, k : k + 1], pred_test[:, k : k + 1]) for k in range(len(pairs))
    ]

    return {
        "domain": domain, "instance": instance, "encoder": encoder, "seed": seed,
        "source": source, "n_samples": n, "n_train": len(train_idx), "n_test": len(test_idx),
        "embed_dim": emb.shape[1], "n_pairs": len(pairs), "pairs": pairs,
        "overall_r2": overall_r2, "per_pair_r2": per_pair_r2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=list(DOMAINS_INSTANCES), default=None)
    parser.add_argument("--instance", default=None)
    parser.add_argument("--encoder", choices=list(ALL_ENCODERS), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument(
        "--quick-train-steps", type=int, default=0,
        help="if no checkpoint exists, train the encoder for this many env_steps first (smoke-test fallback, NOT a real trained encoder)",
    )
    parser.add_argument("--out", type=Path, default=Path("analysis/difference_probe.json"))
    args = parser.parse_args()

    from run_one import _build_instance

    targets = []
    domains = [args.domain] if args.domain else list(DOMAINS_INSTANCES)
    for domain in domains:
        instances = [args.instance] if args.instance else DOMAINS_INSTANCES[domain]
        for instance in instances:
            encoders = [args.encoder] if args.encoder else list(ALL_ENCODERS)
            for encoder in encoders:
                targets.append((domain, instance, encoder))

    results = []
    for domain, instance, encoder in targets:
        instance_obj, ctor = _build_instance(domain, instance)
        try:
            res = run_probe_for_encoder(
                domain, instance, encoder, instance_obj, ctor, args.seed, args.n_samples, args.quick_train_steps,
            )
        except FileNotFoundError as e:
            print(f"[{domain}/{instance}/{encoder}] SKIPPED: {e}")
            continue
        results.append(res)
        print(
            f"[{domain}/{instance}/{encoder}] R^2={res['overall_r2']:.4f}  "
            f"(n={res['n_samples']}, embed_dim={res['embed_dim']}, source={res['source']})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} probe result(s) to {args.out}")


if __name__ == "__main__":
    main()
