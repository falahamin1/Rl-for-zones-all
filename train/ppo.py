"""PPO with GAE over ``train.env.SyncVectorEnv`` + any ``encoders.Encoder``.

Fixed hyperparameters per spec: ``gamma=0.99``, ``gae_lambda=0.95``,
``clip=0.2``, ``entropy_coef=0.01``, ``lr=3e-4``, ``rollout=2048`` (total
transitions collected per update, across all parallel envs combined).
Epoch count / minibatch size / value-loss form are not specified by the
task; this module picks standard, documented defaults (4 epochs, minibatch
256, plain unclipped MSE value loss, advantage normalization per
minibatch) rather than silently inventing something exotic.

Action masking is applied to the raw logits (``-inf`` on disabled actions)
*before* the softmax/Categorical is constructed, never after sampling --
per spec, this is what makes masked actions literally unreachable rather
than merely resampled away.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.base import ActorCriticHead, Encoder
from encoders.cnn_dbm import CNNDBMEncoder
from encoders.cnn_proj import CNNProjEncoder
from encoders.deepset_h import DeepSetHEncoder
from encoders.deepset_v import DeepSetVEncoder
from encoders.gnn import GNNEncoder
from encoders.mlp_point import MLPPointEncoder
from train.env import SymbolicJobShopGymEnv, SyncVectorEnv
from zoneviews.raster import raster_wall_time, reset_raster_wall_time

ENCODER_NAMES = ("mlp", "deepset", "cnn_dbm", "cnn_proj", "deepset_v", "gnn")


def build_head(encoder_name: str, env: SyncVectorEnv, point_dim: int) -> ActorCriticHead:
    """Encoder-by-name factory. New encoders register here -- the rest of
    the trainer (rollout, GAE, PPO update, logging) is encoder-agnostic.

    All four encoders now have a per-element action head where the
    architecture allows it (confound fix, /tmp/ENCODERS_SETUP.md item 1).
    mlp's per-element blocks are grouped by ``env.n_blocks``/
    ``env.mlp_group_idx`` -- on job-shop these default to one block per
    action (the identity broadcast); on modesel a block is a raw task
    shared by its two mode-actions; on gridtimed a block is a move shared
    by its delay-variants. deepset/cnn_dbm/cnn_proj are job/clock-indexed
    and need env.task_job_idx (a STATIC action->clock mapping) to
    broadcast to task slots -- when an env has no such static mapping
    (gridtimed: which clock a move is "about" depends on current position,
    not fixed for the whole episode), env.task_job_idx is None and these
    three fall back to a pooled head (see encoders/base.py's "Pooled
    fallback" note) rather than a fabricated grouping.
    """
    n_tasks, d = env.n_tasks, env.n_clocks
    n_blocks = getattr(env, "n_blocks", n_tasks)
    n_status_categories = getattr(env, "n_status_categories", 3)
    mlp_group_idx = getattr(env, "mlp_group_idx", np.arange(n_tasks))
    task_job_idx = getattr(env, "task_job_idx", None)
    if encoder_name == "mlp":
        enc: Encoder = MLPPointEncoder(point_dim, n_tasks, n_blocks=n_blocks, n_status_categories=n_status_categories)
        return ActorCriticHead(enc, n_tasks, element_group_idx=mlp_group_idx)
    if encoder_name == "deepset":
        enc = DeepSetHEncoder(d)
        return ActorCriticHead(enc, n_tasks, task_job_idx=task_job_idx)
    if encoder_name == "cnn_dbm":
        enc = CNNDBMEncoder(d)
        return ActorCriticHead(enc, n_tasks, task_job_idx=task_job_idx)
    if encoder_name == "cnn_proj":
        enc = CNNProjEncoder(d)
        return ActorCriticHead(enc, n_tasks, task_job_idx=task_job_idx)
    if encoder_name == "deepset_v":
        # Same job/clock-indexed, static-mapping-required treatment as
        # deepset_h -- pooled fallback on gridtimed for the identical
        # reason (task_job_idx is None there).
        enc = DeepSetVEncoder(d)
        return ActorCriticHead(enc, n_tasks, task_job_idx=task_job_idx)
    if encoder_name == "gnn":
        enc = GNNEncoder(d)
        return ActorCriticHead(enc, n_tasks, task_job_idx=task_job_idx)
    raise ValueError(f"Unknown encoder_name {encoder_name!r}; expected one of {ENCODER_NAMES}")


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    rollout_size: int = 2048  # total transitions per update, across all envs
    num_envs: int = 8
    n_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    total_steps: int = 300_000
    plateau_patience_steps: int = 50_000
    plateau_min_delta: float = 1e-3
    #: Uniform-budget requirement (/tmp/PART1_FINAL.md item 1): plateau
    #: early stopping is now opt-IN, default OFF -- every run trains to
    #: exactly total_steps unless this is explicitly set True. The
    #: plateau-tracking machinery itself (_steps_since_improve etc.) is
    #: left in place either way, not deleted -- only whether it's allowed
    #: to actually set should_stop=True is gated on this flag.
    plateau_enabled: bool = False
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        if self.rollout_size % self.num_envs != 0:
            raise ValueError("rollout_size must be divisible by num_envs")
        self.n_steps: int = self.rollout_size // self.num_envs


@dataclass
class UpdateLog:
    update: int
    env_steps: int
    mean_return: float
    std_return: float
    #: Domain metric averaged over episodes completed during this update's
    #: rollout -- makespan (job-shop/modesel) or steps-to-goal (gridtimed).
    #: Named ``mean_makespan`` for backward compat with
    #: train/generate_report.py's job-shop-only reader; run_one.py (the
    #: multi-domain SLURM entry point) maps this onto the jsonl schema's
    #: generic ``domain_metric`` key. NaN if no episode completed, or if no
    #: completed episode reported a defined metric (e.g. all failed on a
    #: domain where failure has no defined metric).
    mean_makespan: float
    mean_ep_length: float
    n_episodes: int
    entropy: float
    value_loss: float
    policy_loss: float
    explained_variance: float
    #: PPO-standard diagnostics (/tmp/ENCODERS_SETUP.md logging schema).
    approx_kl: float
    clip_frac: float
    grad_norm: float
    wall_clock_s: float
    encoder_forward_s: float
    encoder_forward_frac: float
    #: Wall time inside zoneviews.raster.projection_stack/dbm_tensor during
    #: this update's rollout + PPO epochs -- a SUBSET of encoder_forward_s
    #: for cnn_dbm/cnn_proj (rasterization happens inside their forward()),
    #: 0 for mlp/deepset which never call into zoneviews.raster.
    raster_s: float
    #: Wall time spent inside vec_env.step() during rollout collection only
    #: (excludes the PPO update epochs, which touch no env).
    env_step_s: float


def _masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    return torch.distributions.Categorical(logits=masked_logits)


def _explained_variance(returns: np.ndarray, values: np.ndarray) -> float:
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / var_returns)


class _RolloutBuffer:
    """Flat (n_steps * num_envs) storage for one rollout."""

    def __init__(self, n_steps: int, num_envs: int, point_dim: int, n_tasks: int):
        n = n_steps * num_envs
        self.point = np.zeros((n, point_dim), dtype=np.float32)
        self.mask = np.zeros((n, n_tasks), dtype=bool)
        self.zone: List[Any] = [None] * n
        self.actions = np.zeros(n, dtype=np.int64)
        self.log_probs = np.zeros(n, dtype=np.float32)
        self.values = np.zeros(n, dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.dones = np.zeros(n, dtype=bool)
        self.n_steps = n_steps
        self.num_envs = num_envs

    def set(self, t: int, obs, actions, log_probs, values, rewards, dones):
        lo, hi = t * self.num_envs, (t + 1) * self.num_envs
        self.point[lo:hi] = obs["point"]
        self.mask[lo:hi] = obs["mask"]
        self.zone[lo:hi] = obs["zone"]
        self.actions[lo:hi] = actions
        self.log_probs[lo:hi] = log_probs
        self.values[lo:hi] = values
        self.rewards[lo:hi] = rewards
        self.dones[lo:hi] = dones


class PPOTrainer:
    """:param instance_name: job-shop instance name (original interface,
        unchanged) -- mutually exclusive with ``vec_env``.
    :param vec_env: a pre-built, already-constructed vector env (any
        object satisfying the SyncVectorEnv contract: ``.envs``,
        ``.n_tasks``, ``.n_clocks``, ``.reset(seed)``, ``.step(actions)``,
        ``.drain_episode_stats()``) -- how modesel/gridtimed plug in,
        without PPOTrainer needing to know how to construct their envs.
        Its own ``.reset(seed=config.seed)`` is called here.
    """

    def __init__(self, instance_name: Optional[str], encoder_name: str, config: PPOConfig,
                 vec_env: Optional[SyncVectorEnv] = None):
        # instance_name kept FIRST-positional for backward compatibility
        # with existing callers (train/run_experiment.py: PPOTrainer(instance,
        # encoder, cfg)); pass instance_name=None, vec_env=<...> (keyword)
        # for modesel/gridtimed instead.
        if (instance_name is None) == (vec_env is None):
            raise ValueError("exactly one of instance_name or vec_env must be given")
        self.instance_name = instance_name
        self.encoder_name = encoder_name
        self.cfg = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if vec_env is not None:
            self.vec_env = vec_env
        else:
            self.vec_env = SyncVectorEnv(
                [
                    (lambda s=config.seed + i: SymbolicJobShopGymEnv(instance_name, seed=s))
                    for i in range(config.num_envs)
                ]
            )
        self.point_dim = self.vec_env.envs[0]._point_dim
        self.n_tasks = self.vec_env.n_tasks

        self.head = build_head(encoder_name, self.vec_env, self.point_dim).to(config.device)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=config.lr)

        self._obs = self.vec_env.reset(seed=config.seed)
        self._encoder_forward_s = 0.0  # accumulated within the current update cycle
        self._env_step_s = 0.0  # accumulated within the current update cycle (rollout collection only)
        self._best_mean_return = -float("inf")
        self._steps_since_improve = 0
        self.total_env_steps = 0
        self.update_idx = 0
        self.history: List[UpdateLog] = []

    # ------------------------------------------------------------------
    # Checkpoint support (run_one.py). PPOTrainer itself stays agnostic of
    # *where*/*how often* to checkpoint -- it just knows how to serialize
    # and restore everything needed to resume training bit-for-bit-ish:
    # weights, optimizer state (Adam's moments matter for PPO stability),
    # the plateau tracker, and env_steps/update_idx bookkeeping. RNG state
    # (python/numpy/torch) is NOT included here -- it's process-global, not
    # trainer-owned, so run_one.py captures/restores it directly around
    # calls into this trainer (see run_one.py's checkpoint dict).
    def state_dict(self) -> Dict[str, Any]:
        return {
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_env_steps": self.total_env_steps,
            "update_idx": self.update_idx,
            "best_mean_return": self._best_mean_return,
            "steps_since_improve": self._steps_since_improve,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_env_steps = state["total_env_steps"]
        self.update_idx = state["update_idx"]
        self._best_mean_return = state["best_mean_return"]
        self._steps_since_improve = state["steps_since_improve"]

    # ------------------------------------------------------------------
    def _forward(self, obs_batch: Dict[str, Any]):
        t0 = time.perf_counter()
        logits, value = self.head(obs_batch)
        self._encoder_forward_s += time.perf_counter() - t0
        return logits, value

    def _collect_rollout(self) -> _RolloutBuffer:
        cfg = self.cfg
        buf = _RolloutBuffer(cfg.n_steps, cfg.num_envs, self.point_dim, self.n_tasks)

        for t in range(cfg.n_steps):
            obs = self._obs
            mask_t = torch.as_tensor(obs["mask"])
            with torch.no_grad():
                logits, value = self._forward(obs)
                dist = _masked_categorical(logits, mask_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            t_step0 = time.perf_counter()
            next_obs, reward, term, trunc, infos = self.vec_env.step(action.numpy())
            self._env_step_s += time.perf_counter() - t_step0
            done = term | trunc

            buf.set(
                t,
                obs,
                action.numpy(),
                log_prob.numpy(),
                value.numpy(),
                reward,
                done,
            )
            self._obs = next_obs
            self.total_env_steps += cfg.num_envs

        return buf

    def _bootstrap_value(self) -> np.ndarray:
        with torch.no_grad():
            _, value = self._forward(self._obs)
        return value.numpy()

    def _compute_gae(self, buf: _RolloutBuffer, last_value: np.ndarray) -> tuple:
        cfg = self.cfg
        n_steps, num_envs = buf.n_steps, buf.num_envs
        rewards = buf.rewards.reshape(n_steps, num_envs)
        values = buf.values.reshape(n_steps, num_envs)
        dones = buf.dones.reshape(n_steps, num_envs)

        advantages = np.zeros_like(rewards)
        last_gae = np.zeros(num_envs, dtype=np.float32)
        next_value = last_value
        for t in reversed(range(n_steps)):
            next_non_terminal = 1.0 - dones[t].astype(np.float32)
            delta = rewards[t] + cfg.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]

        returns = advantages + values
        return advantages.reshape(-1), returns.reshape(-1)

    def _ppo_update(self, buf: _RolloutBuffer, advantages: np.ndarray, returns: np.ndarray):
        cfg = self.cfg
        n = buf.n_steps * buf.num_envs
        adv_t = torch.as_tensor(advantages, dtype=torch.float32)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)
        old_log_probs_t = torch.as_tensor(buf.log_probs, dtype=torch.float32)
        actions_t = torch.as_tensor(buf.actions, dtype=torch.long)

        total_entropy, total_v_loss, total_p_loss, n_updates = 0.0, 0.0, 0.0, 0
        total_kl, total_clip_frac, total_grad_norm = 0.0, 0.0, 0.0

        for _epoch in range(cfg.n_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                mb_obs = {
                    "point": buf.point[idx],
                    "mask": buf.mask[idx],
                    "zone": [buf.zone[i] for i in idx],
                }
                mb_mask = torch.as_tensor(mb_obs["mask"])
                mb_adv = adv_t[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                logits, value = self._forward(mb_obs)
                dist = _masked_categorical(logits, mb_mask)
                new_log_prob = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                log_ratio = new_log_prob - old_log_probs_t[idx]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, ret_t[idx])

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.head.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator: E[(r-1) - log r]
                    # (always >= 0, unlike the naive -log_ratio.mean()).
                    approx_kl = float(((ratio - 1) - log_ratio).mean().item())
                    clip_frac = float((torch.abs(ratio - 1) > cfg.clip_coef).float().mean().item())

                total_entropy += float(entropy.item())
                total_v_loss += float(value_loss.item())
                total_p_loss += float(policy_loss.item())
                total_kl += approx_kl
                total_clip_frac += clip_frac
                total_grad_norm += float(grad_norm.item())
                n_updates += 1

        return (
            total_entropy / n_updates,
            total_v_loss / n_updates,
            total_p_loss / n_updates,
            total_kl / n_updates,
            total_clip_frac / n_updates,
            total_grad_norm / n_updates,
        )

    # ------------------------------------------------------------------
    def _do_one_update(self) -> "tuple[UpdateLog, List[Dict[str, Any]], bool]":
        """Run exactly one rollout+PPO-update cycle. Returns ``(log,
        raw_episodes, should_stop)`` -- ``raw_episodes`` is the undigested
        per-episode list (return/length/domain_metric/success) so callers
        that need per-episode records (run_one.py's episodes.jsonl) don't
        have to re-derive them from the aggregated ``log``. ``should_stop``
        reflects the plateau tracker (``self._best_mean_return`` /
        ``self._steps_since_improve``, both updated in place here so it
        survives a checkpoint/resume cycle via state_dict()).

        This is the single source of truth for "what happens in one PPO
        update" -- both ``train()`` (in-process, job-shop-compatible) and
        run_one.py's checkpointing loop (modesel/gridtimed, SLURM) call
        this same method, so there is exactly one place update semantics
        can drift.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        self._encoder_forward_s = 0.0
        self._env_step_s = 0.0
        reset_raster_wall_time()

        buf = self._collect_rollout()
        last_value = self._bootstrap_value()
        advantages, returns = self._compute_gae(buf, last_value)
        entropy, v_loss, p_loss, approx_kl, clip_frac, grad_norm = self._ppo_update(buf, advantages, returns)

        wall = time.perf_counter() - t0
        ev = _explained_variance(returns, buf.values)
        raster_s = raster_wall_time()

        episodes = self.vec_env.drain_episode_stats()
        if episodes:
            rets = [e["return"] for e in episodes]
            # "makespan" (job-shop's own SyncVectorEnv) or "domain_metric"
            # (train/vec_env.py, modesel/gridtimed) -- either way, drop
            # episodes where the metric is undefined (e.g. a failed
            # gridtimed episode has no steps-to-goal) rather than
            # feeding None into np.mean.
            metrics = [e["domain_metric"] if "domain_metric" in e else e["makespan"] for e in episodes]
            metrics = [m for m in metrics if m is not None]
            lengths = [e["length"] for e in episodes]
            mean_return, std_return = float(np.mean(rets)), float(np.std(rets))
            mean_makespan = float(np.mean(metrics)) if metrics else float("nan")
            mean_length = float(np.mean(lengths))
        else:
            mean_return = std_return = mean_makespan = mean_length = float("nan")

        log = UpdateLog(
            update=self.update_idx,
            env_steps=self.total_env_steps,
            mean_return=mean_return,
            std_return=std_return,
            mean_makespan=mean_makespan,
            mean_ep_length=mean_length,
            n_episodes=len(episodes),
            entropy=entropy,
            value_loss=v_loss,
            policy_loss=p_loss,
            explained_variance=ev,
            approx_kl=approx_kl,
            clip_frac=clip_frac,
            grad_norm=grad_norm,
            wall_clock_s=wall,
            encoder_forward_s=self._encoder_forward_s,
            encoder_forward_frac=self._encoder_forward_s / wall if wall > 0 else 0.0,
            raster_s=raster_s,
            env_step_s=self._env_step_s,
        )
        self.history.append(log)

        should_stop = False
        if not np.isnan(mean_return):
            if mean_return > self._best_mean_return + cfg.plateau_min_delta:
                self._best_mean_return = mean_return
                self._steps_since_improve = 0
            else:
                self._steps_since_improve += cfg.num_envs * cfg.n_steps
            if cfg.plateau_enabled and self._steps_since_improve >= cfg.plateau_patience_steps:
                should_stop = True

        self.update_idx += 1
        return log, episodes, should_stop

    def train(self, log_every: int = 1, verbose: bool = True) -> List[UpdateLog]:
        cfg = self.cfg
        while self.total_env_steps < cfg.total_steps:
            log, _episodes, should_stop = self._do_one_update()
            if verbose and log.update % log_every == 0:
                print(
                    f"[{self.instance_name}/{self.encoder_name}/seed{cfg.seed}] "
                    f"upd={log.update:4d} steps={log.env_steps:7d} "
                    f"ret={log.mean_return:8.2f}±{log.std_return:6.2f} makespan={log.mean_makespan:7.2f} "
                    f"H={log.entropy:5.3f} Vloss={log.value_loss:8.3f} EV={log.explained_variance:5.2f} "
                    f"wall={log.wall_clock_s:5.2f}s enc_frac={log.encoder_forward_frac * 100:4.1f}%"
                )
            if should_stop:
                if verbose:
                    print(f"  early stop: no improvement for {self._steps_since_improve} steps")
                break
        return self.history

        return self.history
