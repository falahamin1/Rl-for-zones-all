"""MLP over the ``'point'`` observation -- the non-symbolic baseline.

Consumes the flat, already-canonically-ordered per-task feature vector plus
the boolean mask (concatenated, per spec) through a plain feedforward
network to a single pooled embedding. No permutation structure beyond the
per-element head below -- deliberately the simplest possible encoder, so
the symbolic encoders are compared against a real (not handicapped)
non-symbolic baseline.

``has_per_element_head = True`` (confound fix, /tmp/ENCODERS_SETUP.md item
1): the ``point`` vector already has one dedicated feature block per
per-element "block" (a status one-hot -- category count configurable via
``n_status_categories``, since not every domain has job-shop's exact 3
statuses; modesel needs 4: waiting/active/done/failed -- plus an elapsed
fraction). A small MLP with weights SHARED across blocks, applied
independently to each block's own slice (concatenated with the shared
job/clock features every block can see), is a genuinely natural per-element
readout for this encoder, not an invented one. It does not reintroduce any
zone/symbolic information (the encoder still only ever sees ``point``,
never ``zone``) -- it only changes whether the final logits are read off a
single pooled vector or a per-block-decomposed one, which is exactly the
confound under test, so this still leaves MLP-Point the only encoder with
zero access to zone geometry.

``n_blocks`` (the number of point-vector per-element blocks) is a SEPARATE
constructor parameter from ``n_tasks`` (the actual action-space size)
because they don't always coincide: job-shop has one block per task
(``n_blocks == n_tasks``, an identity broadcast -- the degenerate case of
:class:`encoders.base.ActorCriticHead`'s general group-broadcast
mechanism), but modesel has one block per raw task while the action space
is ``2 x n_blocks`` (FAST and SLOW both draw on the same task's block, and
``ActorCriticHead``'s ``element_group_idx``/sub-slot mechanism is what lets
them still end up with different final logits).

``max_point_dim``/``max_blocks`` (both default to the exact size needed and
are then no-ops) exist purely for extensibility: a later experiment that
wants one MLP shared across differently-sized instances can pad every
instance's ``point``/``mask`` up to a common budget without changing this
class.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import Encoder, EncoderOutput


class MLPPointEncoder(Encoder):
    def __init__(
        self,
        point_dim: int,
        n_tasks: int,
        n_blocks: Optional[int] = None,
        n_status_categories: int = 3,
        embed_dim: int = 92,
        hidden: int = 112,
        per_task_dim: int = 20,
        per_task_hidden: int = 48,
        max_point_dim: Optional[int] = None,
        max_blocks: Optional[int] = None,
    ):
        super().__init__()
        self.point_dim = point_dim
        self.n_tasks = n_tasks
        self.n_blocks = n_blocks if n_blocks is not None else n_tasks
        self.n_status_categories = n_status_categories
        self.block_width = n_status_categories + 1  # status one-hot + elapsed fraction
        self.per_task_dim = per_task_dim
        self.max_point_dim = max_point_dim or point_dim
        self.max_blocks = max_blocks or self.n_blocks
        if self.max_point_dim < point_dim or self.max_blocks < self.n_blocks:
            raise ValueError("max_point_dim/max_blocks must be >= the instance's actual size")

        self.embed_dim = embed_dim
        self.shared_dim = point_dim - self.block_width * self.n_blocks
        if self.shared_dim < 0:
            raise ValueError(
                f"point_dim={point_dim} too small for n_blocks={self.n_blocks} x "
                f"block_width={self.block_width}"
            )
        in_dim = self.max_point_dim + self.n_tasks  # point (padded) ++ mask
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Per-block branch: shared weights applied to [shared features ++
        # this block's own (n_status_categories+1)-dim status/elapsed
        # slice] independently per block -- see module docstring.
        self.per_task_net = nn.Sequential(
            nn.Linear(self.shared_dim + self.block_width, per_task_hidden),
            nn.ReLU(),
            nn.Linear(per_task_hidden, per_task_dim),
        )
        self.pooled_readout = nn.Linear(hidden, embed_dim)

    def forward(self, obs_batch: Dict[str, Any], shuffle_tasks: bool = False) -> EncoderOutput:
        """:param shuffle_tasks: permutation-ablation hook -- permutes which
        block's status-onehot/elapsed-fraction features occupy which slot of
        the ``point`` vector (independently per batch item), while leaving
        the mask and the action head's fixed slot-i-means-action-i output
        convention untouched. Since the point vector's per-block slices are
        the only thing that changes, this directly tests whether the MLP's
        weights have learned to rely on genuine block content or on which
        position that content happened to sit at during training. Job-clock
        features (not block-indexed) are left alone.
        """
        point_np = np.asarray(obs_batch["point"])
        d = self.shared_dim
        bw = self.block_width
        nb = self.n_blocks
        if shuffle_tasks:
            point_np = point_np.copy()
            for row in range(point_np.shape[0]):
                perm = np.random.permutation(nb)
                status = point_np[row, d : d + (bw - 1) * nb].reshape(nb, bw - 1)
                elapsed = point_np[row, d + (bw - 1) * nb : d + bw * nb]
                point_np[row, d : d + (bw - 1) * nb] = status[perm].reshape(-1)
                point_np[row, d + (bw - 1) * nb : d + bw * nb] = elapsed[perm]

        point = torch.as_tensor(point_np, dtype=torch.float32)
        mask = torch.as_tensor(np.asarray(obs_batch["mask"]), dtype=torch.float32)
        b = point.shape[0]

        point_padded = point
        if self.max_point_dim > self.point_dim:
            point_padded = torch.cat([point_padded, torch.zeros(b, self.max_point_dim - self.point_dim)], dim=1)

        x = torch.cat([point_padded, mask], dim=1)
        pooled = self.pooled_readout(self.net(x))

        # Per-block branch, on the UNPADDED point (padding is only for a
        # future shared-across-instance-size use case, not part of this
        # sweep -- see class docstring).
        shared = point[:, :d]  # (B, shared_dim)
        status = point[:, d : d + (bw - 1) * nb].reshape(b, nb, bw - 1)
        elapsed = point[:, d + (bw - 1) * nb : d + bw * nb].reshape(b, nb, 1)
        per_block = torch.cat([status, elapsed], dim=-1)  # (B, n_blocks, block_width)
        shared_bcast = shared.unsqueeze(1).expand(-1, nb, -1)  # (B, n_blocks, shared_dim)
        per_task_in = torch.cat([shared_bcast, per_block], dim=-1)  # (B, n_blocks, shared_dim+block_width)
        per_task = self.per_task_net(per_task_in)  # (B, n_blocks, per_task_dim)

        return {"pooled": pooled, "per_task": per_task}
