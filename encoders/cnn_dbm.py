"""Small CNN over ``dbm_tensor`` (CNN-M) -- the architectural control.

Consumes the same underlying information as ``deepset_h`` (every finite DBM
bound/strictness cell, over the same ``(d+1) x (d+1)`` clock grid, plus a
finiteness mask channel) but through a spatial-grid CNN instead of a
set-pooled MLP. This is the intended control for the DeepSet-H comparison:
same information content, different architecture -- so any performance gap
between the two isolates the effect of *architecture* rather than *what the
encoder can see*.

``has_per_element_head = True``: unlike ``deepset_h``'s constraint-indexed
elements, ``dbm_tensor``'s rows/columns are indexed directly by *clock*, and
in this domain clock == job (row ``i`` in ``1..d`` is exactly job clock
``i``). Averaging the conv feature map over the column axis and dropping
the reference-clock row (row 0) therefore gives a genuinely natural
per-job embedding, which the shared :class:`encoders.base.ActorCriticHead`
broadcasts out to each job's task slot(s) via ``task_job_idx``. This is the
one encoder among the four with a principled per-element action head.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from zoneviews.raster import dbm_tensor

from .base import Encoder, EncoderOutput


class CNNDBMEncoder(Encoder):
    has_per_element_head = True

    def __init__(self, d: int, embed_dim: int = 120, channels: Tuple[int, int] = (40, 58), per_task_dim: int = 36):
        super().__init__()
        self.d = d
        self.n = d + 1
        self.embed_dim = embed_dim
        self.per_task_dim = per_task_dim

        c1, c2 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Global average pool over the spatial grid before the readout,
        # not a flatten: flattening would make pool_readout's parameter
        # count scale as O(d^2) (grid is (d+1) x (d+1)), which is exactly
        # the kind of instance-size-dependent blowup the param-matching
        # requirement is trying to avoid -- confirmed empirically (LJS_01's
        # d=6 grid made a flattened readout ~3x every other encoder).
        self.pool_readout = nn.Linear(c2, embed_dim)
        self.job_proj = nn.Linear(c2, per_task_dim)

    def forward(self, obs_batch: Dict[str, Any], shuffle_clocks: bool = False) -> EncoderOutput:
        """:param shuffle_clocks: permutation-ablation hook -- relabels the
        real clocks (rows/cols 1..d) with a random permutation before
        rasterizing, independently per zone in the batch. The reference
        clock (row/col 0) is never moved.
        """
        zones = obs_batch["zone"]
        b = len(zones)

        arrs = []
        for zone in zones:
            t = dbm_tensor(zone)
            if shuffle_clocks:
                t = _permute_dbm_tensor(t, self.d)
            arrs.append(t)
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float()  # (B, 3, n, n)

        feat = self.conv(x)  # (B, c2, n, n)
        pooled = self.pool_readout(feat.mean(dim=(2, 3)))  # global average pool -> (B, c2) -> (B, embed_dim)

        row_feat = feat.mean(dim=3)  # (B, c2, n) -- average over columns
        row_feat = row_feat[:, :, 1:]  # drop the reference-clock row (index 0)
        row_feat = row_feat.permute(0, 2, 1)  # (B, d, c2)
        per_task = self.job_proj(row_feat)  # (B, d, per_task_dim)

        return {"pooled": pooled, "per_task": per_task}


def _permute_dbm_tensor(t: np.ndarray, d: int) -> np.ndarray:
    """Relabel real clocks ``1..d`` by a random permutation (reference
    clock 0 fixed), permuting both rows and columns consistently."""
    perm = np.random.permutation(d)
    order = np.concatenate([[0], perm + 1])
    return t[order][:, order]
