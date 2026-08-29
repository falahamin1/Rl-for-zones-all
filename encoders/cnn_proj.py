"""Small CNN over ``projection_stack`` (CNN-P).

Consumes the ``d(d-1)/2`` pairwise 2D clock-shadow rasters as a
multi-channel image through a standard strided-conv stack. Channel count
varies by instance (``d(d-1)/2``), so ``d`` is a required constructor
argument that sizes the first conv layer's ``in_channels`` -- never
hardcoded, so a later instance with a different clock count needs no code
change, just a different ``d``.

``has_per_element_head = True`` (confound fix, /tmp/ENCODERS_SETUP.md item
1): each channel here is indexed by a
*pair* of clocks ``(i, j)``, not a single clock/job, so a per-job embedding
cannot be recovered from the main conv stack's output (Conv2d mixes all
input channels together before any pooling happens -- there is no way to
un-mix "the contribution of clock i" from the post-conv feature map). A
genuinely natural per-job signal is still available one level earlier,
though: gather and average the ``d-1`` *raw rasterized input* channels that
mention clock ``j`` (before the shared conv stack), and run that single
averaged ``(res, res)`` map through a small dedicated conv branch, shared
across all ``d`` jobs (same weights every time, so it's still one small
network, not ``d`` separate ones). This is a genuinely new, small piece of
machinery (not present in the CNN-DBM row-pooling trick, which works
because CNN-DBM's channels ARE already indexed by clock) -- reported as
such rather than silently presented as equivalent.

The channel stack is treated as an ordinary multi-channel image with a
fixed, meaningful channel order (the conv kernels mix channels together
in the usual dense way) -- NOT a per-channel-shared, order-invariant
design. This is deliberate: item 4 of the report explicitly expects both
CNNs to degrade under the channel/clock permutation ablation, and an
invariant-by-construction CNN-P would silently defeat that comparison.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from zoneviews.raster import projection_stack

from .base import Encoder, EncoderOutput


class CNNProjEncoder(Encoder):
    def __init__(self, d: int, res: int = 32, embed_dim: int = 90, channels: Tuple[int, int] = (11, 19),
                 per_task_dim: int = 24, job_channels: int = 8, per_zone_normalize: bool = False):
        super().__init__()
        if d < 2:
            raise ValueError("CNNProjEncoder requires d >= 2 (need at least one clock pair)")
        self.d = d
        self.res = res
        self.in_channels = d * (d - 1) // 2
        self.embed_dim = embed_dim
        self.per_task_dim = per_task_dim
        #: Confound fix (/tmp/ENCODERS_SETUP.md item 3): rasterize each
        #: zone within its OWN bounding box per clock pair rather than the
        #: global k-box, since global-box occupancy measured under ~0.1%
        #: lit pixels on both new domains -- see zoneviews.raster.projection_stack.
        self.per_zone_normalize = per_zone_normalize

        c1, c2 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.readout = nn.Linear(c2 * 4 * 4, embed_dim)

        # Per-job branch: average the raw (pre-conv) channels mentioning
        # clock j, run the single averaged map through a small SHARED conv,
        # same weights for every job -- see module docstring.
        mask = torch.zeros(d, self.in_channels)
        for c, (p, q) in enumerate(combinations(range(d), 2)):
            mask[p, c] = 1.0
            mask[q, c] = 1.0
        mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer("_job_channel_mask", mask)  # (d, C), rows sum to 1

        self.job_conv = nn.Sequential(
            nn.Conv2d(1, job_channels, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.job_readout = nn.Linear(job_channels * 2 * 2, per_task_dim)

    def forward(self, obs_batch: Dict[str, Any], shuffle_channels: bool = False) -> EncoderOutput:
        """:param shuffle_channels: permutation-ablation hook -- randomly
        permutes channel order per zone in the batch before the conv stack.
        """
        zones = obs_batch["zone"]
        b = len(zones)

        arrs = []
        for zone in zones:
            a = projection_stack(zone, res=self.res, per_zone_normalize=self.per_zone_normalize)
            if shuffle_channels:
                a = a[np.random.permutation(a.shape[0])]
            arrs.append(a)
        x = torch.from_numpy(np.stack(arrs)).float()  # (B, C, res, res)

        feat = self.conv(x)
        pooled = self.readout(feat.reshape(b, -1))

        job_maps = torch.einsum("bcxy,jc->bjxy", x, self._job_channel_mask)  # (B, d, res, res)
        job_maps_flat = job_maps.reshape(b * self.d, 1, self.res, self.res)
        job_feat = self.job_conv(job_maps_flat)
        per_task = self.job_readout(job_feat.reshape(b * self.d, -1)).reshape(b, self.d, self.per_task_dim)

        return {"pooled": pooled, "per_task": per_task}
