"""DeepSets over ``Zone.v_rep()`` -- the V-representation counterpart to
``deepset_h``'s H-representation.

Each element is one exact vertex of the zone (a point in ``Q^d``), cast to
float32. Same shared 2-layer 128-d MLP (``phi``), sum-pool (not mean --
matches deepset_h), single linear readout -- architecturally identical to
``deepset_h`` (same hidden width, same pooling, same per-job broadcast
mechanism) so any H-vs-V performance gap isolates the REPRESENTATION
(constraints vs. vertices), not an incidental architecture difference.

``v_rep()`` is called through ``zoneviews.raster.cached_v_rep`` --
content-keyed (not address-keyed, see /tmp/PRESUBMIT.md), same caching
this repo already uses for ``projection_stack``/``dbm_tensor``. Feasibility
(cost, cache hit rate, degeneracy) verified in /tmp/VREP_FEASIBILITY.md
before this module was written.

VertexEnumerationLimit fallback: exact enumeration is ``raise``-based
(zoneviews.zone.VertexEnumerationLimit) when ``C(m', affine_dim)`` exceeds
the exact-enumeration budget. On that exception, this encoder falls back to
the zone's axis-aligned BOUNDING-BOX corners (``2**d`` points, read directly
off the per-clock DBM bounds -- cheap, no enumeration) rather than crashing
or silently returning an empty set. Every fallback is counted in a
module-level accumulator (``vrep_fallback_stats()``/``reset_vrep_fallback_stats()``,
same pattern as ``zoneviews.raster``'s wall-time accumulators) so
``run_one.py`` can log the true per-run fallback rate as ``vrep_fallback_frac``
rather than this being invisible.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict

import torch
import torch.nn as nn

from zoneviews.raster import cached_v_rep
from zoneviews.zone import VertexEnumerationLimit

from .base import Encoder, EncoderOutput

#: Cumulative fallback/total-call counts across the current logging
#: window -- read + reset once per PPO update, same convention as
#: zoneviews.raster.raster_wall_time()/reset_raster_wall_time().
_vrep_fallback_count = 0
_vrep_total_count = 0


def reset_vrep_fallback_stats() -> None:
    global _vrep_fallback_count, _vrep_total_count
    _vrep_fallback_count = 0
    _vrep_total_count = 0


def vrep_fallback_stats() -> Dict[str, float]:
    frac = _vrep_fallback_count / _vrep_total_count if _vrep_total_count else 0.0
    return {"fallback_count": _vrep_fallback_count, "total_count": _vrep_total_count, "fallback_frac": frac}


def _bounding_box_corners(zone) -> list:
    """2**d axis-aligned bounding-box corners, one coordinate per real
    clock read directly off the DBM's own single-clock bounds (cheap --
    no enumeration, no Fraction Gauss-Jordan). Same bound-extraction as
    projection_stack's per_zone_normalize path."""
    dbm = zone.dbm
    d = zone.d
    ranges = []
    for i in range(1, d + 1):
        hi = float(dbm.bound(i, 0)) if not dbm.is_infinity(i, 0) else float(max(zone.k[i - 1], 0))
        lo = float(-dbm.bound(0, i)) if not dbm.is_infinity(0, i) else 0.0
        if hi < lo:
            hi = lo
        ranges.append((lo, hi))
    return list(product(*ranges))


def _get_vertices(zone) -> list:
    global _vrep_fallback_count, _vrep_total_count
    _vrep_total_count += 1
    try:
        return [tuple(float(x) for x in v) for v in cached_v_rep(zone)]
    except VertexEnumerationLimit:
        _vrep_fallback_count += 1
        return _bounding_box_corners(zone)


class DeepSetVEncoder(Encoder):
    has_per_element_head = True

    def __init__(self, d: int, embed_dim: int = 96, hidden: int = 128, per_task_dim: int = 10):
        super().__init__()
        self.d = d
        self.embed_dim = embed_dim
        self.per_task_dim = per_task_dim
        in_dim = d  # one vertex = d real-clock coordinates (no reference-clock coordinate, always 0)
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.readout = nn.Linear(hidden, embed_dim)
        self.job_readout = nn.Linear(hidden, per_task_dim)
        self._hidden = hidden

    def forward(self, obs_batch: Dict[str, Any], shuffle_elements: bool = False) -> EncoderOutput:
        """:param shuffle_elements: permutation-ablation hook, same contract
        as deepset_h's -- shuffles each zone's vertex list before pooling
        (sum-pool is invariant to this by construction)."""
        zones = obs_batch["zone"]
        b = len(zones)

        feats = []
        batch_idx = []
        for bi, zone in enumerate(zones):
            verts = _get_vertices(zone)
            if shuffle_elements:
                import random

                verts = list(verts)
                random.shuffle(verts)
            for v in verts:
                feats.append(list(v))
                batch_idx.append(bi)

        if len(feats) == 0:
            pooled = torch.zeros(b, self.embed_dim)
            per_task = torch.zeros(b, self.d, self.per_task_dim)
            return {"pooled": pooled, "per_task": per_task}

        feats_t = torch.tensor(feats, dtype=torch.float32)
        batch_idx_t = torch.tensor(batch_idx, dtype=torch.long)

        elem_emb = self.phi(feats_t)  # (M_total, hidden)
        pooled_hidden = torch.zeros(b, self._hidden, dtype=elem_emb.dtype)
        pooled_hidden = pooled_hidden.index_add(0, batch_idx_t, elem_emb)
        pooled = self.readout(pooled_hidden)

        # Per-job pooling: sum phi(vertex) over exactly the elements whose
        # coordinate for job j is nonzero (same "which job does this
        # element touch" convention as deepset_h, applied to vertex
        # coordinates instead of constraint coefficients).
        involves = feats_t != 0  # (M_total, d) bool
        elem_sel, job_sel = involves.nonzero(as_tuple=True)
        job_pool_flat = torch.zeros(b * self.d, self._hidden, dtype=elem_emb.dtype)
        if elem_sel.numel() > 0:
            combined_idx = batch_idx_t[elem_sel] * self.d + job_sel
            job_pool_flat = job_pool_flat.index_add(0, combined_idx, elem_emb[elem_sel])
        per_task = self.job_readout(job_pool_flat.reshape(b, self.d, self._hidden))

        return {"pooled": pooled, "per_task": per_task}
