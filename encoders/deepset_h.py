"""DeepSets over ``Zone.h_rep()``.

Each element is one non-redundant DBM half-space constraint, encoded as
``(a_1..a_d, b_normalized, strict_bit)`` (dimension ``d + 2``). A shared
2-layer 128-d MLP (``phi``, per spec) embeds every element independently;
elements are summed (not averaged -- sum-pool, per spec) per zone; a single
linear readout layer maps the pooled sum to the embedding. Constraint count
varies per zone/decision-point and is handled natively via a flat
concatenation + segment-sum (``index_add``) over a per-element batch index
-- no padding, no fixed max constraint count anywhere in this module.

``has_per_element_head = True`` (confound fix, /tmp/ENCODERS_SETUP.md item
1): the elements here are *constraints*,
not tasks, so there is no *direct* per-task slice -- but every constraint's
half-space coefficient vector ``a`` already tells you exactly which job(s)
it references (at most two nonzero entries, per HalfSpace's own
construction). A per-job embedding is therefore a genuine sum-pool DeepSets
variant: for each job ``j``, sum ``phi(element)`` over exactly the elements
whose ``a[j] != 0`` (order-invariant within each job's group, same as the
original single global pool), then project to ``per_task_dim`` and let
:class:`encoders.base.ActorCriticHead` broadcast it out to that job's task
slot(s) via ``task_job_idx`` -- the same broadcast mechanism ``cnn_dbm``
already uses. This is not a stretch: it reuses ``elem_emb``, which was
already being computed for the global pool anyway.

By construction, this encoder is invariant to the order ``h_rep()`` returns
its constraints in (sum-pool over a set) -- this is the permutation-ablation
control the other three encoders are compared against.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import Encoder, EncoderOutput


class DeepSetHEncoder(Encoder):
    def __init__(self, d: int, embed_dim: int = 96, hidden: int = 128, per_task_dim: int = 8):
        super().__init__()
        self.d = d
        self.embed_dim = embed_dim
        self.per_task_dim = per_task_dim
        in_dim = d + 2  # a (d) + normalized bound (1) + strict bit (1)
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
        """:param shuffle_elements: permutation-ablation hook -- shuffles
        each zone's constraint list before pooling. Sum-pool is invariant to
        this by construction; passed through here (rather than only at the
        call site) so the ablation script exercises the exact same code
        path used at evaluation.
        """
        zones = obs_batch["zone"]
        b = len(zones)

        feats = []
        batch_idx = []
        for bi, zone in enumerate(zones):
            cap = max(zone.k) + 1 if len(zone.k) else 1
            h = list(zone.h_rep())
            if shuffle_elements:
                import random

                random.shuffle(h)
            for hs in h:
                row = list(hs.a) + [hs.b / cap, 1.0 if hs.strict else 0.0]
                feats.append(row)
                batch_idx.append(bi)

        if len(feats) == 0:
            # No constraints at all (never observed in practice, but the
            # interface must stay uniform now that every encoder always
            # returns a real per_task tensor, not None).
            pooled = torch.zeros(b, self.embed_dim)
            per_task = torch.zeros(b, self.d, self.per_task_dim)
            return {"pooled": pooled, "per_task": per_task}

        feats_t = torch.tensor(feats, dtype=torch.float32)
        batch_idx_t = torch.tensor(batch_idx, dtype=torch.long)

        elem_emb = self.phi(feats_t)  # (M_total, hidden)
        pooled_hidden = torch.zeros(b, self._hidden, dtype=elem_emb.dtype)
        pooled_hidden = pooled_hidden.index_add(0, batch_idx_t, elem_emb)
        pooled = self.readout(pooled_hidden)

        # Per-job pooling: sum phi(element) over exactly the elements whose
        # coefficient vector references job j (a[j] != 0), for every job.
        involves = feats_t[:, : self.d] != 0  # (M_total, d) bool
        elem_sel, job_sel = involves.nonzero(as_tuple=True)  # both (K,)
        job_pool_flat = torch.zeros(b * self.d, self._hidden, dtype=elem_emb.dtype)
        if elem_sel.numel() > 0:
            combined_idx = batch_idx_t[elem_sel] * self.d + job_sel
            job_pool_flat = job_pool_flat.index_add(0, combined_idx, elem_emb[elem_sel])
        per_task = self.job_readout(job_pool_flat.reshape(b, self.d, self._hidden))

        return {"pooled": pooled, "per_task": per_task}
