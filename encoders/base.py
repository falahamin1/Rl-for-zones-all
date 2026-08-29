"""Shared encoder interface and actor/critic head.

Every encoder consumes a *batch* of observations, as produced by a
``SyncVectorEnv`` (a dict with ``'point'`` ``(B, point_dim)`` float32
ndarray, ``'zone'`` a length-``B`` Python list of :class:`zoneviews.zone.Zone`,
and ``'mask'`` ``(B, N)`` bool ndarray), and must produce a fixed-size pooled
embedding for the shared value head.

Per-element action head
------------------------
Confound fix (see /tmp/ENCODERS_SETUP.md item 1): ALL FOUR encoders now have
a per-element head, ``has_per_element_head = True`` uniformly. Each
encoder's ``per_task`` output is indexed by whatever *group* is natural for
its own architecture -- ``d`` job/clock rows for ``cnn_dbm``/``cnn_proj``/
``deepset_h`` (see each module's docstring for how that group is computed),
or one row per raw task for ``mlp_point`` (the ``point`` vector already has
a dedicated feature block per task). :class:`ActorCriticHead` broadcasts
whichever grouping an encoder uses out to the actual action space via
``element_group_idx`` -- an ``(n_actions,)`` int array mapping every action
to its group index, ALWAYS required now (there is no more "already
task-indexed, no broadcast needed" special case: job-shop's original
mlp_point happened to have one group per action, i.e. an identity mapping,
which is just the degenerate case of the same general mechanism).

Sub-slot disambiguation (found while wiring up modesel, not merely
theoretical)
-------------------------------------------------------------------------
A group-indexed per-element head gives every action mapped to the same
group the SAME base scalar (one ``Linear(per_task_dim, 1)`` row, broadcast
via ``index_select``). In prob_jobshop_ext this was harmless: cnn_dbm's
groups are jobs, and a job's tasks are a sequential chain, so at most one
of a job's tasks is ever unmasked at a time -- the "collision" only ever
affects actions that can't be picked anyway. modesel breaks that
assumption: FAST and SLOW are BOTH always legal for the same task
simultaneously, so a naive broadcast would give them the identical logit
forever, and a group-indexed encoder could never learn a FAST-vs-SLOW
preference at all. Fixed generically here (not per-encoder, and not only
for the "job" case): every action sharing a group gets a fixed one-hot
"which Nth action mapped to this group am I" feature, computed once from
``element_group_idx`` at construction time and concatenated onto the
broadcast per-element embedding before the final linear readout -- so
same-group actions still share the pooled group context but are no longer
forced to share a logit. For job-shop's mlp (identity mapping, group size
always 1) this is a costless no-op (one-hot of width 1, always ``[1.0]``).

Pooled fallback (gridtimed only)
---------------------------------
``element_group_idx`` must be a FIXED, state-independent mapping (it's
registered as a buffer, shared by every batch element) -- true for
job-shop/modesel, where a task statically belongs to one job for the whole
episode. It is NOT true for gridtimed's zone-based encoders: which clock a
move is "about" depends on the agent's CURRENT position (a move that leads
to a gate at one position leads to a plain cell at another), so there is no
static per-action clock grouping to broadcast from at construction time.
mlp_point's grouping (by MOVE, not clock) stays static there and keeps its
per-element head; cnn_dbm/cnn_proj/deepset_h fall back to the ORIGINAL
pooled head (``Linear(embed_dim, n_tasks)`` over ``out["pooled"]``) for
gridtimed specifically -- pass ``element_group_idx=None`` to get this path.
This is the task's own explicitly-sanctioned escape hatch ("if any [encoder]
cannot [get a natural per-element head], run the full 2x4 grid... and
report it as such") applied per-domain rather than per-encoder-globally.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

EncoderOutput = Dict[str, Optional[torch.Tensor]]
#: {'pooled': (B, embed_dim), 'per_task': (B, n_groups, per_task_dim)}


class Encoder(nn.Module):
    """Base class every encoder subclasses.

    Subclasses must set ``self.embed_dim`` and ``self.per_task_dim`` (int)
    in ``__init__``, and implement :meth:`forward` to always return a real
    ``per_task`` tensor (never ``None`` -- every encoder has a per-element
    head now).
    """

    has_per_element_head: bool = True

    def forward(self, obs_batch: Dict[str, Any]) -> EncoderOutput:  # pragma: no cover - interface
        raise NotImplementedError

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ActorCriticHead(nn.Module):
    """Shared policy/value head: wraps any :class:`Encoder`.

    :param encoder: The encoder producing embeddings.
    :param n_tasks: Size of the (fixed, per-instance) discrete action space.
    :param element_group_idx: ``(n_tasks,)`` int array/tensor mapping every
        action index to the per-element group index that "owns" it (a job,
        a clock, a raw task -- whatever the encoder's own per-element axis
        represents). Required for every encoder now.
    """

    def __init__(self, encoder: Encoder, n_tasks: int, element_group_idx: Optional[np.ndarray] = None,
                 task_job_idx: Optional[np.ndarray] = None):
        super().__init__()
        # task_job_idx accepted as a backward-compatible alias.
        if element_group_idx is None:
            element_group_idx = task_job_idx

        self.encoder = encoder
        self.n_tasks = n_tasks
        self.value_head = nn.Linear(encoder.embed_dim, 1)
        self.per_element = element_group_idx is not None

        if not self.per_element:
            self.element_group_idx = None
            self.action_head = nn.Linear(encoder.embed_dim, n_tasks)
            return

        gi = np.asarray(element_group_idx, dtype=np.int64)
        assert len(gi) == n_tasks, f"element_group_idx has length {len(gi)}, expected n_tasks={n_tasks}"
        self.register_buffer("element_group_idx", torch.as_tensor(gi, dtype=torch.long))

        sub_slot = np.zeros(len(gi), dtype=np.int64)
        counts: Dict[int, int] = {}
        for i, j in enumerate(gi):
            sub_slot[i] = counts.get(int(j), 0)
            counts[int(j)] = counts.get(int(j), 0) + 1
        self._max_slots = max(counts.values()) if counts else 1
        sub_slot_onehot = nn.functional.one_hot(
            torch.as_tensor(sub_slot, dtype=torch.long), num_classes=self._max_slots
        ).float()
        self.register_buffer("_sub_slot_onehot", sub_slot_onehot)  # (n_tasks, max_slots)
        self.action_head = nn.Linear(encoder.per_task_dim + self._max_slots, 1)

    def forward(self, obs_batch: Dict[str, Any], **encoder_kwargs: Any):
        """Return ``(logits, value)``: ``logits`` is ``(B, n_tasks)`` (raw,
        unmasked -- masking is applied by the trainer just before the
        softmax), ``value`` is ``(B,)``.

        :param encoder_kwargs: Forwarded to ``encoder.forward`` unchanged --
            this is how the permutation ablation's ``shuffle_*`` flags reach
            whichever encoder is wrapped, without every call site needing to
            know which encoder-specific kwarg name applies.
        """
        out = self.encoder(obs_batch, **encoder_kwargs)
        value = self.value_head(out["pooled"]).squeeze(-1)

        if not self.per_element:
            logits = self.action_head(out["pooled"])  # (B, n_tasks)
            return logits, value

        per_elem = out["per_task"]  # (B, n_groups, per_task_dim)
        broadcast = per_elem.index_select(1, self.element_group_idx)  # (B, n_tasks, per_task_dim)
        sub_slot = self._sub_slot_onehot.unsqueeze(0).expand(broadcast.shape[0], -1, -1)
        combined = torch.cat([broadcast, sub_slot], dim=-1)  # (B, n_tasks, per_task_dim+max_slots)
        logits = self.action_head(combined).squeeze(-1)  # (B, n_tasks)

        return logits, value
