"""Two-layer GCN over ``Zone.graph()``.

Nodes are clocks (index ``0`` = reference clock, ``1..d`` real clocks, same
labeling ``Zone.graph()`` itself uses). Node features are the clock's own
bounds against the reference: ``(bound(0,i)_normalized, finite(0,i),
bound(i,0)_normalized, finite(i,0))`` -- for the reference node itself
(``i=0``) this collapses to the trivial diagonal encoding ``(0,0,1,1)``,
matching ``dbm_tensor``'s diagonal convention. Edges carry
``(bound_normalized, strict, finiteness)`` for each finite DBM entry --
same 3-channel convention ``cnn_dbm`` uses per pixel, just per edge instead.

No external GNN library (no torch_geometric/dgl in this environment) --
message passing is implemented directly with batched dense tensors, which
is exact (not an approximation) here because every zone in one instance
has the same fixed node count (``d+1``); only which edges are FINITE
varies per zone, handled via an edge mask. ``zone.graph()`` is still the
literal data source per call (not bypassed) -- its node/edge attributes
are read directly into the dense tensors below.

Global pooling is SUM over nodes, matching deepset_h/deepset_v (not
cnn_dbm/cnn_proj's mean pool) -- so all three zone-set encoders
(deepset_h, deepset_v, gnn) share the same aggregation, and any gap
between them isolates the representation (vertices vs. constraints vs.
graph structure), not an incidental pooling choice. cnn_dbm/cnn_proj keep
their own mean pool: they read off a fixed-size tensor, not a variable
zone-derived set, so sum vs. mean doesn't carry the same "set cardinality"
meaning there.

``has_per_element_head = True``: nodes ``1..d`` are already clock-indexed
(same natural grouping as cnn_dbm's rows), so per-task embeddings are read
directly off each clock node's own final hidden state, dropping the
reference node -- same "drop the reference-clock row" convention cnn_dbm
uses on ``dbm_tensor``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import Encoder, EncoderOutput


def _zone_to_dense_graph(zone, cap: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(node_feat (n,4), edge_feat (n,n,3), edge_mask (n,n) bool)."""
    g = zone.graph()
    n = zone.d + 1
    node_feat = np.zeros((n, 4), dtype=np.float32)
    edge_feat = np.zeros((n, n, 3), dtype=np.float32)
    edge_mask = np.zeros((n, n), dtype=bool)

    dbm = zone.dbm
    for i in range(n):
        if i == 0:
            node_feat[i] = (0.0, 1.0, 0.0, 1.0)
            continue
        if dbm.is_infinity(0, i):
            b0i, f0i = 1.0, 0.0
        else:
            b0i, f0i = max(-cap, min(cap, dbm.bound(0, i))) / cap, 1.0
        if dbm.is_infinity(i, 0):
            bi0, fi0 = 1.0, 0.0
        else:
            bi0, fi0 = max(-cap, min(cap, dbm.bound(i, 0))) / cap, 1.0
        node_feat[i] = (b0i, f0i, bi0, fi0)

    for i, j, data in g.edges(data=True):
        clipped = max(-cap, min(cap, data["bound"]))
        edge_feat[i, j] = (clipped / cap, 1.0 if data["strict"] else 0.0, 1.0)
        edge_mask[i, j] = True

    return node_feat, edge_feat, edge_mask


class _GCNLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int = 3):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(hidden + edge_dim, hidden), nn.ReLU())
        self.upd = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())

    def forward(self, h: torch.Tensor, edge_feat: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """h: (B, n, hidden); edge_feat: (B, n, n, edge_dim); edge_mask: (B, n, n) bool.
        Message i<-j along every finite edge (i, j); mean-aggregated per i."""
        b, n, hidden = h.shape
        h_src = h.unsqueeze(1).expand(b, n, n, hidden)  # (B, n_dst, n_src, hidden) -- h_src[:, i, j] = h[j]
        msg_in = torch.cat([h_src, edge_feat], dim=-1)  # (B, n, n, hidden+edge_dim)
        msgs = self.msg(msg_in)  # (B, n, n, hidden)

        mask_f = edge_mask.float().unsqueeze(-1)  # (B, n, n, 1)
        summed = (msgs * mask_f).sum(dim=2)  # (B, n, hidden) -- sum over src j
        counts = mask_f.sum(dim=2).clamp(min=1.0)  # (B, n, 1)
        agg = summed / counts

        h_new = self.upd(torch.cat([h, agg], dim=-1))
        return h_new


class GNNEncoder(Encoder):
    has_per_element_head = True

    def __init__(self, d: int, embed_dim: int = 90, hidden: int = 64, per_task_dim: int = 24):
        super().__init__()
        self.d = d
        self.n = d + 1
        self.embed_dim = embed_dim
        self.per_task_dim = per_task_dim
        self.node_embed = nn.Linear(4, hidden)
        self.layer1 = _GCNLayer(hidden)
        self.layer2 = _GCNLayer(hidden)
        self.readout = nn.Linear(hidden, embed_dim)
        self.job_readout = nn.Linear(hidden, per_task_dim)
        self._hidden = hidden

    def forward(self, obs_batch: Dict[str, Any], shuffle_nodes: bool = False) -> EncoderOutput:
        """:param shuffle_nodes: permutation-ablation hook -- relabels real
        clocks (nodes 1..d) with a random permutation per zone, reference
        node (0) fixed. Same convention as cnn_dbm's shuffle_clocks."""
        zones = obs_batch["zone"]
        b = len(zones)
        n = self.n

        node_feats, edge_feats, edge_masks = [], [], []
        for zone in zones:
            cap = max(max(zone.k, default=0) + 1, 1)
            nf, ef, em = _zone_to_dense_graph(zone, cap)
            if shuffle_nodes:
                perm = np.random.permutation(self.d)
                order = np.concatenate([[0], perm + 1])
                nf = nf[order]
                ef = ef[order][:, order]
                em = em[order][:, order]
            node_feats.append(nf)
            edge_feats.append(ef)
            edge_masks.append(em)

        node_feat_t = torch.from_numpy(np.stack(node_feats)).float()  # (B, n, 4)
        edge_feat_t = torch.from_numpy(np.stack(edge_feats)).float()  # (B, n, n, 3)
        edge_mask_t = torch.from_numpy(np.stack(edge_masks))  # (B, n, n) bool

        h = torch.relu(self.node_embed(node_feat_t))  # (B, n, hidden)
        h = self.layer1(h, edge_feat_t, edge_mask_t)
        h = self.layer2(h, edge_feat_t, edge_mask_t)

        pooled = self.readout(h.sum(dim=1))  # global SUM pool, matches deepset_h/deepset_v

        per_task = self.job_readout(h[:, 1:, :])  # (B, d, per_task_dim) -- drop reference node 0

        return {"pooled": pooled, "per_task": per_task}
