"""Three rasterizations of a :class:`zoneviews.zone.Zone`, all pure numpy/PIL.

No torch, no matplotlib, no scipy. Every function returns a ``float32``
numpy array with a documented, fixed shape and deterministic content given
the same ``Zone`` and parameters.
"""
from __future__ import annotations

import math
import random as _random
import time as _time
from collections import OrderedDict
from itertools import combinations
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .zone import Zone

__all__ = [
    "dbm_tensor", "projection_stack", "graph_image",
    "projection_cache_stats", "clear_projection_cache",
    "raster_wall_time", "reset_raster_wall_time",
]

#: Cumulative wall-clock spent actually rasterizing (dbm_tensor +
#: projection_stack cache misses), separate from any encoder's total
#: forward time -- see /tmp/ENCODERS_SETUP.md item 2. A trainer reads the
#: delta across one update to log "raster_s" distinctly from
#: "encoder_forward_s".
_raster_wall_s = 0.0


def raster_wall_time() -> float:
    return _raster_wall_s


def reset_raster_wall_time() -> None:
    global _raster_wall_s
    _raster_wall_s = 0.0

#: Supersampling factor used by :func:`projection_stack` for anti-aliasing.
_SUPERSAMPLE = 2

#: LRU cache on projection_stack, keyed by a CONTENT-based zone fingerprint
#: (see _zone_content_key below), res, and per_zone_normalize -- see
#: /tmp/ENCODERS_SETUP.md item 2. Rasterization (not the conv itself) was
#: found to dominate CNN-Proj's per-step cost, and PPO revisits the exact
#: same rollout (hence the exact same zones) across every epoch of every
#: update, so this is a large-hit-rate cache in practice, not merely
#: theoretical.
#:
#: NOT keyed on ``str(zone)``: ``Zone`` defines no ``__str__``/``__repr__``,
#: so ``str(zone)`` silently falls back to the default
#: ``<zoneviews.zone.Zone object at 0x...>`` -- an ADDRESS, not content.
#: This was tried first and is a real, confirmed bug, not a hypothetical
#: one: CPython reuses a garbage-collected object's address for a later
#: object, so a stale cache entry from an earlier (different-shape) zone
#: can be returned for an unrelated later zone that happens to land at the
#: same address -- reproduced directly via a training run that crashed
#: with a channel-count mismatch inside CNNProjEncoder.forward's
#: ``np.stack`` after ~40-80 PPO updates (bytes recycled, not on the first
#: call). ``_zone_content_key`` fixes this by hashing the zone's actual DBM
#: entries instead of its address.
_PROJECTION_CACHE_MAXSIZE = 8192
_projection_cache: "OrderedDict[Tuple[tuple, int, bool], np.ndarray]" = OrderedDict()
_cache_hits = 0
_cache_misses = 0


def _zone_content_key(zone: Zone) -> tuple:
    """Canonical, content-based fingerprint of a zone's DBM -- see the
    cache docstring above for why this exists instead of ``str(zone)``.
    O(d^2) simple comparisons, negligible next to the O(d^2 * res^2)
    rasterization it guards (the same order of work ``_dbm_tensor_impl``
    already does per call, just without the pixel grid).
    """
    dbm = zone.dbm
    n = zone.d + 1
    entries = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dbm.is_infinity(i, j):
                entries.append(None)
            else:
                entries.append((dbm.bound(i, j), dbm.is_strict(i, j)))
    return (zone.d, tuple(zone.clock_names), tuple(zone.k), tuple(entries))


def clear_projection_cache() -> None:
    global _cache_hits, _cache_misses
    _projection_cache.clear()
    _cache_hits = 0
    _cache_misses = 0


def projection_cache_stats() -> dict:
    total = _cache_hits + _cache_misses
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": _cache_hits / total if total else float("nan"),
        "size": len(_projection_cache),
        "maxsize": _PROJECTION_CACHE_MAXSIZE,
    }


def dbm_tensor(zone: Zone) -> np.ndarray:
    _t0 = _time.perf_counter()
    try:
        return _dbm_tensor_impl(zone)
    finally:
        global _raster_wall_s
        _raster_wall_s += _time.perf_counter() - _t0


def _dbm_tensor_impl(zone: Zone) -> np.ndarray:
    """Rasterize a zone's raw DBM matrix as a ``(d+1, d+1, 3)`` float32 tensor.

    Channel ``0`` is the bound, clamped to ``[-cap, cap]`` and normalized by
    ``cap`` where ``cap = max(zone.k) + 1`` (falling back to ``1`` if
    ``zone.k`` is empty or all-zero). Channel ``1`` is the strictness bit
    (``1.0`` if strict, ``0.0`` otherwise; ``0.0`` for infinite/diagonal
    cells, where strictness is not meaningful). Channel ``2`` is the
    finiteness mask (``1.0`` for finite cells, including the trivial
    diagonal ``0 <= 0``; ``0.0`` for entries that are ``< inf`` in the DBM).

    Infinite entries are never encoded as a large float — per-instruction,
    they are clamped to exactly ``cap`` (so their normalized value is
    ``1.0``) and flagged via the finiteness mask, which callers must check
    before trusting the bound channel.
    """
    d = zone.d
    n = d + 1
    cap = max(max(zone.k, default=0) + 1, 1)

    out = np.zeros((n, n, 3), dtype=np.float32)
    dbm = zone.dbm
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i, j] = (0.0, 0.0, 1.0)
                continue
            if dbm.is_infinity(i, j):
                out[i, j] = (1.0, 0.0, 0.0)
                continue
            bound = dbm.bound(i, j)
            clipped = max(-cap, min(cap, bound))
            out[i, j] = (clipped / cap, 1.0 if dbm.is_strict(i, j) else 0.0, 1.0)
    return out


def projection_stack(zone: Zone, res: int = 64, use_cache: bool = True, per_zone_normalize: bool = False) -> np.ndarray:
    """Rasterize the zone's pairwise 2D shadows as a ``(d(d-1)/2, res, res)`` tensor.

    For each unordered pair ``(i, j)`` of real clocks (``1``-indexed DBM
    positions, ``0 < i < j``), channel index ``combinations(range(d), 2)``
    order (deterministic, ``i`` then ``j`` ascending) holds the zone's
    projection onto the ``(x_i, x_j)`` plane.

    For a canonical DBM the pairwise shadow is exactly cut out by the six
    matrix entries ``(i,0)``, ``(0,i)``, ``(j,0)``, ``(0,j)``, ``(i,j)``,
    ``(j,i)`` — no projection algorithm is run; those six bounds are read
    directly and rasterized as intersecting half-planes. Anti-aliasing is
    done by rendering at ``2x`` resolution and averaging 2x2 blocks down to
    ``res``, so pixel values are continuous coverage fractions in ``[0, 1]``,
    not a hard binary mask. Rasterization treats every bound as non-strict
    (the topological closure), consistent with :meth:`Zone.v_rep`.

    :param use_cache: LRU-cache the result keyed by ``(str(zone), res,
        per_zone_normalize)`` (see /tmp/ENCODERS_SETUP.md item 2) -- PPO
        revisits the exact same concrete decision points (hence identical
        zones) repeatedly across rollout epochs, so this is a real,
        high-hit-rate cache in training, not a defensive no-op. Set False
        to force recomputation (e.g. for benchmarking raw rasterization
        cost).
    :param per_zone_normalize: If False (default), the box for pair
        ``(i,j)`` is the GLOBAL ``[0, k_i] x [0, k_j]`` (the same box for
        every zone of this instance). If True, the box is instead THIS
        zone's own finite bounding box for that pair -- ``[lo_i, hi_i] x
        [lo_j, hi_j]`` read off the same six DBM entries, falling back to
        the global ``k`` bound on whichever side is unbounded (infinite)
        for that specific zone, and to a 1-unit-wide box if the zone is a
        single point in that pair (``lo == hi``) so the box is never
        degenerate. Confound fix (/tmp/ENCODERS_SETUP.md item 3): the
        global-box scheme measured under ~0.1% mean lit-pixel occupancy on
        both gridtimed and modesel at res=64 -- almost every pixel is
        always off, most of a CNN's receptive field is wasted on empty
        space, and small zones near the origin vs. far from it render as
        visually near-identical tiny specks. Per-zone normalization always
        fills the frame, trading away the (here, apparently not very
        useful) absolute-position-in-clock-space information for shape
        information at every zone's own natural scale.
    """
    global _cache_hits, _cache_misses, _raster_wall_s
    if use_cache:
        key = (_zone_content_key(zone), res, per_zone_normalize)
        cached = _projection_cache.get(key)
        if cached is not None:
            _projection_cache.move_to_end(key)
            _cache_hits += 1
            return cached.copy()
        _cache_misses += 1
        _t0 = _time.perf_counter()
        result = _projection_stack_uncached(zone, res, per_zone_normalize)
        _raster_wall_s += _time.perf_counter() - _t0
        _projection_cache[key] = result
        if len(_projection_cache) > _PROJECTION_CACHE_MAXSIZE:
            _projection_cache.popitem(last=False)
        return result.copy()
    _t0 = _time.perf_counter()
    try:
        return _projection_stack_uncached(zone, res, per_zone_normalize)
    finally:
        _raster_wall_s += _time.perf_counter() - _t0


def _projection_stack_uncached(zone: Zone, res: int, per_zone_normalize: bool = False) -> np.ndarray:
    d = zone.d
    n_channels = d * (d - 1) // 2
    out = np.zeros((n_channels, res, res), dtype=np.float32)
    if n_channels == 0:
        return out

    dbm = zone.dbm
    big = res * _SUPERSAMPLE

    for c, (p, q) in enumerate(combinations(range(d), 2)):
        i, j = p + 1, q + 1  # DBM indices (1-based real clocks)
        ki = max(zone.k[p], 0)
        kj = max(zone.k[q], 0)
        lo_i, hi_i = 0.0, float(ki)
        lo_j, hi_j = 0.0, float(kj)
        if per_zone_normalize:
            if not dbm.is_infinity(i, 0):
                hi_i = float(dbm.bound(i, 0))
            if not dbm.is_infinity(0, i):
                lo_i = float(-dbm.bound(0, i))
            if not dbm.is_infinity(j, 0):
                hi_j = float(dbm.bound(j, 0))
            if not dbm.is_infinity(0, j):
                lo_j = float(-dbm.bound(0, j))
            if hi_i <= lo_i:
                hi_i = lo_i + 1.0
            if hi_j <= lo_j:
                hi_j = lo_j + 1.0
        # Pixel-center coordinates in clock space, axis0 -> x_i, axis1 -> x_j.
        xs = lo_i + (np.arange(big, dtype=np.float64) + 0.5) / big * (hi_i - lo_i)
        ys = lo_j + (np.arange(big, dtype=np.float64) + 0.5) / big * (hi_j - lo_j)
        Xi, Yj = np.meshgrid(xs, ys, indexing="ij")

        mask = np.ones((big, big), dtype=bool)
        if not dbm.is_infinity(i, 0):
            mask &= Xi <= dbm.bound(i, 0)
        if not dbm.is_infinity(0, i):
            mask &= Xi >= -dbm.bound(0, i)
        if not dbm.is_infinity(j, 0):
            mask &= Yj <= dbm.bound(j, 0)
        if not dbm.is_infinity(0, j):
            mask &= Yj >= -dbm.bound(0, j)
        if not dbm.is_infinity(i, j):
            mask &= (Xi - Yj) <= dbm.bound(i, j)
        if not dbm.is_infinity(j, i):
            mask &= (Yj - Xi) <= dbm.bound(j, i)

        coverage = mask.astype(np.float32).reshape(res, _SUPERSAMPLE, res, _SUPERSAMPLE).mean(axis=(1, 3))
        out[c] = coverage

    return out


def graph_image(zone: Zone, res: int = 128, layout_seed: Optional[int] = None) -> np.ndarray:
    """Rasterize the constraint digraph as a ``(1, res, res)`` float32 image.

    Nodes are placed evenly on a circle, ordered by clock index (node ``0``
    is the reference clock). Every finite DBM entry becomes an arrow from
    its row node to its column node, with the integer bound value (and an
    ``s`` suffix if strict) drawn near the arrow midpoint. Rendering uses
    only numpy/PIL primitives (no matplotlib) with PIL's built-in bitmap
    font, so output is deterministic given the same inputs, independent of
    any system font availability.

    :param layout_seed: If ``None`` (the default), nodes keep their natural
        clock-index order around the circle. If an ``int``, a deterministic
        permutation of node positions seeded by this value is used instead
        (for a later encoding ablation on node-position sensitivity) — node
        identity (``clock_index``) is unaffected, only where it is drawn.
    """
    d = zone.d
    n = d + 1

    if layout_seed is None:
        perm = list(range(n))
    else:
        perm = _random.Random(layout_seed).sample(range(n), n)

    img = Image.new("L", (res, res), color=0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    cx, cy = res / 2.0, res / 2.0
    radius = res * 0.38
    positions = []
    for node in range(n):
        slot = perm[node]
        angle = 2.0 * math.pi * slot / n
        positions.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    dbm = zone.dbm
    for i in range(n):
        for j in range(n):
            if i == j or dbm.is_infinity(i, j):
                continue
            x0, y0 = positions[i]
            x1, y1 = positions[j]
            draw.line([(x0, y0), (x1, y1)], fill=255, width=1)

            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy) or 1.0
            ux, uy = dx / length, dy / length
            perp_x, perp_y = -uy, ux

            tip_x, tip_y = x1 - ux * 2.0, y1 - uy * 2.0
            base_x, base_y = x1 - ux * 8.0, y1 - uy * 8.0
            draw.polygon(
                [
                    (tip_x, tip_y),
                    (base_x + perp_x * 3.0, base_y + perp_y * 3.0),
                    (base_x - perp_x * 3.0, base_y - perp_y * 3.0),
                ],
                fill=255,
            )

            bound = dbm.bound(i, j)
            strict = dbm.is_strict(i, j)
            label = "{0}{1}".format(bound, "s" if strict else "")
            mx, my = x0 + dx * 0.5, y0 + dy * 0.5
            side = 1.0 if i < j else -1.0
            draw.text((mx + perp_x * 4.0 * side, my + perp_y * 4.0 * side), label, fill=255, font=font)

    for node in range(n):
        x, y = positions[node]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=255, fill=255)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, :, :]
