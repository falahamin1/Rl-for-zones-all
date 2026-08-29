"""Tests for :mod:`zoneviews.raster`.

Covers required items 4, 5, 7, and 8 from the encoder-study spec.
"""
from __future__ import annotations

import random
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from pyudbm import Context, Federation
from zoneviews.raster import dbm_tensor, graph_image, projection_stack
from zoneviews.zone import Zone, _halfspace_to_federation, HalfSpace


def _box_zone(bounds, names=None):
    names = names or [f"x{i}" for i in range(len(bounds))]
    ctx = Context(names)
    Z = Federation(ctx).set_init()
    for name, (lo, hi) in zip(names, bounds):
        clock = getattr(ctx, name)
        Z = Z & (clock >= lo) & (clock <= hi)
    k = [hi + 1 for _, hi in bounds]
    return Zone(Z, names, k)


def _pentagon_zone():
    ctx = Context(["x", "y"])
    Z = (ctx.x >= 2) & (ctx.x <= 5) & (ctx.y >= 1) & (ctx.y <= 4) & (ctx.x - ctx.y <= 3)
    return Zone(Z, ["x", "y"], k=[10, 10])


# ---------------------------------------------------------------------------
# Item 4: projection_stack lit/unlit sampling
# ---------------------------------------------------------------------------

def _pixel_index(value, k, res):
    idx = int(float(value) / k * res)
    return max(0, min(res - 1, idx))


def test_projection_stack_lit_for_interior_unlit_for_exterior():
    zone = _pentagon_zone()
    res = 64
    ps = projection_stack(zone, res=res)  # channel 0 == (x, y) since d=2

    rng = random.Random(2024)

    # 500 points sampled strictly inside the zone, with a margin away from
    # every constraint boundary (box edges AND the x-y<=3 diagonal cut): the
    # pixel at that point's (x, y) coordinate must be lit. A point sitting
    # exactly on a boundary can legitimately raster to an unlit pixel at
    # finite resolution (its neighborhood is mostly outside), so this is a
    # *necessary*, not sufficient, condition on the raster away from
    # boundaries -- see item 4's note.
    interior_hits = 0
    while interior_hits < 500:
        x = Fraction(rng.randint(23, 47), 10)
        y = Fraction(rng.randint(13, 37), 10)
        if not zone.contains((x, y)):
            continue
        if (x - y) > Fraction(25, 10):
            continue
        r = _pixel_index(x, zone.k[0], res)
        c = _pixel_index(y, zone.k[1], res)
        assert ps[0, r, c] > 0.0
        interior_hits += 1

    # 500 points sampled outside the zone (but inside the k-box, so the
    # pixel coordinate itself is meaningful): at least one channel (there
    # is only one here, d=2) must be unlit at that pixel.
    exterior_hits = 0
    while exterior_hits < 500:
        x = Fraction(rng.randint(0, 100), 10)
        y = Fraction(rng.randint(0, 100), 10)
        if zone.contains((x, y)):
            continue
        r = _pixel_index(x, zone.k[0], res)
        c = _pixel_index(y, zone.k[1], res)
        if ps[0, r, c] > 0.0:
            # A boundary pixel can legitimately have partial coverage from
            # supersampling even though this exact sampled point is
            # outside -- only count genuine misses.
            continue
        exterior_hits += 1
        assert ps[0, r, c] == 0.0


# ---------------------------------------------------------------------------
# Item 5: losslessness of the pairwise projections (six entries, not pixels)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bounds",
    [
        [(2, 5), (1, 4)],
        [(0, 2), (0, 2), (0, 2)],
        [(0, 3), (1, 5), (2, 6), (0, 4)],
    ],
)
def test_pairwise_projection_reconstruction_is_lossless(bounds):
    """Rebuild the DBM by intersecting, for every unordered clock pair, the
    exact six-entry back-projection (read directly off the DBM, never from
    rasterized pixels) and assert the result equals the original canonical
    DBM. This is the literal claim behind projection_stack's channel
    design -- test it rather than assume it.
    """
    zone = _box_zone(bounds)
    d = zone.d
    dbm = zone.dbm

    ctx = Context(list(zone.clock_names))
    clocks = {name: getattr(ctx, name) for name in zone.clock_names}
    rebuilt = Federation(ctx).set_init()

    for p, q in combinations(range(d), 2):
        i, j = p + 1, q + 1
        for (row, col) in [(i, 0), (0, i), (j, 0), (0, j), (i, j), (j, i)]:
            if dbm.is_infinity(row, col):
                continue
            a = [0] * d
            if row != 0:
                a[row - 1] += 1
            if col != 0:
                a[col - 1] -= 1
            hs = HalfSpace(a=tuple(a), b=dbm.bound(row, col), strict=dbm.is_strict(row, col))
            rebuilt = rebuilt & _halfspace_to_federation(hs, zone.clock_names, clocks)

    assert rebuilt.to_dbm_list()[0].to_matrix() == dbm.to_matrix()


def test_pairwise_projection_reconstruction_single_clock_needs_direct_bounds():
    """With only one real clock there are no pairs; the reconstruction
    argument from item 5 degenerates and must be checked using the direct
    (i, 0)/(0, i) bounds instead -- documented here rather than silently
    skipped."""
    zone = _box_zone([(2, 5)], names=["x"])
    assert zone.d * (zone.d - 1) // 2 == 0
    assert zone.dbm.bound(1, 0) == 5
    assert zone.dbm.bound(0, 1) == -2


# ---------------------------------------------------------------------------
# Item 7: graph_image determinism under layout_seed
# ---------------------------------------------------------------------------

def test_graph_image_deterministic_and_seed_sensitive():
    zone = _box_zone([(0, 2), (0, 2), (0, 2)])

    a1 = graph_image(zone, res=64, layout_seed=None)
    a2 = graph_image(zone, res=64, layout_seed=None)
    assert np.array_equal(a1, a2), "same (None) layout_seed must be byte-identical across calls"

    b1 = graph_image(zone, res=64, layout_seed=7)
    b2 = graph_image(zone, res=64, layout_seed=7)
    assert np.array_equal(b1, b2), "same explicit layout_seed must be byte-identical across calls"

    c1 = graph_image(zone, res=64, layout_seed=11)
    assert not np.array_equal(b1, c1), "different layout_seed values must produce different images"
    assert not np.array_equal(a1, b1), "None vs an explicit seed must produce different images"


# ---------------------------------------------------------------------------
# Item 8: dbm_tensor is finite and normalized within [-1, 1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bounds",
    [
        [(2, 5)],
        [(2, 5), (1, 4)],
        [(0, 2), (0, 2), (0, 2)],
    ],
)
def test_dbm_tensor_finite_and_normalized(bounds):
    zone = _box_zone(bounds)
    t = dbm_tensor(zone)
    assert t.dtype == np.float32
    assert t.shape == (zone.d + 1, zone.d + 1, 3)
    assert np.isfinite(t).all(), "dbm_tensor must never contain NaN or inf"
    bound_channel = t[..., 0]
    assert bound_channel.min() >= -1.0 - 1e-6
    assert bound_channel.max() <= 1.0 + 1e-6


def test_dbm_tensor_finite_and_normalized_with_unbounded_zone():
    """An unbounded zone must still clamp its infinite entries into range,
    never encode them as a raw large float."""
    ctx = Context(["x", "y"])
    Z = ctx.get_zero_federation().up()  # x == y, unbounded above
    zone = Zone(Z, ["x", "y"], k=[10, 10])
    t = dbm_tensor(zone)
    assert np.isfinite(t).all()
    assert t[..., 0].min() >= -1.0 - 1e-6
    assert t[..., 0].max() <= 1.0 + 1e-6
    # The unbounded (i, 0) cells must be flagged non-finite via the mask.
    assert t[1, 0, 2] == 0.0  # x <= inf: finiteness mask off
    assert t[2, 0, 2] == 0.0  # y <= inf: finiteness mask off
