"""Correctness tests for :mod:`zoneviews.collect`.

Not one of the 9 numbered spec items (those cover ``zone``/``raster``), but
``collect_zones``/``to_npz``/``from_npz`` are core deliverables of this
phase and need direct coverage too.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from zoneviews.collect import collect_zones, from_npz, to_npz
from zoneviews.zone import Zone


def test_collect_zones_returns_decision_points_with_delta_free_zones():
    zones = collect_zones("PJS_01", n_episodes=5, seed=42)
    assert len(zones) > 0
    for loc, zone in zones:
        assert isinstance(loc, tuple)
        assert isinstance(zone, Zone)
        # "delta" must never appear as a clock in this module's zones.
        assert "delta" not in zone.clock_names
        assert zone.d == len(loc) // 1 or zone.d >= 1  # sanity: at least one job clock
        assert not zone.federation.is_empty()


def test_collect_zones_deterministic_given_same_seed():
    zones_a = collect_zones("PJS_01", n_episodes=10, seed=7)
    zones_b = collect_zones("PJS_01", n_episodes=10, seed=7)
    assert len(zones_a) == len(zones_b)
    for (loc_a, za), (loc_b, zb) in zip(zones_a, zones_b):
        assert loc_a == loc_b
        assert za.dbm.to_matrix() == zb.dbm.to_matrix()


def test_collect_zones_rejects_unknown_policy():
    with pytest.raises(ValueError):
        collect_zones("PJS_01", n_episodes=1, seed=0, policy="greedy")


def test_collect_zones_unknown_instance_raises():
    with pytest.raises(ValueError):
        collect_zones("NOT_AN_INSTANCE", n_episodes=1, seed=0)


def test_to_npz_from_npz_roundtrip_preserves_raster_shapes_and_hrep():
    zones = collect_zones("PJS_01", n_episodes=3, seed=1)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "views.npz")
        to_npz(zones, path, projection_res=32, graph_res=64)
        loaded = from_npz(path)

    assert len(loaded) == len(zones)
    for (loc, zone), view in zip(zones, loaded):
        assert view.location == loc
        assert view.dbm_tensor.shape == (zone.d + 1, zone.d + 1, 3)
        assert view.projection_stack.shape == (zone.d * (zone.d - 1) // 2, 32, 32)
        assert view.graph_image.shape == (1, 64, 64)
        assert np.isfinite(view.dbm_tensor).all()

        expected_h_rep = zone.h_rep()
        assert len(view.h_rep) == len(expected_h_rep)
        assert {(hs.a, hs.b, hs.strict) for hs in view.h_rep} == {
            (hs.a, hs.b, hs.strict) for hs in expected_h_rep
        }

        expected_edges = zone.graph().number_of_edges()
        assert view.graph.number_of_edges() == expected_edges
        assert view.graph.number_of_nodes() == zone.d + 1
