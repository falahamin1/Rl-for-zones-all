"""PTA P3 (Jovanović et al. 2017, Fig. 7); the automaton lives in examples/p3_example.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from p3_example import pta_p3, zeta, zeta_blocks  # noqa: E402
from prestable.algorithm import check_phase1, check_phase2, phase1, phase2  # noqa: E402
from prestable.plot2d import closure_polygon  # noqa: E402
from prestable.zones import ZoneContext  # noqa: E402

pytestmark = pytest.mark.slow  # Phase 1 + Phase 2 take about 4 s


@pytest.fixture(scope="module")
def p3():
    ta = pta_p3()
    ctx = ZoneContext(ta.clocks)
    p1 = phase1(ta, ctx)
    p2 = phase2(ta, ctx, p1.store)
    return ta, ctx, p1, p2


def test_phase1_block_counts(p3):
    ta, ctx, p1, _ = p3
    check_phase1(ta, ctx, p1.store, p1.K)
    assert len(p1.store["l0"]) == 3
    assert len(p1.store["l1"]) == 38  # hand count


def test_l1_blocks_respect_zeta(p3):
    _, ctx, p1, p2 = p3
    for store in (p1.store, p2.store):
        zeta_blocks(store["l1"], ctx)  # raises if a block straddles ζ
    # ζ itself is covered exactly by the final ζ blocks.
    from prestable.zones import subtract_all
    zb = zeta_blocks(p2.store["l1"], ctx)
    assert all(not subtract_all(z, zb) for z in zeta(ctx))


def test_final_store_is_prestable(p3):
    ta, ctx, p1, p2 = p3
    check_phase2(ta, ctx, p2.store, p1.K)


def test_closure_polygons_have_right_dimension(p3):
    _, ctx, _, p2 = p3
    for z in p2.store["l1"]:
        vertices = closure_polygon(z, box=10)
        # Point, segment or polygon; a bounded-or-clipped block is never empty in [0, 10]².
        assert 1 <= len(vertices)
        for x, y in vertices:
            assert 0 <= x <= 10 and 0 <= y <= 10


# With invariants, the invariant faces x = 2, y = 2 are blocks of their own (step 12; before: 17, 72, 234).
@pytest.mark.parametrize("invariants, l1_blocks, nodes, edges", [(False, 38, 122, 452), (True, 26, 88, 319)])
def test_reference_numbers(invariants, l1_blocks, nodes, edges):
    from prestable.graph import build_prestable_graph

    r = build_prestable_graph(pta_p3(invariants=invariants))
    assert len(r.phase1.store["l1"]) == l1_blocks
    assert len(r.graph.nodes) == nodes and len(r.graph.edges) == edges
