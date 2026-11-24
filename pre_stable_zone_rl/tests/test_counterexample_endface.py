import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from counterexample_endface import (block_value, brg_value, ce2_value_by_wait, endface, endface2,  # noqa: E402
                                    max_value_by_wait)
from prestable.block_vi import check_block_mdp  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.model import diff_gt, diff_le, eq, gt, le  # noqa: E402

TOL = 1e-6


def test_value_by_wait():
    assert [max_value_by_wait(t, 0.9) for t in (0, 1, 2)] == pytest.approx([1.8, 2.3, 1.9])
    assert [ce2_value_by_wait(t, 0.9) for t in (0, 1, 2)] == pytest.approx([2.43, 1.72, 1.82])
    assert [ce2_value_by_wait(t, 0.5) for t in (0, 1, 2)] == pytest.approx([0.75, 0.6, 0.7])


# --- invariant_atoms=False: the guard-only partition misses the end face ------------

# (λ, max-form rates (s, m), BRG = integer value, boundary value, boundary delay at s)
CASES = [(0.9, (0.5, 1), 2.3, 1.9, 2),
         (0.5, (0.5, 1), 1.5, 1.5, 2),  # t = 1 and t = 2 tie, so the boundary variant is exact here
         (0.9, (0.5, 2), 4.1, 3.6, 0)]


@pytest.mark.parametrize("lam, rates, v_opt, v_boundary, t_boundary", CASES)
def test_values(lam, rates, v_opt, v_boundary, t_boundary):
    ta = endface(*rates)
    v, step = brg_value(ta, lam)
    assert v == pytest.approx(v_opt, abs=TOL)
    assert step == ("s", (0, 0), "a", 1, "thin")
    graph = build_prestable_graph(ta, invariant_atoms=False).graph
    v, step, r = block_value(ta, lam, "integer", graph)
    check_block_mdp(ta, graph, r.mdp)
    assert v == pytest.approx(v_opt, abs=TOL)
    assert (step[3], step[4]) == ("a", 1)
    v, step, r = block_value(ta, lam, "boundary", graph)
    check_block_mdp(ta, graph, r.mdp)
    assert v == pytest.approx(v_boundary, abs=TOL)
    assert (step[3], step[4]) == ("a", t_boundary)
    # The boundary variant offers only t ∈ {0, 2} at s.
    assert sorted({a.delay for a in r.mdp.actions[r.mdp.initial]}) == [0, 2]


def test_blocks():
    """Differs from the hand prediction (A0 ∋ 0 and U ⊇ the diagonal at m): a at t = 0 puts
    {0} at m, and its delay closure, the diagonal, is a Phase 1 block there too. Phase 1's
    partition separates both from A0 and U; Phase 2 then splits nothing."""
    ta = endface()
    res = build_prestable_graph(ta, invariant_atoms=False)
    ctx = res.graph.nodes[0].zone.ctx  # zones compare equal only within one context
    zero = ctx.zero()
    diag = ctx.from_guard([gt("x", 0), le("x", 2), diff_le("x", "y", 0), diff_le("y", "x", 0)])
    assert res.phase2.splits == 0
    assert res.phase2.store["s"] == {zero, diag}
    assert res.phase2.store["m"] == {
        zero,
        diag,
        ctx.from_guard([gt("x", 0), le("x", 2), *eq("y", 0)]),
        ctx.from_guard([gt("x", 0), le("x", 3), gt("y", 0), le("y", 2), diff_gt("x", "y", 0), diff_le("x", "y", 2)]),
    }


def test_boundary_states_at_m():
    ta = endface()
    _, _, r = block_value(ta, 0.9, "boundary", invariant_atoms=False)
    at_m = sorted((nu, -r.values[i]) for i, (loc, nu, _) in enumerate(r.mdp.states) if loc == "m")
    assert [nu for nu, _ in at_m] == [(0, 0), (0, 0), (2, 0)]
    assert [v for _, v in at_m] == pytest.approx([2.0, 2.0, 1.0])


# --- step 12.3: both settings on CE1 and CE2 -----------------------------------------

# (model, λ, sign, BRG = integer value, boundary value with invariant_atoms=False)
STEP12 = [pytest.param(lambda: endface(0.5, 1), 0.9, -1, 2.3, 1.9, id="CE1-0.9-(0.5,1)"),
          pytest.param(lambda: endface(0.5, 2), 0.9, -1, 4.1, 3.6, id="CE1-0.9-(0.5,2)"),
          pytest.param(endface2, 0.9, 1, 1.72, 1.82, id="CE2-0.9"),
          pytest.param(endface2, 0.5, 1, 0.6, 0.7, id="CE2-0.5")]


@pytest.mark.parametrize("make, lam, sign, v_opt, v_old_boundary", STEP12)
def test_step12_values(make, lam, sign, v_opt, v_old_boundary):
    ta = make()
    v, step = brg_value(ta, lam, sign)
    assert v == pytest.approx(v_opt, abs=TOL)
    assert step == ("s", (0, 0), "a", 1, "thin")
    for atoms, v_boundary in ((False, v_old_boundary), (True, v_opt)):
        graph = build_prestable_graph(ta, invariant_atoms=atoms).graph
        for variant, expected in (("integer", v_opt), ("boundary", v_boundary)):
            v, step, r = block_value(ta, lam, variant, graph, sign)
            check_block_mdp(ta, graph, r.mdp)
            assert v == pytest.approx(expected, abs=TOL), (atoms, variant)
        if atoms:  # with the face x = 3 of m a block boundary, the boundary policy waits 1 in s
            assert (step[3], step[4]) == ("a", 1)


@pytest.mark.parametrize("make, splits", [(endface, 4), (endface2, 10)], ids=["CE1", "CE2"])
def test_step12_blocks_at_s(make, splits):
    """With invariant atoms s is cut at x = 1 (the right diagonal of x = 3 at m, pulled back by
    Phase 2), besides the face x = 2 of its own invariant."""
    ta = make()
    res = build_prestable_graph(ta, invariant_atoms=True)
    ctx = res.graph.nodes[0].zone.ctx
    assert ctx.from_guard([*eq("x", 1), *eq("y", 1)]) in res.phase2.store["s"]
    assert {str(Z) for Z in res.phase2.store["s"]} == {
        "x == 0 and y == 0", "0 < x < 1 and 0 < y < 1 and x - y == 0", "x == 1 and y == 1",
        "1 < x < 2 and 1 < y < 2 and x - y == 0", "x == 2 and y == 2"}
    assert res.phase2.splits == splits
