"""Step 12.2: invariant atoms in G_l (partition_atoms), and byte-identity with the flag off."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_dumps import catalogue, dumps  # noqa: E402
from prestable.algorithm import check_phase1, check_phase2  # noqa: E402
from prestable.graph import build_prestable_graph  # noqa: E402
from prestable.zones import ZoneContext  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "graph_dumps_step12.json"


def test_flag_off_is_byte_identical_to_before_step12():
    """Phase 1/2 results with traces, nodes and edges hash as before step 12 (see graph_dumps.py)."""
    expected = json.loads(DATA.read_text())
    assert dumps(lambda ta: build_prestable_graph(ta, trace=True, invariant_atoms=False)) == expected


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(catalogue()))
def test_checks_follow_the_flag(name):
    ta = catalogue()[name]
    ctx = ZoneContext(ta.clocks)
    for atoms in (False, True):
        r = build_prestable_graph(ta, invariant_atoms=atoms)
        store = {l: frozenset(n.zone for n in r.graph.nodes if n.location == l) for l in ta.locations}
        # Zones of the graph live in its own context; rebuild them in ctx for the checks.
        store = {l: frozenset(ctx.from_guard(Z.constraints()) for Z in P) for l, P in store.items()}
        check_phase2(ta, ctx, store, r.phase1.K, invariant_atoms=atoms)
        check_phase1(ta, ctx, store, r.phase1.K, invariant_atoms=atoms)


@pytest.mark.slow
def test_flag_changes_only_models_with_non_strict_invariants():
    for name, ta in catalogue().items():
        faces = any(not c.strict for l in ta.locations for c in ta.invariant(l))
        off = build_prestable_graph(ta, invariant_atoms=False).phase2.store
        on = build_prestable_graph(ta, invariant_atoms=True).phase2.store
        same = {l: sorted(map(str, P)) for l, P in off.items()} == {l: sorted(map(str, P)) for l, P in on.items()}
        if not faces:
            assert same, name


def test_guard_only_partition_fails_the_invariant_atom_check():
    """CE1 without invariant atoms: the diagonal block at s straddles the faces x = 2 and y = 2
    (which one is reported depends on set order)."""
    from counterexample_endface import endface

    ta = endface()
    ctx = ZoneContext(ta.clocks)
    r = build_prestable_graph(ta, invariant_atoms=False)
    store = {l: frozenset(ctx.from_guard(Z.constraints()) for Z in P) for l, P in r.phase2.store.items()}
    check_phase2(ta, ctx, store, r.phase1.K, invariant_atoms=False)
    with pytest.raises(ValueError, match=r"location s: block 0 < x <= 2 and 0 < y <= 2 and x - y == 0 "
                                         r"is neither inside nor disjoint from guard atom [xy] < 2"):
        check_phase1(ta, ctx, store, r.phase1.K, invariant_atoms=True)
