"""Full dumps of the pre-stable construction for the step 12.2 byte-identity test.

`dumps(build)` maps each catalogue model to the sha256 of `canonical`: both phases' stores,
counters and traces, and the graph's nodes and edges, where build(ta) returns the
PrestableResult (built with trace=True). tests/data/graph_dumps_step12.json holds these
hashes from the code before step 12 (no invariant atoms); regenerate it only for a
deliberate change of the guard-only construction:

    python tests/graph_dumps.py > tests/data/graph_dumps_step12.json
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from brg_examples import EXAMPLES  # noqa: E402
from compare_values import p3_priced  # noqa: E402
from counterexample_endface import endface, endface2  # noqa: E402
from p3_example import pta_p3  # noqa: E402
from pta_examples import MODELS  # noqa: E402


def catalogue():
    out = {name: make() for name, (make, _) in EXAMPLES.items()}
    out.update({"P3": pta_p3(), "P3inv": pta_p3(invariants=True), "P3priced": p3_priced()})
    out.update({name: make() for name, make in MODELS.items()})
    out.update({"CE1": endface(), "CE2": endface2()})
    return out


def _store(store) -> list:
    return [(l, sorted(str(Z) for Z in P)) for l, P in sorted(store.items())]


def canonical(r) -> str:
    """Text of a PrestableResult that does not depend on set iteration order (hash seeds)."""
    p1, p2, g = r.phase1, r.phase2, r.graph
    parts = [_store(p1.store), (p1.K, p1.locations_processed, p1.inserts, p1.changed_inserts)]
    parts += [(e.location, str(e.zone), str(e.extrapolated), e.changed, e.blocks_after) for e in p1.trace or ()]
    parts += [_store(p2.store), p2.splits]
    parts += [(t.source, t.target, t.label, t.edge_index, t.branch_index, str(e.block), str(e.target),
               str(e.z_minus), sorted(map(str, e.pieces))) for e in p2.trace or () for t in (e.transition,)]
    parts += [(n.id, n.location, str(n.zone)) for n in g.nodes]
    parts += [(e.source, e.label, e.target, e.cost, e.edge, e.branch) for e in g.edges]
    return "\n".join(map(repr, parts))


def dumps(build) -> dict[str, str]:
    return {name: hashlib.sha256(canonical(build(ta)).encode()).hexdigest() for name, ta in catalogue().items()}


if __name__ == "__main__":
    from prestable.graph import build_prestable_graph

    print(json.dumps(dumps(lambda ta: build_prestable_graph(ta, trace=True, invariant_atoms=False)), indent=1))
