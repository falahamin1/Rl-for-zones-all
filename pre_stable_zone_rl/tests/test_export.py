import random
import re
import shutil

import pytest

from prestable.export import from_json, render, summary, to_dot, to_json, write_dot
from prestable.graph import build_prestable_graph
from prestable.model import Constraint
from prestable.zones import ZoneContext
from test_phase1 import running_example, two_clock_example
from test_phase2_graph import unreachable_example


def edges_in_dot(dot):
    lines = [l for l in dot.splitlines() if "->" in l]
    delay = [l for l in lines if "style=dashed" in l]
    action = [l for l in lines if "label=" in l]
    return delay, action


@pytest.fixture(scope="module")
def running():
    ta = running_example()
    return ta, build_prestable_graph(ta).graph


# --- Zone.constraints ---------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
def test_constraints_round_trip(n):
    rng = random.Random(40 + n)
    ctx = ZoneContext("xyz"[:n])
    names = ["0", *ctx.clocks]
    for _ in range(200):
        guard = [Constraint(*rng.sample(names, 2), rng.randint(-4, 4), rng.random() < 0.5)
                 for _ in range(rng.randint(0, 5))]
        z = ctx.from_guard(guard)
        assert ctx.from_guard(z.constraints()) == z, str(z)
    assert ctx.from_guard(ctx.empty().constraints()).is_empty
    assert ctx.universe().constraints() == ()


# --- DOT --------------------------------------------------------------------

def test_dot_structure(running):
    ta, g = running
    dot = to_dot(g, ta)
    assert dot.count("subgraph ") == 2
    assert 'subgraph "cluster_l0"' in dot and 'subgraph "cluster_l1"' in dot
    for n in g.nodes:
        assert re.search(rf'n{n.id} \[label="{n.id}: ', dot), n.id
    (init_line,) = [l for l in dot.splitlines() if l.strip().startswith(f"n{g.initial} [")]
    assert "peripheries=2" in init_line
    assert sum("peripheries=2" in l for l in dot.splitlines()) == 1


@pytest.mark.parametrize("mode, n_delay", [("all", 20), ("no_self_loops", 12), ("covering", 6)])
def test_dot_delay_modes(running, mode, n_delay):
    ta, g = running
    delay, action = edges_in_dot(to_dot(g, ta, delay_edges=mode))
    assert len(delay) == n_delay
    assert len(action) == 5


def test_dot_covering_edges_follow_time(running):
    ta, g = running
    delay, _ = edges_in_dot(to_dot(g, ta))
    pairs = {tuple(map(int, re.findall(r"n(\d+)", l))) for l in delay}
    ctx = g.nodes[0].zone.ctx
    zones = ["x == 0", "0 < x < 1", "1 <= x <= 2", "x > 2"]
    expected = set()
    for l in ("l0", "l1"):
        ids = [next(n.id for n in g.nodes if n.location == l and str(n.zone) == z) for z in zones]
        expected |= {(ids[i], ids[i + 1]) for i in range(3)}
    assert pairs == expected


def test_dot_bad_mode(running):
    ta, g = running
    with pytest.raises(ValueError):
        to_dot(g, ta, delay_edges="some")


def test_dot_escaping_and_cost():
    from prestable.model import Edge, TimedAutomaton, ge
    ta = TimedAutomaton(("x",), ('a"b', "c\\d"), 'a"b',
                        (Edge('a"b', "c\\d", "go", guard=(ge("x", 1),), resets={"x"}, cost=2.5),))
    g = build_prestable_graph(ta).graph
    dot = to_dot(g, ta)
    assert 'label="a\\"b";' in dot and 'label="c\\\\d";' in dot
    assert 'label="go / 2.5"' in dot


def test_dot_unreachable():
    ta = unreachable_example()
    g = build_prestable_graph(ta).graph
    assert len(g.nodes) == 8  # island has no blocks; check the flag on a synthetic case below
    from prestable.graph import GraphEdge, Node, ZoneGraph
    ctx = g.nodes[0].zone.ctx
    extra = Node(len(g.nodes), "island", ctx.universe())
    g2 = ZoneGraph(nodes=g.nodes + (extra,), edges=g.edges, initial=g.initial)
    dot = to_dot(g2, ta)
    (line,) = [l for l in dot.splitlines() if l.strip().startswith(f"n{extra.id} [")]
    assert "dashed" in line and "grey" in line
    assert f"n{extra.id} [" not in to_dot(g2, ta, include_unreachable=False)


def test_write_dot(tmp_path, running):
    ta, g = running
    path = tmp_path / "g.dot"
    write_dot(g, ta, path, delay_edges="all")
    assert path.read_text() == to_dot(g, ta, delay_edges="all")


def test_render(tmp_path, running):
    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' not installed")
    ta, g = running
    path = tmp_path / "g.png"
    render(g, ta, path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    render(g, ta, tmp_path / "g.svg", fmt="svg")
    assert "<svg" in (tmp_path / "g.svg").read_text()


def test_render_without_dot(tmp_path, running, monkeypatch):
    ta, g = running
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="not installed"):
        render(g, ta, tmp_path / "g.png")


# --- JSON -------------------------------------------------------------------

@pytest.mark.parametrize("make_ta", [running_example, two_clock_example])
def test_json_round_trip(make_ta):
    import json
    ta = make_ta()
    g = build_prestable_graph(ta).graph
    data = json.loads(json.dumps(to_json(g, ta)))  # survives real serialisation
    assert data["initial"] == 0 and data["K"] == g.K and data["clocks"] == list(ta.clocks)
    ctx = g.nodes[0].zone.ctx
    g2 = from_json(data, ta, ctx)
    assert g2.initial == g.initial and g2.K == g.K
    assert g2.nodes == g.nodes  # same ids, locations, semantically equal zones (same ctx)
    assert g2.edges == g.edges
    for e in g2.edges:
        assert e.ta_edge is None or any(e.ta_edge is te for te in ta.edges)
    # Without a shared context the zones still match by printed form.
    g3 = from_json(data, ta)
    assert [(n.id, n.location, str(n.zone)) for n in g3.nodes] == [(n.id, n.location, str(n.zone)) for n in g.nodes]


def test_json_clock_mismatch(running):
    ta, g = running
    data = to_json(g, ta)
    data["clocks"] = ["y"]
    with pytest.raises(ValueError):
        from_json(data, ta)


# --- summary ------------------------------------------------------------------

def test_summary(running):
    ta, g = running
    text = summary(g)
    for l in ta.locations:
        assert f"location {l}" in text
    assert "edges: 25 (a: 2, b: 3, delay: 20)" in text
    assert "reachable: 8 of 8 nodes" in text
    for n in g.nodes:
        assert f"{n.id}: {n.zone}" in text


def test_json_keeps_indices_of_value_identical_edges():
    import json

    from prestable.model import Edge, TimedAutomaton, ge, le

    def make():
        a = Edge("l0", "g", "a", guard=(ge("x", 1),))
        return TimedAutomaton(("x",), ("l0", "g"), "l0", (a, Edge("l0", "g", "a", guard=(ge("x", 1),))),
                              rates={"l0": 1}, invariants={"l0": (le("x", 2),)})

    ta, ta2 = make(), make()  # equal, with fresh Edge objects
    assert ta.edges[0] == ta.edges[1] and ta2 == ta
    g = build_prestable_graph(ta).graph
    data = to_json(g, ta)
    indices = sorted(e["ta_edge"] for e in data["edges"] if e["ta_edge"] is not None)
    assert indices == [0, 0, 1, 1]  # both edges leave 1 <= x < 2 and the face x = 2
    back = from_json(json.loads(json.dumps(data)), ta2)
    assert sorted(e.edge for e in back.edges if e.edge is not None) == [0, 0, 1, 1]
    assert all(e.ta_edge is (None if e.edge is None else ta2.edges[e.edge]) for e in back.edges)
    assert to_json(back, ta2) == data
