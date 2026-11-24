import pytest

from prestable.model import (
    Constraint,
    Edge,
    TimedAutomaton,
    diff_ge,
    diff_gt,
    diff_le,
    diff_lt,
    eq,
    ge,
    gt,
    le,
    lt,
)


def test_helpers():
    assert le("x", 3) == Constraint("x", "0", 3, False)
    assert lt("x", 3) == Constraint("x", "0", 3, True)
    assert ge("x", 1) == Constraint("0", "x", -1, False)
    assert gt("x", 1) == Constraint("0", "x", -1, True)
    assert eq("x", 2) == (Constraint("x", "0", 2, False), Constraint("0", "x", -2, False))
    assert diff_le("x", "y", 2) == Constraint("x", "y", 2, False)
    assert diff_lt("x", "y", 2) == Constraint("x", "y", 2, True)
    assert diff_ge("x", "y", 2) == Constraint("y", "x", -2, False)
    assert diff_gt("x", "y", 2) == Constraint("y", "x", -2, True)


def test_constraint_str():
    assert str(le("x", 3)) == "x <= 3"
    assert str(gt("x", 1)) == "x > 1"
    assert str(diff_lt("x", "y", 2)) == "x - y < 2"


def small_automaton() -> TimedAutomaton:
    return TimedAutomaton(
        clocks=("x", "y"),
        locations=("l0", "l1", "l2"),
        initial="l0",
        edges=(
            Edge("l0", "l1", "a", guard=(ge("x", 1), diff_lt("x", "y", 2)), resets=frozenset({"y"})),
            Edge("l0", "l2", "b", guard=(le("y", 5),), cost=3.0),
            Edge("l1", "l0", "c", guard=(diff_ge("y", "x", 7),), resets={"x", "y"}),
            Edge("l2", "l2", "d"),
        ),
        rates={"l0": 1.0, "l2": 2.5},
    )


def test_outgoing_and_atomic_guards():
    ta = small_automaton()
    assert [e.action for e in ta.outgoing("l0")] == ["a", "b"]
    assert ta.outgoing("l2")[0].action == "d"
    assert ta.atomic_guards("l0") == frozenset({ge("x", 1), diff_lt("x", "y", 2), le("y", 5)})
    assert ta.atomic_guards("l1") == frozenset({Constraint("x", "y", -7, False)})
    assert ta.atomic_guards("l2") == frozenset()


def test_partition_atoms():
    """Guard atoms plus x < b per non-strict invariant bound x <= b; x < b adds nothing."""
    base = small_automaton()
    ta = TimedAutomaton(base.clocks, base.locations, base.initial, base.edges,
                        invariants={"l0": (le("x", 3), lt("y", 4)), "l1": (lt("x", 2),)})
    assert ta.partition_atoms("l0") == ta.atomic_guards("l0") | {lt("x", 3)}
    assert ta.partition_atoms("l1") == ta.atomic_guards("l1")  # strict bound only
    assert ta.partition_atoms("l2") == frozenset()  # no invariant, no guards
    assert ta.atomic_guards("l0") == base.atomic_guards("l0")  # unchanged by invariants
    # The face atom joins an equal guard atom rather than duplicating it.
    ta = TimedAutomaton(("x",), ("l",), "l", (Edge("l", "l", "a", guard=(lt("x", 3),)),),
                        invariants={"l": (le("x", 3),)})
    assert ta.partition_atoms("l") == frozenset({lt("x", 3)})


def test_max_constant_includes_diagonals():
    assert small_automaton().max_constant() == 7


def test_max_constant_no_guards():
    ta = TimedAutomaton(clocks=("x",), locations=("l",), initial="l", edges=(Edge("l", "l", "a"),))
    assert ta.max_constant() == 0


def test_iterables_are_normalised():
    ta = small_automaton()
    e = ta.edges[2]
    assert e.resets == frozenset({"x", "y"})
    assert isinstance(e.guard, tuple)
    hash(ta)  # frozen and hashable despite the rates dict


def _ta(clocks=("x",), locations=("l0", "l1"), initial="l0", edges=(), rates=None):
    return TimedAutomaton(clocks, locations, initial, edges, rates or {})


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(edges=(Edge("l0", "l1", "a", guard=(le("z", 1),)),)),  # unknown clock in guard
        dict(edges=(Edge("l0", "l1", "a", guard=(diff_le("x", "z", 1),)),)),  # unknown clock, diagonal
        dict(edges=(Edge("l0", "l1", "a", resets={"z"}),)),  # unknown clock in reset
        dict(edges=(Edge("l0", "l1", "a", resets={"0"}),)),  # zero clock cannot be reset
        dict(edges=(Edge("l0", "nowhere", "a"),)),  # unknown target
        dict(edges=(Edge("nowhere", "l1", "a"),)),  # unknown source
        dict(initial="nowhere"),  # initial not a location
        dict(edges=(Edge("l0", "l1", "a", guard=(Constraint("x", "x", 1, False),)),)),  # left == right
        dict(edges=(Edge("l0", "l1", "a", guard=(Constraint("0", "0", 1, False),)),)),  # left == right
        dict(clocks=("x", "x")),  # duplicate clock
        dict(locations=("l0", "l1", "l0")),  # duplicate location
        dict(clocks=("x", "0")),  # clock named "0"
        dict(rates={"nowhere": 1.0}),  # rate for unknown location
    ],
)
def test_validate_rejects(kwargs):
    with pytest.raises(ValueError):
        _ta(**kwargs)


def test_validate_accepts_zero_clock_in_constraints():
    _ta(edges=(Edge("l0", "l1", "a", guard=(ge("x", 1), lt("x", 3))),))


def test_validate_rejects_reserved_delay_action():
    with pytest.raises(ValueError, match="reserved action name 'delay'"):
        _ta(edges=(Edge("l0", "l1", "delay"),))


def test_validate_rejects_initial_invariant_without_zero():
    def ta(inv):
        return TimedAutomaton(("x", "y"), ("l0", "l1"), "l0", (), invariants=inv)
    for bad in [(lt("x", 0),), (le("y", 3), le("x", -1))]:
        with pytest.raises(ValueError, match="zero valuation violates .* initial location 'l0'"):
            ta({"l0": bad})
    ta({"l0": (le("x", 0), lt("y", 1))})  # 0 satisfies x <= 0 and y < 1
    ta({"l1": (lt("x", 0),)})  # only the initial location must contain 0
