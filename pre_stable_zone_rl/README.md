# prestable

Builds a *pre-stable zone graph* from a (priced) timed automaton.

Modules (all implemented and tested):

- `prestable.model`: timed automaton data classes (`Constraint`, `Edge`, `TimedAutomaton`)
  plus constraint helpers (`le`, `lt`, `ge`, `gt`, `eq`, `diff_*`). Location invariants
  are upper bounds on single clocks; the action name `"delay"` is reserved.
- `prestable.zones`: immutable convex zones over [pyudbm](https://github.com/HansBug/pyudbm)
  (`ZoneContext`, `Zone`, `up`, `down`, `reset`, `pre_edge`, `extrapolate`, `subtract`,
  `hull`, `is_k_bounded`). Only this module imports pyudbm.
- `prestable.partition`: the partition operator Π_K (`partition`, `check_partition`).
- `prestable.algorithm`: Phase 1 (forward closure) and Phase 2 (pre-stability), with checkers.
  Blocks are uniform w.r.t. `TimedAutomaton.partition_atoms` (guard atoms plus invariant
  faces); `invariant_atoms=False` gives the guard-only partition of before step 12.
- `prestable.graph`: Phase 3, the pre-stable zone graph (`build_prestable_graph`).
- `prestable.export`: DOT/Graphviz, JSON round trip and a text summary.
- `prestable.plot2d`: plots of 2-clock partitions (needs matplotlib).
- `prestable.brg`: boundary region graph and discounted value iteration (reference solver,
  independent of the zone code).
- `prestable.block_vi`: discounted optimal cost on the pre-stable graph via a block-corner MDP
  ("boundary" and "integer" variants).

Scripts in `examples/` write their output to `examples/out/` (git-ignored).

## Running the tests

You need a Python environment where `import pyudbm` works. pyudbm is not on PyPI for
every platform; if `pip install pyudbm` fails, build it from source (see its README:
`make bin && make build`, then `pip install -e .`).

```shell
cd pre_stable_zone_rl
pytest -m "not slow"   # about 10 s
pytest                 # full suite, about 2 min
```

`pytest` picks up `src/` through `pythonpath` in `pyproject.toml`, so installing the
package is optional (`pip install -e ".[dev]"` also works).
