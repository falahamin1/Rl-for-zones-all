# PARTITION: How Much Room Does a Sound Abstraction Have?

READ-ONLY diagnostic. `env.py`, `dp.py`, and the model are unmodified —
verified: `gridtimed/diagnostics.py` is the only new file, nothing else in
`gridtimed/` (or outside it) is newer than the start of this work. New
analysis only: exact DP over the hand-built T0 instance's full concrete
state space, plus a from-scratch "smallest DBM containing a point set"
construction for Measurement 3 (not a change to `zone_ops.py`/`zones.py`).

Setup: `instance.py` defaults (hand-built T0), `w=0.25`, default delays
`{0,1,2}`.

**Exclusions** (applied to every measurement below, not just some):

| | count |
|---|---:|
| total reachable states (incl. terminal) | 3438 |
| terminal (DONE/FAILED) excluded | 2 |
| deadlocked (0 legal actions) excluded | 0 |
| **usable states analyzed** | **3436** |

(0 deadlocks — expected: the hand-built T0 layout structurally guarantees
every cell has a plain/switch neighbour, unlike some of `/tmp/GRIDSOUND.md`
Task B's generated layouts.)

## Measurement 1 — the ceiling

| | value |
|---|---:|
| usable states | 3436 |
| **distinct optimal-action-SET classes** | **64** |
| class size: mean | 53.69 |
| class size: median | 16 |
| class size: max | 594 |
| singleton classes (size 1) | 10.94% of classes |

**64 is the ceiling** — no partition sound by this repo's definition (every
cell shares >=1 common optimal action) can have fewer than 64 cells,
because 64 is the number of GENUINELY DISTINCT optimal-action requirements
actually present in the reachable state space; anything coarser must, by
pigeonhole, put two states with disjoint requirements in one cell somewhere.

**This is a ceiling on coarseness, not an achievable abstraction.**
Optimal-action classes are defined purely by which actions are argmax —
nothing requires them to be expressible as a zone (a DBM-representable
convex region), and nothing requires them to even be topologically
connected in clock space. Measurement 3 tests this directly, and finds
they are not zone-expressible, at least for the largest ones.

**Compared directly**, as asked:
- vs. **3436 current-construction zones** (Measurement 2's candidate (c),
  recomputed fresh here, not cited from `/tmp/GRIDSOUND.md`): the ceiling
  is **~54x coarser** (64 vs 3436) — the current zone construction has
  essentially zero compression relative to the theoretical best case.
- vs. **8 distinct locations**: the ceiling is **8x finer** than location
  alone (64 vs 8) — even the best-case, not-necessarily-realizable
  partition needs 8x more cells than pure location provides, confirming
  location alone is structurally too coarse (consistent with Measurement 2
  below, where location-only-type candidates are all well under the
  soundness bar).

## Measurement 2 — candidate observation models

One table, sorted by cell count (coarsest first). (f) is reported as three
rows (`g=1,2,4`) since granularity is itself the variable under test —
nine rows total for the seven lettered candidates.

| Candidate | cells | **soundness** | mean cell size | max cell size | singleton frac |
|---|---:|---:|---:|---:|---:|
| (e) gate-reachability bits | 13 | 53.85% | 264.31 | 555 | 0.00% |
| (b) location + activated bitmask | 23 | 52.17% | 149.39 | 810 | 4.35% |
| (d) gate-passability bits | 23 | 43.48% | 149.39 | 533 | 0.00% |
| (g) window-relative position | 44 | 56.82% | 78.09 | 184 | 0.00% |
| (f) discretized clocks, g=4 | 212 | 73.58% | 16.21 | 52 | 2.36% |
| (f) discretized clocks, g=2 | 820 | **87.68%** | 4.19 | 8 | 4.76% |
| (a) concrete state [control] | 3436 | 100.00% | 1.00 | 1 | 100.00% |
| (c) current zone construction | 3436 | 100.00% | 1.00 | 1 | 100.00% |
| (f) discretized clocks, g=1 | 3436 | 100.00% | 1.00 | 1 | 100.00% |

(a), (c), and (f, g=1) coincide exactly (3436 cells, 100% sound, mean size
1) — expected: `floor(c/1)=c` makes g=1 discretization identical to the
raw concrete state, and the current zone construction is already ~1:1 with
concrete states (`/tmp/GRIDSOUND.md` Task A), so all three are, in effect,
the same trivial (non-)abstraction restated three ways.

**None of the six candidates that achieve any real compression clear the
95% soundness bar.** The closest is (f, g=2): 87.68% sound with mean cell
size 4.19 — real compression, but 12.3% of its cells still contain a
genuine action conflict. Every one of (b), (d), (e), (g) — the
topologically/semantically motivated candidates — sits between 43% and
57% sound, i.e. roughly coin-flip odds that a given cell is safe, despite
compressing 3436 states into 13-44 cells.

## Measurement 3 — is the ceiling zone-expressible?

For each of the 10 largest optimal-action classes: the smallest DBM
containing every member point (entrywise-max of pairwise clock
differences — the standard tightest-DBM-hull construction), then checked
for containing any OTHER usable state (a state outside the class).

| class size | optimal action(s) | contamination (states) | clean? |
|---:|---|---:|---|
| 594 | `d0_DOWN` | 1945 | no |
| 525 | `d0_RIGHT` | 1860 | no |
| 170 | `d0_UP`, `d1_UP`, `d2_UP` | 939 | no |
| 166 | `d0_UP` | 2171 | no |
| 141 | `d0_DOWN`, `d0_RIGHT` | 3052 | no |
| 137 | `d0_LEFT`, `d0_UP`, `d1_LEFT`, `d1_UP`, `d2_LEFT`, `d2_UP` | 595 | no |
| 135 | `d0_DOWN`, `d0_RIGHT`, `d1_DOWN`, `d1_RIGHT`, `d2_DOWN`, `d2_RIGHT` | 954 | no |
| 133 | `d0_LEFT` | 1912 | no |
| 110 | `d0_LEFT`, `d0_UP` | 385 | no |
| 103 | `d1_DOWN` | 1335 | no |

**0 of the top 10 are clean — none of the largest optimal-action classes
is zone-expressible.** Every hull is heavily contaminated: the smallest
DBM containing e.g. the 141-member `{d0_DOWN, d0_RIGHT}` class also
contains 3052 *other* usable states (89% of all remaining usable states),
most of which almost certainly require different actions. This means the
ceiling from Measurement 1 is not just "not yet found" by the zone
constructions tried so far — for these classes it is provably unreachable
by *any* DBM-based zone, because their member states are scattered across
clock space in a way no convex region can isolate without sweeping in
unrelated states. (Self-check performed before trusting this: every
class's own defining points were confirmed contained in its own hull —
the hull-construction code is not itself the source of contamination.)

## Measurement 4 — what does each candidate already determine?

**Refinement check**: does knowing the cell determine the optimal-action
*set* exactly (not just share one action)?

| Candidate | refinement of action partition? | cells failing |
|---|---|---:|
| (a) concrete state | yes | 0.00% |
| (b) location + activated | no | 86.96% |
| (c) current zone construction | yes | 0.00% |
| (d) gate-passability bits | no | 100.00% |
| (e) gate-reachability bits | no | 100.00% |
| (f) g=1 | yes | 0.00% |
| (f) g=2 | no | 40.49% |
| (f) g=4 | no | 62.74% |
| (g) window-relative position | no | 100.00% |

Only the three trivial (non-abstracting) candidates are exact refinements.
Every candidate that compresses at all also loses information the DP
actually used.

**Reverse direction (the one that matters for the claim)** — for (d), (e),
(g), the genuinely coarser candidates: how many concrete states actually
share a cell on average, i.e. is mean cell size above 1 (so the candidate
is abstracting at all, as opposed to secretly still being 1:1)?

| Candidate | mean states per cell | above 1? |
|---|---:|---|
| (d) gate-passability bits | 149.39 | **yes** |
| (e) gate-reachability bits | 264.31 | **yes** |
| (g) window-relative position | 78.09 | **yes** |

All three genuinely abstract (mean cell size well above 1 — these are not
disguised 1:1 mappings) — they just aren't SOUND abstractions (Measurement
2: 43-57%). The problem isn't that they fail to compress; it's that the
compression they achieve cuts across states with different optimal
actions.

## Summary

- **The ceiling is 64 classes** (out of 3436 usable states) — a ~54x
  compression is theoretically available, IF optimal-action classes could
  be used directly as an abstraction's cells.
- **They cannot, at least for the 10 largest classes**: 0/10 are
  zone-expressible (Measurement 3). The ceiling is not achievable by any
  DBM-based construction for the states that matter most for compression.
- **No candidate tried is both SOUND (>= 95%) and COARSENING (mean cell
  size >= 3).** Stated plainly, as instructed: this is the result, not an
  incomplete search outcome. The nearest candidate, (f) discretized clocks
  at g=2, reaches 87.68% soundness with mean cell size 4.19 — real but
  imperfect compression, 7.3 points short of the bar. Every topologically/
  semantically motivated candidate (gate-passability, gate-reachability,
  location+activation, window-relative-position) lands between 43% and 57%
  sound despite compressing by 80-260x on average.
- Every sound candidate found (concrete state, the current zone
  construction, g=1 discretization) achieves **zero** compression (mean
  cell size exactly 1) — confirming this report's opening premise
  (`/tmp/GRIDSOUND.md`): every abstraction tried so far is either too fine
  (no compression) or too coarse (unsound), and this diagnostic did not
  find a counterexample to that pattern among the seven candidates tested.

No new abstraction was built, the model was not modified, and no direction
is recommended.

## Files

`gridtimed/diagnostics.py` (new, read-only analysis; `env.py`/`dp.py`/the
model untouched). Entry point: `python -m gridtimed.diagnostics`. Result
JSON: `/tmp/partition_results.json`.
