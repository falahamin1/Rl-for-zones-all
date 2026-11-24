# Zone-side state counts for the Polytope benchmarks (Task B)

Branch `audit-and-zone-measure`, 2026-10-05.

- Script: `measurements/zone_counts.py`.
- Raw data:
  - `measurements/zone_counts.csv`: 48 BFS rows.
  - `measurements/modesel_sampling.csv`: 2 rows.
  - `measurements/results/*.json`: one file per configuration.
- Ran locally with `/Users/falah/miniforge3/bin/python3.10`, which loads
  the compiled pyudbm in `pyudbm/` (a CPython 3.10 build). No Alpine or
  SLURM run was needed: the full set took about 3.5 hours on 3 cores.
- No benchmark source was modified.

## Bottom line

1. **On gridtimed and trmgrid, zones do not compress anything.**
   - With the trained jitter (w = 0.25), each concrete state is reached
     with *several* different zones, because the zone width depends on the
     path. Every w = 0.25 BFS hit the 2M-node cap having seen only
     0.48–0.89M concrete states.
   - Extrapolation (b2) changes nothing. Subsumption (b3) removes only
     about 15%.
   - With w = 0, every zone is a single point, and the zone count equals
     the concrete state count exactly (T2: 533,149; G_SMALL: 1,346,833).
   - Either way, a DBM-keyed Zone RL table would be *larger* than the
     Region RL table: T2 regions = 221,309, G_SMALL = 490,652 (from
     `RECON_polytope_benchmarks.md` §4.2).
2. **modesel: during training the zone depends on the sampled durations.
   This is a bug in the benchmark.**
   - `modesel/zones.py:51-55` seeds jobs that are already running with
     `js.active_duration` from the concrete state. In
     `ModeSelGymEnv.step` that value is the *sampled* real duration, not
     the nominal one. The function's docstring says the opposite ("never
     on which concrete duration was actually realized").
   - Two runs with identical actions end in different zones
     (c2 ∈ [30, 50] vs [26, 44] in pyudbm units, from sampled durations
     2.012 and 1.762).
   - As a result, 10k sampled episodes reach 40,841 (M3) and 74,623 (M4)
     distinct (location, zone) pairs, and most zones contain exactly one
     concrete state.
   - With nominal durations (the BFS below), M3 has only 553 pairs and
     M4 9,649.
3. **Adding the cost clock `delta` changes nothing**, except one modesel
   b3 row. In all three domains `delta` duplicates a clock that already
   exists, never resets and runs at rate 1 (`y` in gridtimed, `g` in
   trmgrid and modesel). Without extrapolation, `delta − y` is pinned
   at 0. M4 b3 is the one difference (8,984 vs 8,998 pairs): `g` is
   extrapolated but `delta` is not, which changes a few subset tests.

## Method

The BFS explores the **joint** (concrete state, zone) pairs that the
training environment can reach. It counts distinct (location, `str(Z)`)
pairs, which is the key a DBM-keyed Zone RL agent would use.

**How each env's zone transition is reproduced:**

- **gridtimed and trmgrid:** the training env object itself is stepped
  (`GridTimedTrainEnv.step`, `TRMGridTrainEnv.step`) after setting its
  `_state` and `_Z`. Only `_get_obs` is stubbed, because building the
  observation is not part of the transition. These envs are
  deterministic, so this covers exactly what training can reach.
- **modesel:** `ModeSelGymEnv.step` samples real durations, which a BFS
  can't enumerate. Here the BFS branches on nominal durations
  (`step_deterministic`) and calls the same zone update the gym env calls
  (`zones.advance_zone_for_transition`). Real-duration behaviour is
  measured separately by sampling.

**Locations:** grid cell for gridtimed; (cell, RM phase) for trmgrid;
per-job (cursor, mode, failed) for modesel (`modesel/diagnostics.py:94`).
Terminal successors are not counted.

**Variants:**

- **b1:** the transition as-is.
- **b2:** b1, plus `extrapolate_max_bounds` after every transition. The
  max constant per clock is the largest guard or invariant constant on
  that clock, in pyudbm units (×20):

  | Instance | Max constants (pyudbm units) |
  |---|---|
  | T2 | s1 = 500, s2 = 200, s3 = 340, y = 540, e = 65 |
  | G_SMALL | t_0 = 260, t_1 = 300, t_2 = 300, g = 600, e = 65 |
  | G_LARGE | t_0 = 240, t_1 = 240, t_2 = 260, g = 600, e = 65 |
  | M3 | c0 = 250, c1 = 250, c2 = 200, g = 600 |
  | M4 | c0 = 250, c1 = 250, c2 = 200, c3 = 200, g = 720 |

  `g` in modesel has no zone guard at all. I used max D_j, because its
  concrete guards are `g + remaining ≤ D_j`.
- **b3:** b2, plus subsumption. A new zone Z at location l is replaced by
  an existing zone Z' at l whenever Z ⊆ Z', and exploration continues
  from Z'. The subset test compares raw DBM matrices with numpy; it
  agreed with pyudbm's own `<=` on all 8,400 pairs checked.
- **b4:** b1–b3 rerun with w = 0 (gridtimed and trmgrid only).
- **delta:** every variant also ran with an extra clock `delta` (rate 1,
  never reset, extrapolation bound ∞).

Each BFS is capped at 2,000,000 joint nodes. Truncated counts are lower
bounds, and they depend on BFS order.

## Results

"Pairs" = distinct (location, zone). "Concrete" = distinct concrete
states among the explored nodes. **T** = truncated at 2M nodes. Each
value is the same with and without `delta` unless shown as
"without / with".

| Instance | w | b1 pairs | b2 pairs | b3 pairs | Concrete (b1) | Truncated |
|---|---|---|---|---|---|---|
| T2 | 0.25 | ≥ 2,000,000 | ≥ 2,000,000 | ≥ 1,669,192 | ≥ 480,325 | T |
| T2 (b4) | 0 | 533,149 | 533,149 | 533,149 | 533,149 | no |
| G_SMALL | 0.25 | ≥ 2,000,000 | ≥ 2,000,000 | ≥ 1,710,213 | ≥ 696,374 | T |
| G_SMALL (b4) | 0 | 1,346,833 | 1,346,833 | 1,346,833 | 1,346,833 | no |
| G_LARGE | 0.25 | ≥ 2,000,000 | ≥ 2,000,000 | ≥ 1,769,408 | ≥ 886,244 | T |
| G_LARGE (b4) | 0 | ≥ 2,000,000 | ≥ 2,000,000 | ≥ 2,000,000 | ≥ 2,000,000 | T |
| M3 (nominal) | 0.25 | 553 | 553 | 548 (544 maximal) | 543 | no |
| M4 (nominal) | 0.25 | 9,649 | 9,649 | 8,984 / 8,998 (8,511 / 8,526 maximal) | 8,481 | no |

For comparison, the Region RL state counts (integer clocks capped at
each max constant + 1, from `RECON_polytope_benchmarks.md` §4.2):

| Instance | Region RL states |
|---|---|
| T2 | 221,309 |
| G_SMALL | 490,652 |
| G_LARGE | 822,622 |
| M3 | 569 (exact DP; includes terminal states) |
| M4 | 8,588 (exact DP; includes terminal states) |

**Why extrapolation (b2) does nothing here:**

- In gridtimed and trmgrid, every clock starts equal to the global clock
  (`y` / `g`) and is reset only at switches or sub-goals.
- Each zone therefore pins every clock's difference from the global clock
  to a narrow interval. That global clock's max constant is the deadline,
  and the bound on the global clock itself (≤ D, enforced on every
  edge) is never exceeded.
- Extra-M extrapolation keeps any diagonal bound that lies within the
  max constants, so the zones stay distinct.
- Coarser extrapolations (Extra-LU+, or treating clocks inactive after
  their last guard) could merge more. They were not tried.

**Why w = 0.25 gives *more* zones than concrete states:** the concrete
step always takes exactly `delay + 1` time units, but the zone widens by
`w` on every edge. Two paths to the same concrete state with different
numbers of edges since a reset therefore end with different zone widths.

### modesel: real-duration sampling (as-is, no extrapolation)

Setup: 10,000 episodes per instance, uniform random over legal actions,
seed 0, the training env `ModeSelGymEnv`.

| Instance | (loc, Z) pairs | Distinct concrete states | Visits | Concrete per zone: mean / median / max | Zones with 1 concrete state | ≥ 10 | ≥ 100 | Pairs also in nominal b1 |
|---|---|---|---|---|---|---|---|---|
| M3 | 40,841 | 75,443 | 85,442 | 1.8 / 1 / 1,698 | 34,772 | 433 | 32 | 173 of 553 |
| M4 | 74,623 | 102,130 | 112,129 | 1.4 / 1 / 1,266 | 69,599 | 377 | 18 | 487 of 9,649 |

The 200-episode test run gave 1,208 pairs for M3 against 40,841 at 10k
episodes, so the count keeps growing with more episodes. I did not check
which zones are the few that hold hundreds of concrete states.

## FAST cost in modesel: can pyudbm add a constant to a clock?

**No.**

- UDBM's C++ has it: `fed_t::updateIncrement(x, v)`, i.e. x := x + v
  (`pyudbm/UDBM/include/dbm/fed.h:478-485`).
- The Python binding does not expose it. The native methods are
  `up`, `down`, `update_value`/`reset_value`, `free_clock`, `predt`,
  `extrapolate_max_bounds`, the intersection/union operations, and a few
  conversions (`pyudbm/pyudbm/binding/_udbm.cpp`).
- `DBM._from_raw_matrix` exists, so an increment could be emulated by
  rebuilding the DBM with every bound involving x shifted by v. That is
  untested.

**How the job-shop code handles edge costs:** it doesn't need to. Every
job-shop dispatch has price 0 (`scheduling/prob_jobshop/zone_ops.py:6-8`:
"increment(Z, 0) is a no-op (all job-shop dispatch prices = 0)"). Cost
is time only, read from `delta`'s lower bound as
`r = min_cost(Z') − min_cost(Z)` (`symbolic_rl.py:347`). For modesel, the
FAST penalty would have to be added to the reward outside the zone, as
`ModeSelGymEnv` already does through `res.cost`.

## Caveats

- **Truncated rows are BFS-order dependent.** For example, T2's b1 and
  b2 explored 480,325 vs 480,345 concrete states before the cap.
- The b3 "maximal" count (zones not strictly inside another kept zone)
  is only computed when a location holds ≤ 20,000 zones. That is why it
  appears for M3 and M4 only.
- The modesel b1–b3 counts use nominal durations, so they describe the
  DP's transition graph, not the training env. Because of the bug above,
  the training env reaches very different zones (only 173 / 487 of the
  sampled pairs are nominal pairs).

## Reproduce

```bash
python3.10 measurements/zone_counts.py --all --jobs 3   # 48 BFS configs, ~3.5 h on 3 cores
python3.10 measurements/zone_counts.py --sample          # modesel sampling, ~20 s
python3.10 measurements/zone_counts.py --instance T2 --variant b3 --w 0 --delta 1   # one config
```
