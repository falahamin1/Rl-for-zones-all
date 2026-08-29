# Supplementary Code: Zone-Based Symbolic RL

This package contains the code for all experiments reported in the paper.
It implements two RL agents — **Zone RL** (zone/symbolic abstraction) and
**Region RL** (exact integer-clock abstraction) — across three benchmark
domains, and scripts to reproduce every result table.

---

## Contents

```
supplementary/
├── README.md                      ← this file
├── requirements.txt               ← Python dependencies
│
├── jobshop/                       ← Probabilistic Job-Shop Scheduling
│   ├── prob_jobshop/              ← environment package
│   │   ├── instance.py            ← TaskDistribution, Job, Instance dataclasses
│   │   ├── pta.py                 ← Probabilistic Timed Automaton representation
│   │   ├── simulator.py           ← discrete-event simulator
│   │   ├── symbolic_rl.py         ← Symbolic (zone) RL agent
│   │   ├── region_rl.py           ← Region RL agent
│   │   ├── strategies.py          ← classical heuristics (SEPT, FIFO, MWR, LEPT, Random)
│   │   ├── zone_ops.py            ← DBM zone operations
│   │   └── benchmarks/            ← 20 benchmark instances (PJS_01–10, LJS_01–10)
│   └── run_jobshop_comparison.py  ← runs Zone RL vs Region RL on all 20 instances
│
├── taxi/                          ← Taxi World with Timed Reward Machine
│   ├── taxi_comparison/           ← environment package
│   │   ├── taxi_env.py            ← 5×5 stochastic Taxi grid
│   │   ├── taxi_trm.py            ← TRM instances (c_min varies, D=200)
│   │   ├── taxi_zone_agent.py     ← Zone RL agent
│   │   └── taxi_region_agent.py   ← Region RL agent
│   └── run_taxi_comparison.py     ← runs Zone RL vs Region RL on 6 Taxi instances
│
└── frozen_lake/                   ← Frozen Lake with Timed Reward Machines
    ├── trm_comparison/            ← environment package (all three tasks share this)
    │   ├── grid.py                ← 6×6 grid for three-goal task
    │   ├── trm.py                 ← three-goal TRM instances and state
    │   ├── zone_agent.py          ← Zone RL agent (three-goal)
    │   ├── region_agent.py        ← Region RL agent (three-goal)
    │   ├── sg_grid.py             ← 6×6 grid for single-goal task
    │   ├── sg_trm.py              ← single-goal TRM instances
    │   ├── sg_zone_agent.py       ← Zone RL agent (single-goal)
    │   ├── sg_region_agent.py     ← Region RL agent (single-goal)
    │   ├── hole_grid.py           ← 6×6 grid with non-terminal holes
    │   ├── hole_trm.py            ← hole-disable TRM instances
    │   ├── hole_zone_agent.py     ← Zone RL agent (hole-disable)
    │   └── hole_region_agent.py   ← Region RL agent (hole-disable)
    ├── run_sg_comparison.py       ← Task (a): Single-Goal TRM
    ├── run_trm_comparison.py      ← Task (b): Three-Goal TRM
    └── run_hole_comparison.py     ← Task (c): Hole-Disable TRM
```

---

## Setup

Requires Python 3.9 or later.

```bash
pip install -r requirements.txt
```

The `pyudbm` package provides Difference Bound Matrix (DBM) operations used
by the zone-based agents.

---

## Running the Experiments

Each script must be run from its own subdirectory so that Python can resolve
the local package imports.

### Probabilistic Job-Shop Scheduling

```bash
cd jobshop
python run_jobshop_comparison.py
```

Trains Zone RL and Region RL on all 20 instances (PJS\_01–10, LJS\_01–10).
Training budget scales with instance size: `min(30 000, max(5 000, 200 × tasks))`.
Outputs written to `jobshop/output/all/`:

| File | Description |
|------|-------------|
| `<NAME>_comparison.png` | Per-instance convergence curves (mean makespan vs episodes) |
| `comparison_summary.csv` | Full results: makespan mean ± std, BFS state counts, heuristic baselines |
| `comparison_summary.png` | Summary table figure |

**Expected runtime**: 10–60 minutes depending on hardware (large instances are slow).

---

### Taxi World with Timed Reward Machine

```bash
cd taxi
python run_taxi_comparison.py
```

Trains Zone RL and Region RL on 6 Taxi TRM instances with c\_min ∈ {2, 5, 10, 15, 20, 25}
and fixed D = 200. Each agent runs for 50 000 episodes across 5 independent seeds.
Outputs written to `taxi/output/taxi/`:

| File | Description |
|------|-------------|
| `<NAME>_convergence.png` | Per-instance learning curve |
| `taxi_summary.csv` | Mean reward ± std, visited Q-table states, BFS counts |
| `taxi_summary.png` | Summary table figure |

**Expected runtime**: ~30 minutes.

---

### Frozen Lake — Task (a): Single-Goal TRM

```bash
cd frozen_lake
python run_sg_comparison.py
```

Trains both agents on 5 instances with c\_min = 10 fixed and D ∈ {30, 60, 100, 200, 400}.
50 000 training episodes, 5 seeds.
Outputs written to `frozen_lake/output/sg/`.

**Zone BFS**: 100 reachable states (constant across all instances).
**Region BFS**: grows linearly with D (835 → 12 675).

---

### Frozen Lake — Task (b): Three-Goal TRM

```bash
cd frozen_lake
python run_trm_comparison.py
```

Trains both agents on 6 instances with D = 200 fixed and c\_min ∈ {2, 5, 10, 15, 20, 25}.
50 000 training episodes, 5 seeds.
Outputs written to `frozen_lake/output/trm/`.

**Zone BFS**: 385 reachable states (constant).
**Region BFS**: grows with c\_min × D (19 439 → 212 102).

---

### Frozen Lake — Task (c): Hole-Disable TRM

```bash
cd frozen_lake
python run_hole_comparison.py
```

Trains both agents on 5 instances with c\_min = 10, D = 200 fixed and
x\_h ∈ {5, 10, 20, 30, 50}. Holes are non-terminal; the direction that
caused the fall is disabled for x\_h steps.
50 000 training episodes, 5 seeds.
Outputs written to `frozen_lake/output/hole/`.

**Zone BFS**: 1 905 reachable states (constant).
**Region BFS**: exceeds 1 million for x\_h = 5; surpasses the 2 million BFS
cap for x\_h ≥ 10.

---

## Key Design Choices

**Zone abstraction.** The Zone RL agent groups clock valuations into
Difference Bound Matrix (DBM) zones — equivalence classes of clock
tuples that are indistinguishable for the purpose of satisfying the
TRM guard. The zone graph size is invariant to the numeric values of
guard parameters (c\_min, D, x\_h); only the *number* of guard thresholds
determines the zone count.

**Region abstraction.** The Region RL agent tracks each clock as an
integer capped at the relevant threshold (min(x, c\_min), min(y, D), etc.).
The state space grows linearly or multiplicatively with the parameter values.

**Q-learning.** Both agents use tabular Q-learning with ε-greedy exploration
(ε: 0.9 → 0.05, linear decay), learning rate α = 0.1, and no discounting
(γ = 1.0). The only difference between agents is the clock-state
representation.

**Hyperparameters** (Frozen Lake and Taxi):

| Parameter | Value |
|-----------|-------|
| Training episodes | 50 000 |
| Evaluation episodes | 100 |
| Seeds | 5 (42, 123, 456, 789, 1337) |
| α (learning rate) | 0.1 |
| γ (discount) | 1.0 |
| ε₀ (initial exploration) | 0.9 |
| ε\_min (final exploration) | 0.05 |
| ε schedule | linear decay over training |

---

## Output Format

All scripts produce a CSV summary file with one row per instance.
Common columns:

| Column | Description |
|--------|-------------|
| `zone_mean` / `zone_std` | Mean ± std reward/makespan over evaluation episodes and seeds |
| `zone_visited` | Distinct Q-table entries populated during training |
| `zone_bfs` | Reachable zone states (BFS from initial state) |
| `region_mean` / `region_std` | Same for Region RL |
| `region_visited` | Distinct Q-table entries for Region RL |
| `region_bfs` | Reachable region states (capped at BFS limit) |
