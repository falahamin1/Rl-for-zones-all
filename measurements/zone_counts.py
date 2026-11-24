"""Zone-side state counts for the Polytope benchmarks (Task B).

For T2 (gridtimed), G_SMALL / G_LARGE (trmgrid) and M3 / M4 (modesel) this
runs a BFS over the JOINT (concrete state, zone) pairs that the training
gym env can reach, and counts the distinct (location, str(Z)) pairs -- the
state key a DBM-keyed Zone RL agent would use. Variants:

  b1  the gym env's online zone transition, as-is (no extrapolation)
  b2  b1 + extrapolation (``extrapolate_max_bounds`` with per-clock max
      constants) applied after every transition; the next transition starts
      from the extrapolated zone
  b3  b2 + subsumption: a new zone Z at location l is replaced by an existing
      zone Z' at l whenever Z <= Z' (Z is subset of Z'); exploration continues from Z'
  b4  b1-b3 again with w = 0 (gridtimed / trmgrid only)

Each variant is run with and without an extra cost clock ``delta`` (rate 1,
never reset, never extrapolated: its bound is UDBM infinity).

How the transition is reproduced (no benchmark source is modified):
  * gridtimed / trmgrid: the training env object itself is stepped
    (``GridTimedTrainEnv.step`` / ``TRMGridTrainEnv.step``) after setting its
    ``_state`` and ``_Z``; only ``_get_obs`` is stubbed on the instance, since
    building the observation (Zone + clip) is not part of the transition.
    These envs are deterministic, so this is exactly what training reaches.
  * modesel: ``ModeSelGymEnv.step`` samples real durations, which a BFS
    cannot enumerate. The BFS therefore branches on the NOMINAL durations
    (``modesel.env.step_deterministic``, the exact DP's transition) and
    updates the zone with ``modesel.zones.advance_zone_for_transition`` --
    the same function ``ModeSelGymEnv.step`` calls (gym_env.py:101). The
    real-duration behaviour is measured separately by sampling (``--sample``).

Locations: gridtimed = grid cell; trmgrid = (grid cell, RM phase);
modesel = per-job (cursor, active_mode, failed) (modesel/diagnostics.py:94).
Terminal successors are not counted (the env resets the zone there).

Usage (repo root, Python 3.10 with the compiled pyudbm under pyudbm/):
    python measurements/zone_counts.py --all --jobs 4      # every BFS config
    python measurements/zone_counts.py --sample             # modesel sampling
    python measurements/zone_counts.py --instance T2 --variant b3 --w 0 --delta 1
Results: measurements/results/*.json, merged into measurements/zone_counts.csv
and measurements/modesel_sampling.csv.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter, defaultdict, deque

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OUT_DIR = os.path.join(REPO, "measurements", "results")

CAP = 2_000_000
UDBM_INF = 1073741823  # UDBM's infinity sentinel, same as scheduling/prob_jobshop/zone_ops.py
DELTA = "delta"

# Instance parameters copied from run_one.py (_build_instance, lines 271-295,
# and the constants at lines 65-135); run_one is not imported because it
# pulls in torch.
GRIDTIMED = {"T2": dict(grid_n=5, n_switches=3, n_gates=3, window_width=2, D=27)}
TRMGRID = {"G_SMALL": dict(grid_n=4, k=3), "G_LARGE": dict(grid_n=5, k=3)}
MODESEL = {"M3": dict(n_jobs=3, n_tasks=3, n_machines=3, tightness=0.25),
           "M4": dict(n_jobs=4, n_tasks=3, n_machines=3, tightness=0.5)}
TRAIN_W = 0.25


def domain_of(name):
    return "gridtimed" if name in GRIDTIMED else "trmgrid" if name in TRMGRID else "modesel"


def build_instance(name, w):
    if name in GRIDTIMED:
        from gridtimed.generator import make_instance
        inst, _ = make_instance(**GRIDTIMED[name], w=TRAIN_W, seed=0, label=name)
    elif name in TRMGRID:
        from trmgrid.generator import make_instance
        inst, _ = make_instance(**TRMGRID[name], alpha=0.25, width=8, w=TRAIN_W, seed=2, label=name)
    else:
        from modesel.generator import make_instance
        return make_instance(**MODESEL[name], fast_slow_ratio=2.0, cost_penalty=6.0,
                             w=w, seed=0, label=name)
    # The generators never use w when placing the layout, so replacing it keeps
    # the trained layout and only changes the zone jitter.
    return dataclasses.replace(inst, w=w)


def add_delta(zctx):
    """Rebuild a benchmark zone context with an extra never-reset clock ``delta``."""
    from pyudbm import Context

    names = list(zctx.clock_names) + [DELTA]
    zctx._clock_names = names
    zctx.ctx = Context(names)
    zctx.clocks = {n: getattr(zctx.ctx, n) for n in names}
    return zctx


def scaled(x, scale):
    return int(round(x * scale))  # identical to every benchmark zone_ops._scale


# ---------------------------------------------------------------------------
# Domain adapters: initial (state, zone), successors, location, max constants
# ---------------------------------------------------------------------------

class GridAdapter:
    """gridtimed and trmgrid: step the training env object itself."""

    def __init__(self, name, w, with_delta):
        self.domain = domain_of(name)
        self.inst = build_instance(name, w)
        if self.domain == "gridtimed":
            from gridtimed.env import is_terminal
            from gridtimed.gym_env import GridTimedTrainEnv as Env
            from gridtimed.instance import SCALE
            from gridtimed.zone_ops import GridZoneContext as Ctx
        else:
            from trmgrid.env import is_terminal
            from trmgrid.gym_env import TRMGridTrainEnv as Env
            from trmgrid.instance import SCALE
            from trmgrid.zone_ops import TRMZoneContext as Ctx
        self.is_terminal = is_terminal
        self.env = Env(self.inst, seed=0)
        self.env.reset()
        self.env._get_obs = lambda: None  # observation building is not part of the transition
        zctx = Ctx(self.inst)
        self.zctx = add_delta(zctx) if with_delta else zctx
        self.env._zctx = self.zctx
        self.scale = SCALE
        self.n_actions = self.env.n_tasks

    def initial(self):
        return self.env._state, self.zctx.ctx.get_zero_federation()

    def loc(self, s):
        return s.pos if self.domain == "gridtimed" else (s.pos, s.phase)

    def successors(self, s, Z):
        env = self.env
        for a in range(self.n_actions):
            env._state, env._Z = s, Z
            env.step(a)
            s2, Z2 = env._state, env._Z
            if self.is_terminal(s2) or (s2 == s and Z2 is Z):  # terminal, or illegal self-loop
                continue
            yield s2, Z2

    def max_constants(self):
        inst, sc = self.inst, self.scale
        e_max = scaled(max(inst.allowed_delays) + 1 + inst.w, sc)
        if self.domain == "gridtimed":
            per = {f"s{i + 1}": scaled(hi, sc) for i, (_lo, hi) in enumerate(inst.gate_windows)}
            per["y"] = scaled(inst.deadline, sc)
        else:
            per = {}
            for i in range(inst.k):
                his = [inst.windows[(i, j)][1] for j in range(inst.k) if j != i]
                per[f"t_{i}"] = scaled(max(his + [inst.caps[i]]), sc)
            per["g"] = scaled(inst.deadline, sc)
        per["e"] = e_max
        return per


class ModeSelAdapter:
    """modesel: nominal-duration branching + the gym env's zone update function."""

    def __init__(self, name, w, with_delta):
        from modesel import env as E
        from modesel.diagnostics import _location_key
        from modesel.instance import MODES, SCALE
        from modesel.zone_ops import ModeSelZoneContext
        from modesel.zones import advance_zone_for_transition

        self.E, self.MODES, self.scale = E, MODES, SCALE
        self.inst = build_instance(name, w)
        self._loc = _location_key
        self.advance = advance_zone_for_transition
        zctx = ModeSelZoneContext(self.inst)
        self.zctx = add_delta(zctx) if with_delta else zctx

    def initial(self):
        return self.E.initial_state(self.inst), self.zctx.ctx.get_zero_federation()

    def loc(self, s):
        return self._loc(self.inst, s)

    def successors(self, s, Z):
        E, inst = self.E, self.inst
        for t in inst.all_tasks:
            for m in self.MODES:
                if not E.is_legal(inst, s, t.task_id, m):
                    continue
                res = E.step_deterministic(inst, s, t.task_id, m)
                if res.done:
                    continue
                Z2 = self.advance(Z, self.zctx, inst, s, res.dispatched_job_idx, m, res.trace)
                yield res.next_state, Z2

    def max_constants(self):
        inst, sc = self.inst, self.scale
        per = {f"c{j}": scaled(max(t.d_slow for t in job.tasks) * (1 + inst.w), sc)
               for j, job in enumerate(inst.jobs)}
        # g appears in no zone guard (modesel/zones.py:13-19); its concrete
        # guards are g + remaining <= D_j, so max_j D_j is its largest constant.
        per["g"] = scaled(max(job.deadline for job in inst.jobs), sc)
        return per


def make_adapter(name, w, with_delta):
    return ModeSelAdapter(name, w, with_delta) if name in MODESEL else GridAdapter(name, w, with_delta)


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------

MAXIMAL_LIMIT = 20_000  # per-location zone count above which the O(n^2) maximal count is skipped


class PassedList:
    """Zones kept at one location, with a vectorised subset test.

    For canonical, non-empty single DBMs, Z <= Z' iff every raw DBM entry of Z
    is <= the matching entry of Z' (UDBM's raw encoding (bound << 1) | non-strict
    is order-preserving, infinity is the largest value). Multi-DBM federations
    fall back to pyudbm's own ``<=``.
    """

    def __init__(self):
        import numpy as np

        self.np = np
        self.rows = None
        self.n = 0
        self.zones = []          # Federation per row
        self.by_str = {}         # str(Z) -> Federation
        self.multi = []          # multi-DBM federations (checked with pyudbm)

    def _raw(self, Z):
        dbms = Z.to_dbm_list()
        if len(dbms) != 1:
            return None
        return self.np.fromiter(dbms[0]._raw_matrix(), dtype=self.np.int64)

    def find_superset(self, Z):
        v = self._raw(Z)
        if v is not None and self.n:
            hit = self.np.flatnonzero(self.np.all(self.rows[: self.n] >= v, axis=1))
            if hit.size:
                return self.zones[hit[0]]
        if v is None:
            for Zp in self.zones + self.multi:
                if Z <= Zp:
                    return Zp
        else:
            for Zp in self.multi:
                if Z <= Zp:
                    return Zp
        return None

    def add(self, Z, zs):
        self.by_str[zs] = Z
        v = self._raw(Z)
        if v is None:
            self.multi.append(Z)
            return
        if self.rows is None:
            self.rows = self.np.empty((64, v.size), dtype=self.np.int64)
        if self.n == len(self.rows):
            self.rows = self.np.concatenate([self.rows, self.np.empty_like(self.rows)])
        self.rows[self.n] = v
        self.zones.append(Z)
        self.n += 1

    def n_maximal(self):
        """Zones not strictly contained in another kept zone (None if too many to check)."""
        if self.n + len(self.multi) > MAXIMAL_LIMIT:
            return None
        R = self.rows[: self.n] if self.n else None
        count = len(self.multi)  # multi-DBM zones are counted as maximal (none occur in practice)
        for i in range(self.n):
            ge = self.np.all(R >= R[i], axis=1)
            ge[i] = False
            strictly = ge & self.np.any(R != R[i], axis=1)
            count += not strictly.any()
        return count

def bfs(name, variant, w, with_delta, cap=CAP):
    t0 = time.time()
    ad = make_adapter(name, w, with_delta)
    extrapolate = variant in ("b2", "b3")
    bounds = None
    if extrapolate:
        per = ad.max_constants()
        bounds = {ad.zctx.clocks[n]: per[n] for n in ad.zctx.clock_names if n != DELTA}
        if with_delta:
            bounds[ad.zctx.clocks[DELTA]] = UDBM_INF

    s0, Z0 = ad.initial()
    joint = {(s0, str(Z0))}
    zone_keys = {(ad.loc(s0), str(Z0))}
    passed = defaultdict(PassedList)  # b3: loc -> kept zones
    passed[ad.loc(s0)].add(Z0, str(Z0))
    n_subsumed = 0
    queue = deque([(s0, Z0)])
    truncated = False

    while queue and not truncated:
        s, Z = queue.popleft()
        for s2, Z2 in ad.successors(s, Z):
            if extrapolate:
                Z2 = Z2.extrapolate_max_bounds(bounds)
            loc = ad.loc(s2)
            zs = str(Z2)
            if variant == "b3":
                at_loc = passed[loc]
                if zs not in at_loc.by_str:
                    rep = at_loc.find_superset(Z2)
                    if rep is None:
                        at_loc.add(Z2, zs)
                    else:
                        n_subsumed += 1
                        Z2, zs = rep, str(rep)
            key = (s2, zs)
            if key in joint:
                continue
            joint.add(key)
            zone_keys.add((loc, zs))
            queue.append((s2, Z2))
            if len(joint) >= cap:
                truncated = True
                break

    per_loc = Counter(loc for loc, _ in zone_keys)
    row = dict(
        instance=name, domain=domain_of(name), variant=variant, w=w, delta=int(with_delta),
        clocks=" ".join(ad.zctx.clock_names),
        joint_nodes=len(joint),
        loc_zone_pairs=len(zone_keys),
        concrete_states=len({s for s, _ in joint}),
        locations=len(per_loc),
        max_zones_per_location=max(per_loc.values()),
        mean_zones_per_location=round(statistics.mean(per_loc.values()), 1),
        subsumed_replacements=n_subsumed if variant == "b3" else "",
        truncated=int(truncated), cap=cap, wall_s=round(time.time() - t0, 1),
    )
    if variant == "b3":
        # Kept zones that are not strictly inside a zone added later (order-independent count).
        counts = [pl.n_maximal() for pl in passed.values()]
        row["b3_maximal_zones"] = "" if any(c is None for c in counts) else sum(counts)
        row["b3_multi_dbm_zones"] = sum(len(pl.multi) for pl in passed.values())
    if extrapolate:
        row["max_constants_scaled"] = " ".join(f"{n}={b}" for n, b in sorted(ad.max_constants().items()))
    return row


def run_one(cfg):
    name, variant, w, delta = cfg
    os.makedirs(OUT_DIR, exist_ok=True)
    row = bfs(name, variant, w, bool(delta))
    path = os.path.join(OUT_DIR, f"bfs__{name}__{variant}__w{w}__delta{int(delta)}.json")
    with open(path, "w") as f:
        json.dump(row, f)
    print(json.dumps(row), flush=True)
    return row


def all_configs():
    cfgs = []
    for name in list(GRIDTIMED) + list(TRMGRID) + list(MODESEL):
        ws = [TRAIN_W] if name in MODESEL else [TRAIN_W, 0.0]
        for w in ws:
            for variant in ("b1", "b2", "b3"):
                for delta in (0, 1):
                    cfgs.append((name, variant, w, delta))
    return cfgs


# ---------------------------------------------------------------------------
# modesel sampling: concrete real-duration states per (loc, Z)
# ---------------------------------------------------------------------------

def sample_modesel(name, n_episodes=10_000, seed=0):
    import numpy as np

    from modesel.env import action_mask
    from modesel.gym_env import ModeSelGymEnv

    t0 = time.time()
    inst = build_instance(name, TRAIN_W)
    env = ModeSelGymEnv(inst, seed=seed)
    env._get_obs = lambda: None
    ad = ModeSelAdapter(name, TRAIN_W, False)
    rng = np.random.default_rng(seed)

    concrete = defaultdict(set)   # (loc, str(Z)) -> distinct real-valued states
    visits = Counter()
    for ep in range(n_episodes):
        env.reset(seed=seed * 1_000_003 + ep)
        while True:
            key = (ad.loc(env._state), str(env._Z))
            concrete[key].add(env._state)
            visits[key] += 1
            mask = action_mask(inst, env._state)
            legal = [i for i, aid in enumerate(env.action_ids) if mask[aid]]
            _obs, _r, done, _trunc, _info = env.step(int(rng.choice(legal)))
            if done:
                break

    # Which sampled (loc, Z) pairs the nominal-duration b1 BFS also reaches.
    nominal = set()
    s0, Z0 = ad.initial()
    seen = {(s0, str(Z0))}
    q = deque([(s0, Z0)])
    while q:
        s, Z = q.popleft()
        nominal.add((ad.loc(s), str(Z)))
        for s2, Z2 in ad.successors(s, Z):
            k = (s2, str(Z2))
            if k not in seen:
                seen.add(k)
                q.append((s2, Z2))

    sizes = sorted(len(v) for v in concrete.values())
    row = dict(
        instance=name, episodes=n_episodes, policy="uniform random over legal actions", seed=seed,
        loc_zone_pairs=len(concrete),
        distinct_concrete_states=sum(sizes),
        visits=sum(visits.values()),
        concrete_per_zone_mean=round(statistics.mean(sizes), 1),
        concrete_per_zone_median=statistics.median(sizes),
        concrete_per_zone_max=sizes[-1],
        zones_with_one_concrete_state=sum(1 for x in sizes if x == 1),
        zones_with_ge_10=sum(1 for x in sizes if x >= 10),
        zones_with_ge_100=sum(1 for x in sizes if x >= 100),
        sampled_pairs_also_in_nominal_b1=len(set(concrete) & nominal),
        nominal_b1_pairs=len(nominal),
        wall_s=round(time.time() - t0, 1),
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"sample__{name}.json"), "w") as f:
        json.dump(row, f)
    print(json.dumps(row), flush=True)
    return row


# ---------------------------------------------------------------------------

def merge():
    for prefix, out in (("bfs", "zone_counts.csv"), ("sample", "modesel_sampling.csv")):
        rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT_DIR, f"{prefix}__*.json")))]
        if not rows:
            continue
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        path = os.path.join(REPO, "measurements", out)
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="run every BFS configuration")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes for --all")
    p.add_argument("--instance", choices=list(GRIDTIMED) + list(TRMGRID) + list(MODESEL))
    p.add_argument("--variant", choices=("b1", "b2", "b3"))
    p.add_argument("--w", type=float, default=TRAIN_W)
    p.add_argument("--delta", type=int, choices=(0, 1), default=0)
    p.add_argument("--sample", action="store_true", help="modesel real-duration sampling (M3, M4)")
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--skip-done", action="store_true", help="skip configs whose result JSON exists")
    p.add_argument("--merge", action="store_true", help="only merge results/*.json into the CSVs")
    a = p.parse_args()

    if a.merge:
        merge()
        return
    if a.sample:
        for name in MODESEL:
            sample_modesel(name, a.episodes)
    if a.all:
        cfgs = all_configs()
        if a.skip_done:
            cfgs = [c for c in cfgs if not os.path.exists(os.path.join(
                OUT_DIR, f"bfs__{c[0]}__{c[1]}__w{c[2]}__delta{c[3]}.json"))]
        # Largest runs first so the pool does not end on a long straggler.
        cfgs.sort(key=lambda c: (c[0] in MODESEL, c[1] != "b3"))
        with mp.get_context("spawn").Pool(a.jobs, maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(run_one, cfgs):
                pass
    elif a.instance:
        run_one((a.instance, a.variant, a.w, a.delta))
    merge()


if __name__ == "__main__":
    main()
