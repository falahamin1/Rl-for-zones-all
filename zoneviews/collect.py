"""Collect (location, Zone) pairs from ``SymbolicJobShopEnv``, delta-free.

Design notes carried forward from prior exploration of this domain:

* ``delta`` (the never-resetting global cost clock used by
  ``prob_jobshop_ext.zone_ops``) is **not** part of the zones built here.
  ``zone_ops.ZoneContext`` always appends ``delta`` to its clock list, so
  this module does not use it; instead it builds its own minimal,
  delta-free zone context (:class:`_JobClockContext`, job clocks only) and
  reuses ``zone_ops``'s generic, clock-list-parameterized propagation
  helpers (``apply_guard``, ``reset_clocks``, ``apply_delay``,
  ``apply_invariant`` — none of which special-case ``delta``) directly on
  top of it. ``zone_ops.normalize``/``min_cost`` *do* hard-code ``delta``
  and are intentionally never called here.
* Elapsed time, when a future phase needs it, should be read off the
  concrete/env side (as ``region_env.py:86-111``'s ``elapsed`` return value
  already does for the region agent) rather than off the zone, since there
  is no ``delta`` clock to read it from any more. Whatever reward is
  eventually attached to a transition should keep the existing convention
  used throughout this repository: reward is negative elapsed time (cost
  minimization framed as reward maximization via negation), never a
  positive/shaped signal for progress. This phase does not compute a
  reward at all (no RL here) — this note only fixes the convention for
  whoever wires one up later.
* pyudbm ``Federation`` objects are not picklable (confirmed both in this
  repository's own audit and empirically). :func:`collect_zones` therefore
  returns live ``Zone``/``Federation`` objects and is meant to be consumed
  within the same process; :func:`to_npz`/:func:`from_npz` persist only
  derived plain-array data (rasterized views, half-space list, vertex list,
  constraint graph edges) to cross a process/file boundary.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import zoneviews  # noqa: F401  (runs the pyudbm import-path bootstrap)

import networkx as nx
import numpy as np
from pyudbm import Context

from prob_jobshop_ext.instance import ProbJobShopInstance
from prob_jobshop_ext.symbolic_env import SymbolicJobShopEnv
from prob_jobshop_ext.zone_ops import apply_delay, apply_guard, apply_invariant, reset_clocks

from .raster import dbm_tensor, graph_image, projection_stack
from .zone import HalfSpace, UnboundedZone, VertexEnumerationLimit, Zone

__all__ = ["collect_zones", "to_npz", "from_npz", "CollectedView"]

Location = Tuple[str, ...]


class _JobClockContext:
    """Delta-free zone context: one clock per job, no cost clock.

    Deliberately does *not* subclass or wrap
    ``prob_jobshop_ext.zone_ops.ZoneContext`` (which always appends
    ``delta``); it only exposes the ``.clocks`` mapping that
    ``zone_ops.apply_guard``/``reset_clocks``/``apply_delay``/
    ``apply_invariant`` need, so those functions can be reused unmodified.
    """

    def __init__(self, instance: ProbJobShopInstance):
        self.clock_names: List[str] = [f"c_{j.job_id}" for j in instance.jobs]
        self.ctx = Context(self.clock_names)
        self.clocks = {name: getattr(self.ctx, name) for name in self.clock_names}


def _load_instance(name: str) -> ProbJobShopInstance:
    """Resolve a benchmark instance name across both the small (PJS_*) and
    large (LJS_*) benchmark suites."""
    from prob_jobshop_ext.benchmarks.all_instances import ALL_INSTANCE_NAMES, get_instance_by_name
    from prob_jobshop_ext.benchmarks.all_large_instances import (
        ALL_LARGE_INSTANCE_NAMES,
        get_large_instance_by_name,
    )

    if name in ALL_INSTANCE_NAMES:
        return get_instance_by_name(name)
    if name in ALL_LARGE_INSTANCE_NAMES:
        return get_large_instance_by_name(name)
    raise ValueError(
        f"Unknown instance name {name!r}; expected one of {ALL_INSTANCE_NAMES + ALL_LARGE_INSTANCE_NAMES}"
    )


def _job_clock_k(instance: ProbJobShopInstance) -> Tuple[int, ...]:
    """One box bound per job clock, uniform across jobs: the instance's
    worst-case total makespan.

    ``region_env.py``'s ``max_constant_dict`` caps each job at its own
    longest task duration — but that convention is paired with a
    *saturating* ``min(x, cap)`` region key, never a hard intersection. A
    job clock that is not currently active (job waiting while other jobs'
    tasks run) keeps ticking with no guard bounding it at all, so it can
    legitimately grow past its own tasks' durations before the job is next
    dispatched. Since :meth:`Zone.clip_to_k_box` is a genuine box
    *intersection* (per spec), a per-job "own max duration" cap can and
    does make the clip empty for such zones (confirmed empirically on
    PJS_04). The only per-clock bound that is safe for every reachable
    zone of an episode is a bound on the elapsed time of the *whole*
    episode: no clock, active or waiting, can exceed how long the entire
    schedule can possibly take.
    """
    bound = instance.worst_case_makespan_upper_bound()
    return tuple(bound for _ in instance.jobs)


def _propagate_branches(Z, zctx: _JobClockContext, action: dict, env: SymbolicJobShopEnv):
    """Mirror ``symbolic_rl.SymbolicRLAgent._propagate_zone``, delta-free.

    Returns a list of ``(sampled_duration, Federation)`` pairs, one per
    duration branch whose guard/reset/delay/completion sequence yields a
    non-empty zone (mirrors the original's empty-branch pruning).
    """
    Z_g = apply_guard(Z, zctx, action["guards"])
    if Z_g is None:
        return []
    Z_r = reset_clocks(Z_g, zctx, action["resets"])

    result = []
    for branch in action["branches"]:
        d = branch["d"]
        Z_curr = apply_delay(Z_r, zctx, env.invariant(branch["next_loc"]), 1)

        valid = True
        for step in branch["path"]:
            for cname, exact_d in step["completion_guards"]:
                Z_curr = Z_curr & (zctx.clocks[cname] >= exact_d)
                if Z_curr.is_empty():
                    valid = False
                    break
            if not valid:
                break
            Z_curr = apply_delay(Z_curr, zctx, env.invariant(step["next_loc"]), 1)

        if not valid or Z_curr.is_empty():
            continue
        result.append((d, Z_curr))

    return result


def collect_zones(
    instance_name: str,
    n_episodes: int,
    seed: int,
    policy: str = "uniform",
) -> List[Tuple[Location, Zone]]:
    """Run ``n_episodes`` episodes of ``SymbolicJobShopEnv`` and return every
    ``(location, Zone)`` pair visited at a decision point.

    :param instance_name: A benchmark instance name, e.g. ``"PJS_01"`` or
        ``"LJS_01"``.
    :param n_episodes: Number of episodes to run.
    :param seed: Seeds both the env's own duration sampling and this
        function's action-selection RNG (independent streams — Python's
        ``random`` for action choice, numpy's ``default_rng`` inside
        ``SymbolicJobShopEnv`` for duration sampling — seeding both with the
        same integer is safe and keeps the whole call reproducible).
    :param policy: Only ``"uniform"`` (uniform-random over enabled actions)
        is implemented.
    :return: A list of ``(location, Zone)`` pairs, one per decision point
        actually reached, in visitation order across all episodes. Terminal
        (all-done) locations are not included, since they offer no actions.
    """
    if policy != "uniform":
        raise ValueError(f"Unsupported policy: {policy!r} (only 'uniform' is implemented)")

    instance = _load_instance(instance_name)
    env = SymbolicJobShopEnv(instance, seed=seed)
    action_rng = random.Random(seed)

    zctx = _JobClockContext(instance)
    k = _job_clock_k(instance)

    out: List[Tuple[Location, Zone]] = []

    for _ in range(n_episodes):
        loc, actions = env.reset()
        Z = zctx.ctx.get_zero_federation()
        Z = apply_invariant(Z, zctx, env.invariant(loc))
        Z = apply_delay(Z, zctx, env.invariant(loc), env.rate(loc))

        while actions:
            out.append((loc, Zone(Z, zctx.clock_names, k)))

            action = action_rng.choice(actions)
            branches = _propagate_branches(Z, zctx, action, env)
            if not branches:
                # Every duration branch produced an empty zone (should not
                # happen for a reachable action, but mirrors the original
                # agent's defensive handling rather than crashing).
                break

            loc, actions, done, info = env.step(action["action_id"])
            sampled_d = info["sampled_d"]
            matching = [Zf for d, Zf in branches if d == sampled_d]
            Z = matching[0] if matching else branches[0][1]

            if done:
                break

    return out


# ---------------------------------------------------------------------------
# npz (de)serialization — plain arrays only, never a Federation
# ---------------------------------------------------------------------------

@dataclass
class CollectedView:
    """One zone's data as loaded back from :func:`from_npz`: plain arrays
    and reconstructed value objects (``HalfSpace``/``Fraction``/``DiGraph``),
    never a live pyudbm ``Federation``."""

    location: Location
    dbm_tensor: np.ndarray
    projection_stack: np.ndarray
    graph_image: np.ndarray
    h_rep: List[HalfSpace]
    v_rep: List[Tuple[Fraction, ...]]
    graph: "nx.DiGraph"
    clock_names: Tuple[str, ...]
    k: Tuple[int, ...]


def _object_array(items: Sequence) -> np.ndarray:
    """Build a proper 1-D object array holding one sub-array per entry.

    ``np.array(list_of_arrays, dtype=object)`` silently does the wrong thing
    when every sub-array has the *same* shape: instead of one Python object
    per row, numpy stacks them into a single dense object-dtype ndarray of
    scalars, discarding per-zone array boundaries (this bit precisely
    because dbm_tensor/projection_stack have a fixed shape for a whole
    collect_zones call). Explicit element-by-element assignment into a
    pre-sized empty object array avoids that pitfall regardless of whether
    shapes happen to be uniform or ragged.
    """
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def to_npz(
    zones: Sequence[Tuple[Location, Zone]],
    path: str,
    *,
    projection_res: int = 64,
    graph_res: int = 128,
) -> None:
    """Serialize ``(location, Zone)`` pairs to one ``.npz`` file.

    Only derived, plain-array data is stored: the three rasterized views,
    the half-space list (parallel ``a``/``b``/``strict`` arrays), the vertex
    list of the *clipped* zone (as exact numerator/denominator integer
    arrays — never a float approximation), and the constraint graph (as an
    edge-list array). Locations are stored as ``"|"``-joined strings. Zones
    whose clipped view raises :class:`UnboundedZone` or
    :class:`VertexEnumerationLimit` get an empty vertex list rather than
    failing the whole call.
    """
    locations = np.array(["|".join(loc) for loc, _ in zones], dtype=object)

    dbm_tensors, proj_stacks, graph_imgs = [], [], []
    h_rep_a, h_rep_b, h_rep_strict = [], [], []
    v_rep_num, v_rep_den = [], []
    graph_edges = []
    clock_names_list, k_list = [], []

    for _, zone in zones:
        dbm_tensors.append(dbm_tensor(zone))
        proj_stacks.append(projection_stack(zone, res=projection_res))
        graph_imgs.append(graph_image(zone, res=graph_res))

        hs_list = zone.h_rep()
        h_rep_a.append(np.array([hs.a for hs in hs_list], dtype=np.int64).reshape(len(hs_list), zone.d))
        h_rep_b.append(np.array([hs.b for hs in hs_list], dtype=np.int64))
        h_rep_strict.append(np.array([hs.strict for hs in hs_list], dtype=bool))

        try:
            verts = zone.clip_to_k_box().v_rep()
        except (UnboundedZone, VertexEnumerationLimit):
            verts = []
        num = np.array([[c.numerator for c in v] for v in verts], dtype=np.int64).reshape(len(verts), zone.d)
        den = np.array([[c.denominator for c in v] for v in verts], dtype=np.int64).reshape(len(verts), zone.d)
        v_rep_num.append(num)
        v_rep_den.append(den)

        g = zone.graph()
        edges = np.array(
            [[u, v, data["bound"], int(data["strict"])] for u, v, data in g.edges(data=True)],
            dtype=np.int64,
        ).reshape(-1, 4)
        graph_edges.append(edges)

        clock_names_list.append(np.array(zone.clock_names, dtype=object))
        k_list.append(np.array(zone.k, dtype=np.int64))

    np.savez(
        path,
        locations=locations,
        dbm_tensor=_object_array(dbm_tensors),
        projection_stack=_object_array(proj_stacks),
        graph_image=_object_array(graph_imgs),
        h_rep_a=_object_array(h_rep_a),
        h_rep_b=_object_array(h_rep_b),
        h_rep_strict=_object_array(h_rep_strict),
        v_rep_num=_object_array(v_rep_num),
        v_rep_den=_object_array(v_rep_den),
        graph_edges=_object_array(graph_edges),
        clock_names=_object_array(clock_names_list),
        k=_object_array(k_list),
    )


def from_npz(path: str) -> List[CollectedView]:
    """Load a ``.npz`` file written by :func:`to_npz` back into
    :class:`CollectedView` records (plain arrays and reconstructed value
    objects, never a live ``Federation``)."""
    data = np.load(path, allow_pickle=True)
    n = len(data["locations"])
    out: List[CollectedView] = []

    for idx in range(n):
        location = tuple(str(data["locations"][idx]).split("|"))
        clock_names = tuple(str(v) for v in data["clock_names"][idx])
        k = tuple(int(v) for v in data["k"][idx])
        d = len(clock_names)

        a_arr, b_arr, s_arr = data["h_rep_a"][idx], data["h_rep_b"][idx], data["h_rep_strict"][idx]
        h_rep = [
            HalfSpace(a=tuple(int(v) for v in a_arr[r]), b=int(b_arr[r]), strict=bool(s_arr[r]))
            for r in range(len(b_arr))
        ]

        num_arr, den_arr = data["v_rep_num"][idx], data["v_rep_den"][idx]
        v_rep = [
            tuple(Fraction(int(num_arr[r, c]), int(den_arr[r, c])) for c in range(d))
            for r in range(num_arr.shape[0])
        ]

        g = nx.DiGraph()
        for node in range(d + 1):
            g.add_node(node, clock_index=node)
        for row in data["graph_edges"][idx]:
            u, v, bound, strict = int(row[0]), int(row[1]), int(row[2]), bool(row[3])
            g.add_edge(u, v, bound=bound, strict=strict)

        out.append(
            CollectedView(
                location=location,
                dbm_tensor=data["dbm_tensor"][idx],
                projection_stack=data["projection_stack"][idx],
                graph_image=data["graph_image"][idx],
                h_rep=h_rep,
                v_rep=v_rep,
                graph=g,
                clock_names=clock_names,
                k=k,
            )
        )

    return out
