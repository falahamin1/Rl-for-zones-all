"""Benchmark: wall-clock cost of each zoneviews raster view, and how often
exact vertex enumeration becomes impractical (``VertexEnumerationLimit``),
across job-shop instances of increasing size.

No torch, no gym, no plotting — this prints a table to stdout and reports
whatever the numbers show; it does not recommend a direction.

Usage::

    python3 -m zoneviews.benchmarks.view_bench
"""
from __future__ import annotations

import signal
import time
from typing import Dict, List

import zoneviews  # noqa: F401  (pyudbm import-path bootstrap)
from zoneviews.collect import collect_zones
from zoneviews.raster import dbm_tensor, graph_image, projection_stack
from zoneviews.zone import UnboundedZone, VertexEnumerationLimit

INSTANCES = ["PJS_01", "PJS_04", "PJS_05", "LJS_01"]
N_EPISODES = 15
SEED = 12345

#: Per-zone wall-clock budget for v_rep(), enforced via SIGALRM. The
#: combinatorial cap (C(m, d) > 200_000 -> VertexEnumerationLimit) is a
#: correctness/tractability boundary the library itself enforces; this is a
#: *separate*, benchmark-only observation that plenty of zones sit well
#: under that cap yet still take a long time (each surviving combination is
#: an exact Fraction Gauss-Jordan solve), which is exactly the kind of
#: practicality boundary this benchmark exists to surface.
V_REP_TIMEOUT_S = 2.0


class _VRepTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _VRepTimeout()


def _bench_instance(name: str) -> Dict:
    t_collect0 = time.perf_counter()
    zones = collect_zones(name, n_episodes=N_EPISODES, seed=SEED)
    collect_wall_s = time.perf_counter() - t_collect0

    n = len(zones)
    d_values = sorted({z.d for _, z in zones})

    view_time_s = {"dbm_tensor": 0.0, "projection_stack": 0.0, "graph_image": 0.0, "v_rep": 0.0}
    view_count = {"dbm_tensor": 0, "projection_stack": 0, "graph_image": 0, "v_rep": 0}
    vertex_limit_failures = 0
    unbounded_failures = 0
    v_rep_timeouts = 0
    max_facet_count = 0

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        for _, zone in zones:
            t0 = time.perf_counter()
            dbm_tensor(zone)
            view_time_s["dbm_tensor"] += time.perf_counter() - t0
            view_count["dbm_tensor"] += 1

            t0 = time.perf_counter()
            projection_stack(zone)
            view_time_s["projection_stack"] += time.perf_counter() - t0
            view_count["projection_stack"] += 1

            t0 = time.perf_counter()
            graph_image(zone)
            view_time_s["graph_image"] += time.perf_counter() - t0
            view_count["graph_image"] += 1

            clipped = zone.clip_to_k_box()
            max_facet_count = max(max_facet_count, clipped.facet_count())

            t0 = time.perf_counter()
            signal.setitimer(signal.ITIMER_REAL, V_REP_TIMEOUT_S)
            try:
                clipped.v_rep()
                view_time_s["v_rep"] += time.perf_counter() - t0
                view_count["v_rep"] += 1
            except VertexEnumerationLimit:
                vertex_limit_failures += 1
            except UnboundedZone:
                # Should not happen: clip_to_k_box() always produces a
                # bounded zone. Counted separately so it stands out if it
                # ever does.
                unbounded_failures += 1
            except _VRepTimeout:
                v_rep_timeouts += 1
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    mean_us = {
        k: (view_time_s[k] / view_count[k] * 1e6 if view_count[k] else float("nan")) for k in view_time_s
    }

    return {
        "instance": name,
        "n_zones": n,
        "d_values": d_values,
        "collect_wall_s": collect_wall_s,
        "mean_us": mean_us,
        "vertex_limit_failures": vertex_limit_failures,
        "unbounded_failures": unbounded_failures,
        "v_rep_timeouts": v_rep_timeouts,
        "max_facet_count": max_facet_count,
        "vertex_attempts": n,
    }


def _print_table(rows: List[Dict]) -> None:
    columns = [
        ("instance", 10),
        ("n_zones", 8),
        ("d", 8),
        ("max_facets", 11),
        ("dbm_us", 9),
        ("proj_us", 9),
        ("graph_us", 9),
        ("vrep_us_ok", 11),
        ("vlim_rate", 10),
        ("timeout_rate", 13),
        ("collect_s", 10),
    ]
    header = "".join(name.ljust(w) for name, w in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        d_str = ",".join(str(x) for x in row["d_values"])
        vlim_rate = row["vertex_limit_failures"] / row["vertex_attempts"] if row["vertex_attempts"] else 0.0
        timeout_rate = row["v_rep_timeouts"] / row["vertex_attempts"] if row["vertex_attempts"] else 0.0
        cells = [
            row["instance"],
            str(row["n_zones"]),
            d_str,
            str(row["max_facet_count"]),
            f"{row['mean_us']['dbm_tensor']:.1f}",
            f"{row['mean_us']['projection_stack']:.1f}",
            f"{row['mean_us']['graph_image']:.1f}",
            f"{row['mean_us']['v_rep']:.1f}",
            f"{vlim_rate * 100:.1f}%",
            f"{timeout_rate * 100:.1f}%",
            f"{row['collect_wall_s']:.2f}",
        ]
        print("".join(cell.ljust(w) for cell, (_, w) in zip(cells, columns)))


def main() -> None:
    rows = []
    for name in INSTANCES:
        print(f"Running {name} ({N_EPISODES} episodes, seed={SEED})...", flush=True)
        t0 = time.perf_counter()
        row = _bench_instance(name)
        rows.append(row)
        print(f"  done in {time.perf_counter() - t0:.1f}s wall", flush=True)
        if row["unbounded_failures"]:
            print(f"  ANOMALY: {row['unbounded_failures']} clipped zones raised UnboundedZone (should be 0)")

    print()
    print(f"(v_rep per-zone timeout budget: {V_REP_TIMEOUT_S}s; vrep_us_ok is the mean over only the "
          f"attempts that finished within budget and under the C(m,d)>200_000 combinatorial cap)")
    _print_table(rows)


if __name__ == "__main__":
    main()
