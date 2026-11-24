"""Plots of 2-clock partitions (matplotlib, imported lazily).

Each block is drawn as its closure clipped to [0, box]²: the box is clipped by every
constraint of `zone.constraints()` (strict treated as non-strict) in exact rational
arithmetic, and the result is drawn by dimension: polygon, segment or point.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from .model import ZERO
from .zones import Zone

Point = tuple[Fraction, Fraction]

FILL = "#cfe0f3"
HIGHLIGHT = "#f4a582"
LINE = "#2b5d8a"
LINE_HIGHLIGHT = "#c0392b"


def _clip(poly: list[Point], a: tuple[int, int], c: int) -> list[Point]:
    """Sutherland–Hodgman: clip a convex polygon to the half-plane a·p <= c."""
    def val(p: Point) -> Fraction:
        return a[0] * p[0] + a[1] * p[1] - c

    out: list[Point] = []
    n = len(poly)
    for k in range(n):
        p, q = poly[k], poly[(k + 1) % n]
        vp, vq = val(p), val(q)
        if vp <= 0:
            out.append(p)
        if (vp < 0 < vq) or (vq < 0 < vp):
            t = vp / (vp - vq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _hull(points: Iterable[Point]) -> list[Point]:
    """Convex hull (monotone chain, collinear points dropped): 0, 1, 2 or >= 3 vertices."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o: Point, a: Point, b: Point) -> Fraction:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def closure_polygon(zone: Zone, clocks: Sequence[str] = ("x", "y"), box: float = 3.0) -> list[Point]:
    """Vertices of the closure of `zone` clipped to [0, box]², in convex order.

    0 vertices: empty; 1: a point; 2: a segment; >= 3: a polygon.
    """
    if zone.is_empty:
        return []
    b = Fraction(box)
    poly: list[Point] = [(Fraction(0), Fraction(0)), (b, Fraction(0)), (b, b), (Fraction(0), b)]
    axis = {clocks[0]: 0, clocks[1]: 1}
    for con in zone.constraints():
        a = [0, 0]
        if con.left != ZERO:
            a[axis[con.left]] += 1
        if con.right != ZERO:
            a[axis[con.right]] -= 1
        poly = _clip(poly, (a[0], a[1]), con.bound)
        if not poly:
            return []
    return _hull(poly)


def _centroid(vertices: list[Point]) -> tuple[float, float]:
    if len(vertices) < 3:
        return (float(sum(p[0] for p in vertices) / len(vertices)),
                float(sum(p[1] for p in vertices) / len(vertices)))
    area = cx = cy = Fraction(0)
    for k in range(len(vertices)):
        (x0, y0), (x1, y1) = vertices[k], vertices[(k + 1) % len(vertices)]
        w = x0 * y1 - x1 * y0
        area += w
        cx += (x0 + x1) * w
        cy += (y0 + y1) * w
    return (float(cx / (3 * area)), float(cy / (3 * area)))


def _check_clocks(blocks: list[Zone], clocks: Sequence[str]) -> None:
    if len(clocks) != 2:
        raise ValueError(f"plot_partition needs exactly 2 clocks, got {list(clocks)}")
    for z in blocks:
        if len(z.ctx.clocks) != 2:
            raise ValueError(f"plot_partition only supports 2-clock zones, got clocks {list(z.ctx.clocks)}")
        if set(z.ctx.clocks) != set(clocks):
            raise ValueError(f"clocks {list(clocks)} do not match the zones' clocks {list(z.ctx.clocks)}")


def plot_partition(
    blocks: Iterable[Zone],
    ax: Any,
    *,
    clocks: Sequence[str] = ("x", "y"),
    box: float = 3.0,
    labels: Optional[Mapping[Zone, str]] = None,
    highlight: Optional[Iterable[Zone]] = None,
    title: Optional[str] = None,
) -> None:
    """Draw a 2-clock partition on a matplotlib Axes."""
    from matplotlib.patches import Polygon, Rectangle

    blocks = sorted(blocks, key=Zone.key)
    _check_clocks(blocks, clocks)
    labels = labels or {}
    hl = set(highlight or ())
    # Polygons first, then segments, then points, so lower-dimensional blocks stay visible.
    shapes = []
    for z in blocks:
        vertices = closure_polygon(z, clocks, box)
        if vertices:
            shapes.append((min(len(vertices), 3), z, vertices))  # 3 polygon, 2 segment, 1 point
    for _, z, vertices in sorted(shapes, key=lambda s: s[0], reverse=True):
        pts = [(float(p[0]), float(p[1])) for p in vertices]
        on = z in hl
        if len(pts) >= 3:
            ax.add_patch(Polygon(pts, closed=True, facecolor=HIGHLIGHT if on else FILL,
                                 edgecolor="black", linewidth=0.5, zorder=1))
        elif len(pts) == 2:
            ax.plot(*zip(*pts), color=LINE_HIGHLIGHT if on else LINE, linewidth=3,
                    solid_capstyle="round", zorder=2)
        else:
            ax.plot(*pts[0], marker="o", markersize=6, color=LINE_HIGHLIGHT if on else LINE, zorder=3)
        if z in labels:
            cx, cy = _centroid(vertices)
            ax.annotate(labels[z], (cx, cy), fontsize=5, ha="center", va="center", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax.add_patch(Rectangle((0, 0), 2, 2, fill=False, linestyle="--", edgecolor="grey", linewidth=1, zorder=5))
    ax.set_xlim(-0.1, box + 0.1)
    ax.set_ylim(-0.1, box + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel(clocks[0])
    ax.set_ylabel(clocks[1])
    if title:
        ax.set_title(title)


def plot_store(
    store: Mapping[str, Iterable[Zone]],
    path: Union[str, Path],
    *,
    labels: Optional[Mapping[str, Mapping[Zone, str]]] = None,
    highlight: Optional[Mapping[str, Iterable[Zone]]] = None,
    **kwargs: Any,
) -> None:
    """One subplot per location, in a row, saved as PNG.

    `labels` and `highlight` are keyed by location first, because the same zone can be a
    block at several locations.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    locations = list(store)
    fig, axes = plt.subplots(1, len(locations), figsize=(4.2 * len(locations), 4.4), squeeze=False)
    for ax, l in zip(axes[0], locations):
        plot_partition(store[l], ax, labels=(labels or {}).get(l), highlight=(highlight or {}).get(l),
                       title=f"{l} ({len(list(store[l]))} blocks)", **kwargs)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
