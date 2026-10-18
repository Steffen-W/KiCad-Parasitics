"""Which copper elements touch, found from their geometry."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, NamedTuple

try:
    from .pcb_types import PAD, WIRE, ZONE, BoardElement, Point, WireElement
except ImportError:
    from pcb_types import PAD, WIRE, ZONE, BoardElement, Point, WireElement

BBox = tuple[float, float, float, float]
End = Literal["conn_start", "conn_end"]

# Extra distance at which two contact points still count as touching.
CONTACT_TOL_M = 1e-9


def _bboxes_overlap(a: BBox, b: BBox) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _bbox(points: Sequence[Point], margin: float = 0.0) -> BBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


class _BBoxGrid:
    """Uniform grid over bounding boxes for overlap queries.

    The cell size is the median box extent, so most boxes occupy one to four
    cells. A box spanning more cells than there are boxes is cheaper to test
    on every query than to enter into the grid, so it is kept aside. Results
    come in insertion order, so callers visit candidates in the same order as
    a scan over all boxes.
    """

    def __init__(self, boxes: list[BBox]) -> None:
        self._boxes = boxes
        extents = sorted(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
        # Any positive size is correct; it only affects speed.
        self._cell = (extents[len(extents) // 2] if extents else 0.0) or 1e-3
        self._cells: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        self._large: list[int] = []
        for i, box in enumerate(boxes):
            xs, ys = self._spans(box)
            if len(xs) * len(ys) > len(boxes):
                self._large.append(i)
                continue
            for x in xs:
                for y in ys:
                    self._cells[x, y].append(i)

    def _spans(self, box: BBox) -> tuple[range, range]:
        c = self._cell
        return (
            range(math.floor(box[0] / c), math.floor(box[2] / c) + 1),
            range(math.floor(box[1] / c), math.floor(box[3] / c) + 1),
        )

    def overlapping(self, box: BBox) -> list[int]:
        """Indices of all boxes overlapping `box`, in insertion order."""
        xs, ys = self._spans(box)
        if len(xs) * len(ys) <= len(self._cells):
            keys = [(x, y) for x in xs for y in ys]
        else:  # box covers more cells than are occupied: scan those instead
            keys = [k for k in self._cells if k[0] in xs and k[1] in ys]
        found = {i for key in keys for i in self._cells.get(key, ())}
        found.update(self._large)
        return sorted(i for i in found if _bboxes_overlap(self._boxes[i], box))


def _nearest_on_segment(p: Point, a: Point, b: Point) -> Point:
    dx, dy = b[0] - a[0], b[1] - a[1]
    len_sq = dx * dx + dy * dy
    t = 0.0
    if len_sq > 0:
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq))
    return (a[0] + t * dx, a[1] + t * dy)


def _dist_sq_to_segment(p: Point, a: Point, b: Point) -> float:
    q = _nearest_on_segment(p, a, b)
    ex, ey = p[0] - q[0], p[1] - q[1]
    return ex * ex + ey * ey


def _segments_dist_sq(a: Point, b: Point, c: Point, d: Point) -> float:
    def side(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    if side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0:
        return 0.0  # the segments cross
    return min(
        _dist_sq_to_segment(a, c, d),
        _dist_sq_to_segment(b, c, d),
        _dist_sq_to_segment(c, a, b),
        _dist_sq_to_segment(d, a, b),
    )


def _segments(line: Sequence[Point]) -> list[tuple[Point, Point]]:
    return list(zip(line, line[1:]))


def _edges(polygon: Sequence[Point]) -> list[tuple[Point, Point]]:
    return _segments([*polygon, polygon[0]])


def _inside(p: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting; the polygon needs no closing point."""
    x, y = p
    inside = False
    for (x1, y1), (x2, y2) in _edges(polygon):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _nearest_on_polyline(p: Point, line: Sequence[Point]) -> Point:
    return min(
        (_nearest_on_segment(p, a, b) for a, b in _segments(line)),
        key=lambda q: math.dist(p, q),
    )


def _near_polyline(p: Point, line: Sequence[Point], margin: float) -> bool:
    return any(
        _dist_sq_to_segment(p, a, b) <= margin * margin for a, b in _segments(line)
    )


def _point_touches_area(p: Point, outline: Sequence[Point], margin: float) -> bool:
    return _inside(p, outline) or any(
        _dist_sq_to_segment(p, a, b) <= margin * margin for a, b in _edges(outline)
    )


def _line_touches_area(
    line: Sequence[Point], outline: Sequence[Point], margin: float
) -> bool:
    return _inside(line[0], outline) or any(
        _segments_dist_sq(a, b, c, d) <= margin * margin
        for a, b in _segments(line)
        for c, d in _edges(outline)
    )


def _areas_touch(p: Sequence[Point], q: Sequence[Point]) -> bool:
    return _inside(q[0], p) or _line_touches_area([*p, p[0]], q, 0.0)


class _Contact(NamedTuple):
    """A point where an element can touch others: a wire end, a via or pad centre."""

    position: Point
    id: str
    end: End  # list of the element that records what touches here
    radius: float  # half the wire width or via diameter; 0 for a pad


def _contacts(e: BoardElement) -> list[_Contact]:
    if e["type"] == WIRE:
        r = e["width"] / 2
        return [
            _Contact(e["start"], e["id"], "conn_start", r),
            _Contact(e["end"], e["id"], "conn_end", r),
        ]
    if e["type"] == ZONE:
        return []
    return [
        _Contact(
            e["position"],
            e["id"],
            "conn_start",
            e["width"] / 2 if e["type"] != PAD else 0.0,
        )
    ]


def _nearer_end(w: WireElement, p: Point) -> End:
    return (
        "conn_start"
        if math.dist(p, w["start"]) <= math.dist(p, w["end"])
        else "conn_end"
    )


def _conn(e: BoardElement, end: End) -> list[str]:
    if e["type"] == WIRE and end == "conn_end":
        return e["conn_end"]
    return e["conn_start"]


def build_connectivity(elements: dict[str, BoardElement]) -> None:
    """Fill conn_start/conn_end of every element with the elements it touches.

    Elements touch only on a shared layer, when
    - two contact points (see _Contact) overlap,
    - a contact point overlaps a wire's body,
    - a contact point overlaps a pad or zone outline,
    - a wire's body overlaps a pad or zone outline, or
    - two pad or zone outlines overlap.
    Contact with a wire's body is recorded at the wire end nearer to the
    contact point or the outline's centre; board_model.split_at_body_contacts
    cuts wires beforehand, so this remains for wires crossing a zone.
    """
    layers = {i: set(e["layer"]) for i, e in elements.items()}
    contacts = [c for e in elements.values() for c in _contacts(e)]
    wires = [e for e in elements.values() if e["type"] == WIRE]
    areas = [
        e
        for e in elements.values()
        if (e["type"] == PAD or e["type"] == ZONE) and len(e["outline"]) >= 3
    ]
    contact_grid = _BBoxGrid([_bbox([c.position], c.radius) for c in contacts])
    area_grid = _BBoxGrid([_bbox(a["outline"]) for a in areas])

    def unlinked(a: str, a_end: End, b: str) -> bool:
        """Whether a and b share a layer but are not yet linked at a's end."""
        return (
            a != b
            and bool(layers[a] & layers[b])
            and b not in _conn(elements[a], a_end)
        )

    def link(a: str, a_end: End, b: str, b_end: End) -> None:
        for x, x_end, y in ((a, a_end, b), (b, b_end, a)):
            conn = _conn(elements[x], x_end)
            if y not in conn:
                conn.append(y)

    for i, c in enumerate(contacts):
        # Twice the tolerance, so rounding cannot drop a candidate.
        query = _bbox([c.position], c.radius + 2 * CONTACT_TOL_M)
        for j in contact_grid.overlapping(query):
            d = contacts[j]
            reach = c.radius + d.radius + CONTACT_TOL_M
            if (
                j > i
                and unlinked(c.id, c.end, d.id)
                and math.dist(c.position, d.position) <= reach
            ):
                link(c.id, c.end, d.id, d.end)

    # Areas before wire bodies: a wire ending in an area is linked at that end.
    for i, a in enumerate(areas):
        for k in contact_grid.overlapping(_bbox(a["outline"])):
            c = contacts[k]
            if unlinked(c.id, c.end, a["id"]) and _point_touches_area(
                c.position, a["outline"], c.radius
            ):
                link(c.id, c.end, a["id"], "conn_start")
        for k in area_grid.overlapping(_bbox(a["outline"])):
            b = areas[k]
            if (
                k > i
                and unlinked(a["id"], "conn_start", b["id"])
                and _areas_touch(a["outline"], b["outline"])
            ):
                link(a["id"], "conn_start", b["id"], "conn_start")

    for w in wires:
        r = w["width"] / 2
        line = w["midline_pts"]
        for k in contact_grid.overlapping(_bbox(line, r)):
            c = contacts[k]
            if unlinked(c.id, c.end, w["id"]) and _near_polyline(
                c.position, line, r + c.radius
            ):
                link(c.id, c.end, w["id"], _nearer_end(w, c.position))
        for k in area_grid.overlapping(_bbox(line, r)):
            a = areas[k]
            if (
                a["id"] not in w["conn_end"]
                and unlinked(w["id"], "conn_start", a["id"])
                and _line_touches_area(line, a["outline"], r)
            ):
                box = _bbox(a["outline"])
                centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                link(w["id"], _nearer_end(w, centre), a["id"], "conn_start")


def body_contacts(
    elements: dict[str, BoardElement],
) -> list[tuple[WireElement, list[Point]]]:
    """The wires touched on their body away from both ends, with the points on
    their midline where that happens.

    A pad whose outline overlaps the body touches at its centre's projection,
    any other contact point (see _Contact) where it lies within reach of the
    midline. A pad that covers an end of the wire, or a contact point that
    reaches one, makes an end contact and is not listed, nor is a point within
    CONTACT_TOL_M of an end.
    """
    layers = {i: set(e["layer"]) for i, e in elements.items()}
    pads = [e for e in elements.values() if e["type"] == PAD and len(e["outline"]) >= 3]
    pad_ids = {p["id"] for p in pads}
    contacts = [
        c for e in elements.values() if e["id"] not in pad_ids for c in _contacts(e)
    ]
    contact_grid = _BBoxGrid([_bbox([c.position], c.radius) for c in contacts])
    pad_grid = _BBoxGrid([_bbox(p["outline"]) for p in pads])
    touched = []
    for w in (e for e in elements.values() if e["type"] == WIRE):
        r = w["width"] / 2
        line = w["midline_pts"]
        ends = (w["start"], w["end"])
        points = []
        for k in contact_grid.overlapping(_bbox(line, r)):
            c = contacts[k]
            if c.id == w["id"] or not layers[c.id] & layers[w["id"]]:
                continue
            reach = r + c.radius + CONTACT_TOL_M
            q = _nearest_on_polyline(c.position, line)
            if math.dist(c.position, q) <= reach and all(
                math.dist(c.position, end) > reach for end in ends
            ):
                points.append(q)
        for k in pad_grid.overlapping(_bbox(line, r)):
            p = pads[k]
            if (
                layers[p["id"]] & layers[w["id"]]
                and _line_touches_area(line, p["outline"], r)
                and not any(_point_touches_area(end, p["outline"], r) for end in ends)
            ):
                points.append(_nearest_on_polyline(p["position"], line))
        points = [
            q for q in points if min(math.dist(q, e) for e in ends) > CONTACT_TOL_M
        ]
        if points:
            touched.append((w, points))
    return touched
