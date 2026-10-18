"""Board input shared by the pcbnew and the IPC plugin.

Each API has a thin adapter (read_board_pcbnew, read_board_ipc) that implements
BoardSource. read_board() then applies the same rules to both: which items the
selection stands for, which nets they join, how the copper connects and how
the stackup is modelled (see README, Board input).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, Union

try:
    from .connectivity import CONTACT_TOL_M, body_contacts, build_connectivity
    from .pcb_types import (
        PAD,
        VIA,
        WIRE,
        ZONE,
        BoardElement,
        CuLayer,
        DielectricInfo,
        PadElement,
        Point,
        ViaElement,
        WireElement,
        ZoneElement,
    )
except ImportError:
    from connectivity import CONTACT_TOL_M, body_contacts, build_connectivity
    from pcb_types import (
        PAD,
        VIA,
        WIRE,
        ZONE,
        BoardElement,
        CuLayer,
        DielectricInfo,
        PadElement,
        Point,
        ViaElement,
        WireElement,
        ZoneElement,
    )

log = logging.getLogger(__name__)


class BoardInputError(ValueError):
    """The board or its selection cannot be read; the message addresses the user."""


def layer_id(name: str) -> int | None:
    """KiCad layer ID of a copper layer name: F.Cu 0, B.Cu 2, In<n>.Cu 2n+2."""
    if name == "F.Cu":
        return 0
    if name == "B.Cu":
        return 2
    match = re.fullmatch(r"In(\d+)\.Cu", name)
    return 2 * int(match.group(1)) + 2 if match else None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class ArcCircle(NamedTuple):
    """Circle through the three points of an arc (m, rad)."""

    center: Point
    radius: float
    start_angle: float


def arc_geometry(
    start: Point, end: Point, mid: Point
) -> tuple[float, float, ArcCircle | None]:
    """Signed sweep (rad, positive = CCW), arc length and circle of an arc.

    For collinear points the result is (0, chord, None).
    """
    sx, sy = start
    mx, my = mid
    ex, ey = end
    D = 2 * (sx * (my - ey) + mx * (ey - sy) + ex * (sy - my))
    if abs(D) < 1e-30:
        return 0.0, math.hypot(ex - sx, ey - sy), None

    ux = (
        (sx * sx + sy * sy) * (my - ey)
        + (mx * mx + my * my) * (ey - sy)
        + (ex * ex + ey * ey) * (sy - my)
    ) / D
    uy = (
        (sx * sx + sy * sy) * (ex - mx)
        + (mx * mx + my * my) * (sx - ex)
        + (ex * ex + ey * ey) * (mx - sx)
    ) / D
    r = math.hypot(sx - ux, sy - uy)

    a_start = math.atan2(sy - uy, sx - ux)
    turn = 2 * math.pi
    to_mid = (math.atan2(my - uy, mx - ux) - a_start) % turn
    to_end = (math.atan2(ey - uy, ex - ux) - a_start) % turn
    sweep = to_end if to_mid < to_end else to_end - turn
    return sweep, abs(sweep) * r, ArcCircle((ux, uy), r, a_start)


def arc_points(
    center: Point,
    radius: float,
    start_angle: float,
    sweep: float,
    width: float = 0.0,
) -> list[Point]:
    """Points along an arc, excluding its start and including its end.

    The chord error stays below a quarter of the track width (1 % of the
    radius for width 0); 2 to 64 points.
    """
    cx, cy = center
    if radius < 1e-15 or abs(sweep) < 1e-12:
        angle = start_angle + sweep
        return [(cx + radius * math.cos(angle), cy + radius * math.sin(angle))]

    tol = max(width / 4, 1e-7) if width > 0 else radius * 0.01
    ratio = tol / radius
    if ratio >= 1.0:
        n_points = 2
    else:
        n_points = math.ceil(abs(sweep) / (2 * math.acos(1 - ratio)))
    n_points = max(2, min(n_points, 64))

    return [
        (
            cx + radius * math.cos(start_angle + sweep * i / n_points),
            cy + radius * math.sin(start_angle + sweep * i / n_points),
        )
        for i in range(1, n_points + 1)
    ]


def arc_midline(
    start: Point, end: Point, mid: Point, width: float = 0.0
) -> list[Point]:
    """Points from start through mid to end along an arc (see arc_points)."""
    sweep, _length, circle = arc_geometry(start, end, mid)
    if circle is None:
        return [start, end]
    return [
        start,
        *arc_points(circle.center, circle.radius, circle.start_angle, sweep, width),
    ]


def polygon_area(points: Sequence[Point]) -> float:
    """Area enclosed by a polygon (shoelace formula)."""
    n = len(points)
    twice = sum(
        points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
        for i in range(n)
    )
    return abs(twice) / 2


# ---------------------------------------------------------------------------
# Elements; the adapters only extract the values from their API
# ---------------------------------------------------------------------------


def wire(
    id: str,
    net: str,
    layers: list[int],
    start: Point,
    end: Point,
    width: float,
    mid: Point | None = None,
) -> WireElement:
    """A track from start to end; an arc if it passes through mid."""
    element: WireElement = {
        "type": WIRE,
        "id": id,
        "net": net,
        "layer": layers,
        "area": 0.0,
        "is_selected": False,
        "conn_start": [],
        "conn_end": [],
        "start": start,
        "end": end,
        "width": width,
        "length": math.dist(start, end),
        "midline_pts": [start, end],
    }
    if mid is not None:
        sweep, length, circle = arc_geometry(start, end, mid)
        element["mid"] = mid
        element["angle"] = sweep
        element["length"] = length
        element["midline_pts"] = arc_midline(start, end, mid, width)
        if circle is not None:
            element["radius"] = circle.radius
    element["area"] = width * element["length"]
    return element


def split_wire(w: WireElement, points: Sequence[Point]) -> list[WireElement]:
    """`w` cut at `points` on its midline into consecutive wires "<id>#<n>".

    A point within CONTACT_TOL_M of an end or of another cut adds no cut; `w`
    itself is returned if none remains.
    """
    start, end = w["start"], w["end"]
    mid = w.get("mid")
    sweep, _length, circle = (
        arc_geometry(start, end, mid) if mid is not None else (0.0, 0.0, None)
    )
    if circle is None:
        dx, dy = end[0] - start[0], end[1] - start[1]

        def fraction(p: Point) -> float:
            return ((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / (
                dx * dx + dy * dy
            )

        def at(f: float) -> Point:
            return (start[0] + f * dx, start[1] + f * dy)

    else:
        (cx, cy), radius, start_angle = circle
        direction = math.copysign(1.0, sweep)

        def fraction(p: Point) -> float:
            turned = (math.atan2(p[1] - cy, p[0] - cx) - start_angle) * direction
            return (turned % (2 * math.pi)) / abs(sweep)

        def at(f: float) -> Point:
            angle = start_angle + sweep * f
            return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    step = CONTACT_TOL_M / w["length"]
    cuts = [0.0]
    for f in sorted(map(fraction, points)):
        if cuts[-1] + step < f < 1.0 - step:
            cuts.append(f)
    if len(cuts) == 1:
        return [w]
    cuts.append(1.0)
    ends = [start, *map(at, cuts[1:-1]), end]
    parts = []
    for n, (a, b) in enumerate(zip(ends, ends[1:])):
        part_mid = at((cuts[n] + cuts[n + 1]) / 2) if circle else None
        part = wire(f"{w['id']}#{n}", w["net"], w["layer"], a, b, w["width"], part_mid)
        part["is_selected"] = w["is_selected"]
        parts.append(part)
    return parts


def via(
    id: str, net: str, layers: list[int], position: Point, width: float, drill: float
) -> ViaElement:
    return {
        "type": VIA,
        "id": id,
        "net": net,
        "layer": layers,
        "area": 0.0,
        "is_selected": False,
        "conn_start": [],
        "position": position,
        "width": width,
        "drill": drill,
    }


def pad(
    id: str,
    net: str,
    layers: list[int],
    position: Point,
    drill: float,
    outline: list[Point],
) -> PadElement:
    return {
        "type": PAD,
        "id": id,
        "net": net,
        "layer": layers,
        "area": polygon_area(outline),
        "is_selected": False,
        "conn_start": [],
        "position": position,
        "drill": drill,
        "outline": outline,
    }


def zone(
    id: str, net: str, layers: list[int], outline: list[Point], area: float
) -> ZoneElement:
    """A copper zone; area is its filled copper on all layers."""
    return {
        "type": ZONE,
        "id": id,
        "net": net,
        "layer": layers,
        "area": area,
        "is_selected": False,
        "conn_start": [],
        "outline": outline,
    }


# ---------------------------------------------------------------------------
# Stackup
# ---------------------------------------------------------------------------


class Copper(NamedTuple):
    name: str  # e.g. "F.Cu"
    thickness: float  # m


# A stackup lists its copper and dielectric layers from top to bottom.
StackupEntry = Union[Copper, DielectricInfo]


class Stackup(NamedTuple):
    """The stackup as an API describes it, top to bottom."""

    entries: list[StackupEntry]
    board_thickness: float | None = None  # if the API knows it apart from entries


_FR4 = {"epsilon_r": 4.3, "loss_tangent": 0.02}
_DEFAULT_BOARD_THICKNESS = 1.6e-3
_DEFAULT_COPPER = 35e-6


def dielectric(
    h: float, epsilon_r: float | None = None, loss_tangent: float | None = None
) -> DielectricInfo:
    """A core or prepreg; missing or zero material values default to FR4."""
    return {
        "h": h,
        "epsilon_r": epsilon_r or _FR4["epsilon_r"],
        "loss_tangent": loss_tangent or _FR4["loss_tangent"],
    }


def build_cu_stack(stackup: Sequence[StackupEntry]) -> dict[int, CuLayer]:
    """Copper layers by layer ID, with height (0 = top surface) and neighbours.

    A copper layer with an unknown name or no positive thickness is skipped.
    """

    def neighbour(i: int) -> DielectricInfo | None:
        entry = stackup[i] if 0 <= i < len(stackup) else None
        return None if entry is None or isinstance(entry, Copper) else entry

    cu_stack: dict[int, CuLayer] = {}
    height = 0.0
    for i, entry in enumerate(stackup):
        if not isinstance(entry, Copper):
            height += entry["h"]
            continue
        layer = layer_id(entry.name)
        if layer is None or not entry.thickness > 0:
            log.warning("Copper layer %r (%s m) skipped", entry.name, entry.thickness)
        else:
            above, below = neighbour(i - 1), neighbour(i + 1)
            cu_stack[layer] = {
                "thickness": entry.thickness,
                "name": entry.name,
                "abs_height": height,
                "die_above": above,
                "die_below": below,
                "model": ("R only", "Microstrip", "Stripline")[
                    (above is not None) + (below is not None)
                ],
                "gap": None,
            }
        height += entry.thickness
    return cu_stack


def _cu_stack(stackup: Stackup) -> tuple[dict[int, CuLayer], str | None]:
    """The copper layers of `stackup`; without any usable one, those of an
    assumed two-layer FR4 board, with a note saying so."""
    cu_stack = build_cu_stack(stackup.entries)
    if cu_stack:
        return cu_stack, None
    total = stackup.board_thickness or _DEFAULT_BOARD_THICKNESS
    assumed: list[StackupEntry] = [
        Copper("F.Cu", _DEFAULT_COPPER),
        dielectric(total - 2 * _DEFAULT_COPPER),
        Copper("B.Cu", _DEFAULT_COPPER),
    ]
    note = (
        f"No usable stackup was found, so a two-layer FR4 board of {total * 1e3:g} mm "
        f"with {_DEFAULT_COPPER * 1e6:g} µm copper is assumed. Define the stackup in "
        "Board Setup (for the pcbnew plugin, then save the board)."
    )
    log.warning("%s", note)
    return build_cu_stack(assumed), note


def slot_drill(x: float, y: float) -> float:
    """Diameter of the round hole whose wall is as long as that of an oval
    (slot) hole of x by y, so that its plated barrel has the same resistance."""
    short, long = sorted((x, y))
    return short + 2 * (long - short) / math.pi


# ---------------------------------------------------------------------------
# Selection and nets
# ---------------------------------------------------------------------------


class Selected(NamedTuple):
    """A selected board item."""

    id: str
    net: str = ""  # net of a track, via or pad; empty for anything else
    pads: tuple[Selected, ...] = ()  # a footprint's pads


class NetTie(NamedTuple):
    """A Net Tie footprint; each group lists the nets its copper joins."""

    ref: str
    groups: tuple[frozenset[str], ...]
    footprint: Any  # the adapter's footprint object


def net_tie(
    ref: str,
    pad_groups: Iterable[Iterable[str]],
    pad_nets: Mapping[str, str],
    footprint: Any,
) -> NetTie:
    """A Net Tie from groups of pad numbers (entries may be comma-separated)."""
    groups: dict[frozenset[str], None] = {}  # ordered set: groups may repeat
    for group in pad_groups:
        numbers = {n.strip() for entry in group for n in entry.split(",")}
        nets = frozenset(pad_nets[n] for n in numbers if pad_nets.get(n))
        if nets:
            groups[nets] = None
    return NetTie(ref, tuple(groups), footprint)


class BoardSource(Protocol):
    """Board access through one of KiCad's APIs."""

    def selection(self) -> list[Selected]: ...

    def net_ties(self) -> list[NetTie]: ...

    def copper(
        self, nets: AbstractSet[str], net_ties: Sequence[NetTie]
    ) -> Iterable[BoardElement]:
        """Tracks, vias, pads and copper zones of `nets`, and the copper of
        `net_ties`."""
        ...

    def unmodelled(self) -> list[str]:
        """The kind of each copper item of the nets and Net Ties of the last
        copper() call that it left out, e.g. a copper shape with a net."""
        ...

    def stackup(self) -> Stackup: ...


@dataclass
class Board:
    elements: dict[str, BoardElement]  # by ID, connected (conn_start/conn_end)
    cu_stack: dict[int, CuLayer]
    nets: list[str]
    net_ties: list[str]  # references of the Net Ties joining the nets
    notes: list[str]  # simplifications the user should know of


def split_at_body_contacts(elements: dict[str, BoardElement]) -> None:
    """Cut the wires of `elements` where something touches their body (see
    connectivity.body_contacts), so that every contact is at a wire end.

    A cut makes new wire ends, which may lie on other wires; cutting repeats
    until no wire is cut any more. Each round but the last cuts at least one
    wire, so the number of elements bounds the rounds.
    """
    for _round in range(len(elements) + 1):
        cuts = [(w, split_wire(w, points)) for w, points in body_contacts(elements)]
        cuts = [(w, parts) for w, parts in cuts if len(parts) > 1]
        if not cuts:
            return
        for w, parts in cuts:
            del elements[w["id"]]
            elements.update((part["id"], part) for part in parts)
    log.warning("Cutting wires at body contacts did not settle; some remain uncut")


def _selected_items(selection: list[Selected]) -> list[Selected]:
    """The selected tracks, vias and pads; a footprint stands for its pads.
    Anything else (zones, text, graphics) is no measurement point and left out."""
    items = [item for s in selection for item in (s.pads or (s,)) if item.net]
    if not items:
        raise BoardInputError(
            "Nothing to measure is selected. Select in KiCad what to measure "
            "across:\n"
            "  - two pads or vias\n"
            "  - a footprint with two pads; as a Net Tie (e.g. a PCB coil)\n"
            "    it may join two nets\n"
            "  - a single track or a whole run of tracks\n"
            "These can be combined, e.g. a coil footprint with its feed tracks."
        )
    return items


def _joined_nets(net: str, net_ties: list[NetTie]) -> tuple[set[str], list[NetTie]]:
    """The nets that Net Ties join to `net`, transitively, and those Net Ties."""
    nets = {net}
    grown = True
    while grown:
        grown = False
        for group in (g for tie in net_ties for g in tie.groups):
            if group & nets and not group <= nets:
                nets |= group
                grown = True
    return nets, [t for t in net_ties if any(g & nets for g in t.groups)]


def read_board(source: BoardSource) -> Board:
    """The connected copper of the nets of the selected items.

    Which points of the selection are measured between is decided on the
    connected nodes, see parasitic.selection_ends.

    Raises:
        BoardInputError: the selection holds no track, via or pad, holds items
            of nets not joined by Net Ties, or items on no enabled copper layer
    """
    items = _selected_items(source.selection())
    nets, net_ties = _joined_nets(items[0].net, source.net_ties())
    if any(item.net not in nets for item in items):
        found = ", ".join(sorted({item.net for item in items}))
        raise BoardInputError(
            "Selected elements must belong to the same net or be connected by "
            f"a Net Tie (found: {found})."
        )
    log.info("Nets %s, joined by %s", sorted(nets), [t.ref for t in net_ties])

    selected = {item.id for item in items}
    elements: dict[str, BoardElement] = {}
    zero_length: set[str] = set()  # no resistance, and both ends coincide
    for element in source.copper(nets, net_ties):
        if element["type"] == WIRE and element["length"] == 0:
            zero_length.add(element["id"])
            continue
        element["is_selected"] = element["id"] in selected
        elements[element["id"]] = element
    on_copper = {id for id, e in elements.items() if e["layer"]}
    unusable = selected - zero_length - on_copper
    if unusable:
        raise BoardInputError(
            f"{len(unusable)} selected item(s) lie on no enabled copper layer, "
            "so they cannot be measured. Deselect them in KiCad, or enable "
            "their copper layer in Board Setup."
        )
    split_at_body_contacts(elements)
    log.info("Read %d elements", len(elements))
    build_connectivity(elements)

    notes = []
    unmodelled = Counter(source.unmodelled())
    if unmodelled:
        kinds = ", ".join(
            f"{kind} ×{n}" if n > 1 else kind for kind, n in sorted(unmodelled.items())
        )
        notes.append(
            f"{sum(unmodelled.values())} copper shape(s) of these nets ({kinds}) "
            "are not part of the model, so current through them is not counted."
        )
    cu_stack, stackup_note = _cu_stack(source.stackup())
    if stackup_note:
        notes.append(stackup_note)
    nets_read, ties = sorted(nets), [t.ref for t in net_ties]
    return Board(elements, cu_stack, nets_read, ties, notes)
