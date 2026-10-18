"""Which points of a selection are measured between (see selection_ends)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from analysis_progress import no_progress
from board_model import pad, via, wire
from connect_nets import Connect_Nets
from connectivity import build_connectivity
from get_parasitic import extract_network
from parasitic import _farthest_pair, analyze_pcb_parasitic, selection_ends
from pcb_types import BoardElement, Point

MM = 1e-3


def _w(id: str, start: Point, end: Point, layer: int = 0) -> BoardElement:
    return wire(id, "N", [layer], (start[0] * MM, start[1] * MM),
                (end[0] * MM, end[1] * MM), 0.2 * MM)  # fmt: skip


def _via(id: str, at: Point) -> BoardElement:
    return via(id, "N", [0, 2], (at[0] * MM, at[1] * MM), 0.6 * MM, 0.3 * MM)


def _pad(id: str, at: Point) -> BoardElement:
    x, y, r = at[0] * MM, at[1] * MM, 0.5 * MM
    square = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
    return pad(id, "N", [0], (x, y), 0.0, square)


def _connected(elements: list[BoardElement], selected: str) -> dict[str, Any]:
    """Copies of `elements` as Connect_Nets returns them, `selected` IDs marked."""
    data = {e["id"]: copy.deepcopy(e) for e in elements}
    build_connectivity(data)
    for e in data.values():
        e["is_selected"] = e["id"] in selected.split()
    return Connect_Nets(data)


def _measured(
    elements: list[BoardElement], selected: str, cu_stack: dict[int, Any]
) -> tuple[int, set[Point]]:
    """Number of free ends of the `selected` IDs and the two measured points (mm)."""
    data = _connected(elements, selected)
    ends = selection_ends(data)
    pair = _farthest_pair(extract_network(data, cu_stack)["graph"], ends, no_progress)
    measured = {
        (round(end.position[0] / MM, 6), round(end.position[1] / MM, 6))
        for end, _layer in pair or ()
    }
    return len(ends), measured


_CHAIN = [_w("w1", (0, 0), (5, 0)), _w("w2", (5, 0), (10, 0))]
_THROUGH_VIA = [
    _w("top", (0, 0), (5, 0)),
    _via("v", (5, 0)),
    _w("bottom", (5, 0), (10, 0), layer=2),
]
_TEE = [
    _w("left", (0, 0), (10, 0)),
    _w("right", (10, 0), (20, 0)),
    _w("stub", (10, 0), (10, 3)),
]
_LOOP = [
    _w("a", (0, 0), (5, 0)),
    _w("b", (5, 0), (5, 5)),
    _w("c", (5, 5), (0, 5)),
    _w("d", (0, 5), (0, 0)),
]
_PADS = [_pad("p1", (0, 0)), _w("w", (0, 0), (5, 0)), _pad("p2", (5, 0))]
_GAP = [*_CHAIN, _w("w3", (10, 0), (15, 0))]
_ISLANDS = [_w("w1", (0, 0), (5, 0)), _w("w3", (10, 0), (15, 0))]


@pytest.mark.parametrize(
    ("elements", "selected", "ends", "measured"),
    [
        (_CHAIN, "w2", 2, {(5, 0), (10, 0)}),
        (_THROUGH_VIA, "top v bottom", 2, {(0, 0), (10, 0)}),
        (_THROUGH_VIA, "top bottom", 2, {(0, 0), (10, 0)}),
        (_TEE, "left right stub", 3, {(0, 0), (20, 0)}),
        (_LOOP, "a b c d", 0, set()),
        (_PADS, "p1 p2", 2, {(0, 0), (5, 0)}),
        (_PADS, "p1 w p2", 2, {(0, 0), (5, 0)}),
        (_GAP, "w1 w3", 4, {(0, 0), (15, 0)}),
        (_ISLANDS, "w1 w3", 4, set()),
    ],
    ids=[
        "one track",
        "run through via",
        "run, via not selected",
        "tee",
        "closed loop",
        "two pads",
        "pads with track",
        "joined by unselected track",
        "copper islands",
    ],
)
def test_measured_between_farthest_free_ends(
    cu_stack: dict[int, Any],
    elements: list[BoardElement],
    selected: str,
    ends: int,
    measured: set[Point],
) -> None:
    assert _measured(elements, selected, cu_stack) == (ends, measured)


def test_copper_islands_are_named_instead_of_measuring_one(
    cu_stack: dict[int, Any],
) -> None:
    result = analyze_pcb_parasitic(_connected(_ISLANDS, "w1 w3"), cu_stack)

    assert "2 separate copper islands" in result["error"]
    assert result["path"] is None
