"""Connectivity, path finding and arc geometry."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from get_distance import find_shortest_path
from get_pcb_elements_ipc import (
    _arc_geometry,
    _bboxes_overlap,
    _BBoxGrid,
    _build_connectivity,
)


def test_connect_nets_assigns_distinct_nodes_along_chain(
    coil_data: dict[Any, dict[str, Any]],
) -> None:
    wires = [d for d in coil_data.values() if d["type"] == "WIRE"]

    for prev, cur in zip(wires, wires[1:]):
        assert prev["net_end"][0] == cur["net_start"][0]
    assert all(w["net_start"][0] != w["net_end"][0] for w in wires)


def test_find_shortest_path() -> None:
    graph = {1: {2: 1.0}, 2: {1: 1.0, 3: 2.0}, 3: {2: 2.0}}

    assert find_shortest_path(graph, 1, 3) == (3.0, [1, 2, 3])


def test_find_shortest_path_unreachable() -> None:
    graph: dict[int, dict[int, float]] = {1: {2: 1.0}, 2: {1: 1.0}, 3: {}}

    with pytest.raises(ValueError, match="No path found"):
        find_shortest_path(graph, 1, 3)


def test_arc_geometry_half_circle() -> None:
    angle, length, circle = _arc_geometry((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))

    assert circle is not None
    assert circle.center == pytest.approx((0.0, 0.0))
    assert circle.radius == pytest.approx(1.0)
    assert angle == pytest.approx(math.pi)
    assert length == pytest.approx(math.pi)


def test_arc_geometry_collinear_falls_back_to_chord() -> None:
    assert _arc_geometry((0.0, 0.0), (2.0, 0.0), (1.0, 0.0)) == (0.0, 2.0, None)


def test_bbox_grid_finds_exactly_the_overlapping_boxes_in_order() -> None:
    rng = random.Random(0)

    def box(size: float) -> tuple[float, float, float, float]:
        x, y = rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)
        return (x, y, x + rng.uniform(0, size), y + rng.uniform(0, size))

    boxes = [box(1e-3) for _ in range(500)] + [box(0.0), box(0.2)]
    grid = _BBoxGrid(boxes)

    for query in [box(1e-3) for _ in range(200)] + [box(0.0), box(0.3)]:
        expected = [i for i, b in enumerate(boxes) if _bboxes_overlap(b, query)]
        assert grid.overlapping(query) == expected


def test_bbox_grid_examines_only_nearby_boxes() -> None:
    class CountingList(list[tuple[float, float, float, float]]):
        reads = 0

        def __getitem__(self, i: Any) -> Any:
            self.reads += 1
            return super().__getitem__(i)

    pitch, size = 1e-3, 0.1e-3
    boxes = CountingList(
        (x * pitch, y * pitch, x * pitch + size, y * pitch + size)
        for x in range(100)
        for y in range(100)
    )
    boxes.append((0.0, 0.0, 0.1, 0.1))  # one box spanning everything
    grid = _BBoxGrid(boxes)
    assert sum(map(len, grid._cells.values())) <= 4 * len(boxes)
    boxes.reads = 0

    query = (50 * pitch, 50 * pitch, 50 * pitch + size, 50 * pitch + size)

    assert grid.overlapping(query) == [50 * 100 + 50, len(boxes) - 1]
    assert boxes.reads <= 4


def test_build_connectivity_links_consecutive_coil_segments() -> None:
    import coil_generator as cg

    coil = cg.generate_rectangular_coil(
        turns=3,
        width=0.0002,
        spacing=0.0003,
        inner_width=0.003,
        inner_height=0.001,
        ctx=cg.new_context(),
    )
    items = cg.items_to_dict(coil.result())
    for d in items.values():
        d["conn_start"], d["conn_end"] = [], []

    _build_connectivity(items)

    wires = [oid for oid, d in items.items() if d["type"] == "WIRE"]
    assert len(wires) == len(items) > 2
    for prev, cur in zip(wires, wires[1:]):
        assert cur in items[prev]["conn_end"]
        assert prev in items[cur]["conn_start"]


def _wire(x0: float, y0: float, x1: float, y1: float, layer: int = 0) -> dict[str, Any]:
    """0.2 mm wide straight track between two points given in mm."""
    start, end = (x0 * 1e-3, y0 * 1e-3), (x1 * 1e-3, y1 * 1e-3)
    return {
        "type": "WIRE",
        "start": start,
        "end": end,
        "width": 0.2e-3,
        "layer": [layer],
        "midline_pts": [start, end],
        "conn_start": [],
        "conn_end": [],
    }


def _square(cx: float, cy: float, half: float) -> list[tuple[float, float]]:
    return [
        ((cx + sx * half) * 1e-3, (cy + sy * half) * 1e-3)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def test_build_connectivity_covers_every_contact_kind() -> None:
    pad = {
        "type": "PAD",
        "position": (20e-3, 0.0),
        "size": (2e-3, 1e-3),
        "shape": 2,
        "orientation": 0,
        "drill": 0.0,
        "layer": [0],
        "_outline": [
            (19e-3, -0.5e-3),
            (21e-3, -0.5e-3),
            (21e-3, 0.5e-3),
            (19e-3, 0.5e-3),
        ],
        "conn_start": [],
    }

    def via(x: float) -> dict[str, Any]:
        return {
            "type": "VIA",
            "position": (x * 1e-3, 0.0),
            "width": 0.6e-3,
            "drill": 0.3e-3,
            "layer": [0, 2],
            "conn_start": [],
        }

    items = {
        "w1": _wire(0, 0, 10, 0),
        "w2": _wire(10, 0, 10, 10),  # endpoint on w1's end
        "w3": _wire(5, -5, 5, 0),  # T-junction on w1's body
        "v1": {**via(10), "position": (10e-3, 10e-3)},  # larger radius than w2
        "w4": _wire(15, 0, 25, 0),  # runs through the pad
        "pad": pad,
        "w5": _wire(20, 5, 20, 0.4),  # ends inside the pad
        "zone": {
            "type": "ZONE",
            "layer": [2],
            "_outline": _square(35, 0, 5),
            "conn_start": [],
        },
        "v2": via(35),  # inside the zone on layer 2
        "w6": _wire(30, 0, 35, 0),  # layer 0 only: reaches v2, not the zone
        "w7": _wire(50, 0, 51, 0),
        "w8": _wire(51.2, 0, 52, 0),  # gap equals both half widths: touching
        "w9": _wire(60, 0, 61, 0),
        "w10": _wire(61.2 + 1e-5, 0, 62, 0),  # 10 nm further: not touching
    }

    _build_connectivity(items)

    links = {
        oid: (sorted(d["conn_start"]), sorted(d.get("conn_end", [])))
        for oid, d in items.items()
    }
    assert links == {
        "w1": (["w3"], ["w2"]),
        "w2": (["w1"], ["v1"]),
        "w3": ([], ["w1"]),
        "v1": (["w2"], []),
        "w4": (["pad"], []),
        "pad": (["w4", "w5"], []),
        "w5": ([], ["pad"]),
        "zone": (["v2"], []),
        "v2": (["w6", "zone"], []),
        "w6": ([], ["v2"]),
        "w7": ([], ["w8"]),
        "w8": (["w7"], []),
        "w9": ([], []),
        "w10": ([], []),
    }


def test_extract_network_skips_defective_elements(
    coil_data: dict[Any, dict[str, Any]],
    cu_stack: dict[int, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    import get_parasitic as gp

    wires = [d for d in coil_data.values() if d["type"] == "WIRE"]
    wires[1]["width"] = 0.0
    via = {
        "type": "VIA",
        "layer": [0, 2],
        "net_start": {0: wires[0]["net_start"][0], 2: 10_000},
        "position": (0.0, 0.0),
        "drill": 0.0,
    }

    network = gp.extract_network({**coil_data, "via": via}, cu_stack)

    assert len(network["resistors"]) == len(wires) - 1
    assert [s.split()[0] for s in network["skipped"]] == ["WIRE", "VIA"]
    assert "Skipping WIRE" in caplog.text


@pytest.mark.parametrize("r_dc", [0.0, -1.0, math.nan, math.inf])
def test_extract_network_skips_unusable_resistance(
    coil_data: dict[Any, dict[str, Any]],
    cu_stack: dict[int, Any],
    monkeypatch: pytest.MonkeyPatch,
    r_dc: float,
) -> None:
    import get_parasitic as gp

    monkeypatch.setattr(gp, "analyze_trace", lambda *_a, **_k: {"r_dc": r_dc})

    network = gp.extract_network(coil_data, cu_stack)

    assert network["resistors"] == []
    assert all(s.startswith("WIRE") for s in network["skipped"])
    assert network["skipped"]
