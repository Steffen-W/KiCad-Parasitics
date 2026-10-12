"""Connectivity, path finding and arc geometry."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from board_model import arc_geometry, pad, via, wire, zone
from connectivity import _bboxes_overlap, _BBoxGrid, build_connectivity
from get_distance import find_shortest_path


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
    angle, length, circle = arc_geometry((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))

    assert circle is not None
    assert circle.center == pytest.approx((0.0, 0.0))
    assert circle.radius == pytest.approx(1.0)
    assert angle == pytest.approx(math.pi)
    assert length == pytest.approx(math.pi)


def test_arc_geometry_collinear_falls_back_to_chord() -> None:
    assert arc_geometry((0.0, 0.0), (2.0, 0.0), (1.0, 0.0)) == (0.0, 2.0, None)


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

    build_connectivity(items)

    wires = [oid for oid, d in items.items() if d["type"] == "WIRE"]
    assert len(wires) == len(items) > 2
    for prev, cur in zip(wires, wires[1:]):
        assert cur in items[prev]["conn_end"]
        assert prev in items[cur]["conn_start"]


def _square(cx: float, cy: float, half: float) -> list[tuple[float, float]]:
    return [
        ((cx + sx * half) * 1e-3, (cy + sy * half) * 1e-3)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def test_build_connectivity_covers_every_contact_kind() -> None:
    def w(oid: str, x0: float, y0: float, x1: float, y1: float) -> Any:
        """0.2 mm wide track on layer 0 between two points given in mm."""
        return wire(
            oid, "N", [0], (x0 * 1e-3, y0 * 1e-3), (x1 * 1e-3, y1 * 1e-3), 0.2e-3
        )

    def v(oid: str, x: float, y: float = 0.0) -> Any:
        return via(oid, "N", [0, 2], (x * 1e-3, y * 1e-3), 0.6e-3, 0.3e-3)

    elements = [
        w("w1", 0, 0, 10, 0),
        w("w2", 10, 0, 10, 10),  # endpoint on w1's end
        w("w3", 5, -5, 5, 0),  # T-junction on w1's body
        v("v1", 10, 10),  # larger radius than w2
        w("w4", 15, 0, 25, 0),  # runs through the pad
        pad("pad", "N", [0], (20e-3, 0.0), 0.0, _square(20, 0, 0.5)),
        w("w5", 20, 5, 20, 0.4),  # ends inside the pad
        zone("zone", "N", [2], _square(35, 0, 5), 1e-4),
        v("v2", 35),  # inside the zone on layer 2
        pad(
            "pad2", "N", [2], (40.2e-3, 0.0), 0.0, _square(40.2, 0, 0.5)
        ),  # overlaps it
        w("w6", 30, 0, 35, 0),  # layer 0 only: reaches v2, not the zone
        w("w7", 50, 0, 51, 0),
        w("w8", 51.2, 0, 52, 0),  # gap equals both half widths: touching
        w("w9", 60, 0, 61, 0),
        w("w10", 61.2 + 1e-5, 0, 62, 0),  # 10 nm further: not touching
        w("w11", 70, -1, 70, 1),  # crosses the zone below on layer 0
        zone(
            "zone0",
            "N",
            [0],
            [(69e-3, -0.5e-3), (71e-3, -0.5e-3), (70e-3, 0.5e-3)],
            1e-6,
        ),
        # ends in a long zone; its other end is nearer to the zone's centre
        w("w12", 121, 0, 130, 5),
        zone(
            "zone1",
            "N",
            [0],
            [(120e-3, -1e-3), (140e-3, -1e-3), (140e-3, 1e-3), (120e-3, 1e-3)],
            4e-5,
        ),
    ]
    items = {e["id"]: e for e in elements}

    build_connectivity(items)

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
        "zone": (["pad2", "v2"], []),
        "pad2": (["zone"], []),
        "v2": (["w6", "zone"], []),
        "w6": ([], ["v2"]),
        "w7": ([], ["w8"]),
        "w8": (["w7"], []),
        "w9": ([], []),
        "w10": ([], []),
        "w11": (["zone0"], []),
        "zone0": (["w11"], []),
        "w12": (["zone1"], []),
        "zone1": (["w12"], []),
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


@pytest.mark.xfail(
    strict=True,
    reason="a contact on a wire's body is moved to the nearer end",
)
def test_branch_off_a_wire_body_splits_its_length(cu_stack: dict[int, Any]) -> None:
    import get_parasitic as gp
    from connect_nets import Connect_Nets

    main = wire("main", "N", [0], (0.0, 0.0), (10e-3, 0.0), 0.2e-3)
    branch = wire("branch", "N", [0], (3e-3, 0.0), (3e-3, 5e-3), 0.2e-3)
    items = {"main": main, "branch": branch}
    build_connectivity(items)
    data = Connect_Nets(items)
    network = gp.extract_network(data, cu_stack)
    tip = data["branch"]["net_end"][0]

    distances = [
        gp.find_path(network, data["main"][end][0], tip)[0]
        for end in ("net_start", "net_end")
    ]

    assert distances == pytest.approx([8e-3, 12e-3])


def _zone_network(
    zones: list[Any], pad_centres: list[tuple[float, float]], cu_stack: dict[int, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Network of zones on layer 0 with 1 mm pads at the given centres (mm)."""
    import get_parasitic as gp
    from connect_nets import Connect_Nets

    pads = [
        pad(f"p{i}", "N", [0], (x * 1e-3, y * 1e-3), 0.0, _square(x, y, 0.5))
        for i, (x, y) in enumerate(pad_centres)
    ]
    items = {e["id"]: e for e in [*zones, *pads]}
    build_connectivity(items)
    data = Connect_Nets(items)
    return data, gp.extract_network(data, cu_stack)


@pytest.mark.xfail(
    strict=True,
    reason="a zone links its N connections pairwise, not to a star point",
)
def test_zone_needs_one_resistor_per_connection(cu_stack: dict[int, Any]) -> None:
    centres = [(x, y) for x in (2, 5, 8) for y in (3, 7)]
    zones = [zone("z", "N", [0], _square(5, 5, 5), 1e-4)]

    _data, network = _zone_network(zones, centres, cu_stack)

    assert len(network["resistors"]) == len(centres)


@pytest.mark.xfail(
    strict=True,
    reason="overlapping zones are linked but extract_network ignores it",
)
def test_overlapping_zones_conduct(cu_stack: dict[int, Any]) -> None:
    import get_parasitic as gp

    zones = [
        zone("a", "N", [0], _square(5, 5, 5), 1e-4),
        zone("b", "N", [0], _square(13, 5, 5), 1e-4),
    ]

    data, network = _zone_network(zones, [(2, 5), (16, 5)], cu_stack)
    assert data["a"]["conn_start"] == ["p0", "b"]  # the geometry is recognised

    distance = gp.find_path(
        network, data["p0"]["net_start"][0], data["p1"]["net_start"][0]
    )[0]

    assert math.isfinite(distance)
