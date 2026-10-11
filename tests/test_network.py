"""Connectivity, path finding and arc geometry."""

from __future__ import annotations

import math
from typing import Any

import pytest

from get_distance import find_shortest_path
from get_pcb_elements_ipc import _arc_geometry


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
