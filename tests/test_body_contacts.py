"""Tracks touched on their body are cut there (see split_at_body_contacts)."""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from board_model import BoardElement, pad, split_at_body_contacts, via, wire

MM = 1e-3
_MAIN = wire("main", "N", [0], (0.0, 0.0), (10 * MM, 0.0), 0.2 * MM)
_ARC_MID = (math.sqrt(0.5) * 10 * MM,) * 2
# A quarter circle of radius 10 mm from (10, 0) to (0, 10) around the origin
_ARC = wire("main", "N", [0], (10 * MM, 0.0), (0.0, 10 * MM), 0.2 * MM, _ARC_MID)


def _square(x: float, y: float, half: float) -> list[tuple[float, float]]:
    return [
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half),
    ]


def _split(main: BoardElement, *others: BoardElement) -> dict[str, BoardElement]:
    elements = {e["id"]: copy.deepcopy(e) for e in (main, *others)}
    split_at_body_contacts(elements)
    return elements


def _lengths_mm(elements: dict[str, Any]) -> list[float]:
    parts = sorted(i for i in elements if i.startswith("main"))
    return [round(elements[i]["length"] / MM, 4) for i in parts]


@pytest.mark.parametrize(
    "contact",
    [
        wire("branch", "N", [0], (3 * MM, 0.0), (3 * MM, 5 * MM), 0.2 * MM),
        wire("offset", "N", [0], (3 * MM, 0.15 * MM), (3 * MM, 5 * MM), 0.2 * MM),
        via("via", "N", [0, 2], (3 * MM, 0.0), 0.6 * MM, 0.3 * MM),
        pad(
            "pad",
            "N",
            [0],
            (3 * MM, 0.4 * MM),
            0.0,
            _square(3 * MM, 0.4 * MM, 0.5 * MM),
        ),
    ],
    ids=["track end", "track end beside the midline", "via", "pad over the body"],
)
def test_contact_on_the_body_cuts_the_track(contact: BoardElement) -> None:
    elements = _split(_MAIN, contact)

    assert _lengths_mm(elements) == [3.0, 7.0]
    assert elements["main#0"]["end"] == pytest.approx((3 * MM, 0.0))


def test_arc_is_cut_into_arcs_on_its_circle() -> None:
    branch = wire("branch", "N", [0], (0.0, 0.0), _ARC_MID, 0.2 * MM)

    elements = _split(_ARC, branch)

    eighth = 10 * math.pi / 4
    assert _lengths_mm(elements) == pytest.approx([eighth] * 2, abs=1e-4)
    radii = [elements[i]["radius"] for i in ("main#0", "main#1")]
    assert radii == pytest.approx([10 * MM] * 2)


@pytest.mark.parametrize(
    "others",
    [
        [wire("end", "N", [0], (10.1 * MM, 0.0), (10.1 * MM, 5 * MM), 0.2 * MM)],
        [wire("other", "N", [2], (3 * MM, 0.0), (3 * MM, 5 * MM), 0.2 * MM)],
        [
            pad(
                "cover",
                "N",
                [0],
                (0.5 * MM, 0.0),
                0.0,
                _square(0.5 * MM, 0.0, 0.6 * MM),
            )
        ],
    ],
    ids=["at the end", "other layer", "pad covering an end"],
)
def test_end_contacts_and_other_layers_do_not_cut(others: list[BoardElement]) -> None:
    elements = _split(_MAIN, *others)

    assert "main" in elements


def test_contacts_at_one_point_make_one_cut() -> None:
    contacts = [
        via("via", "N", [0, 2], (3 * MM, 0.0), 0.6 * MM, 0.3 * MM),
        wire("branch", "N", [0], (3 * MM, 0.0), (3 * MM, 5 * MM), 0.2 * MM),
        wire("later", "N", [0], (6 * MM, 0.0), (6 * MM, -5 * MM), 0.2 * MM),
    ]

    selected = copy.deepcopy(_MAIN)
    selected["is_selected"] = True

    elements = _split(selected, *contacts)

    assert _lengths_mm(elements) == [3.0, 3.0, 4.0]
    assert all(elements[f"main#{n}"]["is_selected"] for n in range(3))


def test_path_and_inductance_geometry_follow_the_cut(cu_stack: dict[int, Any]) -> None:
    import get_parasitic as gp
    from calc_inductance import _extract_segments
    from connect_nets import Connect_Nets
    from connectivity import build_connectivity

    branch = wire("branch", "N", [0], (3 * MM, 0.0), (3 * MM, 5 * MM), 0.2 * MM)
    elements = _split(_MAIN, branch)
    build_connectivity(elements)
    data = Connect_Nets(elements)
    network = gp.extract_network(data, cu_stack)

    distance, path, _ = gp.find_path(
        network, data["main#0"]["net_start"][0], data["branch"]["net_end"][0]
    )
    segments = _extract_segments(path, network["network_info"], cu_stack)

    assert distance == pytest.approx(8 * MM)
    assert sum(s["length"] for s in segments) == pytest.approx(8 * MM)


def test_new_ends_on_other_tracks_cut_them_too() -> None:
    from connectivity import body_contacts

    crossing = wire("crossing", "N", [0], (5 * MM, -5 * MM), (5 * MM, 5 * MM), 0.2 * MM)
    # Its end lies beside the midlines of both and cuts them at its projections,
    # (5.19, 0) and (5, 0.19); these are farther apart than the tracks reach,
    # so each new end lies on the body of the other track.
    end = (5.19 * MM, 0.19 * MM)
    branch = wire("branch", "N", [0], end, (0.0, 5 * MM), 0.2 * MM)

    elements = _split(_MAIN, crossing, branch)

    assert not body_contacts(elements)
    assert len([i for i in elements if i.startswith("crossing")]) == 3


def test_nothing_to_cut_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    elements: dict[str, BoardElement] = {}

    split_at_body_contacts(elements)

    assert not caplog.records
