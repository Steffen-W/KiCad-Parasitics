"""Selection rules, Net Ties and stackup readers."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from kipy.proto.board.board_types_pb2 import BoardLayer

from board_model import (
    BoardInputError,
    Copper,
    NetTie,
    Selected,
    build_cu_stack,
    dielectric,
    net_tie,
    read_board,
    via,
)
from pcb_types import BoardElement, CuLayer
from read_board_ipc import IpcBoard
from stackup_file import read_stackup_file


class FakeBoard:
    """BoardSource with one via per net; records the nets asked for."""

    def __init__(self, selection: list[Selected], ties: Sequence[NetTie] = ()) -> None:
        self._selection = selection
        self._ties = list(ties)
        self.nets: AbstractSet[str] = set()

    def selection(self) -> list[Selected]:
        return self._selection

    def net_ties(self) -> list[NetTie]:
        return self._ties

    def copper(
        self, nets: AbstractSet[str], net_ties: Sequence[NetTie]
    ) -> Iterable[BoardElement]:
        self.nets = nets
        return [via(f"v{net}", net, [0], (0.0, 0.0), 1e-3, 5e-4) for net in "ABCD"]

    def stackup(self) -> dict[int, CuLayer]:
        return build_cu_stack([])


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ([Selected("vA", "A")], "exactly two"),
        ([Selected("vA", "A"), Selected("text")], "with a net"),
        ([Selected("vA", "A"), Selected("vB", "B")], "same net"),
        ([Selected("vA", "A"), Selected("pad", "A")], "pads, vias, tracks or zones"),
    ],
)
def test_invalid_selection(selection: list[Selected], message: str) -> None:
    with pytest.raises(BoardInputError, match=message):
        read_board(FakeBoard(selection))


def test_footprint_stands_for_its_two_pads() -> None:
    footprint = Selected("fp", pads=(Selected("vA", "A"), Selected("vC", "A")))

    board = read_board(FakeBoard([footprint]))

    assert [e["is_selected"] for e in board.elements.values()] == [
        True,
        False,
        True,
        False,
    ]


def test_net_ties_join_nets_transitively() -> None:
    ties = [
        net_tie("NT1", [["1, 2"]], {"1": "A", "2": "B"}, None),
        net_tie("NT2", [["1", "2"]], {"1": "B", "2": "C"}, None),
        net_tie("NT3", [["1,2"]], {"1": "D", "2": "E"}, None),
    ]
    source = FakeBoard([Selected("vA", "A"), Selected("vC", "C")], ties)

    board = read_board(source)

    assert source.nets == {"A", "B", "C"}
    assert board.nets == ["A", "B", "C"]
    assert board.net_ties == ["NT1", "NT2"]


def _copper(layer: Any, user_name: str = "") -> SimpleNamespace:
    return SimpleNamespace(type=1, layer=layer, user_name=user_name, thickness=35_000)


def _core() -> SimpleNamespace:
    sub = SimpleNamespace(thickness=1_500_000, epsilon_r=4.5, loss_tangent=0.02)
    return SimpleNamespace(
        type=2, thickness=0, dielectric=SimpleNamespace(layers=[sub])
    )


def test_ipc_stackup_skips_unknown_copper_layer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    board = mock.MagicMock()
    board.get_stackup.return_value.layers = [
        _copper(BoardLayer.BL_F_Cu),
        _core(),
        _copper(BoardLayer.BL_UNDEFINED, "Mystery"),
        _core(),
        _copper(BoardLayer.BL_B_Cu),
    ]

    with caplog.at_level(logging.WARNING):
        stack = IpcBoard(board).stackup()

    assert sorted(stack) == [0, 2]
    assert stack[2]["abs_height"] == pytest.approx(0.035e-3 * 2 + 1.5e-3 * 2)
    assert "Mystery" in caplog.text


_STACKUP = """(kicad_pcb (setup (stackup
  (layer "F.Mask" (type "Top Solder Mask") (thickness 0.01))
  (layer "F.Cu" (type "copper") (thickness 0.035))
  (layer "dielectric 1" (type "prepreg") (thickness 0.1) (epsilon_r 4.1)
    addsublayer (thickness 0.2) (epsilon_r 3.9))
  (layer "In1.Cu" (type "copper") (thickness 0.035))
  (layer "dielectric 2" (type "core") (thickness 1.0))
  (layer "B.Cu" (type "copper") (thickness 0.035))
)))"""


def test_file_stackup_lists_copper_and_dielectrics(tmp_path: Path) -> None:
    board_file = tmp_path / "board.kicad_pcb"
    board_file.write_text(_STACKUP, encoding="utf-8")

    entries = read_stackup_file(board_file)

    names = [e.name if isinstance(e, Copper) else "-" for e in entries]
    heights = [e.thickness if isinstance(e, Copper) else e["h"] for e in entries]
    assert names == ["F.Cu", "-", "In1.Cu", "-", "B.Cu"]
    assert heights == pytest.approx([35e-6, 0.3e-3, 35e-6, 1e-3, 35e-6])
    assert entries[1] == dielectric(heights[1], 4.1)  # first sublayer's material
    assert entries[3] == dielectric(heights[3])


def test_cu_stack_heights_and_models() -> None:
    stack = build_cu_stack(
        [Copper("F.Cu", 35e-6), dielectric(1e-3), Copper("In1.Cu", 35e-6)]
    )

    assert stack[4]["abs_height"] == pytest.approx(35e-6 + 1e-3)
    assert [stack[k]["model"] for k in (0, 4)] == ["Microstrip", "Microstrip"]


_ZERO_THICKNESS = (
    '(kicad_pcb (setup (stackup (layer "F.Cu" (type "copper") (thickness 0)))))'
)


@pytest.mark.parametrize(
    "content",
    [None, "(kicad_pcb (setup", _ZERO_THICKNESS],
    ids=["missing", "broken", "invalid"],
)
def test_file_stackup_falls_back_to_two_layers(
    tmp_path: Path, content: str | None
) -> None:
    board_file = tmp_path / "board.kicad_pcb"
    if content is not None:
        board_file.write_text(content, encoding="utf-8")

    stack = build_cu_stack(read_stackup_file(board_file), board_thickness=1e-3)

    assert {k: v["name"] for k, v in stack.items()} == {0: "F.Cu", 2: "B.Cu"}
    assert stack[2]["abs_height"] + stack[2]["thickness"] == pytest.approx(1e-3)
