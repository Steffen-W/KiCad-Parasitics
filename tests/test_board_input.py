"""Selection validation and stackup readers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from kipy.proto.board.board_types_pb2 import BoardLayer

from get_pcb_elements_ipc import Get_PCB_Elements_IPC, SelectionError
from get_pcb_stackup import Get_PCB_Stackup_fun
from get_pcb_stackup_ipc import Get_PCB_Stackup_IPC


def _item(net: str | None = None) -> SimpleNamespace:
    item = SimpleNamespace(id=SimpleNamespace(value=object()))
    if net is not None:
        item.net = SimpleNamespace(name=net)
    return item


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ([_item("A")], "exactly two"),
        ([_item("A"), _item()], "with a net"),
        ([_item("A"), _item("B")], "same net"),
    ],
)
def test_invalid_selection(selection: list[SimpleNamespace], message: str) -> None:
    board = mock.MagicMock()
    board.get_selection.return_value = selection
    board.get_footprints.return_value = []

    with pytest.raises(SelectionError, match=message):
        Get_PCB_Elements_IPC(board)


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
        stack = Get_PCB_Stackup_IPC(board)

    assert sorted(stack) == [0, 2]
    assert stack[2]["abs_height"] == pytest.approx(0.035e-3 * 2 + 1.5e-3 * 2)
    assert "Mystery" in caplog.text


_ZERO_THICKNESS = '(kicad_pcb (setup (stackup (layer "F.Cu" (thickness 0)))))'


@pytest.mark.parametrize("content", [None, _ZERO_THICKNESS], ids=["missing", "invalid"])
def test_file_stackup_falls_back_to_two_layers(
    tmp_path: Path, content: str | None
) -> None:
    board_file = tmp_path / "board.kicad_pcb"
    if content is not None:
        board_file.write_text(content, encoding="utf-8")

    stack = Get_PCB_Stackup_fun(board_file)

    assert {k: v["name"] for k, v in stack.items()} == {0: "F.Cu", 2: "B.Cu"}


@pytest.mark.parametrize(
    ("selected", "kept"),
    [
        ({"a1", "a2"}, {"a1", "a2", "a3"}),
        ({"a1", "b1"}, {"a1", "a2", "a3", "b1"}),
        (set(), set()),
    ],
    ids=["same net", "two nets", "nothing selected"],
)
def test_pcbnew_reader_keeps_only_selected_nets(
    monkeypatch: pytest.MonkeyPatch, selected: set[str], kept: set[str]
) -> None:
    monkeypatch.setitem(sys.modules, "pcbnew", mock.MagicMock())
    from get_pcb_elements import _selected_nets_only

    nets = {"a1": 1, "a2": 1, "a3": 1, "b1": 2, "c1": 3}
    items = {
        oid: {"net_code": net, "is_selected": oid in selected}
        for oid, net in nets.items()
    }

    assert set(_selected_nets_only(items)) == kept
