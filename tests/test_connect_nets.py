"""Node assignment in Connect_Nets."""

from __future__ import annotations

import copy
import itertools
from typing import Any

import pytest

from connect_nets import Connect_Nets


def _wire(
    start: tuple[float, float], end: tuple[float, float], layer: int = 0
) -> dict[str, Any]:
    return {
        "type": "WIRE",
        "start": start,
        "end": end,
        "layer": [layer],
        "conn_start": [],
        "conn_end": [],
    }


def _point(
    kind: str, position: tuple[float, float], layers: list[int]
) -> dict[str, Any]:
    return {"type": kind, "position": position, "layer": layers, "conn_start": []}


def _board() -> dict[str, dict[str, Any]]:
    """Pad P ─ W1 ┬ W2 ─ via V ═ W4 (bottom layer), with W3 branching off at the T."""
    data = {
        "P": _point("PAD", (0, 0), [0]),
        "W1": _wire((0, 0), (1, 0)),
        "W2": _wire((1, 0), (2, 0)),
        "W3": _wire((1, 0), (1, 1)),
        "V": _point("VIA", (2, 0), [0, 2]),
        "W4": _wire((2, 0), (3, 0), layer=2),
    }
    links = [
        ("P", "conn_start", "W1", "conn_start"),
        ("W1", "conn_end", "W2", "conn_start"),
        ("W1", "conn_end", "W3", "conn_start"),
        ("W2", "conn_start", "W3", "conn_start"),
        ("W2", "conn_end", "V", "conn_start"),
        ("V", "conn_start", "W4", "conn_start"),
    ]
    for a, a_key, b, b_key in links:
        data[a][a_key].append(b)
        data[b][b_key].append(a)
    return data


_EXPECTED = {
    frozenset({("P", "net_start", 0), ("W1", "net_start", 0)}),
    frozenset({("W1", "net_end", 0), ("W2", "net_start", 0), ("W3", "net_start", 0)}),
    frozenset({("W2", "net_end", 0), ("V", "net_start", 0)}),
    frozenset({("V", "net_start", 2), ("W4", "net_start", 2)}),
    frozenset({("W3", "net_end", 0)}),
    frozenset({("W4", "net_end", 2)}),
}


def _groups(data: dict[str, dict[str, Any]]) -> set[frozenset[tuple[str, str, int]]]:
    nodes: dict[int, set[tuple[str, str, int]]] = {}
    for uid, d in data.items():
        sides = ("net_start", "net_end") if d["type"] == "WIRE" else ("net_start",)
        for side in sides:
            for layer, node in d[side].items():
                assert node > 0
                nodes.setdefault(node, set()).add((uid, side, layer))
    return {frozenset(g) for g in nodes.values()}


def test_grouping_is_independent_of_element_order() -> None:
    board = _board()
    for order in itertools.permutations(board):
        assert (
            _groups(Connect_Nets({k: copy.deepcopy(board[k]) for k in order}))
            == _EXPECTED
        )


@pytest.mark.parametrize("dropped", ["W1", "W2", "V"])
def test_one_sided_connection_lists(dropped: str) -> None:
    board = _board()
    for d in board.values():
        for key in ("conn_start", "conn_end"):
            if dropped in d.get(key, []):
                d[key].remove(dropped)

    assert _groups(Connect_Nets(board)) == _EXPECTED


def test_via_and_pad_ends_mirror_their_start() -> None:
    data = Connect_Nets(_board())

    assert data["V"]["net_end"] == data["V"]["net_start"]
    assert data["V"]["net_start"][0] != data["V"]["net_start"][2]


def test_missing_position_does_not_connect() -> None:
    data = {"A": _wire((0, 0), (1, 0)), "B": _wire((5, 5), (6, 5))}
    del data["B"]["start"]
    data["B"]["conn_start"].append("A")

    groups = _groups(Connect_Nets(data))

    assert all(len(g) == 1 for g in groups)


def test_preassigned_nodes_are_kept() -> None:
    a, b = _wire((0, 0), (1, 0)), _wire((5, 0), (6, 0))
    a["net_start"], a["net_end"] = {0: 7}, {0: 8}
    b["net_start"], b["net_end"] = {0: 8}, {0: 9}

    data = Connect_Nets({"A": a, "B": b})

    assert (data["A"]["net_start"][0], data["A"]["net_end"][0]) == (7, 8)
    assert (data["B"]["net_start"][0], data["B"]["net_end"][0]) == (8, 9)


def test_junction_listed_as_chain_forms_one_node() -> None:
    """Four wires meet at one point, but each lists only its neighbour in a chain."""
    board = {name: _wire((0, 0), (i + 1, 1)) for i, name in enumerate("ABCD")}
    for a, b in zip("ABC", "BCD"):
        board[a]["conn_start"].append(b)
        board[b]["conn_start"].append(a)
    junction = frozenset((name, "net_start", 0) for name in "ABCD")

    for order in itertools.permutations(board):
        data = Connect_Nets({k: copy.deepcopy(board[k]) for k in order})
        assert junction in _groups(data)


def test_two_wires_closing_a_loop_keep_separate_ends() -> None:
    """Two arcs P→Q and Q→P list each other at both ends."""
    board = {
        "P": _point("PAD", (0, 0), [0]),
        "Q": _point("PAD", (1, 0), [0]),
        "A": _wire((0, 0), (1, 0)),
        "B": _wire((1, 0), (0, 0)),
    }
    board["A"]["conn_start"] += ["P", "B"]
    board["A"]["conn_end"] += ["Q", "B"]
    board["B"]["conn_start"] += ["Q", "A"]
    board["B"]["conn_end"] += ["P", "A"]
    board["P"]["conn_start"] += ["A", "B"]
    board["Q"]["conn_start"] += ["A", "B"]

    groups = _groups(Connect_Nets(board))

    assert (
        frozenset({("P", "net_start", 0), ("A", "net_start", 0), ("B", "net_end", 0)})
        in groups
    )
    assert (
        frozenset({("Q", "net_start", 0), ("A", "net_end", 0), ("B", "net_start", 0)})
        in groups
    )


def test_conflicting_preset_nodes_are_rejected() -> None:
    a, b = _wire((0, 0), (1, 0)), _wire((1, 0), (2, 0))
    a["net_start"], a["net_end"] = {0: 1}, {0: 2}
    b["net_start"], b["net_end"] = {0: 3}, {0: 4}
    a["conn_end"].append("B")
    b["conn_start"].append("A")

    with pytest.raises(ValueError, match="different preset nodes"):
        Connect_Nets({"A": a, "B": b})
