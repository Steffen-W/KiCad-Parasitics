"""Loop inductance of reference paths (needs bfieldtools, not installed in CI)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("bfieldtools")

from calc_inductance import _compute_inductance

_W = 0.2e-3


def _wire(
    start: tuple[float, float],
    end: tuple[float, float],
    nodes: tuple[int, int],
    layer_name: str,
    z: float,
    path: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """A wire from node to node; `path` gives the order the path runs through
    them if it runs from end to start."""
    path_from, path_to = path or nodes
    return {
        "type": "WIRE",
        "start": start,
        "end": end,
        "node_at_start": nodes[0],
        "node_at_end": nodes[1],
        "path_from": path_from,
        "path_to": path_to,
        "width": _W,
        "length": ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5,
        "layer_name": layer_name,
        "z": z,
        "thickness": 35e-6,
    }


def _via(
    position: tuple[float, float], z_top: float, z_bot: float, names: tuple[str, str]
) -> dict[str, Any]:
    return {
        "type": "VIA",
        "position": position,
        "drill": 0.3e-3,
        "length": abs(z_top - z_bot),
        "z_top": z_top,
        "z_bot": z_bot,
        "layer1_name": names[0],
        "layer2_name": names[1],
    }


_Z_F, _Z_IN, _Z_B = 0.010e-3, 0.800e-3, 1.555e-3

# Path of a real two-layer board: B.Cu, via, F.Cu.
_TWO_LAYERS = [
    _wire((0.125, 0.020), (0.125, 0.023), (1, 2), "B.Cu", _Z_B, path=(2, 1)),
    _wire((0.130, 0.020), (0.125, 0.020), (3, 1), "B.Cu", _Z_B, path=(1, 3)),
    _via((0.130, 0.020), _Z_B, _Z_F, ("F.Cu", "B.Cu")),
    _wire((0.130, 0.020), (0.130, 0.025), (4, 5), "F.Cu", _Z_F),
    _wire((0.130, 0.025), (0.125, 0.025), (5, 6), "F.Cu", _Z_F),
]

# 10 mm square: bottom on F.Cu, right on In1.Cu, top and left on B.Cu.
_THREE_LAYERS = [
    _wire((0, 0), (0.010, 0), (10, 11), "F.Cu", _Z_F),
    _via((0.010, 0), _Z_IN, _Z_F, ("In1.Cu", "F.Cu")),
    _wire((0.010, 0), (0.010, 0.010), (12, 13), "In1.Cu", _Z_IN),
    _via((0.010, 0.010), _Z_B, _Z_IN, ("B.Cu", "In1.Cu")),
    _wire((0.010, 0.010), (0, 0.010), (14, 15), "B.Cu", _Z_B),
    _wire((0, 0.010), (0, 0), (15, 16), "B.Cu", _Z_B),
]


@pytest.mark.parametrize(
    ("segments", "nanohenry"),
    [(_TWO_LAYERS, 17.31), (_THREE_LAYERS, 38.96)],
    ids=["two layers", "three layers"],
)
def test_reference_loop_inductance(
    segments: list[dict[str, Any]], nanohenry: float
) -> None:
    result = _compute_inductance(segments)

    assert result["inductance"] * 1e9 == pytest.approx(nanohenry, rel=0.01)
