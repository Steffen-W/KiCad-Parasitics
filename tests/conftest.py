"""Shared fixtures. Plugin modules are imported top-level, as KiCad's IPC runner does."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins"))


@pytest.fixture
def no_wx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub wxPython (no PyPI wheels for Linux) for importing the GUI module."""
    monkeypatch.setitem(sys.modules, "wx", mock.MagicMock())


@pytest.fixture
def coil_data() -> dict[Any, dict[str, Any]]:
    """Connected element data of a 3-turn rectangular single-layer coil."""
    import coil_generator as cg
    from connect_nets import Connect_Nets

    ctx = cg.new_context()
    coil = cg.generate_rectangular_coil(
        turns=3,
        width=0.0002,
        spacing=0.0003,
        inner_width=0.003,
        inner_height=0.001,
        ctx=ctx,
    )
    return Connect_Nets(cg.items_to_dict(coil.result()))


@pytest.fixture
def cu_stack() -> dict[int, Any]:
    """Two-layer FR4 stackup (35 µm copper, 1.5 mm core), heights from the top."""
    die = {"h": 1.5e-3, "epsilon_r": 4.5, "loss_tangent": 0.02}
    return {
        0: {
            "thickness": 35e-6,
            "name": "F.Cu",
            "abs_height": 0.0,
            "die_above": None,
            "die_below": die,
            "model": "Microstrip",
            "gap": None,
        },
        2: {
            "thickness": 35e-6,
            "name": "B.Cu",
            "abs_height": 35e-6 + 1.5e-3,
            "die_above": die,
            "die_below": None,
            "model": "Microstrip",
            "gap": None,
        },
    }
