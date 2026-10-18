"""Result table formatting, including failed simulations."""

from __future__ import annotations

from typing import Any

import pytest

import parasitic
from analysis_progress import CancelledError, Stage
from parasitic import format_result_message


def _result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "error": None,
        "distance": 0.01,
        "short_path_resistance": 0.005,
        "resistance_dc": 0.005,
        "impedance_ac": {1e3: complex(0.005, 0.001)},
        "impedance_ac_conn1": {1e3: complex(50.0, 0.0)},
        "impedance_ac_conn2": {1e3: complex(50.0, 0.0)},
        "zin_termination_ohm": 50.0,
        "area": {},
        "spice_unavailable": None,
        "skipped": [],
        "measured": [],
        "free_ends": 0,
        "path": [1, 2],
        "network_info": [],
    }
    result.update(overrides)
    return result


def test_valid_result(cu_stack: dict[int, Any]) -> None:
    text = format_result_message(_result(), cu_stack)

    assert "Shortest path resistance ≈ 5.000 mΩ" in text
    assert "[error]" not in text


def _row(text: str, label: str) -> str:
    return next(line for line in text.splitlines() if line.strip().startswith(label))


@pytest.mark.parametrize(("free_ends", "noted"), [(2, False), (3, True)])
def test_choice_among_free_ends_is_noted(
    cu_stack: dict[int, Any], free_ends: int, noted: bool
) -> None:
    measured = ["(0.000, 0.000) mm on F.Cu", "(5.000, 0.000) mm on F.Cu"]

    text = format_result_message(
        _result(measured=measured, free_ends=free_ends), cu_stack
    )

    assert "to (5.000, 0.000) mm on F.Cu" in text
    assert ("farthest apart of 3 free ends" in text) is noted


@pytest.mark.parametrize(("link_type", "noted"), [("ZONE", True), ("WIRE", False)])
def test_zone_crossing_is_noted(
    cu_stack: dict[int, Any], link_type: str, noted: bool
) -> None:
    link = {"type": link_type, "nodes": (1, 2)}

    text = format_result_message(_result(network_info=[link]), cu_stack)

    assert ("zones on the path count as nearly ideal" in text) is noted


def test_failed_simulations_are_marked(cu_stack: dict[int, Any]) -> None:
    text = format_result_message(
        _result(
            resistance_dc=None,
            impedance_ac={1e3: None, 1e6: complex(0.006, 0.01)},
            impedance_ac_conn1={1e3: None, 1e6: None},
            impedance_ac_conn2={1e6: complex(50.0, 1.0)},
        ),
        cu_stack,
    )

    assert _row(text, "DC").split() == ["DC", "[error]"]
    assert _row(text, "1 kHz").split() == ["1", "kHz", "[error]"]
    assert _row(text, "1 MHz").split()[-3:] == ["[error]", "50.010", "Ω"]


def test_no_path(cu_stack: dict[int, Any]) -> None:
    text = format_result_message(
        _result(short_path_resistance=None, resistance_dc=None, impedance_ac={}),
        cu_stack,
    )

    assert "No connection found" in text
    assert "ngspice simulation failed" in text


def _select_wires(data: dict[Any, dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """`data` with all wires selected: the free ends are the coil's two ends."""
    for d in data.values():
        d["is_selected"] = d["type"] == "WIRE"
    return data


def test_selection_without_two_free_ends_is_reported() -> None:
    touching_pad = {"type": "PAD", "position": (0.0, 0.0), "net_start": {0: 1}}
    data = {i: {**touching_pad, "is_selected": True, "layer": [0]} for i in range(2)}

    result = parasitic.analyze_pcb_parasitic(data, {})

    assert "no two free ends" in result["error"]


def test_missing_ngspice_keeps_path_results(
    monkeypatch: pytest.MonkeyPatch,
    coil_data: dict[Any, dict[str, Any]],
    cu_stack: dict[int, Any],
) -> None:
    from spice_worker import SpiceUnavailableError

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise SpiceUnavailableError("ngspice library could not be loaded")

    monkeypatch.setattr(parasitic, "simulate_network", unavailable)
    result = parasitic.analyze_pcb_parasitic(_select_wires(coil_data), cu_stack)
    text = format_result_message(result, cu_stack)

    assert result["short_path_resistance"] is not None
    assert result["network_info"]
    assert "Shortest path resistance" in text
    assert "Measured from" in text
    assert "ngspice library could not be loaded" in text


@pytest.mark.parametrize("cancel_stage", list(Stage))
def test_cancel_ends_the_analysis(
    coil_data: dict[Any, dict[str, Any]],
    cu_stack: dict[int, Any],
    cancel_stage: Stage,
) -> None:
    with pytest.raises(CancelledError):
        parasitic.analyze_pcb_parasitic(
            _select_wires(coil_data),
            cu_stack,
            progress=lambda stage, _finished, _total: stage != cancel_stage,
        )


@pytest.mark.parametrize("error", [None, "No connection found"])
def test_skipped_elements_are_reported(
    cu_stack: dict[int, Any], error: str | None
) -> None:
    result = _result(skipped=["WIRE on layer 0: resistance nan"], error=error)

    text = format_result_message(result, cu_stack)

    assert text.startswith("Warning: 1 modelling issue(s)")
