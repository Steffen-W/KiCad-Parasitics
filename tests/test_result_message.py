"""Result table formatting, including failed simulations."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def format_result_message(no_wx: None) -> Any:
    from parasitic import format_result_message

    return format_result_message


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
    }
    result.update(overrides)
    return result


def test_valid_result(format_result_message: Any, cu_stack: dict[int, Any]) -> None:
    text = format_result_message(_result(), cu_stack)

    assert "Shortest path resistance ≈ 5.000 mΩ" in text
    assert "[error]" not in text


def _row(text: str, label: str) -> str:
    return next(line for line in text.splitlines() if line.strip().startswith(label))


def test_failed_simulations_are_marked(
    format_result_message: Any, cu_stack: dict[int, Any]
) -> None:
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


def test_no_path(format_result_message: Any, cu_stack: dict[int, Any]) -> None:
    text = format_result_message(
        _result(short_path_resistance=None, resistance_dc=None, impedance_ac={}),
        cu_stack,
    )

    assert "No connection found" in text
    assert "ngspice simulation failed" in text


def test_directly_touching_elements_are_reported(no_wx: None) -> None:
    from parasitic import analyze_pcb_parasitic

    pad = {"type": "PAD", "layer": [0], "net_start": {0: 1}}
    result = analyze_pcb_parasitic({}, {}, pad, dict(pad))

    assert result["error"] is not None
    assert "touch each other directly" in result["error"]


def test_missing_ngspice_keeps_path_results(
    no_wx: None,
    monkeypatch: pytest.MonkeyPatch,
    coil_data: dict[Any, dict[str, Any]],
    cu_stack: dict[int, Any],
) -> None:
    import parasitic
    from get_parasitic import SpiceUnavailableError

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise SpiceUnavailableError("ngspice library could not be loaded")

    monkeypatch.setattr(parasitic, "simulate_network", unavailable)
    wires = [d for d in coil_data.values() if d["type"] == "WIRE"]

    result = parasitic.analyze_pcb_parasitic(coil_data, cu_stack, wires[0], wires[-1])
    text = parasitic.format_result_message(result, cu_stack)

    assert result["short_path_resistance"] is not None
    assert "Shortest path resistance" in text
    assert "ngspice library could not be loaded" in text
