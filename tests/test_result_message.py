"""Result table formatting, including failed simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from analysis_progress import CancelledError, Stage


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
        "skipped": [],
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
    from spice_worker import SpiceUnavailableError

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise SpiceUnavailableError("ngspice library could not be loaded")

    monkeypatch.setattr(parasitic, "simulate_network", unavailable)
    wires = [d for d in coil_data.values() if d["type"] == "WIRE"]

    result = parasitic.analyze_pcb_parasitic(coil_data, cu_stack, wires[0], wires[-1])
    text = parasitic.format_result_message(result, cu_stack)

    assert result["short_path_resistance"] is not None
    assert result["network_info"]
    assert "Shortest path resistance" in text
    assert "ngspice library could not be loaded" in text


def test_zones_are_rejected_as_measurement_points(no_wx: None) -> None:
    from parasitic import analyze_pcb_parasitic

    zone = {"type": "ZONE", "layer": [0], "net_start": {}}
    result = analyze_pcb_parasitic({}, {}, zone, dict(zone))

    assert result["error"] is not None
    assert "zones have no single connection point" in result["error"]


@pytest.mark.parametrize("cancel_stage", list(Stage))
def test_cancel_ends_the_analysis(
    no_wx: None,
    coil_data: dict[Any, dict[str, Any]],
    cu_stack: dict[int, Any],
    cancel_stage: Stage,
) -> None:
    import parasitic

    wires = [d for d in coil_data.values() if d["type"] == "WIRE"]

    with pytest.raises(CancelledError):
        parasitic.analyze_pcb_parasitic(
            coil_data,
            cu_stack,
            wires[0],
            wires[-1],
            progress=lambda stage, _finished, _total: stage != cancel_stage,
        )


def test_dialog_progress_shows_every_new_stage(no_wx: None) -> None:
    import parasitic

    shown: list[str] = []

    class Dialog:
        def Update(self, _value: int, message: str) -> tuple[bool, bool]:
            shown.append(message)
            return True, False

    update = parasitic._dialog_progress(Dialog())
    update(Stage.EXTRACT, 0, 2)
    update(Stage.EXTRACT, 1, 2)
    update(Stage.PATH, 0, 1)

    assert shown == [
        "Step 1 of 4: Extracting network (0 of 2 done)",
        "Step 2 of 4: Finding path (0 of 1 done)",
    ]


def test_import_failure_is_logged_and_shown(
    no_wx: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import logging
    import sys

    import ipc_plugin
    import plugin_logging

    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)
    for name in plugin_logging.QUIET_LOGGERS:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
    log_file = tmp_path / "debug.log"
    monkeypatch.setattr(plugin_logging, "LOG_FILE", str(log_file))
    monkeypatch.setitem(sys.modules, "parasitic", None)

    with pytest.raises(SystemExit) as exit_info:
        ipc_plugin.main()
    for handler in root.handlers:
        handler.close()

    assert exit_info.value.code == 1
    logged = log_file.read_text(encoding="utf-8")
    assert "Importing the plugin failed" in logged
    assert "ModuleNotFoundError" in logged or "ImportError" in logged
    shown = sys.modules["wx"].MessageDialog.call_args.args[1]
    assert "could not start" in shown
    assert "parasitic" in shown


@pytest.mark.parametrize("error", [None, "No connection found"])
def test_skipped_elements_are_reported(
    format_result_message: Any, cu_stack: dict[int, Any], error: str | None
) -> None:
    result = _result(skipped=["WIRE on layer 0: resistance nan"], error=error)

    text = format_result_message(result, cu_stack)

    assert text.startswith("Warning: 1 modelling issue(s)")
