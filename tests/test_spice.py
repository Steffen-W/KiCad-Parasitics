"""ngspice simulation path (requires libngspice)."""

from __future__ import annotations

import math
import sys
import threading
from collections.abc import Iterator
from typing import Any

import pytest

import get_parasitic as gp
from get_parasitic import Resistor


def _assert_ngspice_released() -> None:
    ngspyce = gp.load_ngspyce()
    assert ngspyce.plots() == ["const"]
    assert ngspyce.cmd("setcirc") == []


def test_dc_matches_series_resistance(
    coil_data: dict[Any, dict[str, Any]], cu_stack: dict[int, Any]
) -> None:
    wires = [d for d in coil_data.values() if d["type"] == "WIRE"]
    conn1 = wires[0]["net_start"][0]
    conn2 = wires[-1]["net_start"][0]
    network = gp.extract_network(coil_data, cu_stack)
    _, _, short_path_res = gp.find_path(network, conn1, conn2)

    r_dc, z_ac, zin1, zin2, _ = gp.simulate_network(
        network, conn1, conn2, cu_stack, frequencies=[1e3], rload_far=50.0
    )

    assert short_path_res is not None
    assert r_dc == pytest.approx(short_path_res, rel=1e-6)
    assert z_ac[1e3] is not None
    assert abs(z_ac[1e3]) == pytest.approx(r_dc, rel=1e-3)
    assert zin1[1e3] is not None and zin2[1e3] is not None


def test_failed_simulation_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise gp.SpiceError("forced failure")

    monkeypatch.setattr(gp, "_run_spice", fail)

    r_dc, z_ac, zin1, zin2 = gp.RunSimulation(
        [Resistor(1, 2, 0.1, 1e-3)], 1, 2, frequencies=[1e3], rload_far=50.0
    )

    assert (r_dc, z_ac, zin1, zin2) == (None, {1e3: None}, {1e3: None}, {1e3: None})


def test_simulation_releases_ngspice_state() -> None:
    wire = {"type": "WIRE", "nodes": (1, 2), "resistance": 0.1, "length": 1e-3}
    r_dc, z_ac, zin1, _ = gp.RunSimulation(
        [Resistor(1, 2, 0.1, 1e-3)],
        1,
        2,
        network_info=[wire],
        frequencies=[1e3],
        rload_far=50.0,
    )

    assert r_dc == pytest.approx(0.1)
    assert z_ac[1e3] == pytest.approx(0.1)
    assert zin1[1e3] == pytest.approx(50.1)
    _assert_ngspice_released()


def test_failed_run_does_not_return_previous_result() -> None:
    ngspyce = gp.load_ngspyce()
    good = [("R", 1, 2, 0.1)]
    floating = [("R", 1, 2, 0.1), ("R", 5, 6, 0.1)]

    assert gp._run_spice(ngspyce, good, 1, 2, ac_freq=1e3) == pytest.approx(0.1)
    with pytest.raises(gp.SpiceError):
        gp._run_spice(ngspyce, floating, 1, 2, ac_freq=1e3)
    _assert_ngspice_released()


def test_unconnected_copper_is_not_simulated() -> None:
    island = {"type": "WIRE", "nodes": (5, 6), "resistance": 0.1, "length": 1e-3}
    wire = {"type": "WIRE", "nodes": (1, 2), "resistance": 0.1, "length": 1e-3}

    r_dc, z_ac, *_ = gp.RunSimulation(
        [Resistor(1, 2, 0.1, 1e-3), Resistor(5, 6, 0.1, 1e-3)],
        1,
        2,
        network_info=[wire, island],
        frequencies=[1e3],
    )

    assert r_dc == pytest.approx(0.1)
    assert z_ac[1e3] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("resistors", "rload_far"),
    [
        ([Resistor(1, 2, math.inf, 1e-3)], None),
        ([Resistor(1, 2, -0.1, 1e-3)], None),
        ([Resistor(1, 2, 0.1, 1e-3)], 0.0),
    ],
    ids=["non-finite element", "non-positive DC resistance", "invalid load"],
)
def test_invalid_simulation_yields_none(
    resistors: list[Resistor], rload_far: float | None
) -> None:
    wire = {"type": "WIRE", "nodes": (1, 2), "resistance": 0.1, "length": 1e-3}
    r_dc, _, zin1, _ = gp.RunSimulation(
        resistors, 1, 2, network_info=[wire], frequencies=[1e3], rload_far=rload_far
    )

    if rload_far is None:
        assert r_dc is None
    else:
        assert zin1 == {1e3: None}


def test_self_loop_elements_are_not_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    simulated: list[gp.SpiceElement] = []
    run_spice = gp._run_spice

    def record(ngspyce: Any, elements: list[Any], *args: Any, **kw: Any) -> Any:
        simulated.extend(elements)
        return run_spice(ngspyce, elements, *args, **kw)

    monkeypatch.setattr(gp, "_run_spice", record)
    loop = {"type": "VIA", "nodes": (2, 2), "resistance": 0.05, "length": 1e-3}
    wire = {"type": "WIRE", "nodes": (1, 2), "resistance": 0.1, "length": 1e-3}

    r_dc, *_ = gp.RunSimulation(
        [Resistor(1, 2, 0.1, 1e-3), Resistor(2, 2, 0.05, 1e-3)],
        1,
        2,
        network_info=[wire, loop],
        frequencies=[1e3],
    )

    assert r_dc == pytest.approx(0.1)
    assert simulated
    assert all(na != nb for _, na, nb, _ in simulated)


def test_timed_out_call_blocks_further_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gp, "SPICE_TIMEOUT_S", 0.5)
    monkeypatch.setattr(gp, "_timed_out_call", None)
    release = threading.Event()

    try:
        with pytest.raises(gp.SpiceTimeoutError, match="did not return"):
            gp._run_native_spice_call(release.wait)
        with pytest.raises(gp.SpiceTimeoutError, match="still blocked"):
            gp._run_native_spice_call(lambda: None)
    finally:
        release.set()

    assert gp._timed_out_call is not None
    gp._timed_out_call.join()
    gp._run_native_spice_call(lambda: None)


def test_missing_library_raises_user_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [m for m in sys.modules if m == "ngspyce" or m.startswith("ngspyce.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setenv("LIBNGSPICE", "/nonexistent/libngspice.so")

    with pytest.raises(gp.SpiceUnavailableError, match="LIBNGSPICE"):
        gp.load_ngspyce()


@pytest.fixture
def ngspice_reset() -> Iterator[None]:
    """Unload every circuit and plot a test leaves behind."""
    yield
    ngspyce = gp.load_ngspyce()
    ngspyce.cmd("destroy all")
    while ngspyce.cmd("setcirc"):
        ngspyce.cmd("remcirc")


def test_rejected_netlist_is_a_failed_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ngspyce = gp.load_ngspyce()
    monkeypatch.setattr(ngspyce, "circ", lambda _lines: 1)

    with pytest.raises(gp.SpiceError, match="rejected"):
        gp._run_spice(ngspyce, [("R", 1, 2, 0.1)], 1, 2)

    monkeypatch.undo()
    assert gp._run_spice(ngspyce, [("R", 1, 2, 0.1)], 1, 2) == pytest.approx(0.1)
    _assert_ngspice_released()


@pytest.mark.parametrize("failing_command", ["destroy all", "remcirc"])
@pytest.mark.usefixtures("ngspice_reset")
def test_failed_cleanup_keeps_result_and_original_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failing_command: str,
) -> None:
    ngspyce = gp.load_ngspyce()
    cmd = ngspyce.cmd

    def failing_cmd(command: str) -> Any:
        if command == failing_command:
            raise RuntimeError("cleanup failed")
        return cmd(command)

    monkeypatch.setattr(ngspyce, "cmd", failing_cmd)

    assert gp._run_spice(ngspyce, [("R", 1, 2, 0.1)], 1, 2) == pytest.approx(0.1)
    with pytest.raises(gp.SpiceError, match="Implausible"):
        gp._run_spice(ngspyce, [("R", 1, 2, -0.1)], 1, 2)
    assert "Could not release ngspice state" in caplog.text

    monkeypatch.undo()
    assert gp._run_spice(ngspyce, [("R", 1, 2, 0.2)], 1, 2) == pytest.approx(0.2)
