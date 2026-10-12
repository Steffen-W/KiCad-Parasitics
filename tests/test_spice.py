"""ngspice simulation path (requires libngspice)."""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import get_parasitic as gp
import spice_worker
from analysis_progress import CancelledError, Stage
from get_parasitic import Resistor
from spice_worker import SpiceError, SpiceWorker

R_ONLY: list[gp.SpiceElement] = [("R", 1, 2, 0.1)]


@pytest.fixture
def spice() -> Iterator[SpiceWorker]:
    worker = SpiceWorker(lambda: None)
    yield worker
    worker.close()


def test_dc_matches_series_resistance(
    coil_data: dict[Any, dict[str, Any]], cu_stack: dict[int, Any]
) -> None:
    wires = [d for d in coil_data.values() if d["type"] == "WIRE"]
    conn1 = wires[0]["net_start"][0]
    conn2 = wires[-1]["net_start"][0]
    network = gp.extract_network(coil_data, cu_stack)
    _, _, short_path_res = gp.find_path(network, conn1, conn2)

    r_dc, z_ac, zin1, zin2 = gp.simulate_network(
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


def test_single_wire_impedances() -> None:
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


def test_failed_run_does_not_return_previous_result(
    spice: SpiceWorker, caplog: pytest.LogCaptureFixture
) -> None:
    floating = [*R_ONLY, ("R", 5, 6, 0.1)]

    assert gp._run_spice(spice, R_ONLY, 1, 2, ac_freq=1e3) == pytest.approx(0.1)
    process = spice._child
    with pytest.raises(SpiceError):
        gp._run_spice(spice, floating, 1, 2, ac_freq=1e3)
    assert any(r.name.startswith("ngspyce") for r in caplog.records)
    assert spice._child is process
    assert gp._run_spice(spice, R_ONLY, 1, 2, ac_freq=2e3) == pytest.approx(0.1)


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

    def record(spice: Any, elements: list[Any], *args: Any, **kw: Any) -> Any:
        simulated.extend(elements)
        return run_spice(spice, elements, *args, **kw)

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


def test_timed_out_worker_is_replaced(
    spice: SpiceWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert gp._run_spice(spice, R_ONLY, 1, 2) == pytest.approx(0.1)
    monkeypatch.setattr(spice_worker, "SPICE_TIMEOUT_S", 0.0)
    with pytest.raises(SpiceError, match="did not answer"):
        gp._run_spice(spice, R_ONLY, 1, 2)

    monkeypatch.undo()
    assert gp._run_spice(spice, R_ONLY, 1, 2) == pytest.approx(0.1)


def test_dead_worker_is_replaced(spice: SpiceWorker) -> None:
    assert gp._run_spice(spice, R_ONLY, 1, 2) == pytest.approx(0.1)
    assert spice._child is not None
    process, _ = spice._child
    process.kill()
    process.join()

    with pytest.raises(SpiceError, match="exited unexpectedly"):
        gp._run_spice(spice, R_ONLY, 1, 2)
    assert gp._run_spice(spice, R_ONLY, 1, 2) == pytest.approx(0.1)


def test_missing_library_raises_user_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIBNGSPICE", "/nonexistent/libngspice.so")

    with pytest.raises(spice_worker.SpiceUnavailableError, match="LIBNGSPICE"):
        gp.RunSimulation([Resistor(1, 2, 0.1, 1e-3)], 1, 2)


def test_progress_counts_every_simulation() -> None:
    reported: list[tuple[int, int]] = []

    def progress(stage: str, finished: int, total: int) -> bool:
        if stage is Stage.SPICE:
            reported.append((finished, total))
        return True

    wire = {"type": "WIRE", "nodes": (1, 2), "resistance": 0.1, "length": 1e-3}
    gp.RunSimulation(
        [Resistor(1, 2, 0.1, 1e-3)],
        1,
        2,
        network_info=[wire],
        frequencies=[1e3, 1e6],
        rload_far=50.0,
        progress=progress,
    )

    distinct = [r for i, r in enumerate(reported) if i == 0 or r != reported[i - 1]]
    assert distinct == [(finished, 7) for finished in range(7)]


def test_cancelled_progress_aborts_the_run() -> None:
    with pytest.raises(CancelledError):
        gp.RunSimulation([Resistor(1, 2, 0.1, 1e-3)], 1, 2, progress=lambda *_: False)


def test_unpicklable_child_error_keeps_its_message() -> None:
    class Local(Exception):
        pass

    error = spice_worker._transferable(Local("details"))

    assert isinstance(error, SpiceError)
    assert "details" in str(error)


def test_ac_elements_skip_non_conducting_elements() -> None:
    wire = {"type": "WIRE", "nodes": (1, 2), "resistance": 0.1, "length": 1e-3}
    island = {"type": "ZONE", "nodes": (5, 6), "resistance": 0.2}

    elements = gp._ac_elements([wire, island], 1e3, lambda n1, _n2: n1 == 1)

    assert elements == [("R", 1, 2, 0.1)]


def test_embedded_interpreter_is_derived_from_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "executable", "/usr/bin/kicad")
    monkeypatch.setattr(sys, "exec_prefix", str(tmp_path))
    with pytest.raises(spice_worker.SpiceUnavailableError, match="IPC"):
        spice_worker._python_interpreter()

    interpreter = tmp_path / ("python.exe" if os.name == "nt" else "bin/python3")
    interpreter.parent.mkdir(exist_ok=True)
    interpreter.touch(mode=0o644)
    if os.name != "nt":
        with pytest.raises(spice_worker.SpiceUnavailableError):
            spice_worker._python_interpreter()
        interpreter.chmod(0o755)
    assert spice_worker._python_interpreter() == str(interpreter)


def test_unstartable_interpreter_is_unavailable(
    spice: SpiceWorker, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        spice_worker, "_python_interpreter", lambda: str(tmp_path / "missing")
    )

    with pytest.raises(
        spice_worker.SpiceUnavailableError, match="could not be started"
    ):
        gp._run_spice(spice, R_ONLY, 1, 2)


def test_unload_frees_circuit_and_results() -> None:
    ngspyce = spice_worker._load_ngspyce()
    netlist = ["R1 1 0 2", "Vmeas 1 0 dc 1", ".dc Vmeas 1 1 1"]

    assert spice_worker._measure(ngspyce, netlist, "vmeas#branch") == pytest.approx(
        -0.5
    )
    spice_worker._unload(ngspyce)
    assert ngspyce.plots() == ["const"]
    assert ngspyce.cmd("setcirc") == []


def test_hanging_start_is_unavailable(
    spice: SpiceWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(spice_worker, "SPICE_TIMEOUT_S", 0.0)

    with pytest.raises(spice_worker.SpiceUnavailableError, match="did not answer"):
        gp._run_spice(spice, R_ONLY, 1, 2)


def test_failed_unload_ends_the_process_without_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenNgspyce:
        def plots(self) -> list[str]:
            return ["const"]

        def circ(self, _netlist: list[str]) -> int:
            return 1

        def cmd(self, command: str) -> list[str]:
            raise OSError(f"{command} failed")

    monkeypatch.setattr(spice_worker, "_load_ngspyce", BrokenNgspyce)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    parent, child = multiprocessing.Pipe()
    parent.send((["R1 1 0 1"], "v"))

    with pytest.raises(OSError, match="destroy all failed"):
        spice_worker._serve(child)

    assert parent.recv() == (None, [])
    assert not parent.poll()
