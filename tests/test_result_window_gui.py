"""Lifetime of the result window and its background work, with a real wx.

Needs wxPython and a display (on Linux e.g. `xvfb-run pytest`), so CI skips it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator
from types import ModuleType
from typing import Any, Callable

import pytest

from board_model import Selected, build_cu_stack, via, wire

wx = pytest.importorskip("wx")

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("linux")
    and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
    reason="needs a display",
)


class _Source:
    """Two vias joined by a 5 mm track; `selected` vias are selected."""

    def __init__(self, selected: tuple[str, ...] = ("v1", "v2")) -> None:
        self._selected = selected

    def selection(self) -> list[Selected]:
        return [Selected(i, "N") for i in self._selected]

    def net_ties(self) -> list[Any]:
        return []

    def copper(self, _nets: Any, _ties: Any) -> list[Any]:
        return [
            via("v1", "N", [0, 2], (0.0, 0.0), 0.6e-3, 0.3e-3),
            wire("w", "N", [0], (0.0, 0.0), (5e-3, 0.0), 0.2e-3),
            via("v2", "N", [0, 2], (5e-3, 0.0), 0.6e-3, 0.3e-3),
        ]

    def stackup(self) -> dict[int, Any]:
        return build_cu_stack([])


@pytest.fixture(scope="module")
def app() -> Iterator[Any]:
    app = wx.App()
    keeper = wx.Frame(None)  # keeps the main loop running without other windows
    yield app
    keeper.Destroy()


def _pump(until: Callable[[], bool], timeout: float = 60.0) -> None:
    """Run the main loop until `until()` holds."""
    end = time.monotonic() + timeout

    def check() -> None:
        if until() or time.monotonic() > end:
            wx.GetApp().ExitMainLoop()
        else:
            wx.CallLater(10, check)

    wx.CallLater(10, check)
    wx.GetApp().MainLoop()
    assert until(), "timed out"


@pytest.fixture
def rw(app: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """result_window imported with the real wx; fails on errors in callbacks."""
    monkeypatch.delitem(sys.modules, "result_window", raising=False)
    import result_window

    errors: list[Any] = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    yield result_window
    if result_window._window:
        result_window._window.Destroy()
    _pump(lambda: not result_window._window)
    assert not errors


def _idle(rw: ModuleType) -> bool:
    return (
        bool(rw._window)
        and rw._window._task is None
        and rw._window._status.GetLabel() != ""
    )


def test_window_is_reused_and_shows_the_result(rw: ModuleType) -> None:
    rw.run_plugin(_Source)
    _pump(lambda: _idle(rw))
    first = rw._window

    assert "Shortest distance" in first._summary.GetValue()
    assert first._calc.IsEnabled()
    first._status.SetLabel("")
    rw.run_plugin(_Source)
    _pump(lambda: _idle(rw))
    assert rw._window is first


def test_board_input_error_is_shown(rw: ModuleType) -> None:
    rw.run_plugin(lambda: _Source(selected=("v1",)))
    _pump(lambda: _idle(rw))

    assert "exactly two" in rw._window._summary.GetValue()
    assert rw._window._status.GetLabel().startswith("Not analyzed")


@pytest.mark.parametrize(
    ("error", "shown"),
    [
        (RuntimeError("boom"), "ERROR: RuntimeError: boom"),
        (MemoryError(), "out of memory"),
    ],
)
def test_failure_is_shown_briefly_and_logged(
    rw: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    shown: str,
) -> None:
    def fail(_board: Any, _task: Any) -> None:
        raise error

    monkeypatch.setattr(rw, "_analyze", fail)
    rw.run_plugin(_Source)
    _pump(lambda: _idle(rw))

    text = rw._window._summary.GetValue()
    assert shown in text and "Traceback" not in text
    assert "Traceback" in caplog.text
    assert rw._window._status.GetLabel() == "Failed."


def test_closing_cancels_the_analysis(
    rw: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from analysis_progress import Stage, report

    cancelled = threading.Event()

    def endless(_board: Any, task: Any) -> None:
        try:
            while True:
                report(task.report, Stage.EXTRACT, 0, 1)
                time.sleep(0.01)
        finally:
            cancelled.set()

    monkeypatch.setattr(rw, "_analyze", endless)
    rw.run_plugin(_Source)
    _pump(lambda: bool(rw._window) and rw._window._task is not None)
    first = rw._window

    first.Close()
    _pump(lambda: cancelled.is_set() and not first)

    rw.run_plugin(_Source)
    assert rw._window is not first


def test_result_arriving_after_closing_is_dropped(
    rw: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    workers: list[threading.Thread] = []

    def slow_inductance(_result: Any, _cu_stack: Any) -> dict[str, Any]:
        workers.append(threading.current_thread())
        release.wait(10)
        return {"message": "too late"}

    monkeypatch.setattr(rw, "_inductance", slow_inductance)
    rw.run_plugin(_Source)
    _pump(lambda: _idle(rw))
    window = rw._window
    window._on_calculate_inductance(None)
    window.Close()
    _pump(lambda: not window and bool(workers))

    release.set()
    workers[0].join(10)
    _pump(lambda: True)  # runs the late outcome's CallAfter into the closed window
