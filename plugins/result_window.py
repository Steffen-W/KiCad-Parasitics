"""Result window shared by the pcbnew and the IPC plugin (see README, Result window)."""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import re
import threading
import warnings
from typing import Any, Callable, TextIO, TypeVar

import wx

try:
    from .analysis_progress import CancelledError, Stage
    from .board_model import Board, BoardInputError, BoardSource, read_board
    from .connect_nets import Connect_Nets
    from .network_display import format_network_info
    from .parasitic import analyze_pcb_parasitic, format_result_message
    from .pcb_types import CuLayer
    from .plugin_logging import LOG_FILE
except ImportError:
    from analysis_progress import CancelledError, Stage
    from board_model import Board, BoardInputError, BoardSource, read_board
    from connect_nets import Connect_Nets
    from network_display import format_network_info
    from parasitic import analyze_pcb_parasitic, format_result_message
    from pcb_types import CuLayer
    from plugin_logging import LOG_FILE

log = logging.getLogger(__name__)

T = TypeVar("T")

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_FREQUENCIES = [1e3, 10e3, 100e3, 1e6, 10e6, 100e6, 1e9]
_TICK_MS = 100
# How long a (re)opened window stays above all others before it stacks normally.
_ON_TOP_MS = 1000


# ---------------------------------------------------------------------------
# Work done in the background thread
# ---------------------------------------------------------------------------


class _Task:
    """Background work; its latest progress is read by the GUI thread."""

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.progress: tuple[Stage, int, int] | None = None

    def report(self, stage: Stage, finished: int, total: int) -> bool:
        """Progress callback (see analysis_progress.Progress)."""
        self.progress = (stage, finished, total)
        return not self.cancel.is_set()


def _progress_text(stage: Stage, finished: int, total: int) -> str:
    stages = list(Stage)
    step = f"Step {stages.index(stage) + 1} of {len(stages)}"
    return f"{step}: {stage.value} ({finished} of {total} done)"


def _save_debug_file(name: str, data: Any) -> None:
    """Write `data` as JSON next to the plugin, coordinate pairs on one line."""
    text = re.sub(
        r"\[\s+(-?[\d.e+\-]+),\s+(-?[\d.e+\-]+)\s+\]",
        r"[\1, \2]",
        json.dumps(data, indent=1, default=str),
    )
    try:
        with open(os.path.join(_PLUGIN_DIR, name), "w", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        log.warning("Writing %s failed", name, exc_info=True)


def _analyze(board: Board, task: _Task) -> dict[str, Any]:
    """Analysis between the two selected elements of `board`.

    With PARASITIC_DEBUG set, the elements and the network are also written to
    files (debug_ItemList.json is read by plot_pcb.py).
    """
    debug = bool(os.environ.get("PARASITIC_DEBUG"))
    data = Connect_Nets(board.elements)
    if debug:
        _save_debug_file("debug_ItemList.json", data)
    first, second = (d for d in data.values() if d["is_selected"])
    result = analyze_pcb_parasitic(
        data,
        board.cu_stack,
        first,
        second,
        frequencies=_FREQUENCIES,
        progress=task.report,
    )
    if debug and result["network_info"]:
        network = [
            {k: v for k, v in e.items() if k != "hf"} for e in result["network_info"]
        ]
        _save_debug_file("debug_network_info.json", network)
    return result


def _inductance(result: dict[str, Any], cu_stack: dict[int, CuLayer]) -> dict[str, Any]:
    """Loop inductance of the shortest path (see calc_path_inductance)."""
    # bfieldtools still imports pkg_resources, which warns on every start.
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    try:
        try:
            from .calc_inductance import calc_path_inductance
        except ImportError:
            from calc_inductance import calc_path_inductance
    except ImportError as e:
        log.exception("Inductance dependency import failed")
        missing = f"'{e.name}'" if e.name else str(e)
        return {
            "message": (
                f"ERROR: a required dependency could not be imported ({missing}).\n\n"
                "The inductance dependencies are installed automatically only\n"
                "when the plugin runs via the KiCad IPC API, which manages its\n"
                "own virtual environment:\n"
                "    KiCad -> Settings -> Plugins -> Enable KiCad API\n\n"
                "If you run the plugin outside that environment, install the\n"
                "requirements into that interpreter yourself:\n"
                "    pip install -r requirements.txt"
            )
        }
    log.info("Inductance calculation started (%d path nodes)", len(result["path"]))
    return calc_path_inductance(result["path"], result["network_info"], cu_stack)


def _failure_text(error: Exception) -> str:
    """What the window shows for an unexpected error; its traceback is logged."""
    if isinstance(error, MemoryError):
        return (
            "ERROR: out of memory. The selected net is too large for the available RAM."
        )
    return f"ERROR: {type(error).__name__}: {error}\n\nThe traceback is in {LOG_FILE}"


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


def _bring_to_front(frame: wx.Frame) -> None:
    """Show `frame` above all windows for a moment, then let it stack normally.

    The IPC plugin's window belongs to its own process, and window managers
    often open such a window behind KiCad.
    """
    if frame.IsIconized():
        frame.Iconize(False)
    style = frame.GetWindowStyleFlag() & ~wx.STAY_ON_TOP
    frame.SetWindowStyleFlag(style | wx.STAY_ON_TOP)
    frame.Show()
    frame.Raise()
    wx.CallLater(_ON_TOP_MS, lambda: frame and frame.SetWindowStyleFlag(style))


class ResultFrame(wx.Frame):
    """Progress, result, network details and inductance of one analysis."""

    def __init__(
        self, parent: wx.Window | None, open_board: Callable[[], BoardSource]
    ) -> None:
        super().__init__(parent, title="Parasitics", size=wx.Size(860, 640))
        self._open_board = open_board
        self._board: Board | None = None
        self._result: dict[str, Any] | None = None
        self._inductance: dict[str, Any] | None = None
        self._task: _Task | None = None
        self._timer = wx.Timer(self)

        panel = self._panel = wx.Panel(self)
        font = wx.Font(wx.FontInfo(10).Family(wx.FONTFAMILY_TELETYPE))
        self._status = wx.StaticText(panel, style=wx.ST_ELLIPSIZE_END)
        self._gauge = wx.Gauge(panel, range=100, size=wx.Size(160, -1))
        self._cancel = wx.Button(panel, wx.ID_CANCEL)

        notebook = wx.Notebook(panel)
        self._summary = self._text_page(notebook, "Result", font)
        self._details = self._text_page(notebook, "Network", font)
        page = wx.Panel(notebook)
        self._calc = wx.Button(page, label="Calculate inductance")
        self._plots = wx.Button(page, label="Debug plots")
        self._ind_text = self._text(page, font)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self._calc, 0, wx.RIGHT, 5)
        row.Add(self._plots)
        column = wx.BoxSizer(wx.VERTICAL)
        column.Add(row, 0, wx.ALL, 5)
        column.Add(self._ind_text, 1, wx.EXPAND)
        page.SetSizer(column)
        notebook.AddPage(page, "Inductance")

        self._analyze_btn = wx.Button(panel, label="Analyze selection")
        bottom = wx.BoxSizer(wx.HORIZONTAL)
        bottom.Add(self._status, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        bottom.Add(self._gauge, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        for button in (self._cancel, self._analyze_btn):
            bottom.Add(button, 0, wx.RIGHT, 5)
        bottom.Add(wx.Button(panel, wx.ID_CLOSE))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(bottom, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        panel.SetSizer(sizer)

        self.Bind(wx.EVT_TIMER, self._on_tick, self._timer)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
        self._analyze_btn.Bind(wx.EVT_BUTTON, lambda _e: self.analyze())
        self._calc.Bind(wx.EVT_BUTTON, self._on_calculate_inductance)
        self._plots.Bind(wx.EVT_BUTTON, self._on_debug_plots)
        self.Bind(wx.EVT_BUTTON, lambda _e: self.Close(), id=wx.ID_CLOSE)
        self._set_idle("")

    @staticmethod
    def _text(parent: wx.Window, font: wx.Font) -> wx.TextCtrl:
        text = wx.TextCtrl(parent, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        text.SetFont(font)
        return text

    def _text_page(
        self, notebook: wx.Notebook, title: str, font: wx.Font
    ) -> wx.TextCtrl:
        text = self._text(notebook, font)
        notebook.AddPage(text, title)
        return text

    # --- state ---------------------------------------------------------------

    def _show_progress(self, busy: bool, cancellable: bool) -> None:
        self._gauge.SetValue(0)
        self._gauge.Show(busy)
        self._cancel.Show(cancellable)
        self._cancel.Enable(cancellable)
        self._panel.Layout()

    def _set_busy(self, status: str, cancellable: bool) -> None:
        self._status.SetLabel(status)
        self._show_progress(True, cancellable)
        for button in (self._analyze_btn, self._calc, self._plots):
            button.Disable()
        self._timer.Start(_TICK_MS)

    def _set_idle(self, status: str) -> None:
        self._timer.Stop()
        self._status.SetLabel(status)
        self._show_progress(False, False)
        self._analyze_btn.Enable()
        self._calc.Enable(bool(self._result and self._result.get("path")))
        self._plots.Enable(
            bool(self._inductance and self._inductance.get("_debug_data"))
        )

    def _run(
        self,
        status: str,
        work: Callable[[_Task], T],
        done: Callable[[T], None],
        page: wx.TextCtrl,
        cancellable: bool,
    ) -> None:
        """Run `work` in a background thread, then `done` with its outcome.

        An unexpected error in either is shown on `page` (see _failure_text).
        """
        task = self._task = _Task()
        self._set_busy(status, cancellable)

        def finish(show: Callable[[], None], final: str) -> None:
            if not self:  # the window was closed meanwhile
                return
            self._task = None
            try:
                show()
            except Exception as e:
                log.exception("Showing the outcome of %r failed", status)
                page.SetValue(_failure_text(e))
                final = "Failed."
            self._set_idle(final)

        def run() -> None:
            try:
                outcome = work(task)
            except CancelledError:
                wx.CallAfter(finish, lambda: None, "Cancelled.")
            except Exception as e:
                log.exception("%s failed", status)
                text = _failure_text(e)
                wx.CallAfter(finish, lambda: page.SetValue(text), "Failed.")
            else:
                wx.CallAfter(finish, lambda: done(outcome), "Done.")

        threading.Thread(target=run, daemon=True).start()

    # --- actions ---------------------------------------------------------------

    def analyze(self) -> None:
        """Bring the window to front and analyze the current selection.

        While work is running, only brings the window to front.
        """
        _bring_to_front(self)
        if self._task:
            return
        self._set_busy("Reading board…", cancellable=False)
        self.Update()  # paint before the GUI thread is busy reading
        try:
            board = read_board(self._open_board())
        except BoardInputError as e:
            log.warning("Board input: %s", e)
            self._show_board_message(str(e))
            return
        except Exception as e:
            log.exception("Reading the board failed")
            self._show_board_message(_failure_text(e))
            return
        self._show_board(board)
        self._run(
            "Analyzing…",
            lambda task: _analyze(board, task),
            self._show_result,
            self._summary,
            cancellable=True,
        )

    def _show_board(self, board: Board | None) -> None:
        """Start over with `board`: drop the previous results and their plots."""
        self._board, self._result, self._inductance = board, None, None
        self.SetTitle("Parasitics" + (f" – {', '.join(board.nets)}" if board else ""))
        for text in (self._summary, self._details, self._ind_text):
            text.Clear()
        for child in self.GetChildren():
            if isinstance(child, wx.Frame):
                child.Close()

    def _show_board_message(self, message: str) -> None:
        self._show_board(None)
        self._summary.SetValue(message)
        self._set_idle("Not analyzed: resolve this in KiCad, then analyze again.")

    def _show_result(self, result: dict[str, Any]) -> None:
        board = self._board
        assert board is not None
        self._result = result
        self._summary.SetValue(
            format_result_message(
                result, board.cu_stack, net_ties=board.net_ties, nets=board.nets
            )
        )
        if result["network_info"]:
            self._details.SetValue(
                f"Measuring resistance: n{result['conn1']} <-> n{result['conn2']}\n"
                + format_network_info(
                    result["network_info"], result["graph"], result["path"]
                )
            )

    def _on_calculate_inductance(self, _event: wx.CommandEvent) -> None:
        result, board = self._result, self._board
        assert result is not None and board is not None
        self._run(
            "Calculating inductance…",
            lambda _task: _inductance(result, board.cu_stack),
            self._show_inductance,
            self._ind_text,
            cancellable=False,
        )

    def _show_inductance(self, ind: dict[str, Any]) -> None:
        self._inductance = ind
        self._ind_text.SetValue(ind["message"])

    def _on_debug_plots(self, _event: wx.CommandEvent) -> None:
        ind = self._inductance
        assert ind is not None
        try:
            try:
                from .inductance_plots import show_debug_plots
            except ImportError:
                from inductance_plots import show_debug_plots
            show_debug_plots(ind["_debug_data"], ind["segments"], parent=self)
        except Exception as e:
            log.exception("Debug plots failed")
            self._ind_text.AppendText(f"\n\n{_failure_text(e)}")

    # --- events ------------------------------------------------------------------

    def _on_tick(self, _event: wx.TimerEvent) -> None:
        progress = self._task.progress if self._task else None
        if progress is None:
            self._gauge.Pulse()
            return
        stage, finished, total = progress
        self._gauge.SetValue(100 * finished // total)
        self._status.SetLabel(_progress_text(stage, finished, total))

    def _on_cancel(self, _event: wx.CommandEvent) -> None:
        if self._task:
            self._task.cancel.set()
            self._cancel.Disable()
            self._status.SetLabel("Cancelling…")

    def _on_close(self, event: wx.CloseEvent) -> None:
        if self._task:
            self._task.cancel.set()
        self._timer.Stop()
        event.Skip()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

_window: ResultFrame | None = None
_crash_log: TextIO | None = None


def _enable_crash_log() -> None:
    """Write tracebacks of native crashes, which no try/except can catch, to crash.log.

    faulthandler writes to the file until the process exits, so it stays open.
    """
    global _crash_log
    if _crash_log is not None:
        return
    try:
        _crash_log = open(  # noqa: SIM115
            os.path.join(_PLUGIN_DIR, "crash.log"), "a", buffering=1, encoding="utf-8"
        )
    except OSError:
        log.warning("crash.log not writable, native crashes are not recorded")
        return
    faulthandler.enable(file=_crash_log, all_threads=True)


def run_plugin(
    open_board: Callable[[], BoardSource], parent: wx.Window | None = None
) -> None:
    """Analyze the board's current selection in the result window.

    Creates the window if there is none and returns at once; the analysis
    starts from the event loop. `open_board` is called for every analysis, so
    the window always reads the current board and selection.
    """
    global _window
    _enable_crash_log()
    if not _window:  # none yet, or closed
        _window = ResultFrame(parent, open_board)
    window = _window

    def analyze() -> None:
        if window:  # not closed before the event loop got here
            window.analyze()

    wx.CallAfter(analyze)
