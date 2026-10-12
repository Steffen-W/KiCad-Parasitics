from __future__ import annotations

import faulthandler
import json
import logging
import math
import os
import re
import time
import traceback
import warnings
from typing import Any, TextIO

import wx

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

try:
    from .connect_nets import NOT_ASSIGNED, Connect_Nets
except ImportError:
    from connect_nets import NOT_ASSIGNED, Connect_Nets
try:
    from .get_distance import find_all_reachable_nodes
except ImportError:
    from get_distance import find_all_reachable_nodes
try:
    from .get_parasitic import (
        extract_network,
        find_path,
        simulate_network,
    )
except ImportError:
    from get_parasitic import (
        extract_network,
        find_path,
        simulate_network,
    )
try:
    from .analysis_progress import (
        CancelledError,
        Progress,
        Stage,
        no_progress,
        report,
    )
    from .spice_worker import SpiceUnavailableError
except ImportError:
    from analysis_progress import (
        CancelledError,
        Progress,
        Stage,
        no_progress,
        report,
    )
    from spice_worker import SpiceUnavailableError
try:
    from .network_display import format_network_info
except ImportError:
    from network_display import format_network_info
try:
    from .pcb_types import CuLayer
except ImportError:
    from pcb_types import CuLayer

log = logging.getLogger(__name__)

# far-end termination for single-ended Zin measurement
_ZIN_TERMINATION_OHM: float = 50.0

_PROGRESS_REFRESH_S = 0.1


def _build_path_error(
    element1: dict[str, Any],
    element2: dict[str, Any],
    conn1: int,
    conn2: int,
    graph: dict[int, Any],
    cu_stack: dict[int, CuLayer],
) -> str:
    """Diagnostic message explaining why no path was found between two nodes."""

    def layer_names(elem: dict[str, Any]) -> str:
        return ", ".join(
            cu_stack[layer_id]["name"] if layer_id in cu_stack else str(layer_id)
            for layer_id in elem.get("layer", [])
        )

    def elem_line(label: str, elem: dict[str, Any], conn: int) -> str:
        in_graph = conn in graph
        status = (
            f"in graph ({len(graph.get(conn, {}))} neighbors)"
            if in_graph
            else "NOT in graph"
        )
        return f"  {label}: {elem.get('type', '?')}  net={elem.get('net_name', '?')!r}  layer={layer_names(elem)}  node={conn}  {status}"

    in1, in2 = conn1 in graph, conn2 in graph
    if not in1 and not in2:
        reason = "Neither element is connected to any trace in the copper network."
    elif not in1:
        reason = "Element 1 has no connected traces in the copper network."
    elif not in2:
        reason = "Element 2 has no connected traces in the copper network."
    else:
        reachable = find_all_reachable_nodes(graph, conn1)
        if conn2 in reachable:
            reason = "Nodes are connected by BFS but path finding failed — check log."
        else:
            reason = (
                f"Both nodes exist but are in separate copper islands "
                f"(El.1 can reach {len(reachable)}/{len(graph)} nodes).\n"
                f"Check for copper gaps or missing vias."
            )

    return "\n".join(
        [
            "No path found between selected elements.",
            elem_line("El.1", element1, conn1),
            elem_line("El.2", element2, conn2),
            f"  Network: {len(graph)} nodes total",
            reason,
        ]
    )


def analyze_pcb_parasitic(
    data: dict[str, Any],
    CuStack: dict[int, CuLayer],
    element1: dict[str, Any],
    element2: dict[str, Any],
    frequencies: list[float] | None = None,
    progress: Progress = no_progress,
) -> dict[str, Any]:
    """Analyze parasitic resistance/impedance between two PCB elements.

    Args:
        data: PCB element data from Connect_Nets()
        CuStack: Copper stackup from Get_PCB_Stackup_fun()
        element1, element2: Selected element dicts
        frequencies: List of frequencies for AC analysis (Hz)
        progress: reports each analysis stage; see Progress

    Returns:
        dict with: resistance_dc, impedance_ac, distance, short_path_resistance,
                   area, network_info, graph, path, conn1, conn2, error,
                   spice_unavailable (message when ngspice cannot be started),
                   skipped (elements left out of the network, see extract_network)
    Raises:
        CancelledError: `progress` asked to stop
    """
    result: dict[str, Any] = {
        "resistance_dc": None,
        "impedance_ac": {},
        "distance": float("inf"),
        "short_path_resistance": None,
        "area": {},
        "network_info": None,
        "graph": None,
        "path": None,
        "conn1": None,
        "conn2": None,
        "error": None,
        "spice_unavailable": None,
        "skipped": [],
    }

    network = extract_network(data, CuStack, progress=progress)
    result["skipped"] = network["skipped"]
    best_distance, best_conn = _best_connection(network, element1, element2, progress)

    if not best_conn:
        best_conn = (
            element1["net_start"].get(element1["layer"][0], NOT_ASSIGNED),
            element2["net_start"].get(element2["layer"][0], NOT_ASSIGNED),
            element1["layer"][0],
            element2["layer"][0],
        )

    conn1, conn2 = best_conn[0], best_conn[1]
    result["conn1"], result["conn2"] = conn1, conn2
    if NOT_ASSIGNED in (conn1, conn2):
        result["error"] = (
            "Select pads, vias or tracks as measurement points; "
            "zones have no single connection point."
        )
        return result
    if conn1 == conn2:
        result["error"] = (
            "The selected elements touch each other directly, "
            "so there is no copper path between them to analyze."
        )
        return result

    # Get full path info for best connection
    Distance, path, short_path_RES = find_path(network, conn1, conn2)

    log.info(
        "Best path: conn1=%s, conn2=%s, layers=%s/%s, distance=%.3fmm",
        conn1,
        conn2,
        best_conn[2],
        best_conn[3],
        best_distance * 1000,
    )

    if math.isinf(Distance):
        result["error"] = _build_path_error(
            element1, element2, conn1, conn2, network["graph"], CuStack
        )
        log.error("%s", result["error"])
        return result

    result.update(
        {
            "distance": Distance,
            "short_path_resistance": short_path_RES,
            "area": network["area"],
            "network_info": network["network_info"],
            "graph": network["graph"],
            "path": path,
        }
    )

    try:
        Resistance, Z_ac, Z_ac_conn1, Z_ac_conn2 = simulate_network(
            network,
            conn1,
            conn2,
            CuStack,
            frequencies=frequencies,
            rload_far=_ZIN_TERMINATION_OHM,
            progress=progress,
        )
    except SpiceUnavailableError as e:
        log.error("%s", e)
        result["spice_unavailable"] = str(e)
        return result

    result.update(
        {
            "resistance_dc": Resistance,
            "impedance_ac": Z_ac,
            "impedance_ac_conn1": Z_ac_conn1,
            "impedance_ac_conn2": Z_ac_conn2,
            "zin_termination_ohm": _ZIN_TERMINATION_OHM,
        }
    )
    return result


def _best_connection(
    network: dict[str, Any],
    element1: dict[str, Any],
    element2: dict[str, Any],
    progress: Progress,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Shortest path over all layer combinations of the two elements.

    Returns:
        (distance, (conn1, conn2, layer1, layer2)); (inf, None) if none connects
    """
    layer_pairs = [
        (layer1, layer2)
        for layer1 in element1.get("layer", [])
        for layer2 in element2.get("layer", [])
    ]
    best_distance = float("inf")
    best_conn = None
    for finished, (layer1, layer2) in enumerate(layer_pairs):
        report(progress, Stage.PATH, finished, len(layer_pairs))
        conn1 = element1["net_start"].get(layer1, NOT_ASSIGNED)
        conn2 = element2["net_start"].get(layer2, NOT_ASSIGNED)
        if NOT_ASSIGNED in (conn1, conn2):
            continue
        distance, _, _ = find_path(network, conn1, conn2)
        if distance < best_distance:
            best_distance = distance
            best_conn = (conn1, conn2, layer1, layer2)
    return best_distance, best_conn


def format_result_message(
    result: dict[str, Any],
    CuStack: dict[int, CuLayer],
    net_tie_info: dict[str, Any] | None = None,
) -> str:
    """Format analysis result as human-readable message."""
    lines = []
    if result["skipped"]:
        lines += [
            (
                f"Warning: {len(result['skipped'])} modelling issue(s), so the result "
                "may be incomplete (see debug_parasitic.log)."
            ),
            "",
        ]

    if result["error"]:
        return "\n".join([*lines, str(result["error"])])

    if net_tie_info:
        lines.append("Net tie:")
        for ref in net_tie_info["refs"]:
            lines.append(f"  {ref}")
        lines.append("Bridged nets:")
        for net in net_tie_info["nets"]:
            lines.append(f"  {net}")
        lines.append("")

    Distance = result["distance"]
    short_path_RES = result["short_path_resistance"]
    Resistance = result["resistance_dc"]
    Z_ac = result["impedance_ac"]
    Z_ac_conn1 = result.get("impedance_ac_conn1", {})
    Z_ac_conn2 = result.get("impedance_ac_conn2", {})
    Area = result["area"]

    lines.append(f"Shortest distance ≈ {Distance * 1000:.3f} mm")

    if short_path_RES is not None:
        lines.append(f"Shortest path resistance ≈ {short_path_RES * 1000:.3f} mΩ")
        if short_path_RES == 0:
            lines.append("(Zones treated as ideal conductors)")
    else:
        log_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "debug_parasitic.log"
        )
        lines.append(f"No connection found [DEBUG: {log_file}]")

    # Build impedance table (DC + AC)
    has_valid_dc = Resistance is not None
    has_ac = bool(Z_ac)
    has_zin = bool(Z_ac_conn1) or bool(Z_ac_conn2)
    termination: float | None = result.get("zin_termination_ohm")
    term_str = format_si(termination, "Ω") if termination is not None else "?"

    if has_valid_dc or has_ac:
        lines.append("")
        if has_zin:
            zin1_hdr = f"|Zin1/{term_str}|"
            zin2_hdr = f"|Zin2/{term_str}|"
            lines.append(
                f"{'Freq':^8}  {'|Z|':^12}  {'Re':^12}  {'Im':^12}  {zin1_hdr:^14}  {zin2_hdr:^14}"
            )
            lines.append("─" * (8 + 2 + 12 + 2 + 12 + 2 + 12 + 2 + 14 + 2 + 14))
        else:
            lines.append(f"{'Freq':^8}  {'|Z|':^12}  {'Re':^12}  {'Im':^12}")
            lines.append("─" * (8 + 2 + 12 + 2 + 12 + 2 + 12))

        if has_valid_dc:
            dc_str = format_si(Resistance, "Ω")
            zin_suffix = f"  {term_str:>14}  {term_str:>14}" if has_zin else ""
            lines.append(
                f"{'DC':>8}  {dc_str:>12}  {dc_str:>12}  {'-':>12}{zin_suffix}"
            )
        else:
            lines.append(f"{'DC':>8}  {'[error]':>12}")

        for freq, z in sorted(Z_ac.items()):
            f_str = format_si(freq, "Hz", precision=0)
            if z is None:
                lines.append(f"{f_str:>8}  {'[error]':>12}")
                continue
            z = complex(z)
            row = f"{f_str:>8}  {format_si(abs(z), 'Ω'):>12}  {format_si(z.real, 'Ω'):>12}  {format_si(z.imag, 'Ω'):>12}"
            if has_zin:
                row += "".join(
                    f"  {_format_zin(zin_by_freq, freq):>14}"
                    for zin_by_freq in (Z_ac_conn1, Z_ac_conn2)
                )
            lines.append(row)
    elif result["spice_unavailable"]:
        lines.append("")
        lines.append(result["spice_unavailable"])
    else:
        lines.append("")
        lines.append("ERROR: ngspice simulation failed, see debug_parasitic.log")

    if Area:
        lines.append("")
        lines.append("Copper area (traces only):")
        for layer, area in Area.items():
            info = CuStack[layer]
            lines.append(
                f"  {info['name']}: {area * 1e6:.3f} mm² "
                f"({info['thickness'] * 1e6:.0f} μm, {info['model']} assumption)"
            )

    return "\n".join(lines)


def _format_zin(zin_by_freq: dict[float, Any], freq: float) -> str:
    """|Zin| at freq; "-" when not simulated, "[error]" when the simulation failed."""
    if freq not in zin_by_freq:
        return "-"
    zin = zin_by_freq[freq]
    return "[error]" if zin is None else format_si(abs(zin), "Ω")


_SI_PREFIXES = [
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1, " "),
    (1e-3, "m"),
    (1e-6, "µ"),
    (1e-9, "n"),
    (1e-12, "p"),
]


def format_si(value: float, unit: str, precision: int = 3) -> str:
    """Format value with SI prefix (µ, m, k, M, G, etc.).

    Args:
        value: numeric value in base unit
        unit: base unit string (e.g. "Ω", "Hz", "F", "H")
        precision: decimal places

    Returns:
        Formatted string like "1.234 mΩ" or "56.789 kHz"
    """
    abs_val = abs(value) if value != 0 else 0
    for scale, prefix in _SI_PREFIXES:
        if abs_val >= scale * 0.9999:  # small tolerance for floating point
            scaled = value / scale
            return f"{scaled:.{precision}f} {prefix}{unit}"

    # Fallback for very small values
    return f"{value:.{precision}f} {unit}"


def _orphan_on_top(parent: wx.Window | None) -> int:
    """Window style keeping a parentless window above KiCad.

    In the IPC plugin the window belongs to a separate process, and without a
    parent the window manager may otherwise open it behind KiCad.
    """
    return wx.STAY_ON_TOP if parent is None else 0


def _dialog_progress(dialog: wx.ProgressDialog) -> Progress:
    """Progress that shows each stage as its own step in `dialog`. Within a
    stage it refreshes at most every _PROGRESS_REFRESH_S so per-element reports
    stay cheap; a new stage is always shown."""
    shown: Stage | None = None
    refreshed = -math.inf
    keep_going = True

    stages = list(Stage)

    def update(stage: Stage, finished: int, total: int) -> bool:
        nonlocal shown, refreshed, keep_going
        now = time.monotonic()
        if stage is not shown or now - refreshed >= _PROGRESS_REFRESH_S:
            shown, refreshed = stage, now
            step = f"Step {stages.index(stage) + 1} of {len(stages)}"
            keep_going, _skip = dialog.Update(
                100 * finished // total,
                f"{step}: {stage.value} ({finished} of {total} done)",
            )
        return bool(keep_going)

    return update


def show_message(parent: wx.Window | None, text: str, caption: str, icon: int) -> None:
    with wx.MessageDialog(
        parent, text, caption, wx.OK | icon | _orphan_on_top(parent)
    ) as dlg:
        dlg.ShowModal()


class ResultDialog(wx.Dialog):
    """Dialog to show analysis results with optional details."""

    def __init__(
        self,
        parent: wx.Window | None,
        message: str,
        debug_text: tuple[Any, ...] | str | None = None,
        analysis_result: dict[str, Any] | None = None,
        cu_stack: dict[int, CuLayer] | None = None,
    ) -> None:
        super().__init__(
            parent,
            title="Analysis result",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | _orphan_on_top(parent),
        )
        self.debug_text = debug_text
        self.analysis_result = analysis_result
        self.cu_stack = cu_stack

        sizer = wx.BoxSizer(wx.VERTICAL)
        text = wx.StaticText(self, label=message)
        font = wx.Font(
            10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
        )
        text.SetFont(font)
        sizer.Add(text, 0, wx.ALL | wx.EXPAND, 10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(wx.Button(self, wx.ID_OK, "OK"), 0, wx.ALL, 5)

        if debug_text:
            details_btn = wx.Button(self, label="Details...")
            details_btn.Bind(wx.EVT_BUTTON, self._on_details)
            btn_sizer.Add(details_btn, 0, wx.ALL, 5)

        if analysis_result and analysis_result.get("path"):
            inductance_btn = wx.Button(self, label="Calc Inductance")
            inductance_btn.Bind(wx.EVT_BUTTON, self._on_calc_inductance)
            btn_sizer.Add(inductance_btn, 0, wx.ALL, 5)

        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        self.SetSizerAndFit(sizer)
        self.SetMinSize(wx.Size(400, 150))

    def _on_details(self, _event: wx.CommandEvent) -> None:
        if isinstance(self.debug_text, tuple):
            ni, g, p, c1, c2 = self.debug_text
            txt = f"Measuring resistance: n{c1} <-> n{c2}\n" + format_network_info(
                ni, g, p
            )
        else:
            txt = self.debug_text or ""

        dlg = wx.Dialog(
            self,
            title="Network Details",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        sizer = wx.BoxSizer(wx.VERTICAL)

        text_ctrl = wx.TextCtrl(
            dlg, value=txt, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL
        )
        font = wx.Font(
            10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
        )
        text_ctrl.SetFont(font)

        lines = txt.split("\n")
        dc = wx.ClientDC(dlg)
        dc.SetFont(font)
        char_w, char_h = dc.GetTextExtent("M")
        width = min(max(max(len(line) for line in lines) * char_w + 60, 500), 1000)
        height = min(max(len(lines) * char_h + 100, 300), 800)

        sizer.Add(text_ctrl, 1, wx.ALL | wx.EXPAND, 10)
        sizer.Add(wx.Button(dlg, wx.ID_OK, "Close"), 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        dlg.SetSizer(sizer)
        dlg.SetSize(wx.Size(width, height))
        dlg.ShowModal()
        dlg.Destroy()

    def _on_calc_inductance(self, _event: wx.CommandEvent) -> None:
        result = self.analysis_result
        cu_stack = self.cu_stack
        if not result or cu_stack is None:
            show_message(
                self,
                "No analysis result is available to compute inductance from.",
                "Inductance Calculation",
                wx.ICON_INFORMATION,
            )
            return

        ind = self._compute_inductance(result, cu_stack)

        dlg = wx.Dialog(
            self,
            title="Inductance Calculation",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        sizer = wx.BoxSizer(wx.VERTICAL)

        text_ctrl = wx.TextCtrl(
            dlg,
            value=ind.get("message", "ERROR: calculation returned no message."),
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL,
        )
        font = wx.Font(
            10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
        )
        text_ctrl.SetFont(font)
        text_ctrl.SetInsertionPoint(0)

        sizer.Add(text_ctrl, 1, wx.ALL | wx.EXPAND, 10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        if ind.get("_debug_data"):
            debug_btn = wx.Button(dlg, label="Debug Plots")
            debug_data = ind["_debug_data"]
            segs = ind["segments"]
            debug_btn.Bind(
                wx.EVT_BUTTON,
                lambda _evt: self._show_debug_plots(debug_data, segs),
            )
            btn_sizer.Add(debug_btn, 0, wx.ALL, 5)
        close_btn = wx.Button(dlg, wx.ID_OK, "Close")
        btn_sizer.Add(close_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        dlg.SetSizer(sizer)
        dlg.SetSize(wx.Size(600, 400))
        dlg.ShowModal()
        dlg.Destroy()

    def _compute_inductance(
        self, result: dict[str, Any], cu_stack: dict[int, CuLayer]
    ) -> dict[str, Any]:
        """Run calc_path_inductance with robust, user-visible error handling.

        Always returns a dict containing a ``message`` key. The heavy numeric
        work runs synchronously and blocks the GUI thread, so a busy cursor and
        an info banner are shown to make clear the tool is working (not frozen).

        Note: the dependencies (numpy/scipy/trimesh/bfieldtools) are only
        installed in the venv managed by the KiCad IPC API. When the plugin is
        run any other way they are typically missing, which surfaces here as an
        ImportError -- hence the IPC-API hint in that message.
        """
        # --- import the calculator (a missing dependency lands here) ----------
        try:
            try:
                from .calc_inductance import calc_path_inductance
            except ImportError:
                from calc_inductance import calc_path_inductance
        except ImportError as e:
            log.exception("Inductance dependency import failed")
            missing = getattr(e, "name", None)
            what = f"'{missing}'" if missing else str(e)
            return {
                "message": (
                    f"ERROR: a required dependency could not be imported "
                    f"({what}).\n\n"
                    f"The inductance dependencies are installed automatically\n"
                    f"only when the plugin runs via the KiCad IPC API, which\n"
                    f"manages its own virtual environment:\n"
                    f"    KiCad -> Settings -> Plugins -> Enable KiCad API\n\n"
                    f"If you run the plugin outside that environment, install\n"
                    f"the requirements into that interpreter yourself:\n"
                    f"    pip install -r requirements.txt"
                )
            }

        # --- run the calculation ---------------------------------------------
        path = result.get("path") or []
        log.info("Inductance calculation started (%d path nodes)", len(path))
        try:
            with (
                wx.BusyCursor(),
                wx.BusyInfo(
                    "Calculating inductance...\n"
                    "This can take a while for large nets -- please wait."
                ),
            ):
                ind = calc_path_inductance(path, result["network_info"], cu_stack)
        except MemoryError:
            log.exception("Inductance calculation ran out of memory")
            ind = {
                "message": (
                    "ERROR: ran out of memory while building the inductance\n"
                    "matrix. The selected net is too large/dense for the\n"
                    "available RAM."
                )
            }
        except Exception as e:
            log.exception("Inductance calculation failed")
            log_dir = os.path.dirname(os.path.abspath(__file__))
            ind = {
                "message": (
                    f"ERROR: {type(e).__name__}: {e}\n\n"
                    f"See the log files for details, in:\n    {log_dir}\n"
                    f"(debug_parasitic.log and crash.log)"
                )
            }
        log.info("Inductance calculation finished")

        if not isinstance(ind, dict) or "message" not in ind:
            log.error("calc_path_inductance returned unexpected value: %r", ind)
            return {"message": "ERROR: the calculation returned no result."}
        return ind

    def _show_debug_plots(
        self, debug_data: dict[str, Any], segments: list[dict[str, Any]]
    ) -> None:
        try:
            try:
                from .inductance_plots import show_debug_plots
            except ImportError:
                from inductance_plots import show_debug_plots
            show_debug_plots(debug_data, segments, parent=None)
        except Exception as e:
            log.exception("Debug Plots failed")
            show_message(
                self,
                f"ERROR: could not show debug plots.\n\n{type(e).__name__}: {e}",
                "Debug Plots",
                wx.ICON_ERROR,
            )


def SaveDictToFile(data: dict[str, Any], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


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
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log"),
            "a",
            buffering=1,
            encoding="utf-8",
        )
    except OSError:
        log.warning("crash.log not writable, native crashes are not recorded")
        return
    faulthandler.enable(file=_crash_log, all_threads=True)


def run_plugin(
    ItemList: dict[Any, Any],
    CuStack: dict[int, CuLayer],
    net_tie_info: dict[str, Any] | None = None,
) -> None:
    _enable_crash_log()
    try:
        plugin_path = os.path.dirname(os.path.abspath(__file__))

        log.info("Found %d PCB elements", len(ItemList))

        data = Connect_Nets(ItemList)

        if os.environ.get("PARASITIC_DEBUG"):
            save_file = os.path.join(plugin_path, "debug_ItemList.json")
            SaveDictToFile(data, save_file)

        # Get selected elements
        Selected = [d for d in data.values() if d["is_selected"]]
        log.info("Found %d selected elements", len(Selected))

        analysis_result = None

        if len(Selected) != 2:
            message = "You have to mark exactly two elements. Preferably pads or vias."
            network_debug_text = None
        else:
            frequencies = [1e3, 10e3, 100e3, 1e6, 10e6, 100e6, 1e9]
            parent = wx.GetActiveWindow()
            with wx.ProgressDialog(
                "Parasitics",
                "Analyzing network…",
                parent=parent,
                style=wx.PD_APP_MODAL | wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME,
            ) as busy:
                busy.SetWindowStyleFlag(
                    busy.GetWindowStyleFlag() | _orphan_on_top(parent)
                )
                try:
                    result = analyze_pcb_parasitic(
                        data,
                        CuStack,
                        Selected[0],
                        Selected[1],
                        frequencies=frequencies,
                        progress=_dialog_progress(busy),
                    )
                except CancelledError:
                    log.info("Analysis cancelled")
                    return

            # Save debug info
            if result["network_info"] and log.isEnabledFor(logging.INFO):
                try:
                    with open(
                        os.path.join(plugin_path, "debug_network_info.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        cleaned = [
                            {k: v for k, v in e.items() if k != "hf"}
                            for e in result["network_info"]
                        ]
                        txt = json.dumps(cleaned, indent=1)
                        # Collapse short arrays (coordinates etc.) onto single lines
                        txt = re.sub(
                            r"\[\s+(-?[\d.e+\-]+),\s+(-?[\d.e+\-]+)\s+\]",
                            r"[\1, \2]",
                            txt,
                        )
                        f.write(txt)
                except (OSError, TypeError, ValueError):
                    log.warning("Writing debug_network_info.json failed", exc_info=True)

            message = format_result_message(result, CuStack, net_tie_info=net_tie_info)
            network_debug_text = None
            if result["network_info"]:
                network_debug_text = (
                    result["network_info"],
                    result["graph"],
                    result["path"],
                    result["conn1"],
                    result["conn2"],
                )
            analysis_result = result

        log.info("Opening dialog")
        dlg = ResultDialog(
            wx.GetActiveWindow(),
            message,
            network_debug_text,
            analysis_result=analysis_result,
            cu_stack=CuStack,
        )
        dlg.ShowModal()
        dlg.Destroy()

    except Exception:
        log.exception("Fatal error")
        show_message(
            wx.GetActiveWindow(), traceback.format_exc(), "Fatal Error", wx.ICON_ERROR
        )
