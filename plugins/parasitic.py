import os
import os.path
import re
import sys
import warnings
import traceback
import math
import json
import logging
import wx

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

venv = os.environ.get("VIRTUAL_ENV")
if venv:
    version = "python{}.{}".format(sys.version_info.major, sys.version_info.minor)
    venv_site_packages = os.path.join(venv, "lib", version, "site-packages")
    if venv_site_packages in sys.path:
        sys.path.remove(venv_site_packages)
    sys.path.insert(0, venv_site_packages)


try:
    from .Connect_Nets import Connect_Nets
    from .Get_Parasitic import extract_network, find_path, simulate_network
    from .network_display import format_network_info
except ImportError:
    from Connect_Nets import Connect_Nets
    from Get_Parasitic import extract_network, find_path, simulate_network
    from network_display import format_network_info

log = logging.getLogger(__name__)


def analyze_pcb_parasitic(
    data,
    CuStack,
    element1,
    element2,
    frequencies=None,
):
    """Analyze parasitic resistance/impedance between two PCB elements.

    Args:
        data: PCB element data from Connect_Nets()
        CuStack: Copper stackup from Get_PCB_Stackup_fun()
        element1, element2: Selected element dicts
        frequencies: List of frequencies for AC analysis (Hz)

    Returns:
        dict with: resistance_dc, impedance_ac, distance, short_path_resistance,
                   area, network_info, graph, path, conn1, conn2, error
    """
    result = {
        "resistance_dc": None,
        "impedance_ac": {},
        "distance": float("inf"),
        "short_path_resistance": -1,
        "area": {},
        "network_info": None,
        "graph": None,
        "path": None,
        "conn1": None,
        "conn2": None,
        "error": None,
    }

    # Extract network ONCE (DC only, no HF calculation)
    network = extract_network(data, CuStack)

    # Find best path across all layer combinations (fast, no simulation)
    best_distance = float("inf")
    best_conn = None

    for layer1 in element1.get("layer", []):
        conn1 = element1["net_start"].get(layer1, 0)
        if conn1 == 0:
            continue
        for layer2 in element2.get("layer", []):
            conn2 = element2["net_start"].get(layer2, 0)
            if conn2 == 0:
                continue

            Distance, _, _ = find_path(network, conn1, conn2)

            if not math.isinf(Distance) and Distance < best_distance:
                best_distance = Distance
                best_conn = (conn1, conn2, layer1, layer2)

    if not best_conn:
        best_conn = (
            element1["net_start"][element1["layer"][0]],
            element2["net_start"][element2["layer"][0]],
            element1["layer"][0],
            element2["layer"][0],
        )

    conn1, conn2 = best_conn[0], best_conn[1]
    result["conn1"], result["conn2"] = conn1, conn2

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
        result["error"] = "No path found between selected elements. Check connectivity."
        log.error("%s", result["error"])
        return result

    log.debug("simulate_network start")

    # Simulate with HF parameters
    Resistance, Z_ac, network_info = simulate_network(
        network,
        conn1,
        conn2,
        CuStack,
        frequencies=frequencies,
    )

    log.debug("simulate_network done")

    result.update(
        {
            "resistance_dc": Resistance,
            "impedance_ac": Z_ac,
            "distance": Distance,
            "short_path_resistance": short_path_RES,
            "area": network["area"],
            "network_info": network_info,
            "graph": network["graph"],
            "path": path,
        }
    )
    return result


def format_result_message(result, CuStack, net_tie_info=None):
    """Format analysis result as human-readable message."""
    lines = []

    if result["error"]:
        return result["error"]

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
    Area = result["area"]

    lines.append(f"Shortest distance ≈ {Distance * 1000:.3f} mm")

    if short_path_RES >= 0:
        lines.append(f"Shortest path resistance ≈ {short_path_RES * 1000:.3f} mΩ")
        if short_path_RES == 0:
            lines.append("(Zones treated as ideal conductors)")
    else:
        log_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "debug_parasitic.log"
        )
        lines.append(f"No connection found [DEBUG: {log_file}]")

    # Build impedance table (DC + AC)
    has_valid_dc = (
        Resistance is not None and not math.isinf(Resistance) and Resistance >= 0
    )
    has_ac = bool(Z_ac)

    if has_valid_dc or has_ac:
        lines.append("")
        lines.append(f"{'Freq':^8}  {'|Z|':^12}  {'Re':^12}  {'Im':^12}")
        lines.append("─" * (8 + 2 + 12 + 2 + 12 + 2 + 12))

        if has_valid_dc:
            dc_str = format_si(Resistance, "Ω")
            lines.append(f"{'DC':>8}  {dc_str:>12}  {dc_str:>12}  {'-':>12}")
        elif Resistance is not None and Resistance < 0:
            lines.append("DC         [ngspice error]")

        for freq, z in sorted(Z_ac.items()):
            f = format_si(freq, "Hz", precision=0)
            if isinstance(z, complex):
                lines.append(
                    f"{f:>8}  {format_si(abs(z), 'Ω'):>12}  {format_si(z.real, 'Ω'):>12}  {format_si(z.imag, 'Ω'):>12}"
                )
            elif z >= 0:
                zs = format_si(z, "Ω")
                lines.append(f"{f:>8}  {zs:>12}  {zs:>12}  {'-':>12}")
            else:
                lines.append(f"{f:>8}  {'[error]':>12}")
    elif Resistance is not None and Resistance < 0:
        lines.append("")
        lines.append("ERROR: ngspice simulation failed (check installation)")

    if Area:
        lines.append("")
        lines.append("Copper area (traces only):")
        for layer, area in Area.items():
            info = CuStack[layer]
            lines.append(
                f"  {info['name']}: {area * 1e6:.3f} mm² "
                f"({info['thickness'] * 1e6:.0f} μm, {info['model']})"
            )

    return "\n".join(lines)


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


def format_si(value, unit, precision=3):
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


class ResultDialog(wx.Dialog):
    """Dialog to show analysis results with optional details."""

    def __init__(
        self, parent, message, debug_text=None, analysis_result=None, cu_stack=None
    ):
        super().__init__(
            parent,
            title="Analysis result",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
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

    def _on_details(self, _event):
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

    def _on_calc_inductance(self, _event):
        result = self.analysis_result
        if not result:
            return
        try:
            try:
                from .calc_inductance import calc_path_inductance
            except ImportError:
                from calc_inductance import calc_path_inductance
            ind = calc_path_inductance(
                result["path"],
                result["network_info"],
                self.cu_stack,
            )
        except ImportError as e:
            ind = {
                "message": (
                    f"ERROR: {e}\n\n"
                    f"Use the KiCad IPC API which manages\n"
                    f"dependencies automatically:\n"
                    f"  KiCad -> Settings -> Plugins -> Enable KiCad API"
                )
            }

        dlg = wx.Dialog(
            self,
            title="Inductance Calculation",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        sizer = wx.BoxSizer(wx.VERTICAL)

        text_ctrl = wx.TextCtrl(
            dlg,
            value=ind["message"],
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

    def _show_debug_plots(self, debug_data, segments):
        try:
            try:
                from .calc_inductance import show_debug_plots
            except ImportError:
                from calc_inductance import show_debug_plots
            show_debug_plots(debug_data, segments, parent=None)
        except Exception:
            pass


def SaveDictToFile(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_plugin(ItemList, CuStack):
    try:
        plugin_path = os.path.dirname(os.path.abspath(__file__))

        net_tie_info = ItemList.pop("_net_tie_info", None)

        log.info("Found %d PCB elements", len(ItemList))

        data = Connect_Nets(ItemList)

        if log.isEnabledFor(logging.INFO):
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
            result = analyze_pcb_parasitic(
                data,
                CuStack,
                Selected[0],
                Selected[1],
                frequencies=frequencies,
            )

            # Save debug info
            if result["network_info"] and log.isEnabledFor(logging.INFO):
                try:
                    with open(
                        os.path.join(plugin_path, "debug_network_info.json"), "w"
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
                except Exception:
                    pass

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
            None,
            message,
            network_debug_text,
            analysis_result=analysis_result,
            cu_stack=CuStack,
        )
        dlg.ShowModal()
        dlg.Destroy()

    except Exception:
        error_msg = traceback.format_exc()
        print("[FATAL ERROR]", error_msg)
        wx.MessageDialog(
            None, error_msg, "Fatal Error", wx.OK | wx.ICON_ERROR
        ).ShowModal()


if __name__ == "__main__":
    from kipy import KiCad, errors

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
        filename=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "debug_parasitic.log"
        ),
        filemode="w",
    )
    # Suppress noisy third-party debug output
    for _logger_name in (
        "matplotlib",
        "matplotlib.font_manager",
        "matplotlib.ticker",
        "matplotlib.colorbar",
        "matplotlib.backends",
        "pynng",
        "pynng.nng",
        "ngspyce",
        "sharedspice",
    ):
        logging.getLogger(_logger_name).setLevel(logging.WARNING)
    logging.debug("parasitic.py started")

    app = wx.App()
    try:
        kicad = KiCad()
        logging.debug("Connected to KiCad %s", kicad.get_version())

        try:
            board = kicad.get_board()
        except Exception:
            logging.error("Failed to get board")
            print("Error: No PCB board found. Is a board open in KiCad?")
            sys.exit(1)
        board_FileName = board.document.board_filename
        logging.debug("Board: %s", board.document.board_filename)

        from Get_PCB_Elements_IPC import Get_PCB_Elements_IPC
        from Get_PCB_Stackup_IPC import Get_PCB_Stackup_IPC

        ItemList = Get_PCB_Elements_IPC(board)
        logging.debug("Got %d elements from IPC API", len(ItemList))

        CuStack = Get_PCB_Stackup_IPC(board)
        run_plugin(ItemList, CuStack)
        # Keep event loop alive for any open PlotFrame windows
        app.MainLoop()
    except errors.ConnectionError:
        logging.exception("ConnectionError")
        print("ConnectionError: Failed to connect to KiCad. Is KiCad running?")
    except Exception as e:
        logging.exception("__main__")
        print(f"Error: {e}")
