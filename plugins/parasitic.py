import os
import os.path
import sys
import warnings
import traceback
import math
import json
import time
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


def analyze_pcb_parasitic(
    data, CuStack, element1, element2, frequencies=None, debug=0, debug_print=None
):
    """Analyze parasitic resistance/impedance between two PCB elements.

    Args:
        data: PCB element data from Connect_Nets()
        CuStack: Copper stackup from Get_PCB_Stackup_fun()
        element1, element2: Selected element dicts
        frequencies: List of frequencies for AC analysis (Hz)
        debug: Enable debug output
        debug_print: Custom print function

    Returns:
        dict with: resistance_dc, impedance_ac, distance, short_path_resistance,
                   area, network_info, graph, path, conn1, conn2, error
    """
    if debug_print is None:
        debug_print = print if debug else lambda x: None

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
    network = extract_network(data, CuStack, debug=debug, debug_print=debug_print)

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
    Distance, path, short_path_RES = find_path(
        network, conn1, conn2, debug=debug, debug_print=debug_print
    )

    if debug:
        debug_print(
            f"[{time.strftime('%H:%M:%S')}] Best path: conn1={conn1}, conn2={conn2}, "
            f"layers={best_conn[2]}/{best_conn[3]}, distance={best_distance * 1000:.3f}mm"
        )

    if math.isinf(Distance):
        result["error"] = "No path found between selected elements. Check connectivity."
        print(f"Error: {result['error']}")
        return result

    if debug:
        debug_print(f"[{time.strftime('%H:%M:%S')}] simulate_network start")

    # Simulate with HF parameters
    Resistance, Z_ac, network_info = simulate_network(
        network,
        conn1,
        conn2,
        CuStack,
        frequencies=frequencies,
        debug=debug,
        debug_print=debug_print,
    )

    if debug:
        debug_print(f"[{time.strftime('%H:%M:%S')}] simulate_network done")

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


def format_result_message(result, CuStack, debug_log_file=None, net_tie_info=None):
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
    elif debug_log_file:
        lines.append(f"No connection found [DEBUG: {debug_log_file}]")

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


def format_si(value, unit, precision=3):
    """Format value with SI prefix (µ, m, k, M, G, etc.).

    Args:
        value: numeric value in base unit
        unit: base unit string (e.g. "Ω", "Hz", "F", "H")
        precision: decimal places

    Returns:
        Formatted string like "1.234 mΩ" or "56.789 kHz"
    """
    si_prefixes = [
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

    abs_val = abs(value) if value != 0 else 0
    for scale, prefix in si_prefixes:
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
                debug=1,
            )
        except ImportError as e:
            import sys

            python_exe = sys.executable
            ind = {
                "message": (
                    f"ERROR: {e}\n\n"
                    f"Install missing packages with:\n"
                    f"  {python_exe} -m pip install bfieldtools trimesh scipy\n\n"
                    f"Alternatively, use the KiCad IPC API which manages\n"
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

    @staticmethod
    def _show_debug_plots(debug_data, segments):
        try:
            try:
                from .calc_inductance import show_debug_plots
            except ImportError:
                from calc_inductance import show_debug_plots
            show_debug_plots(debug_data, segments)
        except Exception:
            pass


def SaveDictToFile(dict_name, filename):
    with open(filename, "w") as f:
        f.write("data = {\n")
        for uuid, d in list(dict_name.items()):
            f.write(str(uuid))
            f.write(":")
            f.write(str(d))
            f.write(",\n")
        f.write("}")


def run_plugin(ItemList, CuStack):
    try:
        plugin_path = os.path.dirname(os.path.abspath(__file__))
        debug = 0
        debug_log_file = None
        if debug:
            debug_log_file = os.path.join(plugin_path, "parasitics_debug.log")
            with open(debug_log_file, "w") as f:
                f.write(
                    "=" * 60 + "\nParasitics Plugin Debug Log\n" + "=" * 60 + "\n\n"
                )

        def debug_print(msg):
            if debug:
                print(msg)
                if debug_log_file:
                    try:
                        with open(debug_log_file, "a", encoding="utf-8") as f:
                            f.write(msg + "\n")
                    except Exception:
                        pass

        net_tie_info = ItemList.pop("_net_tie_info", None)

        debug_print(f"[DEBUG] Found {len(ItemList)} PCB elements")

        if debug:
            save_file = os.path.join(plugin_path, "ItemList.py")
            SaveDictToFile(ItemList, save_file)

        data = Connect_Nets(ItemList)

        # Get selected elements
        Selected = [d for d in data.values() if d["is_selected"]]
        debug_print(f"[DEBUG] Found {len(Selected)} selected elements")

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
                debug=debug,
                debug_print=debug_print,
            )

            # Save debug info
            if debug and result["network_info"]:
                try:
                    with open(os.path.join(plugin_path, "network_info.json"), "w") as f:
                        json.dump(result["network_info"], f, indent=2)
                except Exception:
                    pass

            message = format_result_message(
                result, CuStack, debug_log_file, net_tie_info=net_tie_info
            )
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

        debug_print(f"[{time.strftime('%H:%M:%S')}] Opening dialog")
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
            os.path.dirname(os.path.abspath(__file__)), "parasitic.log"
        ),
        filemode="w",
    )
    logging.info("parasitic.py started")

    app = wx.App()
    try:
        kicad = KiCad()
        logging.info(f"Connected to KiCad {kicad.get_version()}")

        try:
            board = kicad.get_board()
        except Exception:
            logging.error("Failed to get board")
            print("Error: No PCB board found. Is a board open in KiCad?")
            sys.exit(1)
        board_FileName = board.document.board_filename
        logging.info(f"Board: {board.document.board_filename}")

        from Get_PCB_Elements_IPC import Get_PCB_Elements_IPC
        from Get_PCB_Stackup_IPC import Get_PCB_Stackup_IPC

        ItemList = Get_PCB_Elements_IPC(board)
        logging.info("Got %d elements from IPC API", len(ItemList))

        CuStack = Get_PCB_Stackup_IPC(board)
        run_plugin(ItemList, CuStack)
    except errors.ConnectionError:
        logging.exception("ConnectionError")
        print("ConnectionError: Failed to connect to KiCad. Is KiCad running?")
    except Exception as e:
        logging.exception("__main__")
        print(f"Error: {e}")
