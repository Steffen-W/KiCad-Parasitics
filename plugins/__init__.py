import os.path
import os
import traceback
from pathlib import Path
import math
import json

import pcbnew
import wx

if __name__ == "__main__":
    from Get_PCB_Elements import Get_PCB_Elements, SaveDictToFile
    from Connect_Nets import Connect_Nets
    from Get_PCB_Stackup import Get_PCB_Stackup_fun
    from Get_Parasitic import Get_Parasitic, format_network_info
else:
    from .Get_PCB_Elements import Get_PCB_Elements, SaveDictToFile
    from .Connect_Nets import Connect_Nets
    from .Get_PCB_Stackup import Get_PCB_Stackup_fun
    from .Get_Parasitic import Get_Parasitic, format_network_info


class KiCadPluginParasitic(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "parasitic"
        self.category = "parasitic"
        self.description = "parasitic"
        self.show_toolbar_button = True
        self.plugin_path = os.path.normpath(os.path.dirname(__file__))
        self.icon_file_name = os.path.join(self.plugin_path, "icon_small.png")
        self.dark_icon_file_name = os.path.join(self.plugin_path, "icon_small.png")

    def Run(self):
        try:
            # Set to 1 to enable debug output and file saving
            debug = 0

            # Debug logging setup
            debug_log_file = None
            if debug:
                debug_log_file = os.path.join(self.plugin_path, "parasitics_debug.log")
                with open(debug_log_file, "w") as f:
                    f.write("=" * 60 + "\n")
                    f.write("Parasitics Plugin Debug Log\n")
                    f.write("=" * 60 + "\n\n")

            def debug_print(msg):
                if debug:
                    print(msg)
                    if debug_log_file:
                        try:
                            with open(debug_log_file, "a", encoding="utf-8") as f:
                                f.write(msg + "\n")
                        except Exception:
                            pass

            board = pcbnew.GetBoard()
            connect = board.GetConnectivity()
            Settings = pcbnew.GetSettingsManager()

            KiCad_SettingsVersion = str(Settings.GetSettingsVersion())
            try:
                new_v9 = int(KiCad_SettingsVersion.split(".")[0]) >= 9
            except Exception:
                if debug:
                    debug_print(f"KiCad_SettingsVersion: {KiCad_SettingsVersion}")
                new_v9 = True
            board_FileName = Path(board.GetFileName())

            ####################################################
            # Get PCB Elements
            ####################################################

            ItemList, BoardThickness = Get_PCB_Elements(board, connect)

            if debug:
                debug_print(f"[DEBUG] Found {len(ItemList)} PCB elements")

            ####################################################
            # save Variable ItemList to file (for debug)
            ####################################################

            if debug:
                save_as_file = os.path.join(self.plugin_path, "ItemList.py")
                debug_print(f"[DEBUG] Saving ItemList to: {save_as_file}")
                SaveDictToFile(ItemList, save_as_file)
                with open(save_as_file, "a") as f:
                    f.write('\nboard_FileName = "')
                    f.write(str(board_FileName))
                    f.write('"')

            ####################################################
            # connect nets together
            ####################################################

            data = Connect_Nets(ItemList)

            if debug:
                from pprint import pprint

                # Check for wires with invalid netStart/netEnd
                invalid_wires = []
                for uuid, d in data.items():
                    if d.get("type") == "WIRE":
                        for layer in d.get("layer", []):
                            netStart = d.get("net_start", {}).get(layer, 0)
                            netEnd = d.get("net_end", {}).get(layer, 0)
                            if netStart == 0 or netEnd == 0:
                                invalid_wires.append((uuid, layer, netStart, netEnd))

                if invalid_wires:
                    debug_print(
                        f"[DEBUG] WARNING: {len(invalid_wires)} wires have invalid nets"
                    )
                    for uuid, layer, ns, ne in invalid_wires[:5]:
                        debug_print(
                            f"[DEBUG]   WIRE {uuid} L{layer}: netStart={ns}, netEnd={ne}"
                        )

            ####################################################
            # read PhysicalLayerStack from file
            ####################################################

            PhysicalLayerStack, CuStack = Get_PCB_Stackup_fun(
                ProjectPath=board_FileName, new_v9=new_v9, board_thickness=BoardThickness
            )
            if debug:
                from pprint import pprint

                pprint(CuStack)

            ####################################################
            # get resistance
            ####################################################

            Selected = [d for uuid, d in data.items() if d["is_selected"]]

            if debug:
                debug_print(f"[DEBUG] Found {len(Selected)} selected elements")
                for i, sel in enumerate(Selected):
                    debug_print(
                        f"[DEBUG] Selected {i + 1}: type={sel.get('type')}, "
                        f"NetCode={sel.get('NetCode')}, Layer={sel.get('Layer')}"
                    )

            message = ""
            NetCode = None
            network_debug_text = None
            if len(Selected) == 2:
                NetCode = Selected[0]["net_code"]

                if NetCode != Selected[1]["net_code"]:
                    message = "The marked points are not in the same network."
            else:
                message = "You have to mark exactly two elements."
                message += " Preferably pads or vias."

            if message == "" and NetCode is not None:
                # Try all layer combinations to find a valid path
                best_result = None
                best_distance = float("inf")
                best_conn = None

                for layer1 in Selected[0].get("layer", []):
                    conn1 = Selected[0]["net_start"].get(layer1, 0)
                    if conn1 == 0:
                        continue
                    for layer2 in Selected[1].get("layer", []):
                        conn2 = Selected[1]["net_start"].get(layer2, 0)
                        if conn2 == 0:
                            continue

                        (
                            Resistance,
                            Distance,
                            short_path_RES,
                            Area,
                            network_info,
                            graph,
                            path,
                        ) = Get_Parasitic(
                            data,
                            CuStack,
                            conn1,
                            conn2,
                            NetCode,
                            debug=debug,
                            debug_print=debug_print,
                        )

                        if not math.isinf(Distance) and Distance < best_distance:
                            best_distance = Distance
                            best_result = (
                                Resistance,
                                Distance,
                                short_path_RES,
                                Area,
                                network_info,
                                graph,
                                path,
                            )
                            best_conn = (conn1, conn2, layer1, layer2)

                if best_result and best_conn:
                    Resistance, Distance, short_path_RES, Area, network_info, graph, path = best_result
                    conn1, conn2 = best_conn[0], best_conn[1]
                    if debug:
                        debug_print(
                            f"[DEBUG] Best path: conn1={conn1}, conn2={conn2}, "
                            f"layers={best_conn[2]}/{best_conn[3]}, distance={Distance:.3f}mm"
                        )
                else:
                    # Fallback to first layer
                    conn1 = Selected[0]["net_start"][Selected[0]["layer"][0]]
                    conn2 = Selected[1]["net_start"][Selected[1]["layer"][0]]
                    (
                        Resistance,
                        Distance,
                        short_path_RES,
                        Area,
                        network_info,
                        graph,
                        path,
                    ) = Get_Parasitic(
                        data,
                        CuStack,
                        conn1,
                        conn2,
                        NetCode,
                        debug=debug,
                        debug_print=debug_print,
                    )

                # Debug output of resistance network (only once for best result)
                if debug and network_info:
                    debug_print(f"Measuring resistance: n{conn1} <-> n{conn2}")
                    debug_print(format_network_info(network_info, graph, path))
                    network_json_file = os.path.join(self.plugin_path, "network_info.json")
                    try:
                        with open(network_json_file, "w", encoding="utf-8") as f:
                            json.dump(network_info, f, indent=2)
                        debug_print(f"[DEBUG] Network info saved to: {network_json_file}")
                    except Exception as e:
                        debug_print(f"[DEBUG] Failed to save network_info: {e}")

                message += "\nShortest distance between the two points ≈ "
                message += "{:.3f} mm".format(Distance)

                message += "\n"
                if not PhysicalLayerStack:
                    message += "\nNo Physical Stackup could be found!"
                if short_path_RES > 0:
                    message += "\nDC resistance (only short path) ≈ "
                    message += "{:.3f} mOhm".format(short_path_RES * 1000)
                elif short_path_RES == 0:
                    message += "\nDC resistance (only short path) ≈ "
                    message += "{:.3f} mOhm".format(short_path_RES * 1000)
                    message += "\nSurfaces of the zones are considered perfectly "
                    message += "conductive and short-circuit points. This is probably the case here."
                else:
                    message += "\nNo connection was found between the two marked points"
                    if debug:
                        message += f"\n[DEBUG: Check {debug_log_file}]"

                if not math.isinf(Resistance) and Resistance >= 0:
                    message += "\nDC resistance between both points ≈ "
                    message += "{:.3f} mOhm".format(Resistance * 1000)
                elif Resistance < 0:
                    message += "\nERROR in Resistance Network calculation."
                    message += " Probably no ngspice installation could be found."
                    message += " The result about the short path"
                    message += " path is however uninfluenced."
                else:
                    message += "\nNo connection was found between the two marked points"
                    if debug:
                        message += f"\n[DEBUG: Check {debug_log_file}]"

                message += "\n\nNote: AC resistance is typically much higher due to skin effect."

                message += "\n"
                if len(Area) > 0:
                    message += "\nRough area estimation of the signal"
                    message += " (without zones and vias):"
                    for layer in Area.keys():
                        message += "\nLayer {}: {:.3f} mm², {} μm copper".format(
                            CuStack[layer]["name"],
                            Area[layer],
                            CuStack[layer]["thickness"] * 1000,
                        )

                # Build network debug text for details dialog
                if network_info:
                    network_debug_text = f"Measuring resistance: n{conn1} <-> n{conn2}\n"
                    network_debug_text += format_network_info(network_info, graph, path)

            # Custom dialog with optional Details button
            class ResultDialog(wx.Dialog):
                def __init__(self, parent, message, debug_text=None):
                    super().__init__(parent, title="Analysis result",
                                     style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
                    self.debug_text = debug_text

                    sizer = wx.BoxSizer(wx.VERTICAL)

                    # Main message
                    text = wx.StaticText(self, label=message)
                    sizer.Add(text, 0, wx.ALL | wx.EXPAND, 10)

                    # Button sizer
                    btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
                    ok_btn = wx.Button(self, wx.ID_OK, "OK")
                    btn_sizer.Add(ok_btn, 0, wx.ALL, 5)

                    if debug_text:
                        details_btn = wx.Button(self, label="Details...")
                        details_btn.Bind(wx.EVT_BUTTON, self.on_details)
                        btn_sizer.Add(details_btn, 0, wx.ALL, 5)

                    sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
                    self.SetSizerAndFit(sizer)
                    self.SetMinSize(wx.Size(400, 150))

                def on_details(self, _event):
                    # Show network details in a simple text dialog
                    detail_dlg = wx.Dialog(self, title="Network Details",
                                           style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
                    sizer = wx.BoxSizer(wx.VERTICAL)

                    # Monospace text control
                    text_ctrl = wx.TextCtrl(detail_dlg, value=self.debug_text or "",
                                            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
                    font = wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
                    text_ctrl.SetFont(font)

                    # Calculate size based on content
                    lines = (self.debug_text or "").split("\n")
                    max_line_len = max((len(line) for line in lines), default=40)
                    num_lines = len(lines)

                    # Estimate character size with monospace font
                    dc = wx.ClientDC(detail_dlg)
                    dc.SetFont(font)
                    char_width, char_height = dc.GetTextExtent("M")

                    # Calculate needed size with padding
                    width = min(max(max_line_len * char_width + 60, 500), 1000)
                    height = min(max(num_lines * char_height + 100, 300), 800)

                    sizer.Add(text_ctrl, 1, wx.ALL | wx.EXPAND, 10)

                    close_btn = wx.Button(detail_dlg, wx.ID_OK, "Close")
                    sizer.Add(close_btn, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

                    detail_dlg.SetSizer(sizer)
                    detail_dlg.SetSize(wx.Size(width, height))
                    detail_dlg.ShowModal()
                    detail_dlg.Destroy()

            dlg = ResultDialog(None, message, network_debug_text)
            dlg.ShowModal()
            dlg.Destroy()

        except Exception:
            error_msg = traceback.format_exc()
            print("[FATAL ERROR]", error_msg)
            dlg = wx.MessageDialog(
                None,
                error_msg,
                "Fatal Error",
                wx.OK | wx.ICON_ERROR,
            )
            dlg.ShowModal()
            dlg.Destroy()


if not __name__ == "__main__":
    KiCadPluginParasitic().register()


if __name__ == "__main__":
    from ItemList import data, board_FileName
    from pprint import pprint

    # Get PCB Elements
    ItemList = data

    # connect nets together
    data = Connect_Nets(ItemList)
    # pprint(data)

    # read PhysicalLayerStack from file
    PhysicalLayerStack, CuStack = Get_PCB_Stackup_fun(ProjectPath=board_FileName)
    pprint(CuStack)

    # get resistance
    Selected = [d for uuid, d in list(data.items()) if d["is_selected"]]
    if len(Selected) == 2:
        conn1 = Selected[0]["net_start"][Selected[0]["layer"][0]]
        conn2 = Selected[1]["net_start"][Selected[1]["layer"][0]]
        NetCode = Selected[0]["net_code"]
        if not NetCode == Selected[1]["net_code"]:
            print("The marked points are not in the same network.")

        Resistance, Distance, short_path_RES, Area, network_info = Get_Parasitic(
            data, CuStack, conn1, conn2, NetCode, debug=1
        )
        print(f"Distance {Distance} mm")
        print(f"Resistance {Resistance} mOhm")
        print(f"Resistance {short_path_RES} mOhm (only short path)")

        if len(Area) > 0:
            for layer in Area.keys():
                txt = "Layer {}: {:.3f} mm²".format(layer, Area[layer])
                print(txt)
    else:
        print("You have to mark exactly two elements.")
