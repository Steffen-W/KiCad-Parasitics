import pcbnew
import os.path
import wx
import traceback
from pathlib import Path
import math

try:
    if not __name__ == "__main__":
        from .Get_PCB_Elements import Get_PCB_Elements, SaveDictToFile
        from .Connect_Nets import Connect_Nets
        from .Get_PCB_Stackup import Get_PCB_Stackup_fun
        from .Get_Parasitic import Get_Parasitic
except Exception as e:
    print(traceback.format_exc())


class KiCadPluginParasitic(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "parasitic"
        self.category = "parasitic"
        self.description = "parasitic"
        self.show_toolbar_button = True
        self.plugin_path = os.path.dirname(__file__)
        self.icon_file_name = os.path.join(self.plugin_path, "icon_small.png")
        self.dark_icon_file_name = os.path.join(self.plugin_path, "icon_small.png")

    def Run(self):
        try:
            # Enable debug mode for troubleshooting
            # Set to 1 to enable debug output and file saving
            debug = 0
            
            # Debug log file location
            debug_log_file = None
            if debug:
                debug_log_file = os.path.join(self.plugin_path, "parasitics_debug.log")
                # Clear previous log
                with open(debug_log_file, "w") as f:
                    f.write("=" * 60 + "\n")
                    f.write("Parasitics Plugin Debug Log\n")
                    f.write("=" * 60 + "\n\n")

            def debug_print(msg):
                """Print debug message to both console and log file"""
                if debug:
                    print(msg)  # Still print to console (if available)
                    if debug_log_file:
                        try:
                            with open(debug_log_file, "a", encoding="utf-8") as f:
                                f.write(msg + "\n")
                        except:
                            pass  # Don't fail if we can't write to log

            board = pcbnew.GetBoard()
            connect = board.GetConnectivity()
            Settings = pcbnew.GetSettingsManager()

            # KiCad_CommonSettings = Settings.GetCommonSettings()
            KiCad_UserSettingsPath = Settings.GetUserSettingsPath()
            KiCad_SettingsVersion = str(Settings.GetSettingsVersion())
            try:
                new_v9 = int(KiCad_SettingsVersion.split(".")[0]) >= 9
            except:
                debug_print(f"KiCad_SettingsVersion: {KiCad_SettingsVersion}")
                new_v9 = True
            board_FileName = Path(board.GetFileName())

            ####################################################
            # Get PCB Elements
            ####################################################

            ItemList = Get_PCB_Elements(board, connect)
            
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
                # Check for wires with invalid netStart/netEnd after Connect_Nets
                invalid_wires = []
                for uuid, d in data.items():
                    if d.get("type") == "WIRE":
                        for layer in d.get("Layer", []):
                            netStart = d.get("netStart", {}).get(layer, 0)
                            netEnd = d.get("netEnd", {}).get(layer, 0)
                            if netStart == 0 or netEnd == 0:
                                invalid_wires.append((uuid, layer, netStart, netEnd, d.get("Start"), d.get("End"), d.get("connStart"), d.get("connEnd")))
                
                if invalid_wires:
                    debug_print(f"[DEBUG] WARNING: {len(invalid_wires)} wires have invalid netStart/netEnd after Connect_Nets!")
                    for uuid, layer, ns, ne, start, end, connS, connE in invalid_wires[:5]:
                        debug_print(f"[DEBUG]   WIRE {uuid} L{layer}: netStart={ns}, netEnd={ne}, Start={start}, End={end}")
                        debug_print(f"[DEBUG]     connStart={connS}, connEnd={connE}")
                
                from pprint import pprint
                # pprint(data)  # Too verbose, comment out

            ####################################################
            # read PhysicalLayerStack from file
            ####################################################

            PhysicalLayerStack, CuStack = Get_PCB_Stackup_fun(
                ProjectPath=board_FileName, new_v9=new_v9
            )
            if debug:
                from pprint import pprint
                pprint(CuStack)

            ####################################################
            # get resistance
            ####################################################

            Selected = [d for uuid, d in data.items() if d["IsSelected"]]
            
            debug_print(f"[DEBUG] Found {len(Selected)} selected elements")
            for i, sel in enumerate(Selected):
                debug_print(f"[DEBUG] Selected {i+1}: type={sel.get('type')}, NetCode={sel.get('NetCode')}, "
                          f"Netname={sel.get('Netname')}, Layer={sel.get('Layer')}")

            message = ""
            if len(Selected) == 2:
                # Get connection points - try all layers to find a path
                NetCode = Selected[0]["NetCode"]
                
                debug_print(f"[DEBUG] Selected[0] Layer: {Selected[0]['Layer']}")
                debug_print(f"[DEBUG] Selected[1] Layer: {Selected[1]['Layer']}")
                # Show all netStart values for each selected via on all layers
                for i, sel in enumerate(Selected):
                    debug_print(f"[DEBUG] Selected[{i}] netStart values by layer:")
                    for layer in sel.get("Layer", []):
                        netStart_val = sel.get("netStart", {}).get(layer, "NOT_SET")
                        debug_print(f"[DEBUG]   Layer {layer}: netStart={netStart_val}")
                
                if not NetCode == Selected[1]["NetCode"]:
                    message = "The marked points are not in the same network."
                    debug_print(f"[DEBUG] ERROR: NetCodes don't match: {NetCode} != {Selected[1]['NetCode']}")
            else:
                message = "You have to mark exactly two elements."
                message += " Preferably pads or vias."
                debug_print(f"[DEBUG] ERROR: Expected 2 selected elements, found {len(Selected)}")

            if message == "":
                # Try to find a path using any layer combination
                # Start with first layer of each, but try all combinations if that fails
                best_result = None
                best_distance = float("inf")
                
                for layer1 in Selected[0].get("Layer", []):
                    conn1 = Selected[0]["netStart"].get(layer1, 0)
                    if conn1 == 0:
                        continue
                    for layer2 in Selected[1].get("Layer", []):
                        conn2 = Selected[1]["netStart"].get(layer2, 0)
                        if conn2 == 0:
                            continue
                        
                        debug_print(f"[DEBUG] Trying path: conn1={conn1} (L{layer1}) -> conn2={conn2} (L{layer2})")
                        
                        (
                            Resistance,
                            Distance,
                            inductance_nH,
                            short_path_RES,
                            Area,
                        ) = Get_Parasitic(data, CuStack, conn1, conn2, NetCode, debug=debug, debug_print=debug_print)
                        
                        # If we found a path, use it (prefer shorter distances)
                        if not math.isinf(Distance) and Distance < best_distance:
                            best_distance = Distance
                            best_result = (Resistance, Distance, inductance_nH, short_path_RES, Area)
                            debug_print(f"[DEBUG] Found path! Distance={Distance}, using this combination")
                
                if best_result:
                    Resistance, Distance, inductance_nH, short_path_RES, Area = best_result
                else:
                    # Fall back to first layer if no path found
                    conn1 = Selected[0]["netStart"][Selected[0]["Layer"][0]]
                    conn2 = Selected[1]["netStart"][Selected[1]["Layer"][0]]
                    debug_print(f"[DEBUG] No path found with any layer combination, using first layer: conn1={conn1}, conn2={conn2}")
                    (
                        Resistance,
                        Distance,
                        inductance_nH,
                        short_path_RES,
                        Area,
                    ) = Get_Parasitic(data, CuStack, conn1, conn2, NetCode, debug=debug, debug_print=debug_print)

                message += "\nShortest distance between the two points ≈ "
                message += "{:.3f} mm".format(Distance)

                message += "\n"
                if not PhysicalLayerStack:
                    message += "\nNo Physical Stackup could be found!"
                if short_path_RES > 0:
                    message += "\nResistance (only short path) ≈ "
                    message += "{:.3f} mOhm".format(short_path_RES * 1000)
                elif short_path_RES == 0:
                    message += "\nResistance (only short path) ≈ "
                    message += "{:.3f} mOhm".format(short_path_RES * 1000)
                    message += "\nSurfaces of the zones are considered perfectly "
                    message += "conductive and short-circuit points. This is probably the case here."
                else:
                    message += "\nNo connection was found between the two marked points"
                    if debug:
                        message += f"\n[DEBUG: Check debug log file for details]"
                        message += f"\nDebug log: {debug_log_file}"

                if not math.isinf(Resistance) and Resistance >= 0:
                    message += "\nResistance between both points  ≈ "
                    message += "{:.3f} mOhm".format(Resistance * 1000)
                elif Resistance < 0:
                    message += "\nERROR in Resistance Network calculation."
                    message += " Probably no ngspice installation could be found."
                    message += " The result about the short path"
                    message += " path is however uninfluenced."
                else:
                    message += "\nNo connection was found between the two marked points"
                    if debug:
                        message += f"\n[DEBUG: Check debug log file for details]"
                        message += f"\nDebug log: {debug_log_file}"

                # message += "\n"
                # if inductance_nH > 0:
                #     message += "\nThe determined self-inductance ≈ "
                #     message += "{:.3f} nH".format(inductance_nH)
                #     message += "\nHere it was assumed that the line is free without ground planes."
                #     message += "\nThe result is to be taken with special caution!"
                # else:
                #     message += "\nThe determined self-inductance ≈ NAN"
                #     message += "\nFor direct and uninterrupted connections the calculation is not applicable."

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

            dlg = wx.MessageDialog(
                None,
                message,
                "Analysis result",
                wx.OK,
            )
            dlg.ShowModal()
            dlg.Destroy()

            ####################################################
            # print pcb in matplotlib
            ####################################################

            # if debug:
            #     from Plot_PCB import Plot_PCB
            #     Plot_PCB(data)

        except Exception as e:
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
    from ItemList import data, board_FileName  # instead: import Get_PCB_Elements
    from Connect_Nets import Connect_Nets
    from Get_PCB_Stackup import Get_PCB_Stackup_fun
    from Get_Parasitic import Get_Parasitic
    from Plot_PCB import Plot_PCB
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
    Selected = [d for uuid, d in list(data.items()) if d["IsSelected"]]
    if len(Selected) == 2:
        conn1 = Selected[0]["netStart"][Selected[0]["Layer"][0]]
        conn2 = Selected[1]["netStart"][Selected[1]["Layer"][0]]
        NetCode = Selected[0]["NetCode"]
        if not NetCode == Selected[1]["NetCode"]:
            print("The marked points are not in the same network.")

        Resistance, Distance, inductance_nH, short_path_RES, Area = Get_Parasitic(
            data, CuStack, conn1, conn2, NetCode
        )
        print(f"Distance {Distance} mm")
        print(f"Resistance {Resistance} mOhm")
        print(f"Resistance {short_path_RES} mOhm (only short path)")
        print(f"inductance {inductance_nH} nH")
        # print(f"Area {Area} mm2")

        if len(Area) > 0:
            for layer in Area.keys():
                txt = "Layer {}: {:.3f} mm²".format(layer, Area[layer])
                print(txt)
    else:
        print("You have to mark exactly two elements.")

    # print pcb in matplotlib
    # Plot_PCB(data)
