import pcbnew
from importlib import reload
import sys
import os
import wx
import traceback
from pprint import pprint
from pathlib import Path
import math


class ActionKiCadPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "parasitic"
        self.category = "parasitic"
        self.description = "parasitic"
        self.show_toolbar_button = True
        self.plugin_path = os.path.dirname(__file__)
        self.icon_file_name = os.path.join(self.plugin_path, "icon.png")
        self.dark_icon_file_name = os.path.join(self.plugin_path, "icon.png")

        # Füge das aktuelle Verzeichnis zum Modulpfad hinzu
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)

    def Run(self):
        try:
            print("###############################################################")

            import Get_PCB_Elements
            import Connect_Nets
            import Get_PCB_Stackup
            import Get_Parasitic

            board = pcbnew.GetBoard()
            connect = board.GetConnectivity()
            Settings = pcbnew.GetSettingsManager()

            # KiCad_CommonSettings = Settings.GetCommonSettings()
            KiCad_UserSettingsPath = Settings.GetUserSettingsPath()
            KiCad_SettingsVersion = Settings.GetSettingsVersion()
            board_FileName = Path(board.GetFileName())

            ####################################################
            # Get PCB Elements
            ####################################################

            reload(Get_PCB_Elements)
            from Get_PCB_Elements import Get_PCB_Elements, SaveDictToFile

            ItemList = Get_PCB_Elements(board, connect)

            ####################################################
            # save Variable ItemList to file (for debug)
            ####################################################

            save_as_file = os.path.join(self.plugin_path, "ItemList.py")
            print("save_as_file", save_as_file)
            SaveDictToFile(ItemList, save_as_file)
            with open(save_as_file, "a") as f:
                f.write('\nboard_FileName = "')
                f.write(str(board_FileName))
                f.write('"')

            ####################################################
            # connect nets together
            ####################################################

            reload(Connect_Nets)
            from Connect_Nets import Connect_Nets

            data = Connect_Nets(ItemList)
            # pprint(data)

            ####################################################
            # read PhysicalLayerStack from file
            ####################################################

            reload(Get_PCB_Stackup)
            from Get_PCB_Stackup import Get_PCB_Stackup

            PhysicalLayerStack, CuStack = Get_PCB_Stackup(ProjectPath=board_FileName)
            # pprint(CuStack)

            ####################################################
            # get resistance
            ####################################################

            reload(Get_Parasitic)
            from Get_Parasitic import Get_Parasitic

            Selected = [d for uuid, d in list(data.items()) if d["IsSelected"]]

            message = ""
            if len(Selected) == 2:
                conn1 = Selected[0]["netStart"][Selected[0]["Layer"][0]]
                conn2 = Selected[1]["netStart"][Selected[1]["Layer"][0]]
                NetCode = Selected[0]["NetCode"]
                if not NetCode == Selected[1]["NetCode"]:
                    message = "The marked points are not in the same network."
            else:
                message = "You have to mark exactly two elements."

            if message == "":
                Resistance, Distance, inductance_nH = Get_Parasitic(
                    data, CuStack, conn1, conn2, NetCode
                )

                if not math.isinf(Resistance):
                    message = "Resistance between both points  ≈ "
                    message += "{:.3f} mOhm".format(Resistance * 1000)
                    message += "\n\nShortest distance between the two points ≈ "
                    message += "{:.3f} mm".format(Distance)
                else:
                    message = "No connection was found between the two marked points"
                if inductance_nH > 0:
                    message += "\n\nThe determined self-inductance ≈ "
                    message += "{:.3f} nH".format(inductance_nH)
                    message += "\nHere it was assumed that the line is free without ground planes."
                    message += "\nThe result is to be taken with special caution!"

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

            # from Plot_PCB import Plot_PCB
            # Plot_PCB(data)

        except Exception as e:
            dlg = wx.MessageDialog(
                None,
                traceback.format_exc(),
                "Fatal Error",
                wx.OK | wx.ICON_ERROR,
            )
            dlg.ShowModal()
            dlg.Destroy()

        pcbnew.Refresh()


if not __name__ == "__main__":
    ActionKiCadPlugin().register()


if __name__ == "__main__":
    from ItemList import data, board_FileName  # instead: import Get_PCB_Elements
    from Connect_Nets import Connect_Nets
    from Get_PCB_Stackup import Get_PCB_Stackup
    from Get_Parasitic import Get_Parasitic
    from Plot_PCB import Plot_PCB

    # Get PCB Elements
    ItemList = data

    # connect nets together
    data = Connect_Nets(ItemList)
    # pprint(data)

    # read PhysicalLayerStack from file
    PhysicalLayerStack, CuStack = Get_PCB_Stackup(ProjectPath=board_FileName)
    pprint(CuStack)

    # get resistance
    Selected = [d for uuid, d in list(data.items()) if d["IsSelected"]]
    if len(Selected) == 2:
        conn1 = Selected[0]["netStart"][Selected[0]["Layer"][0]]
        conn2 = Selected[1]["netStart"][Selected[1]["Layer"][0]]
        NetCode = Selected[0]["NetCode"]
        if not NetCode == Selected[1]["NetCode"]:
            print("The marked points are not in the same network.")

        Resistance, Distance, inductance_nH = Get_Parasitic(
            data, CuStack, conn1, conn2, NetCode
        )
        print("Distance mm", Distance)
        print("Resistance mOhm", Resistance)
        print("inductance_nH", inductance_nH)
    else:
        print("You have to mark exactly two elements.")

    # print pcb in matplotlib
    Plot_PCB(data)
