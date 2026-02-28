import logging
import os
from pathlib import Path
import pcbnew

from .Get_PCB_Elements import Get_PCB_Elements
from .Get_PCB_Stackup import Get_PCB_Stackup_fun
from .parasitic import run_plugin

if os.environ.get("PARASITIC_DEBUG"):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
        filename=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "debug_parasitic.log"
        ),
        filemode="w",
    )

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


class KiCadPluginParasitic(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "Parasitics (pcbnew)"
        self.category = "Parasitics"
        self.description = "Parasitics (pcbnew)"
        self.show_toolbar_button = True
        self.plugin_path = os.path.normpath(os.path.dirname(__file__))
        self.icon_file_name = os.path.join(self.plugin_path, "icon_small.png")
        self.dark_icon_file_name = os.path.join(self.plugin_path, "icon_small.png")

    def Run(self):
        board = pcbnew.GetBoard()
        connect = board.GetConnectivity()
        Settings = pcbnew.GetSettingsManager()

        try:
            new_v9 = int(str(Settings.GetSettingsVersion()).split(".")[0]) >= 9
        except Exception:
            new_v9 = True

        board_FileName = Path(board.GetFileName())
        ItemList, _BoardThickness = Get_PCB_Elements(board, connect)
        CuStack = Get_PCB_Stackup_fun(board_FileName, new_v9=new_v9)

        run_plugin(ItemList, CuStack)


KiCadPluginParasitic().register()
