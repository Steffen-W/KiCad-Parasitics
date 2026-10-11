import logging
import os
from pathlib import Path

import pcbnew

from .get_pcb_elements import Get_PCB_Elements
from .get_pcb_stackup import Get_PCB_Stackup_fun
from .parasitic import run_plugin

if os.environ.get("PARASITIC_DEBUG"):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
        filename=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "debug_parasitic.log"
        ),
        filemode="w",
        encoding="utf-8",
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
    def defaults(self) -> None:
        self.name = "Parasitics (pcbnew)"
        self.category = "Parasitics"
        self.description = "Parasitics (pcbnew)"
        self.show_toolbar_button = True
        self.plugin_path = os.path.normpath(os.path.dirname(__file__))
        self.icon_file_name = os.path.join(self.plugin_path, "icon_small.png")
        self.dark_icon_file_name = os.path.join(self.plugin_path, "icon_small.png")

    def Run(self) -> None:
        board = pcbnew.GetBoard()
        connect = board.GetConnectivity()
        board_FileName = Path(board.GetFileName())
        ItemList, board_thickness = Get_PCB_Elements(board, connect)
        # KiCad 9 renumbered the copper layers (B.Cu: 31 -> 2).
        CuStack = Get_PCB_Stackup_fun(
            board_FileName, new_v9=pcbnew.B_Cu == 2, board_thickness=board_thickness
        )

        run_plugin(ItemList, CuStack)


KiCadPluginParasitic().register()
