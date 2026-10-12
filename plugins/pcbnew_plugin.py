import os

import pcbnew
import wx

from .plugin_logging import configure_logging
from .read_board_pcbnew import PcbnewBoard
from .result_window import run_plugin

configure_logging(__name__.rpartition(".")[0])  # the plugin package


class KiCadPluginParasitic(pcbnew.ActionPlugin):
    def defaults(self) -> None:
        self.name = "Parasitics (pcbnew)"
        self.category = "Parasitics"
        self.description = "Parasitics (pcbnew)"
        self.show_toolbar_button = True
        self.plugin_path = os.path.dirname(os.path.abspath(__file__))
        self.icon_file_name = os.path.join(self.plugin_path, "icon_small.png")
        self.dark_icon_file_name = os.path.join(self.plugin_path, "icon_small.png")

    def Run(self) -> None:
        run_plugin(
            lambda: PcbnewBoard(pcbnew.GetBoard()), wx.FindWindowByName("PcbFrame")
        )


KiCadPluginParasitic().register()
