import os

import pcbnew

from .parasitic import run_plugin
from .plugin_logging import configure_logging
from .read_board_pcbnew import PcbnewBoard

if os.environ.get("PARASITIC_DEBUG"):
    configure_logging()


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
        run_plugin(PcbnewBoard(pcbnew.GetBoard()))


KiCadPluginParasitic().register()
