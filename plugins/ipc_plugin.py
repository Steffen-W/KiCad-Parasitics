"""IPC plugin entry point (plugin.json).

Logging starts before the plugin's modules are imported, so a missing
dependency or an incompatible Python still leaves its reason in the log file
and, if wx can be imported, in a message box.
"""

from __future__ import annotations

import logging
import os
import sys
import sysconfig
import traceback

from plugin_logging import configure_logging

log = logging.getLogger(__name__)


def _prefer_venv_packages() -> None:
    """Search the plugin's virtual environment before KiCad's own packages."""
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return
    venv_layout = "nt" if os.name == "nt" else "posix_prefix"
    venv_site_packages = sysconfig.get_path(
        "purelib", venv_layout, vars={"base": venv, "platbase": venv}
    )
    if venv_site_packages in sys.path:
        sys.path.remove(venv_site_packages)
    sys.path.insert(0, venv_site_packages)


def _show_import_error(details: str) -> None:
    try:
        import wx
    except ImportError:
        return
    _app = wx.App()
    text = (
        f"The Parasitics plugin could not start (Python {sys.version.split()[0]})."
        f"\n\n{details}"
    )
    with wx.MessageDialog(
        None, text, "Parasitics", wx.OK | wx.ICON_ERROR | wx.STAY_ON_TOP
    ) as dlg:
        dlg.ShowModal()


def main() -> None:
    configure_logging()
    log.debug("Plugin started with Python %s", sys.version)

    try:
        import wx

        _prefer_venv_packages()

        from kipy import KiCad, errors

        from parasitic import run_plugin, show_message
        from read_board_ipc import IpcBoard
    except Exception:
        log.exception("Importing the plugin failed")
        _show_import_error(traceback.format_exc())
        sys.exit(1)

    app = wx.App()
    try:
        kicad = KiCad()
        log.debug("Connected to KiCad %s", kicad.get_version())

        try:
            board = kicad.get_board()
        except errors.ApiError:
            log.exception("Failed to get board")
            show_message(
                None, "No PCB is open in KiCad.", "Parasitics", wx.ICON_WARNING
            )
            sys.exit(1)
        log.debug("Board: %s", board.document.board_filename)

        run_plugin(IpcBoard(board))
        # Keep event loop alive for any open PlotFrame windows
        app.MainLoop()
    except errors.ConnectionError:
        log.exception("ConnectionError")
        show_message(
            None,
            "Could not connect to KiCad. Is KiCad running with the API enabled?",
            "Parasitics",
            wx.ICON_ERROR,
        )
    except Exception:
        log.exception("Plugin failed")
        show_message(None, traceback.format_exc(), "Fatal Error", wx.ICON_ERROR)


if __name__ == "__main__":
    main()
