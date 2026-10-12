"""IPC plugin entry point (plugin.json).

Logging starts before the plugin's modules are imported, so a missing
dependency or an incompatible Python still leaves its reason in the log file
and, if wx can be imported, in a message box.

Later starts hand their request to the running instance (see README,
Result window).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import sysconfig
import tempfile
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
from typing import Callable

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


# A fixed key only keeps stray connections out. The sole request is "analyze
# again", so a foreign process at this address could at most trigger an
# analysis or, listening first, swallow the starts.
_AUTHKEY = b"kicad-parasitics"


def _instance_address() -> str:
    """Address of the instance serving this user's KiCad (by its API socket).

    Short, as a Unix socket path is limited to about 100 characters and the
    temporary directory on macOS alone takes about 50. The user's runtime
    directory is private; the shared /tmp is only the fallback.
    """
    kicad = os.environ.get("KICAD_API_SOCKET", "")
    name = "kparasitics-" + hashlib.sha256(kicad.encode()).hexdigest()[:12]
    if os.name == "nt":
        return rf"\\.\pipe\{name}"
    directory = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return os.path.join(directory, f"{name}-{os.getuid()}")


def _hand_over(address: str) -> bool:
    """Ask a running instance to analyze again; False if none is listening."""
    try:
        with Client(address, authkey=_AUTHKEY) as conn:
            conn.send("analyze")
    except (OSError, EOFError, AuthenticationError):
        return False
    return True


@contextmanager
def _serve_hand_overs(address: str, on_request: Callable[[], None]) -> Iterator[None]:
    """Call `on_request` (from a helper thread) for every hand-over.

    Without a listener the plugin still works, but every start opens a window.
    """
    try:
        if os.name != "nt" and os.path.exists(address):
            os.unlink(address)  # left behind by an instance that crashed
        listener = Listener(address, authkey=_AUTHKEY)
    except OSError:
        log.warning(
            "Cannot listen at %s, starts are not handed over", address, exc_info=True
        )
        yield
        return

    def serve() -> None:
        while True:
            try:
                conn = listener.accept()
            except AuthenticationError:
                continue
            except OSError:
                return  # listener closed
            with conn:
                try:
                    conn.recv()
                except (EOFError, OSError):
                    continue
            on_request()

    threading.Thread(target=serve, daemon=True).start()
    try:
        yield
    finally:
        listener.close()


def main() -> None:
    address = _instance_address()
    if _hand_over(address):
        return  # the running instance analyzes and logs
    configure_logging()
    log.debug("Plugin started with Python %s", sys.version)

    try:
        import wx

        _prefer_venv_packages()

        from kipy import KiCad, errors

        from board_model import BoardInputError, BoardSource
        from read_board_ipc import IpcBoard
        from result_window import run_plugin
    except Exception:
        log.exception("Importing the plugin failed")
        _show_import_error(traceback.format_exc())
        sys.exit(1)

    def open_board() -> BoardSource:
        try:
            board = KiCad().get_board()
        except errors.ConnectionError as e:
            raise BoardInputError(
                "Could not connect to KiCad. Is KiCad running with the API enabled?"
            ) from e
        except errors.ApiError as e:
            raise BoardInputError("No PCB is open in KiCad.") from e
        log.debug("Board: %s", board.document.board_filename)
        return IpcBoard(board)

    app = wx.App()
    with _serve_hand_overs(address, lambda: wx.CallAfter(run_plugin, open_board)):
        run_plugin(open_board)
        app.MainLoop()  # ends when the window is closed


if __name__ == "__main__":
    main()
