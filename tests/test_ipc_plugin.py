"""Start of the IPC plugin: hand-over to a running instance, import errors."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_later_start_hands_over_to_the_running_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import uuid

    import ipc_plugin

    # A KiCad of its own, so no plugin instance running on this machine answers.
    monkeypatch.setenv("KICAD_API_SOCKET", f"ipc://test-{uuid.uuid4()}")
    address = ipc_plugin._instance_address()
    requested = threading.Event()
    assert not ipc_plugin._hand_over(address)

    with ipc_plugin._serve_hand_overs(address, requested.set):
        assert ipc_plugin._hand_over(address)
        assert requested.wait(5)

    assert not ipc_plugin._hand_over(address)


def test_import_failure_is_logged_and_shown(
    no_wx: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import logging
    import sys

    import ipc_plugin
    import plugin_logging

    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)
    for name in plugin_logging.QUIET_LOGGERS:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
    log_file = tmp_path / "debug.log"
    monkeypatch.setattr(plugin_logging, "LOG_FILE", str(log_file))
    # Not handed to a plugin instance that happens to run on this machine.
    monkeypatch.setattr(ipc_plugin, "_instance_address", lambda: str(tmp_path / "s"))
    monkeypatch.setitem(sys.modules, "result_window", None)

    with pytest.raises(SystemExit) as exit_info:
        ipc_plugin.main()
    for handler in root.handlers:
        handler.close()

    assert exit_info.value.code == 1
    logged = log_file.read_text(encoding="utf-8")
    assert "Importing the plugin failed" in logged
    assert "ModuleNotFoundError" in logged or "ImportError" in logged
    shown = sys.modules["wx"].MessageDialog.call_args.args[1]
    assert "could not start" in shown
    assert "result_window" in shown
