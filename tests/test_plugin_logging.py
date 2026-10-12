"""Logging setup shared by both entry points."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import pytest

import plugin_logging


def test_reconfiguring_replaces_only_its_own_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_file = tmp_path / "debug.log"
    monkeypatch.setattr(plugin_logging, "LOG_FILE", str(log_file))
    name = f"plugin-{uuid.uuid4()}"
    logger = logging.getLogger(name)
    foreign = logging.NullHandler()
    logger.addHandler(foreign)

    plugin_logging.configure_logging(name)
    plugin_logging.configure_logging(name)  # as when KiCad reloads its plugins
    logging.getLogger(f"{name}.module").info("once")

    assert log_file.read_text(encoding="utf-8").count("once") == 1
    assert foreign in logger.handlers and len(logger.handlers) == 2
    assert not logger.propagate
    for handler in logger.handlers:
        handler.close()
