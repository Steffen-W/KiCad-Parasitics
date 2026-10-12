"""Logging setup shared by the IPC and pcbnew entry points (standard library only)."""

from __future__ import annotations

import logging
import os

LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "debug_parasitic.log"
)

# Third-party loggers whose debug output would drown the plugin's own.
QUIET_LOGGERS = (
    "matplotlib",
    "matplotlib.font_manager",
    "matplotlib.ticker",
    "matplotlib.colorbar",
    "matplotlib.backends",
    "pynng",
    "pynng.nng",
    "ngspyce",
    "sharedspice",
)


def configure_logging() -> None:
    """Log everything of the plugin to LOG_FILE, replacing the previous run."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
        filename=LOG_FILE,
        filemode="w",
        encoding="utf-8",
    )
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
