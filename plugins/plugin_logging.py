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


def configure_logging(name: str = "") -> None:
    """Log everything of logger `name` to LOG_FILE, replacing the previous run.

    The IPC plugin owns its process and configures the root logger. Inside
    KiCad, the pcbnew plugin configures only its package's logger, so neither
    KiCad nor other plugins write to the file. Called again, as when KiCad
    reloads its plugins, it replaces only its own handler.
    """
    logger = logging.getLogger(name)
    for own in [
        h for h in logger.handlers if getattr(h, "baseFilename", None) == LOG_FILE
    ]:
        logger.removeHandler(own)
        own.close()
    handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = not name
    for quiet in QUIET_LOGGERS:
        logging.getLogger(f"{name}.{quiet}" if name else quiet).setLevel(
            logging.WARNING
        )
