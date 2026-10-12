"""Progress reporting with cancellation for the steps of an analysis."""

from __future__ import annotations

from enum import Enum
from typing import Callable


class Stage(Enum):
    """Analysis steps in the order they run."""

    EXTRACT = "Extracting network"
    PATH = "Finding path"
    HF = "Computing HF parameters"
    SPICE = "SPICE simulation"


# Receives (stage, items finished, items in stage) while a stage runs;
# returning False cancels.
Progress = Callable[[Stage, int, int], bool]


def no_progress(_stage: Stage, _finished: int, _total: int) -> bool:
    return True


class CancelledError(Exception):
    """Raised when Progress asks to stop; the message addresses the user."""


def report(progress: Progress, stage: Stage, finished: int, total: int) -> None:
    """Pass the state to `progress` and raise CancelledError if it asks to stop."""
    if not progress(stage, finished, total):
        raise CancelledError("Analysis cancelled.")
