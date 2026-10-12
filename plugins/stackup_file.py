"""Stackup from a .kicad_pcb file, for pcbnew, whose Python API lacks it."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .board_model import Copper, StackupEntry, dielectric
    from .s_expression_parse import parse_sexp
except ImportError:
    from board_model import Copper, StackupEntry, dielectric
    from s_expression_parse import parse_sexp

log = logging.getLogger(__name__)


def _entry(layer: list[Any]) -> StackupEntry | None:
    """A copper or dielectric layer of the stackup; None for other layers."""
    fields: defaultdict[str, list[Any]] = defaultdict(list)
    for field in layer[2:]:  # A dielectric repeats its fields per sublayer.
        if isinstance(field, list) and len(field) > 1:
            fields[field[0]].append(field[1])
    kind = fields["type"][0] if fields["type"] else None
    thickness = sum(fields["thickness"]) / 1000  # mm → m
    if kind == "copper":
        return Copper(str(layer[1]), thickness)
    if kind in ("core", "prepreg"):
        return dielectric(
            thickness,
            next(iter(fields["epsilon_r"]), None),
            next(iter(fields["loss_tangent"]), None),
        )
    return None


def read_stackup_file(path: Path) -> list[StackupEntry]:
    """Copper and dielectric layers of the board file's stackup, top to bottom.

    Empty if the file has no stackup or cannot be read.
    """
    try:
        text = path.read_text(encoding="utf-8")
        start = text.find("(stackup")
        stackup = parse_sexp(text, start) if start >= 0 else []
        layers = [
            e for e in stackup if isinstance(e, list) and len(e) > 1 and e[0] == "layer"
        ]
        entries = [_entry(layer) for layer in layers]
    except (OSError, ValueError, TypeError):
        log.exception("Reading the stackup from %s failed", path)
        return []
    return [e for e in entries if e is not None]
