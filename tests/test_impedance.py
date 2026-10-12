"""Consistency of the quasi-static line formulas in impedance.py."""

from __future__ import annotations

import math
from typing import Callable

import pytest

import impedance as imp

_C0 = 299_792_458.0
_EPS0 = 8.8541878128e-12
_W, _H, _T, _ER, _GAP = 0.2e-3, 0.2e-3, 35e-6, 4.5, 0.15e-3

# (Z0, C per m, L per m, effective permittivity) of each line type
_LINES: dict[str, Callable[[], tuple[float, float, float, float]]] = {
    "microstrip": lambda: (
        imp.get_Microstrip_Z0(_W, _H, _ER),
        imp.get_Microstrip_Cap(_W, _H, 1.0, _ER),
        imp.get_Microstrip_Ind(_W, _H, 1.0, _ER),
        imp.get_Microstrip_eps_eff(_W, _H, _ER),
    ),
    "stripline": lambda: (
        imp.get_Stripline_Z0(_W, 2 * _H, _T, _ER),
        imp.get_Stripline_Cap(_W, 2 * _H, 1.0, _T, _ER),
        imp.get_Stripline_Ind(_W, 2 * _H, 1.0, _T, _ER),
        _ER,
    ),
    "coplanar": lambda: (  # on an infinitely thick substrate
        imp.get_Coplanar_Z0(_W, _GAP, _ER),
        imp.get_Coplanar_Cap(_W, _GAP, 1.0, _ER),
        imp.get_Coplanar_Ind(_W, _GAP, 1.0, _ER),
        (_ER + 1) / 2,
    ),
}


@pytest.mark.parametrize("line", _LINES)
def test_line_parameters_agree(line: str) -> None:
    z0, cap, ind, eps_eff = _LINES[line]()

    assert math.sqrt(ind / cap) == pytest.approx(z0, rel=1e-3)
    assert 1 / math.sqrt(ind * cap) == pytest.approx(_C0 / math.sqrt(eps_eff), rel=1e-3)


def test_quasi_static_microstrip_matches_the_frequency_model() -> None:
    thin = imp.analyze_microstrip(
        w=_W, h=_H, t=1e-9, epsilon_r=_ER, frequency=1e6, length=0.01
    )

    assert imp.get_Microstrip_Z0(_W, _H, _ER) == pytest.approx(thin["z0"], rel=0.01)
    assert imp.get_Microstrip_eps_eff(_W, _H, _ER) == pytest.approx(
        thin["epsilon_eff"], rel=0.01
    )


def test_plate_and_zone_capacitance() -> None:
    area, h = 1e-6, 1e-3
    plate = _EPS0 * _ER * area / h

    assert imp.get_Plate_Cap(1e-3, h, 1e-3, _ER) == pytest.approx(plate)
    assert imp.get_Zone_Cap(area, h, _ER) == pytest.approx(plate)
    assert imp.get_Zone_Cap(area, h, _ER, h, _ER) == pytest.approx(2 * plate)


@pytest.mark.parametrize(
    ("sketch", "planes"),
    [
        (imp.ascii_microstrip, 1),
        (imp.ascii_stripline, 2),
        (imp.ascii_coplanar, 2),
        (imp.ascii_coplanar_grounded, 3),
    ],
)
def test_cross_sections_show_signal_and_ground(
    sketch: Callable[[], str], planes: int
) -> None:
    drawing = sketch()

    assert drawing.count("SIG") == 1
    assert drawing.count("GND") == planes
