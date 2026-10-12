# Central definitions for element-type strings and layer IDs used throughout
# the plugin suite.  Import these constants instead of using literal strings so
# that typos are caught at import time and a single change propagates everywhere.

from __future__ import annotations

from typing import Any, Final, Literal, TypedDict, Union

# Element types (value of the "type" key in the element dicts)
WIRE: Final = "WIRE"
VIA: Final = "VIA"
PAD: Final = "PAD"
ZONE: Final = "ZONE"

# Layer indices used in the coil-generator JSON format (not KiCad layer IDs).
# plot_pcb.py maps these to colours: 0 → red (top), 2 → blue (bottom).
TOP_LAYER = 0
BOTTOM_LAYER = 2

# x, y in m
Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Board elements, as read from KiCad (see board_model.py)
# ---------------------------------------------------------------------------


class _ElementBase(TypedDict):
    id: str  # KiCad UUID
    net: str  # net name
    layer: list[int]  # copper layers, KiCad layer IDs
    area: float  # copper area in m²
    is_selected: bool
    conn_start: list[str]  # IDs of touching elements (at the start of a wire)


class _WireBase(_ElementBase):
    type: Literal["WIRE"]
    start: Point
    end: Point
    width: float
    length: float  # along the midline
    midline_pts: list[Point]  # start … end; intermediate points follow an arc
    conn_end: list[str]  # IDs of elements touching the end


class WireElement(_WireBase, total=False):
    """A straight track, or an arc through mid with signed sweep angle (rad).

    radius is missing for an arc whose three points are collinear.
    """

    mid: Point
    radius: float
    angle: float


class ViaElement(_ElementBase):
    type: Literal["VIA"]
    position: Point
    width: float  # pad diameter
    drill: float


class PadElement(_ElementBase):
    type: Literal["PAD"]
    position: Point
    drill: float  # 0 for SMD pads
    outline: list[Point]  # copper outline on the first copper layer


class ZoneElement(_ElementBase):
    type: Literal["ZONE"]
    outline: list[Point]


BoardElement = Union[WireElement, ViaElement, PadElement, ZoneElement]


# ---------------------------------------------------------------------------
# Resistance network
# ---------------------------------------------------------------------------


class _NetworkElementBase(TypedDict):
    """Mandatory fields always present in every network element."""

    type: str
    nodes: tuple[int, int]
    resistance: float
    length: float


class NetworkElement(_NetworkElementBase, total=False):
    """A single element in the PCB resistance network (WIRE / VIA / ZONE).

    The four base fields (type, nodes, resistance, length) are always present.
    All other fields are type-specific:

    WIRE  – width, layer, layer_name, start, end  (+ arc fields mid/radius/angle/midline_pts)
    VIA   – drill, layer1, layer2, layer1_name, layer2_name, inductance, capacitance
    ZONE  – layer, layer_name
    """

    # ---- WIRE & ZONE ----
    layer: int
    layer_name: str

    # ---- WIRE ----
    width: float
    start: tuple[float, float]
    end: tuple[float, float]
    # arc / midline extras (optional even for WIREs)
    mid: tuple[float, float]
    radius: float
    angle: float
    midline_pts: list[tuple[float, float]]

    # ---- VIA ----
    position: tuple[float, float]
    drill: float
    layer1: int
    layer2: int
    layer1_name: str
    layer2_name: str
    inductance: float
    capacitance: float

    # ---- AC analysis (added dynamically by simulate_network) ----
    hf: dict[float, dict[str, Any]]


# ---------------------------------------------------------------------------
# Copper stackup
# ---------------------------------------------------------------------------


class DielectricInfo(TypedDict):
    """Dielectric layer between two copper layers (core or prepreg)."""

    h: float  # thickness in m
    epsilon_r: float  # relative permittivity
    loss_tangent: float


class _CuLayerBase(TypedDict):
    """Mandatory fields always present in every CuStack entry."""

    thickness: float  # copper thickness in m
    name: str  # e.g. "F.Cu", "In1.Cu", "B.Cu"
    abs_height: float  # absolute z-position in m (0 = bottom)


class CuLayer(_CuLayerBase, total=False):
    """One copper layer entry in the CuStack dict (key = KiCad layer ID).

    The three base fields (thickness, name, abs_height) are always present.
    Model-specific fields are optional and only populated by the stackup parsers.
    """

    die_above: DielectricInfo | None
    die_below: DielectricInfo | None
    model: str  # "Microstrip" | "Stripline" | "Coplanar" | "R only"
    gap: float | None  # coplanar gap in m, None for other models


# SPICE node: int for real PCB nodes, str for synthetic segmentation nodes.
SpiceNode = Union[int, str]

# SPICE netlist element tuple (component, node1, node2, value).
SpiceElement = tuple[str, SpiceNode, SpiceNode, float]
