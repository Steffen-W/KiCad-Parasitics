# Central definitions for element-type strings and layer IDs used throughout
# the plugin suite.  Import these constants instead of using literal strings so
# that typos are caught at import time and a single change propagates everywhere.

from typing import Any, Required, TypedDict

# Element types (value of the "type" key in the element dicts)
WIRE = "WIRE"

VIA = "VIA"
PAD = "PAD"
ZONE = "ZONE"

# Layer indices used in the coil-generator JSON format (not KiCad layer IDs).
# Plot_PCB.py maps these to colours: 0 → red (top), 2 → blue (bottom).
TOP_LAYER = 0
BOTTOM_LAYER = 2


class NetworkElement(TypedDict, total=False):
    """A single element in the PCB resistance network (WIRE / VIA / ZONE).

    The four ``Required`` fields are present in every element type.
    All other fields are type-specific:

    WIRE  – width, layer, layer_name, start, end  (+ arc fields mid/radius/angle/midline_pts)
    VIA   – drill, layer1, layer2, layer1_name, layer2_name, inductance, capacitance
    ZONE  – layer, layer_name
    """

    # ---- common (always present) ----
    type: Required[str]
    nodes: Required[tuple[int, int]]
    resistance: Required[float]
    length: Required[float]

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


# Type alias for the SPICE netlist element tuple (component, node1, node2, value).
# Nodes are int for real PCB nodes, str for synthetic segmentation nodes.
SpiceElement = tuple[str, int | str, int | str, float]
