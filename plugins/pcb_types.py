# Central definitions for element-type strings and layer IDs used throughout
# the plugin suite.  Import these constants instead of using literal strings so
# that typos are caught at import time and a single change propagates everywhere.

# Element types (value of the "type" key in the element dicts)
WIRE = "WIRE"
VIA = "VIA"
PAD = "PAD"
ZONE = "ZONE"

# Layer indices used in the coil-generator JSON format (not KiCad layer IDs).
# Plot_PCB.py maps these to colours: 0 → red (top), 2 → blue (bottom).
TOP_LAYER = 0
BOTTOM_LAYER = 2
