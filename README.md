![Static Badge](https://img.shields.io/badge/Supports_KiCad-v9%2B-%23314cb0)
![Static Badge](https://img.shields.io/badge/Supports-Windows%2C_Mac%2C_Linux-Green)

[![GitHub Release](https://img.shields.io/github/release/Steffen-W/KiCad-Parasitics.svg)](https://github.com/Steffen-W/KiCad-Parasitics/releases/latest)
[![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/Steffen-W/KiCad-Parasitics/total)](https://github.com/Steffen-W/KiCad-Parasitics/releases/latest/download/KiCad-Parasitics.zip)

# KiCad-Parasitics

Plugin to analyze the wires in the PCB editor. To use the plugin, two points must be marked on the board. This is best two pads that are connected with a wire. The tool then determines the DC resistance between the two points.

## Development

The minimum supported Python version is 3.9, the interpreter bundled with KiCad on macOS. Modules that use `X | Y` annotations therefore start with `from __future__ import annotations`, and runtime-evaluated type aliases use `typing.Union`.

Before every nightly release, CI runs ruff (including rule FA102), mypy, vermin, a dry-run install of `plugins/requirements.txt` for Python 3.9 on macOS, and the tests under Python 3.9 and 3.12. The tests need libngspice (e.g. `apt install libngspice0`) and only part of the plugin dependencies:

```bash
pip install pytest numpy scipy kicad-python -c plugins/requirements.txt
pytest
mypy plugins
ruff check --fix --unsafe-fixes && ruff format
```

### Board input

Both plugins read the board the same way; only the API access differs. An adapter per API (`read_board_pcbnew.py`, `read_board_ipc.py`) implements `BoardSource` from `board_model.py` and only converts KiCad objects into the element dicts of `pcb_types.py` (lengths in m, layers as KiCad layer IDs, UUID strings as IDs). `read_board()` then applies the shared rules:

1. **Selection**: exactly two copper items with a net; a single selected footprint with two pads stands for its pads.
2. **Nets**: the net of the first item plus every net that Net Tie footprints join to it, transitively. The second item must belong to one of them. Only the copper of these nets is read from KiCad.
3. **Copper**: tracks, arcs, vias, pads, copper zones (no rule areas or teardrops) and the copper of the involved Net Ties (segments and arcs as tracks, polygons as zones). Pads and zones are described by their outline.
4. **Connectivity** (`connectivity.py`): elements touch on a shared layer when wire ends or via/pad centres overlap, such a point lies on a wire or within a pad or zone outline, a wire crosses an outline, or two outlines overlap. KiCad's own connectivity is not used, since the IPC API does not provide it.
5. **Stackup**: copper and dielectric layers from top to bottom, from the IPC API or, for pcbnew, whose Python API lacks the stackup, from the saved board file. Without a usable stackup, a two-layer FR4 board is assumed.

## API Support

- **pcbnew API** (KiCad 9+, deprecated by KiCad): The classic scripting API, runs as a KiCad plugin inside the PCB editor. Works without further setup.
- **IPC API** (KiCad 9+): The new KiCad IPC API via [kipy](https://docs.kicad.org/kicad-python-main/). Its button only appears after enabling the API in KiCad (Preferences → Plugins → Enable KiCad API). Can also be started externally outside of KiCad (e.g. from the command line) while KiCad is running:
  ```bash
  git clone https://github.com/Steffen-W/KiCad-Parasitics.git
  cd KiCad-Parasitics
  pip install -r plugins/requirements.txt
  python plugins/ipc_plugin.py
  ```
  KiCad must be running with a board open.

The simulation uses the ngspice library: a system installation, else the one bundled with KiCad on macOS and Windows. If neither is found, install ngspice (macOS: `brew install libngspice`) or set `LIBNGSPICE` to the library path.

# How It Works

Select exactly two elements (pads or vias) that belong to the **same net**, or to nets joined by a Net Tie footprint. The plugin then:

1. **Extracts all traces** in that net and builds a resistance network
2. **Calculates wire resistance** based on trace length, width, and copper thickness from the stackup
3. **Simulates the network** using ngspice to determine the total DC resistance between the two points
4. **Finds the shortest path** and reports both the path length and estimated resistance
5. **Estimates copper area** per layer for the entire net (traces only, excluding zones and vias)

## Electrical Models

### Trace Analysis

The stackup determines automatically whether a trace is **Microstrip** or **Stripline**:

```
Outer layers (F.Cu, B.Cu):       Inner layers (In1.Cu, In2.Cu, ...):
Microstrip                       Stripline

    ====     SIG                 ------------ GND
░░░░░░░░░░░░ DIE                 ░░░░░░░░░░░░ DIE
------------ GND                     ====     SIG
                                 ░░░░░░░░░░░░ DIE
                                 ------------ GND
```

**Future:** Coplanar waveguides are also supported in the calculation engine:

```
Coplanar:                        Coplanar with Ground:

GND SIG GND                      GND SIG GND
=== === ===                      === === ===
░░░░░░░░░░░ DIE                  ░░░░░░░░░░░ DIE
                                 ----------- GND
```

### Long Traces (AC)

Short traces use a **lumped** model. When a trace exceeds λ/20, it is automatically split into segments (**distributed** RLGC model):

```
Lumped (< λ/20):              Distributed (> λ/20, e.g. 3 segments):

n1 ──R──L──┬── n2             n1 ─R─L─┬─ s1 ─R─L─┬─ s2 ─R─L─┬─ n2
           │                          │          │          │
           C                         C/3        C/3        C/3
          GND                        GND        GND        GND
```

### Vias

```
Top ---- o
░░░░░░░░ | ░░░░░░░░
Bot ---- o

Model: R-L in series, C parallel to planes
```

### Zones

All elements connected to the same zone are treated as low-resistance connections (1 mΩ).

### Calc Inductance

The **Calc Inductance** button appears next to "Details" when a shortest path exists. It computes the loop inductance of that path using [bfieldtools](https://bfieldtools.github.io/) (BEM on triangulated surface meshes). The trace center-line is buffered to its width, triangulated, and assigned z-coordinates per layer from the stackup. Multi-layer paths with vias are supported.

Requires additional packages: `bfieldtools`, `shapely`, `triangle`, `trimesh`, `scipy<1.14`.
<img width="1017" height="665" alt="grafik" src="https://github.com/user-attachments/assets/bbfc86a4-480f-494b-8597-f3ede231d3c2" />

## Limitations

- **Pads**: Treated as ideal connection points with negligible resistance.
- **Wire connections**: Connections are always made to the **start or end of a trace**, never to a point in the middle.
- **Path length**: The reported distance is the sum of trace lengths along the path, not straight-line distance.
- **Stacked traces**: Multiple traces on different layers connected by vias at both ends may be treated as parallel paths.
- **Coupled lines**: Mutual inductance/capacitance between adjacent traces is not considered. This also means **PCB inductors** (spirals, meanders) cannot be calculated or detected. *(Contributions welcome!)*
- **Inductance – shortest path only**: Calc Inductance uses only the shortest path between the two selected points. Parallel paths or alternative routes are ignored.
- **Inductance – no plane influence**: Metal planes (ground, power) near the trace reduce the effective loop inductance. This effect is currently not modelled.
- **Inductance – arcs**: Arc tracks are treated as straight lines (chord) for the inductance mesh.
- **Connectivity from geometry**: Connections are determined from shapes and positions, not from KiCad's connectivity data (see [Board input](#board-input)). Zones count by their outline, not their fill, so copper inside a zone outline is connected to the zone even where the fill is cut away.
- **pcbnew stackup**: The pcbnew plugin reads the stackup from the saved board file; unsaved stackup changes are not used.

<img width="825" height="713" alt="grafik" src="https://github.com/user-attachments/assets/eae13447-8f02-461b-81b4-e6a2fd24f130" />

# Example

First install Kicad-Parasitic from the Kicad "Plugin and Content Manager", then:
- open Kicad
- go to File -> Open Demo Project ...
- select Stickhub folder
- select StickHub.kicad_pro
- open StickHub.kicad_pcb
- zoom in to the front layer, close to the USB connector
- select the two D- vias
- press on the "parasitic" icon
<img width="1609" height="1115" alt="grafik" src="https://github.com/user-attachments/assets/fb002977-aebe-409e-8f98-523c0edcc2c3" />


# Tested until now

Operating systems
- [x] Windows
- [x] Linux
- [x] Mac

KiCad versions
- [x] KiCad 9
- [x] KiCad 10

# Feedback & Support

If you notice an error then please write me an issue. If you want to change the GUI or the functionality, I am also open for ideas.

[![Create Issue](https://img.shields.io/badge/Create%20Issue-blue.svg)](https://github.com/Steffen-W/KiCad-Parasitics/issues/new)

If you like the plugin, feel free to support me:

<a href="https://ko-fi.com/steffenw1" target="_blank"><img src="https://storage.ko-fi.com/cdn/brandasset/kofi_button_stroke.png" alt="Support via Ko-fi" height="30"></a>
