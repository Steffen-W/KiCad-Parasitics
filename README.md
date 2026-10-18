![Static Badge](https://img.shields.io/badge/Supports_KiCad-v9%2B-%23314cb0)
![Static Badge](https://img.shields.io/badge/Supports-Windows%2C_Mac%2C_Linux-Green)

[![GitHub Release](https://img.shields.io/github/release/Steffen-W/KiCad-Parasitics.svg)](https://github.com/Steffen-W/KiCad-Parasitics/releases/latest)
[![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/Steffen-W/KiCad-Parasitics/total)](https://github.com/Steffen-W/KiCad-Parasitics/releases/latest/download/KiCad-Parasitics.zip)

# KiCad-Parasitics

Plugin to analyze the wires in the PCB editor. To use the plugin, two points must be marked on the board. This is best two pads that are connected with a wire. The tool then determines the DC resistance between the two points.

## Development

The minimum supported Python version is 3.9, the interpreter bundled with KiCad on macOS. Modules that use `X | Y` annotations therefore start with `from __future__ import annotations`, and runtime-evaluated type aliases use `typing.Union`.

Before every nightly release, CI runs ruff (including rule FA102), mypy, vermin, a dry-run install of `plugins/requirements.txt` for Python 3.9 on macOS, and the tests under Python 3.9 and 3.12. The tests need libngspice (e.g. `apt install libngspice0`) and only part of the plugin dependencies. Without bfieldtools the inductance tests are skipped, without wxPython and a display the result window tests (on Linux run `xvfb-run pytest`):

```bash
pip install pytest numpy scipy kicad-python -c plugins/requirements.txt
pytest
mypy plugins
ruff check --fix --unsafe-fixes && ruff format
```

### Board input

Both plugins read the board the same way; only the API access differs. An adapter per API (`read_board_pcbnew.py`, `read_board_ipc.py`) implements `BoardSource` from `board_model.py` and only converts KiCad objects into the element dicts of `pcb_types.py` (lengths in m, layers as KiCad layer IDs, UUID strings as IDs). `read_board()` then applies the shared rules:

1. **Selection**: the selected tracks, arcs, vias and pads; a footprint stands for its pads. Anything else (zones, text, graphics) is no measurement point and left out. A selected item on no enabled copper layer is an error, since it would silently change what is measured.
2. **Nets**: the net of the first item plus every net that Net Tie footprints join to it, transitively. All other items must belong to one of them. Only the copper of these nets is read from KiCad.
3. **Copper**: tracks, arcs, vias, pads, copper zones (no rule areas or teardrops) and the copper of the involved Net Ties (segments and arcs as tracks, polygons as zones). Pads and zones are described by their outline.
4. **Connectivity** (`connectivity.py`): elements touch on a shared layer when wire ends or via/pad centres overlap, such a point lies on a wire or within a pad or zone outline, a wire crosses an outline, or two outlines overlap. KiCad's own connectivity is not used, since the IPC API does not provide it.
5. **Stackup**: copper and dielectric layers from top to bottom, from the IPC API or, for pcbnew, whose Python API lacks the stackup, from the saved board file. Without a usable stackup, a two-layer FR4 board is assumed.
6. **Measurement points** (`parasitic.selection_ends`, on the nodes of `connect_nets.py`): selected tracks join the nodes at their ends, a pad or via, selected or not, is one point on all its layers. The free ends are the points of the selection touched by at most one selected track. The analysis measures between the two free ends with the longest shortest path between them; for a pad or via the layer with the shortest path is used. If the free ends lie on separate copper islands, no single path stands for the selection: the analysis stops and lists the ends of each island.

### Result window

Both plugins show their results in the same non-modal window (`result_window.py`), so KiCad stays usable while it is open. The analysis and the inductance calculation run in a background thread; the window shows their progress and can cancel the analysis. The thread never calls wx: the window polls its progress with a timer and receives its outcome via `wx.CallAfter`. Clicking the plugin button again, or Refresh in the window, re-reads the selection and analyzes it in the same window; Calculate inductance is enabled once an analysis found a path. KiCad starts a new process for every click on the IPC plugin's button; while a window is open for the same KiCad, that process hands the request to it over a local socket (in the user's runtime directory, or a named pipe on Windows) and exits. A reopened window stays above all others for a moment so it does not end up behind KiCad, then stacks normally. Debug plots stay above the window and close with it.

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

Select what to measure across, all in the **same net** or in nets joined by a Net Tie footprint: two pads or vias, a footprint with two pads, a single track, or a whole run of tracks. The plugin measures between the two free ends of the selection that lie farthest apart along the copper and names them in the result. It then:

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

A zone is a nearly ideal conductor: every element touching it is linked with 1 mΩ to one star node of the zone, and zones that overlap on a layer share their star node. The links have no length, so the reported path length does not count the way across a zone.

### Inductance

The **Inductance** tab of the result window computes the loop inductance of the shortest path with [bfieldtools](https://bfieldtools.github.io/) (BEM on a triangulated surface mesh). The path's center line, closed into a loop, becomes a strip of the trace width at the height of each layer from the stackup; vias join the layers. **Debug plots** show the center line, the inductance along the path and the magnetic field with the current distribution.

Requires `bfieldtools`, `trimesh` and `scipy`, which `plugins/requirements.txt` installs with their version limits.
<img width="1017" height="665" alt="grafik" src="https://github.com/user-attachments/assets/bbfc86a4-480f-494b-8597-f3ede231d3c2" />

## Limitations

- **Pads**: Treated as ideal connection points with negligible resistance.
- **Wire connections**: Connections are always made to the **start or end of a trace**, never to a point in the middle.
- **Path length**: The reported distance is the sum of trace and via lengths along the path, not straight-line distance; crossing a zone adds no length.
- **Stacked traces**: Multiple traces on different layers connected by vias at both ends may be treated as parallel paths.
- **Coupled lines**: Mutual inductance/capacitance between adjacent traces is not considered. This also means **PCB inductors** (spirals, meanders) cannot be calculated or detected. *(Contributions welcome!)*
- **Inductance – shortest path only**: The inductance uses only the shortest path between the two selected points. Parallel paths or alternative routes are ignored.
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
