![Static Badge](https://img.shields.io/badge/Supports_KiCad-v6%2C_v7%2C_v8%2C_v9-%23314cb0)
![Static Badge](https://img.shields.io/badge/Supports-Windows%2C_Mac%2C_Linux-Green)

[![GitHub Release](https://img.shields.io/github/release/Steffen-W/KiCad-Parasitics.svg)](https://github.com/Steffen-W/KiCad-Parasitics/releases/latest)
[![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/Steffen-W/KiCad-Parasitics/total)](https://github.com/Steffen-W/KiCad-Parasitics/releases/latest/download/KiCad-Parasitics.zip)

# KiCad-Parasitics

Plugin to analyze the wires in the PCB editor. To use the plugin, two points must be marked on the board. This is best two pads that are connected with a wire. The tool then determines the DC resistance between the two points.

## API Support

- **pcbnew API** (KiCad 7–9): The classic scripting API, runs as a KiCad plugin inside the PCB editor.
- **IPC API** (KiCad 9+): The new KiCad IPC API via [kipy](https://docs.kicad.org/kicad-python-main/). Can also be started externally outside of KiCad (e.g. from the command line) while KiCad is running:
  ```bash
  git clone https://github.com/Steffen-W/KiCad-Parasitics.git
  cd KiCad-Parasitics
  pip install -r plugins/requirements.txt
  python plugins/parasitic.py
  ```
  KiCad must be running with a board open.

# How It Works

Select exactly two elements (pads or vias) that belong to the **same net**. The plugin then:

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

## Limitations

- **Pads**: Treated as ideal connection points with negligible resistance.
- **Wire connections**: Connections are always made to the **start or end of a trace**, never to a point in the middle.
- **Path length**: The reported distance is the sum of trace lengths along the path, not straight-line distance.
- **Stacked traces**: Multiple traces on different layers connected by vias at both ends may be treated as parallel paths.
- **Coupled lines**: Mutual inductance/capacitance between adjacent traces is not considered. This also means **PCB inductors** (spirals, meanders) cannot be calculated or detected. *(Contributions welcome!)*
- **Inductance – shortest path only**: Calc Inductance uses only the shortest path between the two selected points. Parallel paths or alternative routes are ignored.
- **Inductance – no plane influence**: Metal planes (ground, power) near the trace reduce the effective loop inductance. This effect is currently not modelled.
- **Inductance – arcs**: Arc tracks are treated as straight lines (chord) for the inductance mesh.
- **IPC API – no native connectivity**: The IPC API does not provide connectivity data (unlike pcbnew's `CONNECTIVITY_DATA`). Connections are determined by geometric coordinate matching. This may reduce performance on very large nets (please open an issue if this is noticeable). Additionally, traces that connect mid-segment (not at a track endpoint) are currently not detected as connected.

<img width="969" height="872" alt="image" src="https://github.com/user-attachments/assets/b588b122-91ba-4538-b0dd-d0f684971774" />

# Example

First install Kicad-Parasitic from the Kicad "Plugin and Content Manager", then:
- open Kicad
- go to File -> Open Demo Project ...
- select Stickhub folder
- select StickHub.kicad_pro
![grafik](https://user-images.githubusercontent.com/3403218/274055069-4780a4f3-2c2f-4d14-8325-577f7d687760.png)
- open StickHub.kicad_pcb
- zoom in to the front layer, close to the USB connector
- select the two D- vias
- press on the "parasitic" icon
![grafik](https://user-images.githubusercontent.com/3403218/274056663-e2c870e7-c23e-4c59-855d-cbc0a39c98f6.png)

# Tested until now

Operating systems
- [x] Windows
- [x] Linux
- [x] Mac

KiCad versions
- [x] KiCad 7
- [x] KiCad 8
- [x] KiCad 9

# Feedback & Support

If you notice an error then please write me an issue. If you want to change the GUI or the functionality, I am also open for ideas.

[![Create Issue](https://img.shields.io/badge/Create%20Issue-blue.svg)](https://github.com/Steffen-W/KiCad-Parasitics/issues/new)

If you like the plugin, feel free to support me:

<a href="https://ko-fi.com/steffenw1" target="_blank"><img src="https://storage.ko-fi.com/cdn/brandasset/kofi_button_stroke.png" alt="Support via Ko-fi" height="30"></a>
