# KiCad-Parasitics

Plugin to analyze the wires in the PCB editor. To use the plugin, two points must be marked on the board. This is best two pads that are connected with a wire. The tool then determines the DC resistance between the two points.

# How It Works

Select exactly two elements (pads or vias) that belong to the **same net**. The plugin then:

1. **Extracts all traces** in that net and builds a resistance network
2. **Calculates wire resistance** based on trace length, width, and copper thickness from the stackup
3. **Simulates the network** using ngspice to determine the total DC resistance between the two points
4. **Finds the shortest path** and reports both the path length and estimated resistance
5. **Estimates copper area** per layer for the entire net (traces only, excluding zones and vias)

## Simplifications and Limitations

- **Zones (copper pours)**: All elements connected to the same zone are treated as low-resistance connections (1 mΩ). The actual resistance distribution within a zone is not calculated.
- **Vias**: Resistance is calculated based on drill diameter and copper thickness. Multi-layer vias connect all their layers.
- **Pads**: Treated as ideal connection points with negligible resistance.
- **Wire connections**: When traces do not perfectly align, the plugin attempts to find the connection point. However, connections are always made to the **start or end of a trace**, never to a point in the middle. This means if a via physically connects to the center of a trace, it will be electrically connected to the nearest trace endpoint in the model.
- **Path length**: The reported "shortest distance" is the sum of all trace lengths along the shortest electrical path, not the straight-line distance.
- **Stacked traces**: If multiple traces are placed on top of each other (e.g., on different layers connected by vias at both ends), they may be treated as parallel paths, resulting in a lower calculated resistance than the actual value.
- **DC resistance only**: Only the DC resistance is calculated. The AC resistance (at higher frequencies) is typically much higher due to skin effect and proximity effect.

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
- ![grafik](https://user-images.githubusercontent.com/3403218/274056663-e2c870e7-c23e-4c59-855d-cbc0a39c98f6.png)



# Tested until now

Operating systems
- [x] Windows
- [x] Linux
- [x] Mac

KiCad versions
- [x] KiCad 7
- [x] KiCad 8
- [x] KiCad 9
