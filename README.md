# KiCad-Parasitics

Plugin to analyze the wires in the PCB editor. To use the plugin, two points must be marked on the board. This is best two pads that are connected with a wire. The tool then determines the DC resistance between the two points. A parasitic inductance wired also estimated. In future versions, the parasitic capacitance to the ground plane will be determined.

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
- [ ] Mac

KiCad versions
- [x] KiCad 7
- [x] KiCad 8
- [x] KiCad 9
