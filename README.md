# KiCad-Parasitics

... Documentation is still being created

Plugin to analyze the wires in the PCB editor. To use the plugin, two points must be marked on the board. This is best two pads that are connected with a wire. The tool then determines the DC resistance between the two points. A parasitic inductance wired also estimated. In future versions, the parasitic capacitance to the ground plane will be determined.

# Example

First install Kicad-Parasitic from the Kicad "Plugin and Content Manager", then:
- open Kicad
- go to File -> Open Demo Project ...
- select Stickhub folder
- select StickHub.kicad_pro
![grafik](https://github.com/nopeppermint/KiCad-Parasitics/assets/3403218/4780a4f3-2c2f-4d14-8325-577f7d687760)
- open StickHub.kicad_pcb
- zoom in to the front layer, close to the USB connector
- select the two D- vias
- press on the "parasitic" icon
- ![grafik](https://github.com/nopeppermint/KiCad-Parasitics/assets/3403218/e2c870e7-c23e-4c59-855d-cbc0a39c98f6)



# Tested until now

Operating systems
- [x] Windows
- [x] Linux
- [ ] Mac

KiCad versions
- [ ] KiCad 6 (will also not work in the future)
- [x] KiCad 7.0.8 (up to commit [cd378574](https://gitlab.com/kicad/code/kicad/-/commit/cd3785744c3f6654cb151ac6edae8142dd62173a))
- [x] KiCad 7.99.0-unknow (up to commit [099e5dd9](https://gitlab.com/kicad/code/kicad/-/commit/099e5dd9e7baec59cd68b9da01a6b35cdc972eb5))
