import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from matplotlib.transforms import Affine2D
import numpy as np


def Plot_PCB(data):
    figure, axes = plt.subplots()
    axes.set_aspect(1)
    axes.invert_yaxis()


    Color = {0: "red", 1: "green", 2: "orange", 3: "cyan", 4: "pink", 31: "blue"}
    for i in range(0, 32):
        if i not in Color:
            Color[i] = "silver"

    def NameNetInPlot(uuid, layer, active=True, netStart=True):
        if netStart:
            text = data[uuid]["netStart"]
            if "Start" in data[uuid]:
                pos = data[uuid]["Start"]
            else:
                pos = data[uuid]["Position"]
        else:
            text = data[uuid]["netEnd"]
            pos = data[uuid]["End"]
        if layer == 0:
            plt.text(
                *pos,
                text[layer],
                horizontalalignment="left",
                verticalalignment="bottom",
            )
        elif layer == 31:
            plt.text(
                *pos, text[layer], horizontalalignment="left", verticalalignment="top"
            )
        elif layer == 1:
            plt.text(
                *pos,
                text[layer],
                horizontalalignment="right",
                verticalalignment="bottom",
            )
        else:
            plt.text(
                *pos, text[layer], horizontalalignment="right", verticalalignment="top"
            )

    for uuid, d in list(data.items()):
        if d["type"] == "VIA":
            circ = Circle(d["Position"], d["Width"] / 2, color="grey", alpha=0.5)
            axes.add_artist(circ)
            if "Drill" in d:
                axes.add_artist(Circle(d["Position"], d["Drill"] / 2, color="w"))
            # plt.text(*d["Position"], str(data[uuid]["netStart"]))
            for layer in d["Layer"]:
                NameNetInPlot(uuid, layer)

    def plotwire(Start, End, Width, layer, uuid):
        plt.arrow(
            Start[0],
            Start[1],
            End[0] - Start[0],
            End[1] - Start[1],
            width=Width,
            head_length=0,
            head_width=Width,
            color=Color[layer],
            alpha=0.5,
        )
        axes.add_artist(Circle(Start, Width / 2, color=Color[layer], alpha=0.25))
        axes.add_artist(Circle(End, Width / 2, color=Color[layer], alpha=0.25))
        NameNetInPlot(uuid, layer, netStart=True)
        NameNetInPlot(uuid, layer, netStart=False)

    for uuid, d in list(data.items()):
        if d["type"] == "WIRE":
            plotwire(d["Start"], d["End"], d["Width"], d["Layer"][0], uuid)
            # data[uuid]["R"] = calcResWIRE(d["Start"], d["End"], d["Width"])

    for uuid, d in list(data.items()):
        if d["type"] == "PAD":
            if d["Shape"] in {0, 2}:  # oval
                ellip = Ellipse(
                    d["Position"],
                    *d["Size"],
                    color=Color[d["Layer"][0]],
                    alpha=0.5,
                    angle=d["Orientation"],
                )
                axes.add_patch(ellip)
            else:
                rec = Rectangle(
                    np.array(d["Position"]) - np.array(d["Size"]) / 2,
                    width=d["Size"][0],
                    height=d["Size"][1],
                    color=Color[d["Layer"][0]],
                    alpha=0.5,
                    transform=Affine2D().rotate_deg_around(
                        *d["Position"], d["Orientation"]
                    )
                    + axes.transData,
                )
                axes.add_patch(rec)
            NameNetInPlot(uuid, d["Layer"][0])

    plt.show()
