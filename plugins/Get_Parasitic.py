import numpy as np
import os
import math
import traceback

from Get_Self_Inductance import calculate_self_inductance, interpolate_vertices

from Get_Distance import find_shortest_path, get_graph_from_edges


def round_n(n, decimals=0):
    if math.isinf(n):
        return n
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def RunSimulation(resistors, conn1, conn2):
    import ngspyce

    # https://github.com/ignamv/ngspyce/
    filename = os.path.join(os.path.dirname(__file__), "TempNetlist.net")

    Rshunt = 0.1
    with open(filename, "w") as f:
        f.write("* gnetlist -g spice-sdb\n")

        for i, res in enumerate(resistors):
            entry = "R{} {} {} {:.10f}\n".format(i + 1, res[0], res[1], res[2])
            f.write(entry)

        f.write("v1 {} 0 1\n".format(conn1))
        f.write("R{} 0 {} {}\n".format(i + 2, conn2, Rshunt))
        f.write(".end")

    ngspyce.source(filename)
    ngspyce.dc("v1", 1, 1, 1)  # set v1 to 1V
    os.remove(filename)
    vout = ngspyce.vector(str(conn2))[0]

    if not vout == 0:
        R = (1 - vout) / (vout / Rshunt)
    else:
        R = -1
    return R


# in https://www.youtube.com/watch?v=hNHTwpegFBw
# rho_cu = 1/47e6  # Ohm * m # 26% more than 1.68e-8

rho_cu = 1.68e-8  # Ohm * m


def calcResWIRE(Length, Width, cu_thickness=0.035, freq=0):
    # https://learnemc.com/EXT/calculators/Resistance_Calculator/rect.html

    if freq == 0:
        return Length * rho_cu / (cu_thickness * Width) * 1000.0
    else:  # TODO
        # mu = 1
        # SkinDepth = 1 / np.sqrt(freq * np.pi * mu / rho_cu) # in m
        return Length * rho_cu / (cu_thickness * Width) * 1000.0


def calcResVIA(Drill, Length, cu_thickness=0.035):
    radius = Drill / 2
    area = np.pi * ((radius + cu_thickness) ** 2 - radius**2)
    return Length * rho_cu / area * 1000


def Get_shortest_path_RES(path, resistors):
    def get_res(x1, x2):
        x = next(x for x in resistors if {x1, x2} == set(x[0:2]))
        return x[2]

    RES = 0
    for i in range(1, len(path)):
        RES += get_res(path[i - 1], path[i])

    return RES


def Get_Parasitic(data, CuStack, conn1, conn2, netcode):
    resistors = []
    coordinates = {}

    Area = {l: 0 for l in range(32)}  # for all layer

    for uuid, d in list(data.items()):
        if not netcode == d["NetCode"]:
            continue

        if len(d["Layer"]) > 1:
            for i in range(1, len(d["Layer"])):
                Layer1 = d["Layer"][i - 1]
                Layer2 = d["Layer"][i]
                thickness = CuStack[0]["thickness"]  # from Layer Top
                distance = CuStack[Layer2]["abs_height"] - CuStack[Layer1]["abs_height"]
                if "Drill" not in d:
                    continue
                resistor = calcResVIA(d["Drill"], distance, cu_thickness=thickness)
                resistors.append(
                    [d["netStart"][Layer1], d["netStart"][Layer2], resistor, distance]
                )
                coordinates[d["netStart"][Layer1]] = (
                    d["Position"][0],
                    d["Position"][1],
                    CuStack[Layer1]["abs_height"],
                )
                coordinates[d["netStart"][Layer2]] = (
                    d["Position"][0],
                    d["Position"][1],
                    CuStack[Layer2]["abs_height"],
                )

        else:
            Layer = d["Layer"][0]
            Area[Layer] += d["Area"]
            if d["type"] == "WIRE":
                netStart = d["netStart"][Layer]
                netEnd = d["netEnd"][Layer]
                thickness = CuStack[Layer]["thickness"]
                resistor = calcResWIRE(d["Length"], d["Width"], cu_thickness=thickness)
                resistors.append([netStart, netEnd, resistor, d["Length"]])

                coordinates[d["netStart"][Layer]] = (
                    d["Start"][0],
                    d["Start"][1],
                    CuStack[Layer]["abs_height"],
                )
                coordinates[d["netEnd"][Layer]] = (
                    d["End"][0],
                    d["End"][1],
                    CuStack[Layer]["abs_height"],
                )

    Area_reduc = {l: Area[l] for l in Area.keys() if Area[l] > 0}

    for res in resistors:
        if res[2] <= 0:
            raise ValueError("Error in resistance calculation!")

    # edges = list( (node1, node2, distance) )
    edges = [(i[0], i[1], i[3]) for i in resistors]
    graph = get_graph_from_edges(edges)
    try:
        Distance, path = find_shortest_path(graph, conn1, conn2)
        path3d = [coordinates[p] for p in path]
        short_path_RES = Get_shortest_path_RES(path, resistors)
    except Exception as e:
        short_path_RES = -1
        Distance, path3d = float("inf"), []
        print(traceback.format_exc())
        print("ERROR in find_shortest_path")

    inductance_nH = 0
    try:
        if len(path3d) > 2:
            vertices = interpolate_vertices(path3d, num_points=1000)
            inductance_nH = calculate_self_inductance(vertices, current=1) * 1e9
    except Exception as e:
        inductance_nH = 0
        print(traceback.format_exc())
        print("ERROR in calculate_self_inductance")

    try:
        Resistance = RunSimulation(resistors, conn1, conn2)
    except Exception as e:
        Resistance = -1
        print(traceback.format_exc())
        print("ERROR in RunSimulation")
    return Resistance, Distance, inductance_nH, short_path_RES, Area_reduc
