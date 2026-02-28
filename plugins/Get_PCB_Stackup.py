from os.path import exists
from pathlib import Path

try:
    from .s_expression_parse import parse_sexp
except ImportError:
    from s_expression_parse import parse_sexp

import re


def extract_layer_from_string_old(input_string):  # kicad <9.0
    if input_string == "F.Cu":
        return 0
    elif input_string == "B.Cu":
        return 31
    else:
        match = re.search(r"In(\d+)\.Cu", input_string)
        if match:
            return int(match.group(1))
    return None


def extract_layer_from_string(input_string):  # kicad >=9.0
    # https://gitlab.com/kicad/code/kicad/-/commit/5e0abadb23425765e164f49ee2f893e94ddb97fc
    if input_string == "F.Cu":
        return 0
    elif input_string == "B.Cu":
        return 2
    else:
        match = re.match(r"In(\d+)\.Cu", input_string)
        if match:
            inner_index = int(match.group(1))
            return 2 * inner_index + 2  # In1_Cu = 4, In2_Cu = 6, ...
    return None


def search_recursive(line: list, entry: str, all=False):
    if isinstance(line[0], str) and line[0] == entry:
        if all:
            return line
        else:
            return line[1]

    for e in line:
        if isinstance(e, list):
            res = search_recursive(line=e, entry=entry, all=all)
            if res is not None:
                return res
    return None


def Get_PCB_Stackup_fun(
    ProjectPath: str | Path = "./test.kicad_pcb", new_v9=True, board_thickness=None
):
    layers = []
    CuStack = {}
    parsed = None
    try:
        if exists(ProjectPath):
            with open(ProjectPath, "r") as f:
                txt = f.read()
            parsed = parse_sexp(txt)

            while True:
                setup = search_recursive(parsed, "setup", all=True)
                if not setup:
                    break

                stackup = search_recursive(setup, "stackup", all=True)
                if not stackup:
                    break

                abs_height = 0.0
                for layer in stackup:
                    tmp = {}
                    tmp["layer"] = search_recursive(layer, "layer")
                    tmp["thickness"] = search_recursive(layer, "thickness")
                    tmp["epsilon_r"] = search_recursive(layer, "epsilon_r")
                    tmp["loss_tangent"] = search_recursive(layer, "loss_tangent")
                    tmp["type"] = search_recursive(layer, "type")

                    if tmp["thickness"] is not None:
                        if new_v9:
                            tmp["cu_layer"] = extract_layer_from_string(tmp["layer"])
                        else:
                            tmp["cu_layer"] = extract_layer_from_string_old(
                                tmp["layer"]
                            )
                        tmp["abs_height"] = abs_height
                        abs_height += float(tmp["thickness"])
                        layers.append(tmp)
                break

            for i, Layer in enumerate(layers):
                if Layer["cu_layer"] is not None:
                    t = float(Layer["thickness"]) / 1000  # mm → m
                    if t <= 0:
                        raise Exception("Problematic layer thickness detected")

                    def _die_info(idx):
                        if 0 <= idx < len(layers):
                            d = layers[idx]
                            if d["type"] in ("core", "prepreg"):
                                return {
                                    "h": float(d["thickness"]) / 1000,
                                    "epsilon_r": float(d["epsilon_r"])
                                    if d["epsilon_r"]
                                    else 4.3,
                                    "loss_tangent": float(d["loss_tangent"])
                                    if d["loss_tangent"]
                                    else 0.02,
                                }
                        return None

                    die_above = _die_info(i - 1)
                    die_below = _die_info(i + 1)

                    if die_above and die_below:
                        model = "Stripline"
                    elif die_above or die_below:
                        model = "Microstrip"
                    else:
                        model = "R only"

                    CuStack[Layer["cu_layer"]] = {
                        "thickness": t,
                        "name": Layer["layer"],
                        "abs_height": Layer["abs_height"] / 1000,  # mm → m
                        "die_above": die_above,
                        "die_below": die_below,
                        "model": model,
                        "gap": None,
                    }
    except Exception:
        print("ERROR: Reading the CuStack")

    # Fallback: 2-layer board with default values
    if not CuStack:
        total = board_thickness if board_thickness else 1.6e-3  # 1.6 mm in m
        t_cu = 35e-6
        h_die = total - 2 * t_cu
        print(
            f"WARNING: No layer info found. Using 2-layer default: {total * 1000}mm, 35µm copper"
        )
        b_cu = 2 if new_v9 else 31
        die = {"h": h_die, "epsilon_r": 4.3, "loss_tangent": 0.02}
        CuStack[0] = {
            "thickness": t_cu,
            "name": "F.Cu",
            "abs_height": 0.0,
            "die_above": None,
            "die_below": die,
            "model": "Microstrip",
            "gap": 0.2e-3,
        }
        CuStack[b_cu] = {
            "thickness": t_cu,
            "name": "B.Cu",
            "abs_height": total,
            "die_above": die,
            "die_below": None,
            "model": "Microstrip",
            "gap": 0.2e-3,
        }

    return CuStack


if __name__ == "__main__":
    test_file = "./test_kicad.kicad_pcb"
    print(f"Testing: {test_file}\n")

    CuStack = Get_PCB_Stackup_fun(test_file)

    print("=== CuStack ===")
    for layer_id, info in sorted(CuStack.items()):
        print(f"\nLayer {layer_id} ({info['name']}):")
        print(f"  thickness:    {info['thickness'] * 1e6:.1f} µm")
        print(f"  abs_height:   {info['abs_height'] * 1000:.3f} mm")
        print(f"  die_above:    {info['die_above']}")
        print(f"  die_below:    {info['die_below']}")
