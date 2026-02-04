from os.path import exists
from pathlib import Path

if __name__ == "__main__":
    from s_expression_parse import parse_sexp
else:
    from .s_expression_parse import parse_sexp

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


def Get_PCB_Stackup_fun(ProjectPath: str | Path = "./test.kicad_pcb", new_v9=True, board_thickness=None):
    def readFile2var(path):
        if not exists(path):
            return None

        with open(path, "r") as file:
            data = file.read()
        return data

    PhysicalLayerStack = []
    CuStack = {}
    parsed = None
    try:
        if exists(ProjectPath):
            txt = readFile2var(ProjectPath)
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
                        PhysicalLayerStack.append(tmp)
                break

            for Layer in PhysicalLayerStack:
                if Layer["cu_layer"] is not None:
                    CuStack[Layer["cu_layer"]] = {
                        "thickness": Layer["thickness"],
                        "name": Layer["layer"],
                        "abs_height": Layer["abs_height"],
                    }
                    if Layer["thickness"] <= 0:
                        raise Exception("Problematic layer thickness detected")
    except Exception:
        print("ERROR: Reading the CuStack")

    if not CuStack:
        layers = search_recursive(parsed, "layers", all=True) if parsed else None
        if layers:
            cu_layers = []
            for layer in layers:
                if isinstance(layer, list) and "signal" in layer:
                    layer_name = layer[1]
                    extract_fn = extract_layer_from_string if new_v9 else extract_layer_from_string_old
                    cu_layer = extract_fn(layer_name)
                    if cu_layer is not None:
                        cu_layers.append((cu_layer, layer_name))

            cu_layers.sort(key=lambda x: x[0])
            n = len(cu_layers)
            total_height = board_thickness if board_thickness else 1.6

            for i, (cu_layer, layer_name) in enumerate(cu_layers):
                CuStack[cu_layer] = {
                    "thickness": 0.035,
                    "name": layer_name,
                    "abs_height": i * total_height / max(1, n - 1) if n > 1 else 0.0,
                }
            if CuStack:
                print("WARNING: No stackup found, estimated CuStack:", CuStack)

    if not CuStack:
        total = board_thickness if board_thickness else 1.6
        print(f"WARNING: No layer info found. Using 2-layer default: {total}mm, 35µm copper")
        b_cu = 2 if new_v9 else 31
        CuStack[0] = {"thickness": 0.035, "name": "F.Cu", "abs_height": 0.0}
        CuStack[b_cu] = {"thickness": 0.035, "name": "B.Cu", "abs_height": total}

    return PhysicalLayerStack, CuStack
