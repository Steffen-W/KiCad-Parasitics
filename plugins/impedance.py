from math import log, pow, pi, sqrt

v_0 = 299792458  # m/s
epsilon_0 = 1e-9 / (36 * pi)


def get_Microstrip_Cap(w: float, h: float, l: float, epsilon_r: float):
    # https://www.emisoftware.com/calculator/microstrip-capacitance/

    if w < h:
        C = epsilon_r * l / (60 * v_0 * log(8 * h / w + w / 4 / h))
    else:
        C = (
            (epsilon_r * l)
            * (w / h + 1.393 + 0.667 * log(w / h + 1.444))
            / (120 * pi * v_0)
        )
    return C


def get_Microstrip_Z0(w: float, h: float, l: float, epsilon_r: float):
    # https://www.everythingrf.com/rf-calculators/microstrip-calculator
    wh = w / h

    if wh < 1:
        eps_e = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (
            1 / sqrt(1 + 12 / wh) + 0.4 * (1 - wh) * (1 - wh)
        )
        Z0 = 60 / sqrt(eps_e) * log(8 / wh + 0.25 * wh)
    else:
        eps_e = (epsilon_r + 1) / 2 + (epsilon_r - 1) / (2 * sqrt(1 + 12 / wh))
        Z0 = 120 * pi / (sqrt(eps_e) * (wh + 1.393 + 2 / 3 * log(wh + 1.444)))
    return Z0


def get_Plate_Cap(w: float, h: float, l: float, epsilon_r: float):
    C = epsilon_0 * epsilon_r * w * l / h
    return C


def get_Coplanar_Cap(w: float, gap: float, l: float, epsilon_r: float):
    # https://www.emisoftware.com/calculator/coplanar-capacitance/

    s = gap / (gap + 2 * w)
    if gap <= 1 / sqrt(2):
        x = pow(1 - s * s, 1 / 4)
        C = (epsilon_r * l) * log(-2 / (x - 1) * (x + 1)) / (377 * pi * v_0)
    else:
        C = (epsilon_r * l) / (120 * v_0 * log(-2 / (sqrt(s) - 1)) * (sqrt(s) + 1))
    return C


def get_Stripline_Cap(w: float, h: float, l: float, epsilon_r: float):
    ws = w / (h / 2)
    factor = epsilon_r * l / (30 * pi * v_0)

    if ws >= 0.35:
        C = factor * (ws + 0.441)
    else:
        C = factor * (ws - (0.35 - ws) * (0.35 - ws) + 0.441)
    return C


if __name__ == "__main__":
    w = 0.4 * 1e-3  # Width in m
    h = 1.55 * 1e-3  # Height above ground in m
    l = 1000 * 1e-3  # Length in m
    epsilon_r = 4.6  # Relative Permittivity

    gap = 100 * 1e-3  # Gap in m (only Coplanar)

    print("Microstrip C:", get_Microstrip_Cap(w, h, l, epsilon_r) * 1e12, "pF")
    print("Microstrip Z0:", get_Microstrip_Z0(w, h, l, epsilon_r), "Ohm")
    print("Stripline C:", get_Stripline_Cap(w, h * 2, l, epsilon_r) * 1e12, "pF")
    print("Plate C:", get_Plate_Cap(w, h, l, epsilon_r) * 1e12, "pF")
    print("Coplanar C:", get_Coplanar_Cap(w, gap, l, epsilon_r) * 1e12, "pF")
