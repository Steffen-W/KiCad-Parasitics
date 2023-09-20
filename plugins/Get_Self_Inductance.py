import numpy as np


def calculate_self_inductance(vertices, current):  # TODO Must be checked in more detail
    """
    Berechnet die Selbstinduktivität einer polygonalen Spule.

    Args:
    vertices (list of tuples): Eine Liste von Eckpunkten der Spule [(x1, y1, z1), (x2, y2, z2), ...].
    current (float): Der Strom, der durch die Spule fließt.

    Returns:
    float: Selbstinduktivität der Spule in Henry (H).
    """
    mu_0 = 4 * np.pi * 1e-7  # Magnetische Permeabilität des Vakuums

    total_inductance = 0.0

    for i in range(len(vertices)):
        p1 = np.array(vertices[i])
        p2 = np.array(vertices[(i + 1) % len(vertices)])  # Schließe den Kreis

        # Vektor vom Punkt p1 zum Punkt p2
        delta = p2 - p1

        # Berechne den Mittelpunkt zwischen p1 und p2
        midpoint = (p1 + p2) / 2.0

        # Berechne den Abstand vom Mittelpunkt zum Ursprung
        r = np.linalg.norm(midpoint)

        # Berechne die Selbstinduktivität dieses Leiterstücks
        dL = np.linalg.norm(delta)  # Länge des Leiterstücks
        dL_hat = delta / dL  # Einheitsvektor in Richtung des Stroms

        contribution = mu_0 * current * dL / (4 * np.pi * r)  # Beitrag zum Gesamtfeld

        total_inductance += contribution

    return total_inductance


def interpolate_vertices(vertices, num_points):
    """
    Interpoliert zwischen den gegebenen Eckpunkten, um mehr Punkte zu erzeugen.

    Args:
    vertices (list of tuples): Eine Liste von Eckpunkten der Spule [(x1, y1, z1), (x2, y2, z2), ...].
    num_points (int): Die gewünschte Anzahl von interpolierten Punkten zwischen den Eckpunkten.

    Returns:
    list of tuples: Eine Liste von interpolierten Punkten.
    """
    interpolated_points = []

    for i in range(len(vertices)):
        p1 = np.array(vertices[i])
        p2 = np.array(vertices[(i + 1) % len(vertices)])  # Schließe den Kreis

        for j in range(num_points):
            # Lineare Interpolation zwischen p1 und p2
            t = j / float(num_points)
            interpolated_point = tuple(p1 + t * (p2 - p1))
            interpolated_points.append(interpolated_point)

    return interpolated_points


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Beispielaufruf:
    vertices = [
        (0, 0, 0),
        (2, 0, 0),
        (2, 2, 0),
        (0, 2, 0),
    ]  # Beispiel-Eckpunkte einer quadratischen Spule
    current = 1.0  # Beispielstrom in Ampere

    num_points = 1000  # Beispielanzahl von interpolierten Punkten
    vertices = interpolate_vertices(vertices, num_points)

    xpoints = [i[0] for i in vertices]
    ypoints = [i[1] for i in vertices]
    plt.plot(xpoints, ypoints)
    plt.show()

    inductance = calculate_self_inductance(vertices, current) * 1000 * 1000
    print(f"Die Selbstinduktivität der Spule beträgt {inductance:.6f} uHenry (H).")
