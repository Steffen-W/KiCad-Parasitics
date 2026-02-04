from math import log, log1p, pi, sqrt

v_0 = 299792458  # m/s speed of light (exact by SI definition)
epsilon_0 = 8.8541878128e-12  # F/m permittivity of free space (CODATA 2018)
mu_0 = 1 / (epsilon_0 * v_0 * v_0)  # H/m permeability (derived)


def _validate_microstrip(w: float, h: float, epsilon_r: float):
    """Validate microstrip parameters."""
    if w <= 0 or h <= 0:
        raise ValueError(f"w and h must be > 0, got w={w}, h={h}")
    if epsilon_r <= 0:
        raise ValueError(f"epsilon_r must be > 0, got {epsilon_r}")


def _validate_stripline(w: float, h: float, t: float, epsilon_r: float):
    """Validate stripline parameters."""
    if w <= 0 or h <= 0:
        raise ValueError(f"w and h must be > 0, got w={w}, h={h}")
    if epsilon_r <= 0:
        raise ValueError(f"epsilon_r must be > 0, got {epsilon_r}")
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    b = h / 2
    if t >= b:
        raise ValueError(f"t must be < h/2, got t={t}, h/2={b}")
    if t > 0:
        tb = t / b
        if tb > 0.25:
            raise ValueError(f"t/b={tb:.3f} too large for closed-form model (max ~0.25)")


def _validate_coplanar(w: float, gap: float, epsilon_r: float):
    """Validate coplanar waveguide parameters."""
    if w <= 0 or gap <= 0:
        raise ValueError(f"w and gap must be > 0, got w={w}, gap={gap}")
    if epsilon_r <= 0:
        raise ValueError(f"epsilon_r must be > 0, got {epsilon_r}")


def get_Microstrip_eps_eff(w: float, h: float, epsilon_r: float):
    """Calculate effective permittivity of microstrip (quasi-static, t=0).

    Args:
        w: trace width in m
        h: height above ground plane in m
        epsilon_r: relative permittivity of substrate
    Returns:
        effective permittivity (dimensionless)
    Reference:
        Hammerstad & Jensen (1980), IEEE MTT-S Digest, pp. 407-409
    """
    _validate_microstrip(w, h, epsilon_r)

    wh = w / h
    if wh <= 1:
        # Note: coefficient is 0.04, not 0.4 (Hammerstad 1975)
        eps_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (
            1 / sqrt(1 + 12 / wh) + 0.04 * (1 - wh) * (1 - wh)
        )
    else:
        eps_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / (2 * sqrt(1 + 12 / wh))
    return eps_eff


def get_Microstrip_Z0(w: float, h: float, epsilon_r: float):
    """Calculate characteristic impedance of microstrip (quasi-static, t=0).

    Args:
        w: trace width in m
        h: height above ground plane in m
        epsilon_r: relative permittivity of substrate
    Returns:
        characteristic impedance in Ohm
    Reference:
        Hammerstad & Jensen (1980), IEEE MTT-S Digest, pp. 407-409
    """
    _validate_microstrip(w, h, epsilon_r)

    wh = w / h
    eps_eff = get_Microstrip_eps_eff(w, h, epsilon_r)

    if wh < 1:
        Z0 = 60 / sqrt(eps_eff) * log(8 / wh + 0.25 * wh)
    else:
        Z0 = 120 * pi / (sqrt(eps_eff) * (wh + 1.393 + 2 / 3 * log(wh + 1.444)))
    return Z0


def get_Microstrip_Cap(w: float, h: float, length: float, epsilon_r: float):
    """Calculate capacitance of microstrip trace.

    Derived consistently from Z0 and phase velocity: C = length / (Z0 * v_p)

    Args:
        w: trace width in m
        h: height above ground plane in m
        length: trace length in m
        epsilon_r: relative permittivity of substrate
    Returns:
        capacitance in F
    """
    eps_eff = get_Microstrip_eps_eff(w, h, epsilon_r)
    Z0 = get_Microstrip_Z0(w, h, epsilon_r)
    v_p = v_0 / sqrt(eps_eff)
    return length / (Z0 * v_p)


def get_Microstrip_Ind(w: float, h: float, length: float, epsilon_r: float):
    """Calculate inductance of microstrip trace.

    Derived consistently from Z0 and phase velocity: L = Z0 * length / v_p

    Args:
        w: trace width in m
        h: height above ground plane in m
        length: trace length in m
        epsilon_r: relative permittivity of substrate
    Returns:
        inductance in H
    Note:
        This is the partial self-inductance of a single trace segment.
        Mutual inductance between segments is not considered.
    """
    eps_eff = get_Microstrip_eps_eff(w, h, epsilon_r)
    Z0 = get_Microstrip_Z0(w, h, epsilon_r)
    v_p = v_0 / sqrt(eps_eff)
    return Z0 * length / v_p


def get_Stripline_Z0(w: float, h: float, t: float, epsilon_r: float):
    """Calculate characteristic impedance of stripline (symmetric).

    Args:
        w: trace width in m
        h: total height between ground planes (not h/2!)
        t: trace thickness in m
        epsilon_r: relative permittivity of substrate
    Returns:
        characteristic impedance in Ohm
    Reference:
        Steer, "Microwave and RF Design II", Section 3.7
    """
    _validate_stripline(w, h, t, epsilon_r)

    b = h / 2  # distance from trace center to ground
    wb = w / b

    if t > 0:
        # Finite thickness case
        tb = t / b

        # Effective width
        if wb < 0.35:
            w_eff_b = wb - (0.35 - wb) ** 2 / (1 + 12 * tb)
        else:
            w_eff_b = wb

        if w_eff_b <= 0:
            raise ValueError(
                f"Effective width non-positive (w/b={wb:.3f}, t/b={tb:.3f}); "
                "geometry outside model range"
            )

        # Fringing capacitance coefficient
        # C_f = (2/pi)*log(1/(1-tb)+1) - (tb/pi)*log(1/(1-tb)^2 - 1)
        if tb < 0.01:
            # Series expansion for small tb (avoids log(small) instability)
            C_f = (2 / pi) * log(2) + tb * (2 / pi + 1 / pi * log(4 / tb))
        else:
            inv_1_minus_tb = 1 / (1 - tb)
            term1 = (2 / pi) * log1p(inv_1_minus_tb)
            term2 = (tb / pi) * log(inv_1_minus_tb * inv_1_minus_tb - 1)
            C_f = term1 - term2

        Z0 = (30 * pi / sqrt(epsilon_r)) * (1 - tb) / (w_eff_b + C_f)
    else:
        # Zero thickness case
        if wb < 0.35:
            w_eff_b = wb - (0.35 - wb) ** 2
        else:
            w_eff_b = wb

        if w_eff_b <= 0:
            raise ValueError(
                f"Effective width non-positive (w/b={wb:.3f}); "
                "geometry outside model range"
            )

        Z0 = 94.25 / (sqrt(epsilon_r) * (w_eff_b + 0.441))

    return Z0


def get_Stripline_Cap(w: float, h: float, length: float, t: float, epsilon_r: float):
    """Calculate capacitance of stripline trace.

    Derived consistently from Z0 and phase velocity: C = length / (Z0 * v_p)

    Args:
        w: trace width in m
        h: total height between ground planes in m
        length: trace length in m
        t: trace thickness in m
        epsilon_r: relative permittivity of substrate
    Returns:
        capacitance in F
    """
    # For stripline, eps_eff = epsilon_r (fully embedded in dielectric)
    Z0 = get_Stripline_Z0(w, h, t, epsilon_r)
    v_p = v_0 / sqrt(epsilon_r)
    return length / (Z0 * v_p)


def get_Stripline_Ind(w: float, h: float, length: float, t: float, epsilon_r: float):
    """Calculate inductance of stripline trace.

    Derived consistently from Z0 and phase velocity: L = Z0 * length / v_p

    Args:
        w: trace width in m
        h: total height between ground planes in m
        length: trace length in m
        t: trace thickness in m
        epsilon_r: relative permittivity of substrate
    Returns:
        inductance in H
    Note:
        This is the partial self-inductance of a single trace segment.
        Mutual inductance between segments is not considered.
    """
    Z0 = get_Stripline_Z0(w, h, t, epsilon_r)
    v_p = v_0 / sqrt(epsilon_r)
    return Z0 * length / v_p


def get_Plate_Cap(w: float, h: float, length: float, epsilon_r: float):
    """Calculate parallel plate capacitance.

    Args:
        w: plate width in m
        h: plate separation in m
        length: plate length in m
        epsilon_r: relative permittivity of dielectric
    Returns:
        capacitance in F
    """
    if w <= 0 or h <= 0 or length <= 0:
        raise ValueError("w, h, length must be > 0")
    if epsilon_r <= 0:
        raise ValueError("epsilon_r must be > 0")
    return epsilon_0 * epsilon_r * w * length / h


def _elliptic_K_ratio(k: float):
    """Approximate ratio K(k)/K'(k) of complete elliptic integrals.

    Args:
        k: modulus (0 < k < 1)
    Returns:
        K(k)/K'(k)
    Reference:
        Qucs technical documentation, Equations 12.4-12.5
        https://qucs.sourceforge.net/tech/node86.html
    """
    if k <= 0 or k >= 1:
        raise ValueError(f"k must be in (0,1), got {k}")

    # Complementary modulus
    k_prime = sqrt(1 - k * k)

    if k <= 1 / sqrt(2):
        # For small k, use formula with k'
        ratio = (1 / pi) * log(2 * (1 + sqrt(k_prime)) / (1 - sqrt(k_prime)))
        return 1 / ratio
    else:
        # For large k, use formula with k
        return (1 / pi) * log(2 * (1 + sqrt(k)) / (1 - sqrt(k)))


def get_Coplanar_Cap(w: float, gap: float, length: float, epsilon_r: float):
    """Calculate capacitance of CPW (infinite substrate, t=0).

    Args:
        w: center conductor width in m
        gap: gap between center and ground in m
        length: trace length in m
        epsilon_r: relative permittivity of substrate
    Returns:
        capacitance in F
    Reference:
        Qucs technical documentation, Section 12.2
    """
    _validate_coplanar(w, gap, epsilon_r)

    # k = w / (w + 2*gap) is the modulus for elliptic integrals
    k = w / (w + 2 * gap)

    # C = C_dielectric + C_air = 2*eps0*eps_r*K/K' + 2*eps0*K/K'
    # For infinite substrate: eps_eff = (eps_r + 1) / 2
    K_ratio = _elliptic_K_ratio(k)
    C_per_length = 2 * epsilon_0 * (epsilon_r + 1) * K_ratio
    return C_per_length * length


def get_Coplanar_Z0(w: float, gap: float, epsilon_r: float):
    """Calculate characteristic impedance of CPW (infinite substrate, t=0).

    Args:
        w: center conductor width in m
        gap: gap between center and ground in m
        epsilon_r: relative permittivity of substrate
    Returns:
        characteristic impedance in Ohm
    Reference:
        Qucs technical documentation, Section 12.2
    """
    _validate_coplanar(w, gap, epsilon_r)

    k = w / (w + 2 * gap)
    eps_eff = (epsilon_r + 1) / 2

    K_ratio = _elliptic_K_ratio(k)
    return 30 * pi / (sqrt(eps_eff) * K_ratio)


def get_Coplanar_Ind(w: float, gap: float, length: float, epsilon_r: float):
    """Calculate inductance of coplanar waveguide (infinite substrate).

    Derived consistently from Z0 and phase velocity: L = Z0 * length / v_p

    Args:
        w: center conductor width in m
        gap: gap between center and ground in m
        length: trace length in m
        epsilon_r: relative permittivity of substrate
    Returns:
        inductance in H
    """
    eps_eff = (epsilon_r + 1) / 2
    Z0 = get_Coplanar_Z0(w, gap, epsilon_r)
    v_p = v_0 / sqrt(eps_eff)
    return Z0 * length / v_p


if __name__ == "__main__":
    w = 0.4e-3  # Width in m
    h = 1.55e-3  # Height above ground in m
    length = 1.0  # Length in m (1m for per-meter values)
    epsilon_r = 4.6  # Relative permittivity (FR4)
    t = 35e-6  # Trace thickness in m (35um = 1oz copper)

    gap = 0.2e-3  # Gap in m (for Coplanar)

    print("=== Microstrip ===")
    eps_eff = get_Microstrip_eps_eff(w, h, epsilon_r)
    Z0 = get_Microstrip_Z0(w, h, epsilon_r)
    print(f"  eps_eff: {eps_eff:.3f}")
    print(f"  Z0:      {Z0:.2f} Ohm")
    print(f"  C:       {get_Microstrip_Cap(w, h, length, epsilon_r) * 1e12:.2f} pF/m")
    print(f"  L:       {get_Microstrip_Ind(w, h, length, epsilon_r) * 1e9:.2f} nH/m")

    print("=== Stripline ===")
    Z0_strip = get_Stripline_Z0(w, h * 2, t, epsilon_r)
    print(f"  Z0:      {Z0_strip:.2f} Ohm")
    print(f"  C:       {get_Stripline_Cap(w, h * 2, length, t, epsilon_r) * 1e12:.2f} pF/m")
    print(f"  L:       {get_Stripline_Ind(w, h * 2, length, t, epsilon_r) * 1e9:.2f} nH/m")

    print("=== Coplanar (infinite substrate) ===")
    Z0_cpw = get_Coplanar_Z0(w, gap, epsilon_r)
    print(f"  Z0:      {Z0_cpw:.2f} Ohm")
    print(f"  C:       {get_Coplanar_Cap(w, gap, length, epsilon_r) * 1e12:.2f} pF/m")
    print(f"  L:       {get_Coplanar_Ind(w, gap, length, epsilon_r) * 1e9:.2f} nH/m")

    print("=== Plate Capacitor ===")
    print(f"  C:       {get_Plate_Cap(w, h, length, epsilon_r) * 1e12:.2f} pF/m")
