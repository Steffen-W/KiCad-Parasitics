from math import atan, cosh, exp, log, log1p, pi, sinh, sqrt, tanh

# Physical constants (SI units)
v_0 = 299792458  # m/s speed of light (exact by SI definition)
epsilon_0 = 8.8541878128e-12  # F/m permittivity of free space (CODATA 2018)
mu_0 = 1 / (epsilon_0 * v_0 * v_0)  # H/m permeability (derived)
Z_F0 = 376.730313668  # Ohm, wave impedance of free space
RHO_CU = 1.68e-8  # Ohm*m, resistivity of copper at 20°C


# ==============================================================================
# Helper functions
# ==============================================================================


def _elliptic_K(k: float) -> float:
    """Complete elliptic integral of the first kind K(k).

    Uses arithmetic-geometric mean (AGM) algorithm.
    Args:
        k: modulus (0 <= k < 1)
    Returns:
        K(k)
    """
    if k < 0 or k >= 1:
        if k == 1:
            return float("inf")
        raise ValueError(f"k must be in [0,1), got {k}")
    if k == 0:
        return pi / 2

    a, b = 1.0, sqrt(1 - k * k)
    while abs(a - b) > 1e-15:
        a, b = (a + b) / 2, sqrt(a * b)
    return pi / (2 * a)


def _skin_depth(frequency: float, sigma: float, mu_r: float = 1.0) -> float:
    """Calculate skin depth.

    Args:
        frequency: frequency in Hz
        sigma: conductivity in S/m
        mu_r: relative permeability of conductor (default 1.0)
    Returns:
        skin depth in m
    """
    if frequency <= 0 or sigma <= 0:
        return float("inf")
    return 1.0 / sqrt(pi * frequency * mu_r * mu_0 * sigma)


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
            raise ValueError(
                f"t/b={tb:.3f} too large for closed-form model (max ~0.25)"
            )


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


# ==============================================================================
# Microstrip - Frequency-dependent analysis (KiCad PCB Calculator)
# ==============================================================================


def _microstrip_filling_factor(u: float, epsilon_r: float) -> float:
    """Filling factor for microstrip (no cover, t=0).

    Args:
        u: normalized width w/h
        epsilon_r: relative permittivity
    Returns:
        filling factor q
    Reference:
        Hammerstad & Jensen (1980)
    """
    u2, u3, u4 = u * u, u * u * u, u * u * u * u
    a = 1 + log((u4 + u2 / 2704) / (u4 + 0.432)) / 49 + log(1 + u3 / 5929.741) / 18.7
    b = 0.564 * pow((epsilon_r - 0.9) / (epsilon_r + 3), 0.053)
    return pow(1 + 10 / u, -a * b)


def _microstrip_Z0_homogeneous(u: float) -> float:
    """Impedance for homogeneous stripline (no cover).

    Args:
        u: normalized width w/h
    Returns:
        Z0 in Ohm
    """
    freq = 6 + (2 * pi - 6) * exp(-pow(30.666 / u, 0.7528))
    return (Z_F0 / (2 * pi)) * log(freq / u + sqrt(1 + 4 / (u * u)))


def _microstrip_delta_u_thickness(u: float, t_h: float, epsilon_r: float) -> float:
    """Thickness correction for normalized width.

    Args:
        u: normalized width w/h
        t_h: normalized thickness t/h
        epsilon_r: relative permittivity
    Returns:
        delta_u correction
    """
    if t_h <= 0:
        return 0.0
    delta_u = (t_h / pi) * log(1 + 4 * exp(1) * pow(tanh(sqrt(6.517 * u)), 2) / t_h)
    return 0.5 * delta_u * (1 + 1 / cosh(sqrt(epsilon_r - 1)))


def _microstrip_delta_q_cover(h2h: float) -> float:
    """Cover effect on filling factor.

    Args:
        h2h: ratio h_cover/h
    Returns:
        correction factor
    """
    return tanh(1.043 + 0.121 * h2h - 1.164 / h2h)


def _microstrip_delta_q_thickness(u: float, t_h: float) -> float:
    """Thickness effect on filling factor.

    Args:
        u: normalized width w/h
        t_h: normalized thickness t/h
    Returns:
        correction factor
    """
    return (2 * log(2) / pi) * (t_h / sqrt(u))


def _microstrip_er_effective(epsilon_r: float, q: float) -> float:
    """Effective dielectric constant from filling factor.

    Args:
        epsilon_r: relative permittivity
        q: filling factor
    Returns:
        effective permittivity
    """
    return 0.5 * (epsilon_r + 1) + 0.5 * q * (epsilon_r - 1)


def _microstrip_er_dispersion(u: float, epsilon_r: float, f_n: float) -> float:
    """Dispersion correction for effective permittivity.

    Args:
        u: normalized width w/h
        epsilon_r: relative permittivity
        f_n: normalized frequency [GHz * mm]
    Returns:
        dispersion factor P
    Reference:
        Kirschning & Jansen, "Accurate Model for Effective Dielectric Constant
        of Microstrip with Validity up to Millimetre-Wave Frequencies",
        Electronics Letters, Vol. 18, No. 6, 1982.
        (QUCS technical docs, eq. 11.17–11.21)
    """
    P1 = (
        0.27488
        + u * (0.6315 + 0.525 / pow(1 + 0.0157 * f_n, 20))
        - 0.065683 * exp(-8.7513 * u)
    )
    P2 = 0.33622 * (1 - exp(-0.03442 * epsilon_r))
    P3 = 0.0363 * exp(-4.6 * u) * (1 - exp(-pow(f_n / 38.7, 4.97)))
    P4 = 1 + 2.751 * (1 - exp(-pow(epsilon_r / 15.916, 8)))
    return P1 * P2 * pow((P3 * P4 + 0.1844) * f_n, 1.5763)


def _microstrip_Z0_dispersion(
    u: float, epsilon_r: float, er_eff_0: float, er_eff_f: float, f_n: float
) -> float:
    """Dispersion correction factor for Z0.

    Args:
        u: normalized width w/h
        epsilon_r: relative permittivity
        er_eff_0: static effective permittivity
        er_eff_f: frequency-dependent effective permittivity
        f_n: normalized frequency [GHz * mm]
    Returns:
        dispersion factor D, so that Z0(f) = Z0(0) * (R13/R14)^R17
    Reference:
        Jansen & Kirschning, "Arguments and an Accurate Model for the
        Power-Current Formulation of Microstrip Characteristic Impedance",
        AEÜ, Vol. 37, No. 3/4, 1983.
        (QUCS technical docs, eq. 11.33–11.50)
    """
    # R1–R17: intermediate terms per QUCS eq. (11.33)–(11.49)
    R1 = 0.03891 * pow(epsilon_r, 1.4)
    R2 = 0.267 * pow(u, 7.0)
    R3 = 4.766 * exp(-3.228 * pow(u, 0.641))
    R4 = 0.016 + pow(0.0514 * epsilon_r, 4.524)
    R5 = pow(f_n / 28.843, 12.0)
    R6 = 22.2 * pow(u, 1.92)
    R7 = 1.206 - 0.3144 * exp(-R1) * (1 - exp(-R2))
    R8 = 1 + 1.275 * (
        1 - exp(-0.004625 * R3 * pow(epsilon_r, 1.674) * pow(f_n / 18.365, 2.745))
    )
    tmpf = pow(epsilon_r - 1, 6)
    R9 = (
        5.086
        * R4
        * (R5 / (0.3838 + 0.386 * R4))
        * (exp(-R6) / (1 + 1.2992 * R5))
        * (tmpf / (1 + 10 * tmpf))
    )
    R10 = 0.00044 * pow(epsilon_r, 2.136) + 0.0184
    tmpf = pow(f_n / 19.47, 6)
    R11 = tmpf / (1 + 0.0962 * tmpf)
    R12 = 1 / (1 + 0.00245 * u * u)
    R13 = 0.9408 * pow(er_eff_f, R8) - 0.9603
    R14 = (0.9408 - R9) * pow(er_eff_0, R8) - 0.9603
    R15 = 0.707 * R10 * pow(f_n / 12.3, 1.097)
    R16 = 1 + 0.0503 * epsilon_r * epsilon_r * R11 * (1 - exp(-pow(u / 15, 6)))
    R17 = R7 * (1 - 1.1241 * (R12 / R16) * exp(-0.026 * pow(f_n, 1.15656) - R15))
    return pow(R13 / R14, R17)


def analyze_microstrip(
    w: float,
    h: float,
    t: float,
    epsilon_r: float,
    frequency: float,
    length: float,
    tan_d: float = 0.0,
    sigma: float = 5.8e7,
    roughness: float = 0.0,
    h_t: float | None = None,
    mu_r: float = 1.0,
    mu_rc: float = 1.0,
) -> dict:
    """Frequency-dependent analysis of microstrip transmission line.

    Calculates Z0, effective permittivity, losses, and electrical length
    using the KiCad PCB Calculator algorithms (Hammerstad & Jensen, Kirschning & Jansen).

    Args:
        w: trace width in m
        h: substrate height in m
        t: trace thickness in m
        epsilon_r: relative permittivity of substrate
        frequency: operating frequency in Hz
        length: trace length in m
        tan_d: loss tangent of substrate (default 0)
        sigma: conductor conductivity in S/m (default 5.8e7 for copper)
        roughness: surface roughness RMS in m (default 0)
        h_t: height to cover/top in m (default: None = infinite)
        mu_r: relative permeability of substrate (default 1.0)
        mu_rc: relative permeability of conductor (default 1.0)
    Returns:
        dict with keys:
            z0: characteristic impedance in Ohm
            epsilon_eff: effective permittivity (dimensionless)
            skin_depth: skin depth in m
            loss_conductor: conductor loss in dB
            loss_dielectric: dielectric loss in dB
            angle_electrical: electrical length in radians
            delay: propagation delay in s
            capacitance: capacitance in F
            inductance: inductance in H
    Reference:
        KiCad PCB Calculator, based on:
        - Hammerstad & Jensen (1980)
        - Kirschning & Jansen (1982)
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"w and h must be > 0, got w={w}, h={h}")
    if epsilon_r <= 1:
        raise ValueError(f"epsilon_r must be > 1, got {epsilon_r}")
    if frequency <= 0:
        raise ValueError(f"frequency must be > 0, got {frequency}")
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    # Default cover height to very large value (no cover effect)
    if h_t is None:
        h_t = 1e6 * h

    u = w / h
    t_h = t / h
    h2h = h_t / h

    # Effective permeability
    mu_eff = (2 * mu_r) / ((1 + mu_r) + (1 - mu_r) * pow(1 + 10 * h / w, -0.5))

    # Width correction for thickness (epsilon_r = 1)
    delta_u_1 = _microstrip_delta_u_thickness(u, t_h, 1.0)
    Z0_h_1 = _microstrip_Z0_homogeneous(u + delta_u_1)

    # Width correction for thickness (actual epsilon_r)
    delta_u_r = _microstrip_delta_u_thickness(u, t_h, epsilon_r)
    u_eff = u + delta_u_r
    Z0_h_r = _microstrip_Z0_homogeneous(u_eff)

    # Filling factor with corrections
    q_inf = _microstrip_filling_factor(u_eff, epsilon_r)
    q_c = _microstrip_delta_q_cover(h2h)
    q_t = _microstrip_delta_q_thickness(u_eff, t_h)
    q = (q_inf - q_t) * q_c

    # Effective permittivity (static)
    er_eff_t = _microstrip_er_effective(epsilon_r, q)
    er_eff_0 = er_eff_t * pow(Z0_h_1 / Z0_h_r, 2)

    # Static impedance
    Z0_0 = Z0_h_r / sqrt(er_eff_t)

    # Dispersion (frequency dependence)
    f_n = frequency * h / 1e6  # normalized frequency [GHz * mm]
    P = _microstrip_er_dispersion(u, epsilon_r, f_n)
    er_eff_f = epsilon_r - (epsilon_r - er_eff_0) / (1 + P)

    D = _microstrip_Z0_dispersion(u, epsilon_r, er_eff_0, er_eff_f, f_n)
    Z0_f = Z0_0 * D

    # Skin depth
    delta = _skin_depth(frequency, sigma, mu_rc)

    # Conductor losses
    alpha_c = 0.0
    if frequency > 0 and delta > 0:
        K = exp(-1.2 * pow(Z0_h_1 / Z_F0, 0.7))  # current distribution factor
        R_s = 1 / (sigma * delta)  # skin resistance
        # Surface roughness correction
        R_s *= 1 + (2 / pi) * atan(1.40 * pow(roughness / delta, 2))
        # Strip inductive quality factor
        w_eff = u_eff * h
        Q_c = (pi * Z0_h_1 * w_eff * frequency) / (R_s * v_0 * K)
        alpha_c = (20 * pi / log(10)) * frequency * sqrt(er_eff_0) / (v_0 * Q_c)

    # Dielectric losses
    alpha_d = 0.0
    if tan_d > 0:
        alpha_d = (
            (20 * pi / log(10))
            * (frequency / v_0)
            * (epsilon_r / sqrt(er_eff_0))
            * ((er_eff_0 - 1) / (epsilon_r - 1))
            * tan_d
        )

    # Electrical length
    v_p = v_0 / sqrt(er_eff_f * mu_eff)
    lambda_g = v_p / frequency
    angle_l = 2 * pi * length / lambda_g

    # Propagation delay
    delay = length / v_p

    # Capacitance and inductance from Z0 and velocity
    capacitance = length / (Z0_f * v_p)
    inductance = Z0_f * length / v_p

    return {
        "z0": Z0_f,
        "epsilon_eff": er_eff_f,
        "skin_depth": delta,
        "loss_conductor": alpha_c * length,
        "loss_dielectric": alpha_d * length,
        "angle_electrical": angle_l,
        "delay": delay,
        "capacitance": capacitance,
        "inductance": inductance,
    }


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


# ==============================================================================
# Stripline - Frequency-dependent analysis (KiCad PCB Calculator)
# ==============================================================================


def _stripline_line_impedance(
    w: float, h_eff: float, t: float, epsilon_r: float, frequency: float, sigma: float
) -> tuple[float, float]:
    """Calculate stripline impedance and conductor loss factor.

    Args:
        w: trace width in m
        h_eff: effective height (distance from trace to ground) in m
        t: trace thickness in m
        epsilon_r: relative permittivity
        frequency: frequency in Hz
        sigma: conductivity in S/m
    Returns:
        tuple (Z0 in Ohm, alpha_c factor)
    Reference:
        KiCad PCB Calculator stripline implementation
    """
    hmt = h_eff - t
    ac = sqrt(frequency / sigma / 17.2) if sigma > 0 and frequency > 0 else 0.0

    if w / hmt >= 0.35:
        # Wide trace formula
        ZL = (
            w
            + (
                2 * h_eff * log((2 * h_eff - t) / hmt)
                - t * log(h_eff * h_eff / hmt / hmt - 1)
            )
            / pi
        )
        ZL = Z_F0 * hmt / sqrt(epsilon_r) / 4 / ZL

        ac *= 2.02e-6 * epsilon_r * ZL / hmt
        ac *= 1 + 2 * w / hmt + (h_eff + t) / hmt / pi * log(2 * h_eff / t - 1)
    else:
        # Narrow trace formula
        tdw = t / w
        if t / w > 1:
            tdw = w / t

        de = 1 + tdw / pi * (1 + log(4 * pi / tdw)) + 0.236 * pow(tdw, 1.65)

        if t / w > 1:
            de *= t / 2
        else:
            de *= w / 2

        ZL = Z_F0 / 2 / pi / sqrt(epsilon_r) * log(4 * h_eff / pi / de)

        ac *= 0.01141 / ZL / de
        ac *= (
            de / h_eff
            + 0.5
            + tdw / 2 / pi
            + 0.5 / pi * log(4 * pi / tdw)
            + 0.1947 * pow(tdw, 0.65)
            - 0.0767 * pow(tdw, 1.65)
        )

    return ZL, ac


def analyze_stripline(
    w: float,
    h: float,
    t: float,
    a: float,
    epsilon_r: float,
    frequency: float,
    length: float,
    tan_d: float = 0.0,
    sigma: float = 5.8e7,
    mu_rc: float = 1.0,
) -> dict:
    """Frequency-dependent analysis of asymmetric stripline transmission line.

    The trace is positioned at distance 'a' from the bottom ground plane.
    For symmetric stripline, set a = (h - t) / 2.

    Args:
        w: trace width in m
        h: total height between ground planes in m
        t: trace thickness in m
        a: distance from trace bottom to bottom ground plane in m
        epsilon_r: relative permittivity of substrate
        frequency: operating frequency in Hz
        length: trace length in m
        tan_d: loss tangent of substrate (default 0)
        sigma: conductor conductivity in S/m (default 5.8e7 for copper)
        mu_rc: relative permeability of conductor (default 1.0)
    Returns:
        dict with keys:
            z0: characteristic impedance in Ohm
            epsilon_eff: effective permittivity (= epsilon_r for stripline)
            skin_depth: skin depth in m
            loss_conductor: conductor loss in dB
            loss_dielectric: dielectric loss in dB
            angle_electrical: electrical length in radians
            delay: propagation delay in s
            capacitance: capacitance in F
            inductance: inductance in H
    Reference:
        KiCad PCB Calculator stripline implementation
    Note:
        For stripline, epsilon_eff = epsilon_r (no dispersion, fully embedded)
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"w and h must be > 0, got w={w}, h={h}")
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    if a < 0:
        raise ValueError(f"a must be >= 0, got {a}")
    if a + t >= h:
        raise ValueError(f"a + t must be < h, got a+t={a + t}, h={h}")
    if epsilon_r <= 1:
        raise ValueError(f"epsilon_r must be > 1, got {epsilon_r}")
    if frequency <= 0:
        raise ValueError(f"frequency must be > 0, got {frequency}")
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    # Skin depth
    delta = _skin_depth(frequency, sigma, mu_rc)

    # Effective permittivity (no dispersion for stripline)
    epsilon_eff = epsilon_r

    # Impedance as parallel combination of upper and lower halves
    h1 = 2 * a + t  # effective height to bottom
    h2 = 2 * (h - a) - t  # effective height to top

    Z1, ac1 = _stripline_line_impedance(w, h1, t, epsilon_r, frequency, sigma)
    Z2, ac2 = _stripline_line_impedance(w, h2, t, epsilon_r, frequency, sigma)

    Z0 = 2 / (1 / Z1 + 1 / Z2)

    # Conductor loss in dB
    loss_conductor = length * (ac1 + ac2)

    # Dielectric loss in dB
    LOG2DB = 20 / log(10)
    loss_dielectric = LOG2DB * length * (pi / v_0) * frequency * sqrt(epsilon_r) * tan_d

    # Electrical length in radians
    angle_l = 2 * pi * length * sqrt(epsilon_r) * frequency / v_0

    # Propagation delay
    v_p = v_0 / sqrt(epsilon_r)
    delay = length / v_p

    # Capacitance and inductance
    capacitance = length / (Z0 * v_p)
    inductance = Z0 * length / v_p

    return {
        "z0": Z0,
        "epsilon_eff": epsilon_eff,
        "skin_depth": delta,
        "loss_conductor": loss_conductor,
        "loss_dielectric": loss_dielectric,
        "angle_electrical": angle_l,
        "delay": delay,
        "capacitance": capacitance,
        "inductance": inductance,
    }


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


def get_Zone_Cap(
    area: float,
    h1: float,
    epsilon_r1: float,
    h2: float = 0.0,
    epsilon_r2: float = 0.0,
):
    """Parallel-plate capacitance of a copper zone to one or two GND planes.

    One GND plane  (h2 <= 0):  C = eps0 * eps_r1 * A / h1
    Two GND planes (h2 > 0):   C = eps0 * A * (eps_r1/h1 + eps_r2/h2)

    Args:
        area: overlap area in m^2  (width * length)
        h1: dielectric thickness to first GND plane in m
        epsilon_r1: relative permittivity of first dielectric
        h2: dielectric thickness to second GND plane in m (0 = not present)
        epsilon_r2: relative permittivity of second dielectric (ignored if h2 <= 0)
    Returns:
        capacitance in F
    """
    if area <= 0 or h1 <= 0 or epsilon_r1 <= 0:
        raise ValueError("area, h1, epsilon_r1 must be > 0")
    c = epsilon_0 * area * epsilon_r1 / h1
    if h2 > 0 and epsilon_r2 > 0:
        c += epsilon_0 * area * epsilon_r2 / h2
    return c


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


# ==============================================================================
# Coplanar Waveguide - Frequency-dependent analysis (KiCad PCB Calculator)
# ==============================================================================


def analyze_coplanar(
    w: float,
    s: float,
    h: float,
    t: float,
    epsilon_r: float,
    frequency: float,
    length: float,
    tan_d: float = 0.0,
    sigma: float = 5.8e7,
    mu_rc: float = 1.0,
    with_ground: bool = False,
) -> dict:
    """Frequency-dependent analysis of coplanar waveguide (CPW).

    Supports both standard CPW (with_ground=False) and grounded CPW (with_ground=True).

    Args:
        w: center conductor width in m
        s: gap between center and ground conductors in m
        h: substrate height in m
        t: conductor thickness in m
        epsilon_r: relative permittivity of substrate
        frequency: operating frequency in Hz
        length: trace length in m
        tan_d: loss tangent of substrate (default 0)
        sigma: conductor conductivity in S/m (default 5.8e7 for copper)
        mu_rc: relative permeability of conductor (default 1.0)
        with_ground: True for grounded CPW (metal backside), False for standard CPW
    Returns:
        dict with keys:
            z0: characteristic impedance in Ohm
            epsilon_eff: effective permittivity (dimensionless)
            skin_depth: skin depth in m
            loss_conductor: conductor loss in dB
            loss_dielectric: dielectric loss in dB
            angle_electrical: electrical length in radians
            delay: propagation delay in s
            capacitance: capacitance in F
            inductance: inductance in H
    Reference:
        KiCad PCB Calculator coplanar implementation
    """
    if w <= 0 or s <= 0 or h <= 0:
        raise ValueError(f"w, s, h must be > 0, got w={w}, s={s}, h={h}")
    if epsilon_r <= 1:
        raise ValueError(f"epsilon_r must be > 1, got {epsilon_r}")
    if frequency <= 0:
        raise ValueError(f"frequency must be > 0, got {frequency}")
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    # Skin depth
    delta = _skin_depth(frequency, sigma, mu_rc)

    # Quasi-static approximation using elliptic integrals
    # k1 = w / (w + 2*s)
    k1 = w / (w + 2 * s)
    kk1 = _elliptic_K(k1)
    kpk1 = _elliptic_K(sqrt(1 - k1 * k1))
    q1 = kk1 / kpk1

    # Calculate q3 for grounded CPW (needed for thickness correction)
    k3 = tanh((pi / 4) * (w / h)) / tanh((pi / 4) * (w + 2 * s) / h)
    q3 = _elliptic_K(k3) / _elliptic_K(sqrt(1 - k3 * k3))

    if with_ground:
        # Grounded CPW: backside is metal
        qz = 1 / (q1 + q3)
        er0 = 1 + q3 * qz * (epsilon_r - 1)
        zl_factor = Z_F0 / 2 * qz
    else:
        # Standard CPW: backside is air
        k2 = sinh((pi / 4) * (w / h)) / sinh((pi / 4) * (w + 2 * s) / h)
        q2 = _elliptic_K(k2) / _elliptic_K(sqrt(1 - k2 * k2))
        er0 = 1 + (epsilon_r - 1) / 2 * q2 / q1
        zl_factor = Z_F0 / 4 / q1

    # Thickness correction
    if t > 0:
        d = (t * 1.25 / pi) * (1 + log(4 * pi * w / t))
        se = s - d
        We = w + d
        ke = We / (We + 2 * se)
        qe = _elliptic_K(ke) / _elliptic_K(sqrt(1 - ke * ke))

        if with_ground:
            qz = 1 / (qe + q3)
            er0 = 1 + q3 * qz * (epsilon_r - 1)
            zl_factor = Z_F0 / 2 * qz
        else:
            zl_factor = Z_F0 / 4 / qe

        # Thickness correction to er0
        er0 = er0 - (0.7 * (er0 - 1) * t / s) / (q1 + 0.7 * t / s)

    sr_er = sqrt(epsilon_r)
    sr_er0 = sqrt(er0)

    # Cutoff frequency of TE0 mode
    fte = (v_0 / 4) / (h * sqrt(epsilon_r - 1))

    # Dispersion factor G
    p = log(w / h)
    u = 0.54 - (0.64 - 0.015 * p) * p
    v = 0.43 - (0.86 - 0.54 * p) * p
    G = exp(u * log(w / s) + v)

    # Frequency-dependent effective permittivity
    sr_er_f = sr_er0 + (sr_er - sr_er0) / (1 + G * pow(frequency / fte, -1.8))
    epsilon_eff = sr_er_f * sr_er_f

    # Characteristic impedance
    Z0 = zl_factor / sr_er_f

    # Conductor loss
    ac = 0.0
    if t > 0 and sigma > 0 and frequency > 0:
        n = (1 - k1) * 8 * pi / (t * (1 + k1))
        a_half = w / 2
        b = a_half + s
        ac_geom = (pi + log(n * a_half)) / a_half + (pi + log(n * b)) / b
        ac_factor = ac_geom / (4 * Z_F0 * kk1 * kpk1 * (1 - k1 * k1))
        ac = (
            20
            / log(10)
            * length
            * ac_factor
            * sr_er0
            * sqrt(pi * mu_0 * frequency / sigma)
        )

    # Dielectric loss
    ad_factor = (epsilon_r / (epsilon_r - 1)) * tan_d * pi / v_0
    ad = (
        20
        / log(10)
        * length
        * ad_factor
        * frequency
        * (sr_er_f * sr_er_f - 1)
        / sr_er_f
    )

    # Electrical length in radians
    angle_l = 2 * pi * length * sr_er_f * frequency / v_0

    # Propagation delay
    v_p = v_0 / sr_er_f
    delay = length / v_p

    # Capacitance and inductance
    capacitance = length / (Z0 * v_p)
    inductance = Z0 * length / v_p

    return {
        "z0": Z0,
        "epsilon_eff": epsilon_eff,
        "skin_depth": delta,
        "loss_conductor": ac,
        "loss_dielectric": ad,
        "angle_electrical": angle_l,
        "delay": delay,
        "capacitance": capacitance,
        "inductance": inductance,
    }


def _validate_via(
    hole_diameter: float,
    plating_thickness: float,
    via_length: float,
):
    """Validate via parameters."""
    if hole_diameter <= 0:
        raise ValueError(f"hole_diameter must be > 0, got {hole_diameter}")
    if plating_thickness <= 0:
        raise ValueError(f"plating_thickness must be > 0, got {plating_thickness}")
    if via_length <= 0:
        raise ValueError(f"via_length must be > 0, got {via_length}")


def get_Via_Resistance(
    hole_diameter: float,
    plating_thickness: float,
    via_length: float,
    resistivity: float = 1.72e-8,
):
    """Calculate DC resistance of a plated via.

    Args:
        hole_diameter: finished hole diameter in m
        plating_thickness: copper plating thickness in m
        via_length: via barrel length in m
        resistivity: plating resistivity in Ohm*m (default: copper 1.72e-8)
    Returns:
        resistance in Ohm
    Reference:
        KiCad PCB Calculator, CircuitCalculator.com [1], IPC-2221A [4]
    """
    _validate_via(hole_diameter, plating_thickness, via_length)
    if resistivity <= 0:
        raise ValueError(f"resistivity must be > 0, got {resistivity}")

    # Cross-sectional area of cylindrical copper shell
    # A = pi * (d_outer - d_inner) * t = pi * (d + t) * t
    area = pi * (hole_diameter + plating_thickness) * plating_thickness
    return resistivity * via_length / area


def get_Via_Capacitance(
    via_length: float,
    pad_diameter: float,
    clearance_diameter: float,
    epsilon_r: float = 4.5,
):
    """Calculate parasitic capacitance of a via to surrounding copper.

    Args:
        via_length: via barrel length in m
        pad_diameter: via pad diameter in m
        clearance_diameter: antipad (clearance) diameter in m
        epsilon_r: relative permittivity of substrate (default: 4.5 for FR4)
    Returns:
        capacitance in F
    Reference:
        Johnson & Graham, "High Speed Digital Design", Equation 7.6
    Note:
        Returns 0 if clearance_diameter <= pad_diameter (via inside copper pour)
    """
    if via_length <= 0:
        raise ValueError(f"via_length must be > 0, got {via_length}")
    if pad_diameter <= 0:
        raise ValueError(f"pad_diameter must be > 0, got {pad_diameter}")
    if epsilon_r <= 0:
        raise ValueError(f"epsilon_r must be > 0, got {epsilon_r}")

    if clearance_diameter <= pad_diameter:
        # Via is inside copper pour, capacitance model not applicable
        return 0.0

    # Equation 7.6 from Johnson & Graham [7], as implemented in KiCad:
    # C [pF] = 55.51 * eps_r * h * D1 / (D2 - D1) with h, D1, D2 in meters
    capacitance_pf = 55.51 * epsilon_r * via_length * pad_diameter
    capacitance_pf /= clearance_diameter - pad_diameter
    return capacitance_pf * 1e-12  # pF to F


def get_Via_Inductance(
    hole_diameter: float,
    via_length: float,
):
    """Calculate parasitic inductance of a via.

    Args:
        hole_diameter: finished hole diameter in m
        via_length: via barrel length in m
    Returns:
        inductance in H
    Reference:
        Johnson & Graham, "High Speed Digital Design", Equation 7.9
    """
    _validate_via(hole_diameter, 1e-6, via_length)  # plating not used

    # Equation 7.9 from Johnson & Graham [7], as implemented in KiCad:
    # L [nH] = 200 * h * (ln(4h/d) + 1) with h, d in meters
    inductance_nh = 200 * via_length * (log(4 * via_length / hole_diameter) + 1)
    return inductance_nh * 1e-9  # nH to H


def get_Via_Parasitics(
    hole_diameter: float,
    plating_thickness: float,
    via_length: float,
    pad_diameter: float | None = None,
    clearance_diameter: float | None = None,
    resistivity: float = 1.72e-8,
    epsilon_r: float = 4.5,
):
    """Calculate all parasitic properties of a plated via.

    Args:
        hole_diameter: finished hole diameter in m
        plating_thickness: copper plating thickness in m
        via_length: via barrel length in m
        pad_diameter: via pad diameter in m (optional, for capacitance)
        clearance_diameter: antipad diameter in m (optional, for capacitance)
        resistivity: plating resistivity in Ohm*m (default: copper 1.72e-8)
        epsilon_r: relative permittivity of substrate (default: 4.5 for FR4)
    Returns:
        dict with keys:
            resistance: DC resistance in Ohm
            inductance: parasitic inductance in H
            capacitance: parasitic capacitance in F (or None if pad/clearance not given)
    Reference:
        KiCad PCB Calculator, Johnson & Graham "High Speed Digital Design"
    """
    resistance = get_Via_Resistance(
        hole_diameter, plating_thickness, via_length, resistivity
    )
    inductance = get_Via_Inductance(hole_diameter, via_length)

    capacitance = None
    if pad_diameter is not None and clearance_diameter is not None:
        capacitance = get_Via_Capacitance(
            via_length, pad_diameter, clearance_diameter, epsilon_r
        )

    return {
        "resistance": resistance,
        "inductance": inductance,
        "capacitance": capacitance,
    }


def ascii_microstrip() -> str:
    return "    ====     SIG\n░░░░░░░░░░░░ DIE\n------------ GND"


def ascii_stripline() -> str:
    return (
        "------------ GND\n"
        "░░░░░░░░░░░░ DIE\n"
        "    ====     SIG\n"
        "░░░░░░░░░░░░ DIE\n"
        "------------ GND"
    )


def ascii_coplanar() -> str:
    return "GND SIG GND\n=== === ===\n░░░░░░░░░░░ DIE"


def ascii_coplanar_grounded() -> str:
    return "GND SIG GND\n=== === ===\n░░░░░░░░░░░ DIE\n----------- GND"


def ascii_via() -> str:
    return "Top ---- o\n░░░░░░░░ | ░░░░░░░░\nBot ---- o"


if __name__ == "__main__":
    # Test parameters
    w = 0.2e-3  # Width in m (0.2mm)
    h = 0.2e-3  # Substrate height in m (0.2mm)
    t = 35e-6  # Trace thickness in m (35um = 1oz copper)
    epsilon_r = 4.5  # Relative permittivity (FR4)
    frequency = 1e9  # 1 GHz
    length = 0.01  # 10mm trace length

    gap = 0.15e-3  # Gap in m (for Coplanar)

    print("=" * 60)
    print("Quasi-static functions (simple, no frequency dependence)")
    print("=" * 60)

    print("\n=== Microstrip (quasi-static) ===")
    print(ascii_microstrip())
    eps_eff = get_Microstrip_eps_eff(w, h, epsilon_r)
    Z0 = get_Microstrip_Z0(w, h, epsilon_r)
    print(f"  eps_eff: {eps_eff:.3f}")
    print(f"  Z0:      {Z0:.2f} Ohm")
    print(f"  C:       {get_Microstrip_Cap(w, h, 1.0, epsilon_r) * 1e12:.2f} pF/m")
    print(f"  L:       {get_Microstrip_Ind(w, h, 1.0, epsilon_r) * 1e9:.2f} nH/m")

    print("\n=== Stripline (quasi-static, symmetric) ===")
    print(ascii_stripline())
    Z0_strip = get_Stripline_Z0(w, h * 2, t, epsilon_r)
    print(f"  Z0:      {Z0_strip:.2f} Ohm")
    print(
        f"  C:       {get_Stripline_Cap(w, h * 2, 1.0, t, epsilon_r) * 1e12:.2f} pF/m"
    )
    print(f"  L:       {get_Stripline_Ind(w, h * 2, 1.0, t, epsilon_r) * 1e9:.2f} nH/m")

    print("\n=== Coplanar (quasi-static, infinite substrate) ===")
    print(ascii_coplanar())
    Z0_cpw = get_Coplanar_Z0(w, gap, epsilon_r)
    print(f"  Z0:      {Z0_cpw:.2f} Ohm")
    print(f"  C:       {get_Coplanar_Cap(w, gap, 1.0, epsilon_r) * 1e12:.2f} pF/m")
    print(f"  L:       {get_Coplanar_Ind(w, gap, 1.0, epsilon_r) * 1e9:.2f} nH/m")

    print("\n" + "=" * 60)
    print(f"Frequency-dependent analysis @ {frequency / 1e9:.1f} GHz (KiCad-style)")
    print("=" * 60)

    print("\n=== Microstrip (frequency-dependent) ===")
    print(ascii_microstrip())
    ms = analyze_microstrip(
        w=w,
        h=h,
        t=t,
        epsilon_r=epsilon_r,
        frequency=frequency,
        length=length,
        tan_d=0.02,
    )
    print(f"  Z0:           {ms['z0']:.2f} Ohm")
    print(f"  eps_eff:      {ms['epsilon_eff']:.3f}")
    print(f"  skin_depth:   {ms['skin_depth'] * 1e6:.2f} um")
    print(f"  loss_cond:    {ms['loss_conductor']:.4f} dB")
    print(f"  loss_diel:    {ms['loss_dielectric']:.4f} dB")
    print(f"  elec_angle:   {ms['angle_electrical'] * 180 / pi:.1f} deg")
    print(f"  delay:        {ms['delay'] * 1e12:.2f} ps")
    print(f"  C:            {ms['capacitance'] * 1e12:.3f} pF")
    print(f"  L:            {ms['inductance'] * 1e9:.3f} nH")

    print("\n=== Stripline (frequency-dependent, symmetric) ===")
    print(ascii_stripline())
    a_sym = (h * 2 - t) / 2  # symmetric position
    sl = analyze_stripline(
        w=w,
        h=h * 2,
        t=t,
        a=a_sym,
        epsilon_r=epsilon_r,
        frequency=frequency,
        length=length,
        tan_d=0.02,
    )
    print(f"  Z0:           {sl['z0']:.2f} Ohm")
    print(f"  eps_eff:      {sl['epsilon_eff']:.3f}")
    print(f"  skin_depth:   {sl['skin_depth'] * 1e6:.2f} um")
    print(f"  loss_cond:    {sl['loss_conductor']:.4f} dB")
    print(f"  loss_diel:    {sl['loss_dielectric']:.4f} dB")
    print(f"  elec_angle:   {sl['angle_electrical'] * 180 / pi:.1f} deg")
    print(f"  delay:        {sl['delay'] * 1e12:.2f} ps")
    print(f"  C:            {sl['capacitance'] * 1e12:.3f} pF")
    print(f"  L:            {sl['inductance'] * 1e9:.3f} nH")

    print("\n=== Coplanar Waveguide (frequency-dependent) ===")
    print(ascii_coplanar())
    cpw = analyze_coplanar(
        w=w,
        s=gap,
        h=h,
        t=t,
        epsilon_r=epsilon_r,
        frequency=frequency,
        length=length,
        tan_d=0.02,
        with_ground=False,
    )
    print(f"  Z0:           {cpw['z0']:.2f} Ohm")
    print(f"  eps_eff:      {cpw['epsilon_eff']:.3f}")
    print(f"  skin_depth:   {cpw['skin_depth'] * 1e6:.2f} um")
    print(f"  loss_cond:    {cpw['loss_conductor']:.4f} dB")
    print(f"  loss_diel:    {cpw['loss_dielectric']:.4f} dB")
    print(f"  elec_angle:   {cpw['angle_electrical'] * 180 / pi:.1f} deg")
    print(f"  delay:        {cpw['delay'] * 1e12:.2f} ps")
    print(f"  C:            {cpw['capacitance'] * 1e12:.3f} pF")
    print(f"  L:            {cpw['inductance'] * 1e9:.3f} nH")

    print("\n=== Grounded Coplanar Waveguide (frequency-dependent) ===")
    print(ascii_coplanar_grounded())
    gcpw = analyze_coplanar(
        w=w,
        s=gap,
        h=h,
        t=t,
        epsilon_r=epsilon_r,
        frequency=frequency,
        length=length,
        tan_d=0.02,
        with_ground=True,
    )
    print(f"  Z0:           {gcpw['z0']:.2f} Ohm")
    print(f"  eps_eff:      {gcpw['epsilon_eff']:.3f}")
    print(f"  skin_depth:   {gcpw['skin_depth'] * 1e6:.2f} um")
    print(f"  loss_cond:    {gcpw['loss_conductor']:.4f} dB")
    print(f"  loss_diel:    {gcpw['loss_dielectric']:.4f} dB")
    print(f"  elec_angle:   {gcpw['angle_electrical'] * 180 / pi:.1f} deg")
    print(f"  delay:        {gcpw['delay'] * 1e12:.2f} ps")
    print(f"  C:            {gcpw['capacitance'] * 1e12:.3f} pF")
    print(f"  L:            {gcpw['inductance'] * 1e9:.3f} nH")

    print("\n" + "=" * 60)
    print("Via Parasitics")
    print("=" * 60)
    print(ascii_via())

    # KiCad default values for comparison
    hole_dia = 0.4e-3  # 0.4 mm finished hole diameter
    plating_t = 0.035e-3  # 35 um plating thickness
    via_len = 1.6e-3  # 1.6 mm via length (typical 4-layer board)
    pad_dia = 0.6e-3  # 0.6 mm pad diameter
    clearance_dia = 1.0e-3  # 1.0 mm antipad diameter

    via = get_Via_Parasitics(
        hole_dia, plating_t, via_len, pad_dia, clearance_dia, epsilon_r=4.5
    )
    print(f"  R:       {via['resistance'] * 1e3:.6f} mOhm")
    print(f"  L:       {via['inductance'] * 1e9:.5f} nH")
    print(f"  C:       {via['capacitance'] * 1e12:.6f} pF")
