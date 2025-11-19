import numpy as np

# -------------------------------------------------
# Improved surrogate aerodynamic model (no XFOIL)
# -------------------------------------------------
def fake_aero_score(m, p, t, alpha=4.0):
    """
    Smooth surrogate model for Cl, Cd at a given alpha (deg).
    m : max camber (0.0 - 0.06)
    p : camber position (0.1 - 0.7)
    t : thickness (0.08 - 0.18)
    alpha : angle of attack in degrees
    """

    # clip parameters to reasonable ranges
    m = float(np.clip(m, 0.0, 0.06))
    p = float(np.clip(p, 0.1, 0.7))
    t = float(np.clip(t, 0.08, 0.18))
    a = float(alpha)

    # reference "nice" airfoil
    m0, p0, t0 = 0.02, 0.4, 0.12

    # ---------------- LIFT MODEL ----------------
    camber_term = 18.0 * (m - m0)
    thickness_term = 4.0 * (t - t0)
    alpha_term = 0.06 * (a - 4.0)
    Cl = 0.6 + camber_term + thickness_term + alpha_term

    # ---------------- DRAG MODEL ----------------
    drag_base = 0.02

    camber_penalty = 60.0 * (m - m0) ** 2

    # Thinner than 0.11 causes large drag penalty (realistic stall + structural)
    if t < 0.11:
        thickness_penalty = 200.0 * (0.11 - t) ** 2
    else:
        thickness_penalty = 50.0 * (t - t0) ** 2

    position_penalty = 30.0 * (p - p0) ** 2
    alpha_penalty = 0.0015 * (a - 4.0) ** 2

    Cd = drag_base + camber_penalty + thickness_penalty + position_penalty + alpha_penalty
    Cd = max(Cd, 1e-4)

    return float(Cl), float(Cd)


def aero_score_single(m, p, t, alpha=4.0):
    """Single-angle score."""
    return fake_aero_score(m, p, t, alpha=alpha)


def aero_score_multi(m, p, t, alphas=(0.0, 4.0, 8.0)):
    """Multi-angle score."""
    cls, cds, lds = [], [], []
    for alpha in alphas:
        Cl, Cd = aero_score_single(m, p, t, alpha=alpha)
        cls.append(Cl)
        cds.append(Cd)
        lds.append(Cl / (Cd + 1e-6))
    return cls, cds, lds


def aero_score(m, p, t):
    """Backwards-compatible single-angle score."""
    return aero_score_single(m, p, t, alpha=4.0)


if __name__ == "__main__":
    # Quick sanity test for baseline airfoil
    m, p, t = 0.02, 0.4, 0.12
    alphas = (0.0, 4.0, 8.0)
    cls, cds, lds = aero_score_multi(m, p, t, alphas=alphas)
    print("Testing improved surrogate aero model")
    for a, cl, cd, ld in zip(alphas, cls, cds, lds):
        print(f"alpha={a:4.1f}°  Cl={cl:.4f}  Cd={cd:.5f}  L/D={ld:.2f}")
