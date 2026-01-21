import numpy as np

def naca4(m, p, t, num_points=200):
    """
    Generate NACA 4-digit airfoil coordinates.
    m: max camber (e.g. 0.02 for 2%)
    p: location of max camber (e.g. 0.4 for 40%)
    t: thickness (e.g. 0.12 for 12%)
    Returns: x_upper, y_upper, x_lower, y_lower
    """
    x = np.linspace(0, 1, num_points)

    # Thickness distribution
    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    # Camber line & slope
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    for i, xi in enumerate(x):
        if xi < p:
            yc[i] = m / p**2 * (2 * p * xi - xi**2)
            dyc_dx[i] = 2 * m / p**2 * (p - xi)
        else:
            yc[i] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xi - xi**2)
            dyc_dx[i] = 2 * m / (1 - p)**2 * (p - xi)

    theta = np.arctan(dyc_dx)

    # Upper and lower surfaces
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example: roughly like NACA 2412
    m, p, t = 0.02, 0.4, 0.12
    xu, yu, xl, yl = naca4(m, p, t)

    plt.figure()
    plt.plot(xu, yu, label="Upper surface")
    plt.plot(xl, yl, label="Lower surface")
    plt.axis("equal")
    plt.legend()
    plt.title(f"NACA-like airfoil: m={m}, p={p}, t={t}")
    plt.xlabel("x")
    plt.ylabel("y")

    # Save as image
    plt.savefig("airfoil_plot.png", dpi=300)
    print("Saved airfoil_plot.png in current folder.")
