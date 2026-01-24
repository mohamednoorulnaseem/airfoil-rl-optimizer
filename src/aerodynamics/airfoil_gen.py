import numpy as np

def generate_naca_4digit(m, p, t, n_points=100):
    """
    Generate NACA 4-digit airfoil coordinates
    Args:
        m (float): Max camber (0 to 0.095)
        p (float): Max camber position (0 to 0.9)
        t (float): Max thickness (0 to 0.30)
        n_points (int): Number of points per surface
    Returns:
        numpy.ndarray: (2*n_points, 2) array of x,y coordinates
    """
    # X coordinates (cosine spacing)
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    
    # Thickness distribution
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # Camber line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    # Forward of max camber
    idx_fwd = x <= p
    if np.any(idx_fwd):
        yc[idx_fwd] = (m / p**2) * (2 * p * x[idx_fwd] - x[idx_fwd]**2)
        dyc_dx[idx_fwd] = (2 * m / p**2) * (p - x[idx_fwd])
    
    # Aft of max camber
    idx_aft = x > p
    if np.any(idx_aft):
        yc[idx_aft] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[idx_aft] - x[idx_aft]**2)
        dyc_dx[idx_aft] = (2 * m / (1 - p)**2) * (p - x[idx_aft])
    
    # Theta
    theta = np.arctan(dyc_dx)
    
    # Upper and lower surfaces
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Combine (upper surface from TE to LE, then lower surface from LE to TE)
    # XFOIL usually likes TE -> LE -> TE or similar.
    # Standard format: Upper surface (TE to LE) then Lower surface (LE to TE)
    
    # Reverse upper surface to go TE -> LE
    xu = xu[::-1]
    yu = yu[::-1]
    
    # Combine
    coords = np.column_stack((np.concatenate([xu, xl[1:]]), 
                              np.concatenate([yu, yl[1:]])))
    
    return coords

def naca4(m, p, t, num_points=100):
    """
    Legacy wrapper for compatibility
    Returns individual arrays: xu, yu, xl, yl
    """
    coords = generate_naca_4digit(m, p, t, num_points)
    # Split back into upper/lower is tricky because they are merged. 
    # But for now, let's just re-implement simple version if needed or leave it out if not used.
    # Actually, let's just conform to the new standard.
    pass

if __name__ == "__main__":
    # Test
    coords = generate_naca_4digit(0.02, 0.4, 0.12)
    print(f"Generated {len(coords)} points")
