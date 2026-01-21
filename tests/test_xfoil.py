"""
Tests for XFOIL Integration
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_surrogate_available():
    """Test that surrogate model always works."""
    from xfoil_integration import xfoil_analysis
    cl, cd, ld = xfoil_analysis(0.02, 0.4, 0.12, alpha=4.0)
    
    assert 0.3 < cl < 1.5, f"Cl={cl} out of expected range"
    assert 0.005 < cd < 0.05, f"Cd={cd} out of expected range"
    assert 10 < ld < 100, f"L/D={ld} out of expected range"


def test_polar_sweep():
    """Test polar sweep across angles."""
    from xfoil_integration import xfoil_polar
    
    polar = xfoil_polar(0.02, 0.4, 0.12, alphas=[0, 4, 8])
    
    assert len(polar['alpha']) == 3
    assert len(polar['Cl']) == 3
    assert len(polar['Cd']) == 3
    
    # Cl should increase with alpha
    assert polar['Cl'][1] > polar['Cl'][0]
    assert polar['Cl'][2] > polar['Cl'][1]


def test_coords_generation():
    """Test airfoil coordinate generation."""
    from xfoil_integration import coords_from_naca
    
    coords = coords_from_naca(0.02, 0.4, 0.12)
    
    assert coords.shape[1] == 2, "Should have x, y columns"
    assert len(coords) > 50, "Should have many points"
    
    # Check trailing edge closure
    assert np.allclose(coords[0], coords[-1], atol=0.01), "TE should be closed"


def test_different_airfoils():
    """Test various NACA configurations."""
    from xfoil_integration import xfoil_analysis
    
    configs = [
        (0.00, 0.4, 0.12),  # Symmetric
        (0.02, 0.4, 0.12),  # NACA 2412
        (0.04, 0.4, 0.12),  # High camber
        (0.02, 0.4, 0.08),  # Thin
        (0.02, 0.4, 0.18),  # Thick
    ]
    
    for m, p, t in configs:
        cl, cd, ld = xfoil_analysis(m, p, t)
        assert not np.isnan(cl), f"Cl is NaN for ({m}, {p}, {t})"
        assert not np.isnan(cd), f"Cd is NaN for ({m}, {p}, {t})"
        assert cd > 0, f"Cd should be positive for ({m}, {p}, {t})"


def test_reynolds_effect():
    """Test Reynolds number effect on drag."""
    from xfoil_integration import xfoil_analysis
    
    # Higher Re should give lower Cd (thinner boundary layer)
    _, cd_low, _ = xfoil_analysis(0.02, 0.4, 0.12, reynolds=1e5)
    _, cd_high, _ = xfoil_analysis(0.02, 0.4, 0.12, reynolds=1e6)
    
    assert cd_low > cd_high, "Cd should decrease with Re"


if __name__ == "__main__":
    print("Running XFOIL tests...")
    test_surrogate_available()
    print("✓ Surrogate available")
    test_polar_sweep()
    print("✓ Polar sweep")
    test_coords_generation()
    print("✓ Coords generation")
    test_different_airfoils()
    print("✓ Different airfoils")
    test_reynolds_effect()
    print("✓ Reynolds effect")
    print("\nAll tests passed!")
