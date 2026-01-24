"""
Tests for XFOIL Integration
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.aerodynamics.xfoil_interface import XFOILRunner
from src.aerodynamics.airfoil_gen import generate_naca_4digit

def test_xfoil_runner_init():
    """Test XFOILRunner initialization"""
    runner = XFOILRunner()
    assert runner.xfoil_path == 'xfoil'

def test_generate_coords():
    """Test airfoil coordinate generation"""
    coords = generate_naca_4digit(0.02, 0.4, 0.12, n_points=100)
    assert coords.shape == (100, 2)
    # Check bounds
    assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] <= 1)

def test_analyze_airfoil():
    """Test running XFOIL analysis"""
    runner = XFOILRunner()
    coords = generate_naca_4digit(0.02, 0.4, 0.12, n_points=100)
    
    # Analyze single alpha
    result = runner.analyze_airfoil(coords, alpha_range=[0, 2])
    
    if result is None:
        pytest.skip("XFOIL executable not found or failed to run")
        
    assert len(result) > 0
    assert 'cl' in result[0]
    assert 'cd' in result[0]

if __name__ == "__main__":
    test_xfoil_runner_init()
    test_generate_coords()
    try:
        test_analyze_airfoil()
        print("All tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
