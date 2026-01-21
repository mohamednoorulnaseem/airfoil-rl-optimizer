"""
Tests for Manufacturing Constraints
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_valid_airfoil():
    """Test that baseline NACA 2412 is manufacturable."""
    from manufacturing_constraints import check_manufacturability
    
    is_valid, results = check_manufacturability(0.02, 0.4, 0.12)
    
    assert is_valid, f"NACA 2412 should be manufacturable: {results}"


def test_thickness_constraints():
    """Test thickness ratio constraints."""
    from manufacturing_constraints import check_manufacturability
    
    # Too thin
    is_valid, results = check_manufacturability(0.02, 0.4, 0.08)
    assert not results['thickness_ratio']['passed']
    
    # Too thick
    is_valid, results = check_manufacturability(0.02, 0.4, 0.22)
    assert not results['thickness_ratio']['passed']
    
    # Just right
    is_valid, results = check_manufacturability(0.02, 0.4, 0.12)
    assert results['thickness_ratio']['passed']


def test_camber_constraints():
    """Test camber constraints."""
    from manufacturing_constraints import check_manufacturability
    
    # Excessive camber
    is_valid, results = check_manufacturability(0.08, 0.4, 0.12)
    assert not results['max_camber']['passed']
    
    # Acceptable camber
    is_valid, results = check_manufacturability(0.04, 0.4, 0.12)
    assert results['max_camber']['passed']


def test_position_constraints():
    """Test camber position constraints."""
    from manufacturing_constraints import check_manufacturability
    
    # Too far forward
    is_valid, results = check_manufacturability(0.02, 0.10, 0.12)
    assert not results['camber_position']['passed']
    
    # Too far aft
    is_valid, results = check_manufacturability(0.02, 0.70, 0.12)
    assert not results['camber_position']['passed']
    
    # Optimal
    is_valid, results = check_manufacturability(0.02, 0.40, 0.12)
    assert results['camber_position']['passed']


def test_penalty_calculation():
    """Test manufacturing penalty calculation."""
    from manufacturing_constraints import get_manufacturing_penalty
    
    # Valid design = no penalty
    penalty = get_manufacturing_penalty(0.02, 0.4, 0.12)
    assert penalty == 0
    
    # Invalid design = positive penalty
    penalty = get_manufacturing_penalty(0.08, 0.4, 0.08)  # High camber, thin
    assert penalty > 0


def test_structural_score():
    """Test structural feasibility score."""
    from manufacturing_constraints import calculate_structural_score
    
    # Optimal thickness
    score = calculate_structural_score(0.02, 0.4, 0.12)
    assert score >= 0.8
    
    # Thin airfoil = lower score
    score_thin = calculate_structural_score(0.02, 0.4, 0.08)
    assert score_thin < score


def test_cnc_machinability():
    """Test CNC machinability score."""
    from manufacturing_constraints import calculate_cnc_machinability
    
    # Standard airfoil
    score = calculate_cnc_machinability(0.02, 0.4, 0.12)
    assert score >= 0.7
    
    # Complex geometry = lower score
    score_complex = calculate_cnc_machinability(0.05, 0.2, 0.08)
    assert score_complex < score


if __name__ == "__main__":
    print("Running Manufacturing tests...")
    test_valid_airfoil()
    print("✓ Valid airfoil")
    test_thickness_constraints()
    print("✓ Thickness constraints")
    test_camber_constraints()
    print("✓ Camber constraints")
    test_position_constraints()
    print("✓ Position constraints")
    test_penalty_calculation()
    print("✓ Penalty calculation")
    test_structural_score()
    print("✓ Structural score")
    test_cnc_machinability()
    print("✓ CNC machinability")
    print("\nAll tests passed!")
