"""
Performance metrics calculation for airfoil assessment.
"""

import numpy as np

def calculate_lift_drag_ratio(cl, cd):
    """Calculate L/D ratio, handling divide by zero."""
    if abs(cd) < 1e-9:
        return 0.0
    return cl / cd

def calculate_improvement(baseline, new_value, metric='higher_is_better'):
    """
    Calculate percentage improvement.
    
    Args:
        baseline (float): Baseline value
        new_value (float): New value
        metric (str): 'higher_is_better' or 'lower_is_better'
        
    Returns:
        float: Percentage improvement
    """
    if baseline == 0:
        return 0.0
        
    diff = new_value - baseline
    pct = (diff / baseline) * 100
    
    if metric == 'lower_is_better':
        return -pct
    return pct

def calculate_range_breguet(L_D, velocity, fuel_fraction, sfc=0.5):
    """
    Calculate aircraft range using Breguet range equation.
    
    Args:
        L_D (float): Lift-to-drag ratio
        velocity (float): Cruise velocity (m/s)
        fuel_fraction (float): Ratio of fuel weight to gross weight
        sfc (float): Specific Fuel Consumption (lb/lbf/hr or similar unit scaled)
        
    Note: This is a simplified estimation.
    """
    # Simply proportional to L/D
    return L_D * np.log(1 / (1 - fuel_fraction)) * (velocity / sfc)

def fuel_savings_estimation(baseline_ld, optimized_ld, annual_fuel_cost):
    """
    Estimate financial savings based on L/D improvement.
    Assumes fuel consumption is roughly inversely proportional to L/D.
    """
    if baseline_ld <= 0 or optimized_ld <= 0:
        return 0.0
        
    # Fuel consumed ~ 1 / (L/D)
    # Savings = 1 - (NewFuel / OldFuel) = 1 - (OldLD / NewLD)
    
    savings_pct = 1 - (baseline_ld / optimized_ld)
    return savings_pct * annual_fuel_cost
