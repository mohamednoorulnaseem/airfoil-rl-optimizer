"""
Aerodynamic Evaluation Module (Legacy Wrapper)

This module has been upgraded to use the Stanford-grade solvers in src/aerodynamics.
It is kept for backward compatibility with older scripts.

Author: Mohamed Noorul Naseem
"""

from src.aerodynamics.aero_coefficients import compute_coefficients, get_solver

def aero_score(m, p, t, alpha=4.0):
    """
    Calculate Cl, Cd for a given airfoil.
    Now wraps the production-grade AeroSolver.
    """
    cl, cd, ld = compute_coefficients(m, p, t, alpha=alpha)
    return cl, cd

def aero_score_multi(m, p, t, alphas=None):
    """
    Calculate coefficients for multiple angles (used in old scripts).
    Returns (cls, cds, lds).
    """
    if alphas is None:
        alphas = [0, 2, 4, 6]
        
    cls, cds, lds = [], [], []
    solver = get_solver()
    
    for alpha in alphas:
        res = solver.analyze(m, p, t, alpha=alpha)
        cls.append(res.cl)
        cds.append(res.cd)
        lds.append(res.ld)
        
    return cls, cds, lds

# Legacy alias
fake_aero_score = aero_score
