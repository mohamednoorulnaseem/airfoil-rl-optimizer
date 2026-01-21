"""
Aerodynamic Coefficients Module

Core functions for computing lift, drag, and moment coefficients
using multiple solver backends.

Author: Mohamed Noorul Naseem
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AeroCoefficients:
    """Container for aerodynamic coefficients."""
    cl: float          # Lift coefficient
    cd: float          # Drag coefficient
    cm: float          # Moment coefficient
    ld: float          # Lift-to-drag ratio
    alpha: float       # Angle of attack (deg)
    reynolds: float    # Reynolds number
    mach: float        # Mach number
    source: str        # Solver used ('xfoil', 'su2', 'pinn', 'surrogate')
    converged: bool    # Did solver converge


class AeroSolver:
    """
    Unified aerodynamic solver interface.
    
    Automatically selects best available solver:
    1. XFOIL (if installed)
    2. SU2 (if installed)
    3. PINN surrogate
    4. Analytical surrogate
    """
    
    def __init__(self, preferred_solver: str = 'auto'):
        self.preferred = preferred_solver
        self.solvers = self._detect_solvers()
        
    def _detect_solvers(self) -> Dict[str, bool]:
        """Detect available solvers."""
        available = {
            'xfoil': self._check_xfoil(),
            'su2': self._check_su2(),
            'pinn': True,  # Always available (Python-based)
            'surrogate': True  # Always available
        }
        return available
    
    def _check_xfoil(self) -> bool:
        try:
            import subprocess
            subprocess.run(['xfoil'], input=b'QUIT\n', capture_output=True, timeout=2)
            return True
        except:
            return False
    
    def _check_su2(self) -> bool:
        try:
            import subprocess
            subprocess.run(['SU2_CFD', '--help'], capture_output=True, timeout=2)
            return True
        except:
            return False
    
    def analyze(
        self,
        m: float, p: float, t: float,
        alpha: float = 4.0,
        reynolds: float = 1e6,
        mach: float = 0.0
    ) -> AeroCoefficients:
        """
        Compute aerodynamic coefficients.
        
        Automatically selects best available solver.
        """
        solver = self._select_solver()
        
        if solver == 'xfoil':
            return self._run_xfoil(m, p, t, alpha, reynolds, mach)
        elif solver == 'su2':
            return self._run_su2(m, p, t, alpha, reynolds, mach)
        elif solver == 'pinn':
            return self._run_pinn(m, p, t, alpha, reynolds)
        else:
            return self._run_surrogate(m, p, t, alpha, reynolds)
    
    def _select_solver(self) -> str:
        """Select best available solver."""
        if self.preferred != 'auto' and self.solvers.get(self.preferred, False):
            return self.preferred
        
        # Priority order
        for solver in ['xfoil', 'su2', 'pinn', 'surrogate']:
            if self.solvers.get(solver, False):
                return solver
        
        return 'surrogate'
    
    def _run_xfoil(self, m, p, t, alpha, reynolds, mach) -> AeroCoefficients:
        """Run XFOIL analysis."""
        try:
            from src.aerodynamics.xfoil_interface import XFOILAnalyzer
            analyzer = XFOILAnalyzer()
            result = analyzer.analyze(m, p, t, alpha, reynolds, mach)
            return AeroCoefficients(
                cl=result['Cl'], cd=result['Cd'], cm=result.get('Cm', 0),
                ld=result['L/D'], alpha=alpha, reynolds=reynolds, mach=mach,
                source='xfoil', converged=result.get('converged', True)
            )
        except:
            return self._run_surrogate(m, p, t, alpha, reynolds)
    
    def _run_su2(self, m, p, t, alpha, reynolds, mach) -> AeroCoefficients:
        """Run SU2 analysis."""
        try:
            from src.aerodynamics.su2_interface import SU2Interface
            su2 = SU2Interface()
            from airfoil_gen import naca4
            xu, yu, xl, yl = naca4(m, p, t)
            coords = np.column_stack([np.concatenate([xu[::-1], xl[1:]]),
                                     np.concatenate([yu[::-1], yl[1:]])])
            result = su2.run_analysis(coords)
            return AeroCoefficients(
                cl=result['cl'], cd=result['cd'], cm=result['cm'],
                ld=result['ld'], alpha=alpha, reynolds=reynolds, mach=mach,
                source='su2', converged=result.get('converged', True)
            )
        except:
            return self._run_surrogate(m, p, t, alpha, reynolds)
    
    def _run_pinn(self, m, p, t, alpha, reynolds) -> AeroCoefficients:
        """Run PINN surrogate."""
        try:
            from src.aerodynamics.pinn_surrogate import get_pretrained_pinn
            pinn = get_pretrained_pinn()
            cl, cd, ld = pinn.predict(m, p, t, alpha, reynolds)
            return AeroCoefficients(
                cl=cl, cd=cd, cm=-0.025 - 0.1 * m,
                ld=ld, alpha=alpha, reynolds=reynolds, mach=0.0,
                source='pinn', converged=True
            )
        except:
            return self._run_surrogate(m, p, t, alpha, reynolds)
    
    def _run_surrogate(self, m, p, t, alpha, reynolds) -> AeroCoefficients:
        """Run analytical surrogate."""
        alpha_rad = np.radians(alpha)
        
        # Lift
        cl_alpha = 2 * np.pi * (1 + 0.77 * t)
        alpha_zl = -1.15 * m * 100
        cl = cl_alpha * np.radians(alpha - alpha_zl)
        cl = np.clip(cl, -0.5, 1.8)
        
        # Drag
        cf = 0.074 / (reynolds ** 0.2)
        cd = 2 * cf * (1 + 2 * t + 60 * t**4)
        cd += cl**2 / (np.pi * 6 * 0.95)
        cd = max(cd, 0.005)
        
        return AeroCoefficients(
            cl=float(cl), cd=float(cd), cm=-0.025 - 0.1 * m,
            ld=float(cl/cd), alpha=alpha, reynolds=reynolds, mach=0.0,
            source='surrogate', converged=True
        )
    
    def polar_sweep(
        self, m: float, p: float, t: float,
        alphas: List[float] = None, reynolds: float = 1e6
    ) -> List[AeroCoefficients]:
        """Run analysis across multiple angles."""
        if alphas is None:
            alphas = list(np.arange(-4, 16, 1))
        return [self.analyze(m, p, t, a, reynolds) for a in alphas]


# Convenience functions
_solver = None

def get_solver() -> AeroSolver:
    global _solver
    if _solver is None:
        _solver = AeroSolver()
    return _solver

def compute_coefficients(m, p, t, alpha=4.0, re=1e6) -> Tuple[float, float, float]:
    """Quick coefficient computation. Returns (Cl, Cd, L/D)."""
    result = get_solver().analyze(m, p, t, alpha, re)
    return result.cl, result.cd, result.ld


if __name__ == "__main__":
    print("Testing Aero Coefficients Module...")
    solver = AeroSolver()
    print(f"Available solvers: {solver.solvers}")
    
    result = solver.analyze(0.02, 0.4, 0.12, alpha=4.0)
    print(f"\nNACA 2412 @ α=4°:")
    print(f"  Cl = {result.cl:.4f}")
    print(f"  Cd = {result.cd:.5f}")
    print(f"  L/D = {result.ld:.1f}")
    print(f"  Source: {result.source}")
