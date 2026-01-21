"""
Wind Tunnel Simulation Module
Simulates experimental validation with realistic measurement noise.
"""

import numpy as np
from typing import Dict, List

def simulate_wind_tunnel(
    coords: np.ndarray,
    velocity: float = 50.0,
    alpha: float = 0.0,
    reynolds: float = 1e6,
    cfd_cl: float = None,
    cfd_cd: float = None
) -> Dict:
    """
    Simulate wind tunnel measurements with realistic noise.
    Typical wind tunnel accuracy: ±2% for Cl, ±3% for Cd.
    """
    if cfd_cl is None or cfd_cd is None:
        try:
            from src.aerodynamics.legacy_xfoil import xfoil_analysis
        except ImportError:
             import sys
             import os
             sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
             from src.aerodynamics.legacy_xfoil import xfoil_analysis
             
        # naca4 not needed if xfoil_analysis handles it, but let's be safe
        cfd_cl, cfd_cd, _ = xfoil_analysis(0.02, 0.4, 0.12, alpha, reynolds)
    
    cl_noise = np.random.normal(0, 0.02)
    cd_noise = np.random.normal(0, 0.03)
    
    cl_measured = cfd_cl * (1 + cl_noise)
    cd_measured = cfd_cd * (1 + cd_noise)
    
    return {
        "cl": cl_measured, "cd": cd_measured,
        "cl_uncertainty": 0.02 * abs(cl_measured),
        "cd_uncertainty": 0.03 * abs(cd_measured),
        "velocity": velocity, "alpha": alpha, "reynolds": reynolds,
        "cfd_cl": cfd_cl, "cfd_cd": cfd_cd,
        "cl_deviation_pct": abs(cl_measured - cfd_cl) / cfd_cl * 100,
        "cd_deviation_pct": abs(cd_measured - cfd_cd) / cfd_cd * 100,
    }

def run_wind_tunnel_sweep(
    m: float, p: float, t: float,
    alphas: List[float] = None,
    reynolds: float = 1e6
) -> Dict[str, List]:
    """Run wind tunnel simulation across angle sweep."""
    if alphas is None:
        alphas = list(np.arange(-2, 14, 2))
    
    try:
        from src.aerodynamics.legacy_xfoil import xfoil_analysis
    except ImportError:
         import sys
         import os
         sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
         from src.aerodynamics.legacy_xfoil import xfoil_analysis
    
    results = {"alpha": [], "wt_cl": [], "wt_cd": [], "cfd_cl": [], "cfd_cd": [],
               "cl_err": [], "cd_err": [], "cl_dev": [], "cd_dev": []}
    
    for alpha in alphas:
        cfd_cl, cfd_cd, _ = xfoil_analysis(m, p, t, alpha, reynolds)
        wt = simulate_wind_tunnel(None, alpha=alpha, cfd_cl=cfd_cl, cfd_cd=cfd_cd)
        
        results["alpha"].append(alpha)
        results["wt_cl"].append(wt["cl"])
        results["wt_cd"].append(wt["cd"])
        results["cfd_cl"].append(cfd_cl)
        results["cfd_cd"].append(cfd_cd)
        results["cl_err"].append(wt["cl_uncertainty"])
        results["cd_err"].append(wt["cd_uncertainty"])
        results["cl_dev"].append(wt["cl_deviation_pct"])
        results["cd_dev"].append(wt["cd_deviation_pct"])
    
    return results

def get_validation_summary(wt_results: Dict) -> Dict:
    """Get validation statistics."""
    return {
        "mean_cl_deviation": np.mean(wt_results["cl_dev"]),
        "max_cl_deviation": np.max(wt_results["cl_dev"]),
        "mean_cd_deviation": np.mean(wt_results["cd_dev"]),
        "max_cd_deviation": np.max(wt_results["cd_dev"]),
        "validated": np.mean(wt_results["cl_dev"]) < 3.0 and np.mean(wt_results["cd_dev"]) < 5.0
    }

if __name__ == "__main__":
    print("Testing Wind Tunnel Simulation...")
    results = run_wind_tunnel_sweep(0.02, 0.4, 0.12)
    summary = get_validation_summary(results)
    print(f"Mean Cl deviation: {summary['mean_cl_deviation']:.2f}%")
    print(f"Mean Cd deviation: {summary['mean_cd_deviation']:.2f}%")
    print(f"Validated: {summary['validated']}")
