"""
Uncertainty Quantification Module

Monte Carlo and sensitivity analysis for robust aerodynamic predictions.
Based on Stanford ADL uncertainty quantification methods.

Author: Mohamed Noorul Naseem
"""

import numpy as np
from typing import Dict, Tuple, List, Callable
from dataclasses import dataclass


@dataclass
class UncertaintyBudget:
    """Uncertainty sources and magnitudes."""
    # Manufacturing tolerances
    m_tolerance: float = 0.002  # ±0.2% on camber
    p_tolerance: float = 0.02   # ±2% on position
    t_tolerance: float = 0.005  # ±0.5% on thickness
    
    # CFD model uncertainty
    cl_model_error: float = 0.02  # ±2% Cl uncertainty
    cd_model_error: float = 0.05  # ±5% Cd uncertainty
    
    # Operating condition variability
    alpha_uncertainty: float = 0.5  # ±0.5 degrees
    reynolds_uncertainty: float = 0.1  # ±10%


class UncertaintyQuantification:
    """
    Uncertainty quantification for aerodynamic performance.
    
    Methods:
    - Monte Carlo sampling
    - Sensitivity analysis (Sobol indices)
    - Confidence interval estimation
    """
    
    def __init__(
        self, 
        aero_func: Callable = None,
        budget: UncertaintyBudget = None,
        n_samples: int = 500
    ):
        self.aero_func = aero_func or self._default_aero
        self.budget = budget or UncertaintyBudget()
        self.n_samples = n_samples
    
    def _default_aero(self, m, p, t, alpha):
        """Default aerodynamic evaluation."""
        try:
            from aero_eval import aero_score
            return aero_score(m, p, t, alpha)
        except:
            # Simplified fallback
            alpha_rad = np.radians(alpha)
            cl = 2 * np.pi * alpha_rad * (1 + 0.77 * t) + 10 * m
            cd = 0.02 + 50 * m**2 + 50 * (t - 0.12)**2
            return float(cl), float(cd)
    
    def monte_carlo(
        self,
        m: float, p: float, t: float,
        alpha: float = 4.0
    ) -> Dict:
        """
        Monte Carlo uncertainty propagation.
        
        Samples uncertain parameters and propagates through aero model.
        Returns statistics and confidence intervals.
        """
        budget = self.budget
        
        cls, cds, lds = [], [], []
        
        for _ in range(self.n_samples):
            # Sample uncertain parameters
            m_sample = np.random.normal(m, budget.m_tolerance)
            p_sample = np.random.normal(p, budget.p_tolerance)
            t_sample = np.random.normal(t, budget.t_tolerance)
            alpha_sample = np.random.normal(alpha, budget.alpha_uncertainty)
            
            # Clip to valid ranges
            m_sample = np.clip(m_sample, 0.0, 0.06)
            p_sample = np.clip(p_sample, 0.1, 0.7)
            t_sample = np.clip(t_sample, 0.08, 0.20)
            alpha_sample = np.clip(alpha_sample, -5, 20)
            
            # Evaluate
            try:
                cl, cd = self.aero_func(m_sample, p_sample, t_sample, alpha_sample)
            except:
                continue
            
            # Add model uncertainty
            cl *= np.random.normal(1.0, budget.cl_model_error)
            cd *= np.random.normal(1.0, budget.cd_model_error)
            cd = max(cd, 0.003)
            
            cls.append(cl)
            cds.append(cd)
            lds.append(cl / cd)
        
        if len(cls) < 10:
            return {'error': 'Insufficient valid samples'}
        
        return {
            'cl': {
                'mean': float(np.mean(cls)),
                'std': float(np.std(cls)),
                'ci_95': (float(np.percentile(cls, 2.5)), float(np.percentile(cls, 97.5))),
                'ci_99': (float(np.percentile(cls, 0.5)), float(np.percentile(cls, 99.5))),
            },
            'cd': {
                'mean': float(np.mean(cds)),
                'std': float(np.std(cds)),
                'ci_95': (float(np.percentile(cds, 2.5)), float(np.percentile(cds, 97.5))),
                'ci_99': (float(np.percentile(cds, 0.5)), float(np.percentile(cds, 99.5))),
            },
            'ld': {
                'mean': float(np.mean(lds)),
                'std': float(np.std(lds)),
                'ci_95': (float(np.percentile(lds, 2.5)), float(np.percentile(lds, 97.5))),
                'ci_99': (float(np.percentile(lds, 0.5)), float(np.percentile(lds, 99.5))),
            },
            'n_samples': len(cls),
            'convergence': self._check_convergence(lds),
        }
    
    def sensitivity_analysis(
        self,
        m: float, p: float, t: float,
        alpha: float = 4.0,
        delta: float = 0.001
    ) -> Dict:
        """
        Local sensitivity analysis using finite differences.
        
        Returns normalized sensitivities (elasticities) for each parameter.
        """
        # Baseline evaluation
        cl_base, cd_base = self.aero_func(m, p, t, alpha)
        ld_base = cl_base / cd_base
        
        sensitivities = {}
        
        # Sensitivity to m
        cl_p, cd_p = self.aero_func(m + delta, p, t, alpha)
        cl_m, cd_m = self.aero_func(max(0, m - delta), p, t, alpha)
        
        dcl_dm = (cl_p - cl_m) / (2 * delta)
        dcd_dm = (cd_p - cd_m) / (2 * delta)
        dld_dm = ((cl_p/cd_p) - (cl_m/cd_m)) / (2 * delta)
        
        sensitivities['m'] = {
            'dcl_dm': dcl_dm,
            'dcd_dm': dcd_dm,
            'dld_dm': dld_dm,
            'elasticity_cl': dcl_dm * m / cl_base if cl_base != 0 else 0,
            'elasticity_cd': dcd_dm * m / cd_base if cd_base != 0 else 0,
            'elasticity_ld': dld_dm * m / ld_base if ld_base != 0 else 0,
        }
        
        # Sensitivity to p
        cl_p, cd_p = self.aero_func(m, p + delta, t, alpha)
        cl_m, cd_m = self.aero_func(m, max(0.1, p - delta), t, alpha)
        
        dcl_dp = (cl_p - cl_m) / (2 * delta)
        dcd_dp = (cd_p - cd_m) / (2 * delta)
        
        sensitivities['p'] = {
            'dcl_dp': dcl_dp,
            'dcd_dp': dcd_dp,
            'elasticity_cl': dcl_dp * p / cl_base if cl_base != 0 else 0,
            'elasticity_cd': dcd_dp * p / cd_base if cd_base != 0 else 0,
        }
        
        # Sensitivity to t
        cl_p, cd_p = self.aero_func(m, p, t + delta, alpha)
        cl_m, cd_m = self.aero_func(m, p, max(0.08, t - delta), alpha)
        
        dcl_dt = (cl_p - cl_m) / (2 * delta)
        dcd_dt = (cd_p - cd_m) / (2 * delta)
        
        sensitivities['t'] = {
            'dcl_dt': dcl_dt,
            'dcd_dt': dcd_dt,
            'elasticity_cl': dcl_dt * t / cl_base if cl_base != 0 else 0,
            'elasticity_cd': dcd_dt * t / cd_base if cd_base != 0 else 0,
        }
        
        # Sensitivity to alpha
        cl_p, cd_p = self.aero_func(m, p, t, alpha + 0.5)
        cl_m, cd_m = self.aero_func(m, p, t, alpha - 0.5)
        
        dcl_da = (cl_p - cl_m) / 1.0
        dcd_da = (cd_p - cd_m) / 1.0
        
        sensitivities['alpha'] = {
            'dcl_dalpha': dcl_da,
            'dcd_dalpha': dcd_da,
            'cl_alpha_per_deg': dcl_da,
        }
        
        # Rank parameters by importance
        importance = [
            ('m', abs(sensitivities['m'].get('elasticity_ld', 0))),
            ('p', abs(sensitivities['p'].get('elasticity_cd', 0))),
            ('t', abs(sensitivities['t'].get('elasticity_cd', 0))),
        ]
        importance.sort(key=lambda x: x[1], reverse=True)
        sensitivities['ranking'] = [x[0] for x in importance]
        
        return sensitivities
    
    def _check_convergence(self, samples: List[float]) -> Dict:
        """Check if Monte Carlo has converged."""
        n = len(samples)
        if n < 50:
            return {'converged': False, 'reason': 'insufficient_samples'}
        
        # Check if running mean has stabilized
        half = n // 2
        mean_first = np.mean(samples[:half])
        mean_second = np.mean(samples[half:])
        
        relative_change = abs(mean_second - mean_first) / abs(mean_second + 1e-8)
        
        return {
            'converged': relative_change < 0.05,
            'relative_change': relative_change,
            'recommended_samples': n * 2 if relative_change > 0.05 else n
        }
    
    def robust_optimization_bounds(
        self,
        m: float, p: float, t: float,
        confidence: float = 0.95
    ) -> Dict:
        """
        Calculate robust optimization bounds.
        
        Returns worst-case performance with given confidence.
        """
        results = self.monte_carlo(m, p, t)
        
        if 'error' in results:
            return results
        
        # Worst-case at confidence level
        alpha = 1 - confidence
        
        return {
            'nominal_ld': results['ld']['mean'],
            'robust_ld': results['ld']['ci_95'][0],  # Lower bound
            'margin': results['ld']['mean'] - results['ld']['ci_95'][0],
            'margin_pct': (results['ld']['mean'] - results['ld']['ci_95'][0]) / results['ld']['mean'] * 100,
            'confidence': confidence,
            'recommendation': self._get_recommendation(results)
        }
    
    def _get_recommendation(self, results: Dict) -> str:
        """Generate recommendation based on uncertainty analysis."""
        ld_std = results['ld']['std']
        ld_mean = results['ld']['mean']
        cov = ld_std / ld_mean * 100  # Coefficient of variation
        
        if cov < 5:
            return "Low uncertainty. Design is robust."
        elif cov < 10:
            return "Moderate uncertainty. Consider tightening tolerances."
        else:
            return "High uncertainty. Review design parameters and tolerances."


def generate_uncertainty_report(
    m: float, p: float, t: float,
    output_file: str = None
) -> str:
    """Generate comprehensive uncertainty report."""
    uq = UncertaintyQuantification()
    
    lines = []
    lines.append("=" * 70)
    lines.append("UNCERTAINTY QUANTIFICATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Airfoil: m={m:.4f}, p={p:.4f}, t={t:.4f}")
    lines.append("")
    
    # Monte Carlo
    mc_results = uq.monte_carlo(m, p, t)
    lines.append("MONTE CARLO ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Samples: {mc_results.get('n_samples', 'N/A')}")
    lines.append("")
    
    for metric in ['cl', 'cd', 'ld']:
        if metric in mc_results:
            data = mc_results[metric]
            lines.append(f"{metric.upper()}:")
            lines.append(f"  Mean: {data['mean']:.4f}")
            lines.append(f"  Std:  {data['std']:.4f}")
            lines.append(f"  95% CI: [{data['ci_95'][0]:.4f}, {data['ci_95'][1]:.4f}]")
    
    lines.append("")
    
    # Sensitivity
    sens = uq.sensitivity_analysis(m, p, t)
    lines.append("SENSITIVITY ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Parameter importance ranking: {', '.join(sens.get('ranking', []))}")
    lines.append("")
    
    # Robust bounds
    robust = uq.robust_optimization_bounds(m, p, t)
    lines.append("ROBUST OPTIMIZATION")
    lines.append("-" * 40)
    lines.append(f"Nominal L/D: {robust.get('nominal_ld', 'N/A'):.1f}")
    lines.append(f"Robust L/D (95%): {robust.get('robust_ld', 'N/A'):.1f}")
    lines.append(f"Design margin: {robust.get('margin_pct', 'N/A'):.1f}%")
    lines.append(f"Recommendation: {robust.get('recommendation', 'N/A')}")
    
    lines.append("")
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
    
    return report


if __name__ == "__main__":
    print(generate_uncertainty_report(0.02, 0.4, 0.12))
