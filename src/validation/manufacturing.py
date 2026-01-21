"""
Manufacturing Constraints Module for Airfoil Design

This module implements real-world manufacturing feasibility checks
based on industry standards for aircraft wing production.

Author: Mohamed Noorul Naseem
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ManufacturingSpec:
    """Industry manufacturing specifications for airfoils."""
    
    # Thickness constraints
    min_thickness: float = 0.10      # < 10% thickness is structurally challenging
    max_thickness: float = 0.20      # > 20% creates excessive drag and weight
    optimal_thickness: float = 0.12  # Industry sweet spot for subsonic
    
    # Camber constraints
    max_camber: float = 0.06         # > 6% camber is difficult to machine
    optimal_camber: float = 0.02     # Typical for cruise efficiency
    
    # Camber position constraints  
    min_camber_pos: float = 0.15     # Too far forward = stress concentration
    max_camber_pos: float = 0.60     # Too far back = laminar-turbulent issues
    optimal_camber_pos: float = 0.40 # Typical NACA design point
    
    # Trailing edge constraints
    min_trailing_edge_angle: float = 5.0   # Degrees - too sharp = manufacturing difficulty
    max_trailing_edge_angle: float = 20.0  # Degrees - too blunt = excessive base drag
    
    # Leading edge constraints
    min_le_radius_ratio: float = 0.02  # LE radius / chord - too sharp = stress concentration
    max_le_radius_ratio: float = 0.05  # LE radius / chord - too blunt = drag penalty
    
    # Surface quality
    max_surface_roughness: float = 1.6e-5  # m - typical machined aluminum
    

def check_manufacturability(
    m: float, 
    p: float, 
    t: float,
    spec: ManufacturingSpec = None
) -> Tuple[bool, Dict[str, Dict]]:
    """
    Comprehensive manufacturing feasibility check.
    
    Args:
        m: Max camber (0-0.06)
        p: Camber position (0.1-0.7)  
        t: Thickness ratio (0.08-0.20)
        spec: Manufacturing specifications (uses defaults if None)
        
    Returns:
        Tuple of (is_manufacturable, detailed_results)
    """
    if spec is None:
        spec = ManufacturingSpec()
    
    results = {}
    
    # 1. Thickness check
    results["thickness_ratio"] = {
        "value": t,
        "min": spec.min_thickness,
        "max": spec.max_thickness,
        "optimal": spec.optimal_thickness,
        "passed": spec.min_thickness <= t <= spec.max_thickness,
        "message": _get_thickness_message(t, spec)
    }
    
    # 2. Camber check
    results["max_camber"] = {
        "value": m,
        "max": spec.max_camber,
        "optimal": spec.optimal_camber,
        "passed": m <= spec.max_camber,
        "message": _get_camber_message(m, spec)
    }
    
    # 3. Camber position check
    results["camber_position"] = {
        "value": p,
        "min": spec.min_camber_pos,
        "max": spec.max_camber_pos,
        "optimal": spec.optimal_camber_pos,
        "passed": spec.min_camber_pos <= p <= spec.max_camber_pos,
        "message": _get_position_message(p, spec)
    }
    
    # 4. Leading edge radius (estimated from parameters)
    le_radius = estimate_le_radius(m, p, t)
    le_ratio = le_radius  # Already normalized to chord
    results["leading_edge"] = {
        "value": le_ratio,
        "min": spec.min_le_radius_ratio,
        "max": spec.max_le_radius_ratio,
        "passed": spec.min_le_radius_ratio <= le_ratio <= spec.max_le_radius_ratio,
        "message": _get_le_message(le_ratio, spec)
    }
    
    # 5. Trailing edge angle
    te_angle = estimate_te_angle(m, p, t)
    results["trailing_edge"] = {
        "value": te_angle,
        "min": spec.min_trailing_edge_angle,
        "max": spec.max_trailing_edge_angle,
        "passed": spec.min_trailing_edge_angle <= te_angle <= spec.max_trailing_edge_angle,
        "message": _get_te_message(te_angle, spec)
    }
    
    # 6. Structural feasibility
    structural_score = calculate_structural_score(m, p, t)
    results["structural"] = {
        "value": structural_score,
        "min": 0.6,  # Minimum acceptable score
        "passed": structural_score >= 0.6,
        "message": _get_structural_message(structural_score)
    }
    
    # 7. CNC machinability
    cnc_score = calculate_cnc_machinability(m, p, t)
    results["cnc_machinability"] = {
        "value": cnc_score,
        "min": 0.7,
        "passed": cnc_score >= 0.7,
        "message": _get_cnc_message(cnc_score)
    }
    
    # Overall assessment
    all_passed = all(r["passed"] for r in results.values())
    
    return all_passed, results


def estimate_le_radius(m: float, p: float, t: float) -> float:
    """
    Estimate leading edge radius from NACA parameters.
    
    For NACA 4-digit: LE radius ≈ 1.1019 * t²
    """
    return 1.1019 * (t ** 2)


def estimate_te_angle(m: float, p: float, t: float) -> float:
    """
    Estimate trailing edge angle from NACA parameters.
    
    Returns angle in degrees between upper and lower surfaces at TE.
    """
    # For standard NACA 4-digit, TE angle ≈ 2 * arctan(0.2 * 5t)
    te_angle = 2 * np.degrees(np.arctan(1.0 * t))
    
    # Camber correction
    te_angle += 5 * m
    
    return te_angle


def calculate_structural_score(m: float, p: float, t: float) -> float:
    """
    Calculate structural feasibility score (0-1).
    
    Based on:
    - Thickness ratio (main factor for internal structure)
    - Camber position (affects spar placement)
    - Maximum camber (affects skin stress)
    """
    score = 1.0
    
    # Thickness penalty (linear decrease below 0.12)
    if t < 0.12:
        score -= 0.3 * (0.12 - t) / 0.04
    
    # Camber position penalty (issues at extremes)
    if p < 0.25 or p > 0.55:
        score -= 0.2
    
    # High camber penalty (skin stress)
    if m > 0.04:
        score -= 0.3 * (m - 0.04) / 0.02
    
    return max(0.0, min(1.0, score))


def calculate_cnc_machinability(m: float, p: float, t: float) -> float:
    """
    Calculate CNC machinability score (0-1).
    
    Based on:
    - Curvature continuity
    - Minimum tool access
    - Surface complexity
    """
    score = 1.0
    
    # Very thin airfoils harder to machine
    if t < 0.10:
        score -= 0.4
    
    # High camber creates complex curvature
    if m > 0.04:
        score -= 0.2
    
    # Extreme camber positions create access issues
    if p < 0.20 or p > 0.60:
        score -= 0.15
    
    return max(0.0, min(1.0, score))


def _get_thickness_message(t: float, spec: ManufacturingSpec) -> str:
    if t < spec.min_thickness:
        return f"❌ Thickness {t*100:.1f}% too thin for structural integrity (min: {spec.min_thickness*100:.0f}%)"
    elif t > spec.max_thickness:
        return f"❌ Thickness {t*100:.1f}% too thick (max: {spec.max_thickness*100:.0f}%)"
    elif abs(t - spec.optimal_thickness) < 0.02:
        return f"✅ Optimal thickness {t*100:.1f}% for subsonic flight"
    else:
        return f"✅ Acceptable thickness {t*100:.1f}%"


def _get_camber_message(m: float, spec: ManufacturingSpec) -> str:
    if m > spec.max_camber:
        return f"❌ Camber {m*100:.1f}% exceeds manufacturing limit (max: {spec.max_camber*100:.0f}%)"
    elif m < 0.01:
        return f"✅ Low camber {m*100:.1f}% - symmetric-like, easy to manufacture"
    else:
        return f"✅ Camber {m*100:.1f}% within manufacturing limits"


def _get_position_message(p: float, spec: ManufacturingSpec) -> str:
    if p < spec.min_camber_pos:
        return f"❌ Camber position {p*100:.0f}% too far forward (stress issues)"
    elif p > spec.max_camber_pos:
        return f"❌ Camber position {p*100:.0f}% too far aft (laminar flow issues)"
    else:
        return f"✅ Camber position {p*100:.0f}% chord is optimal"


def _get_le_message(le_ratio: float, spec: ManufacturingSpec) -> str:
    if le_ratio < spec.min_le_radius_ratio:
        return f"❌ LE radius too sharp - stress concentration risk"
    elif le_ratio > spec.max_le_radius_ratio:
        return f"⚠️ LE radius blunt - may increase drag"
    else:
        return f"✅ LE radius {le_ratio*100:.2f}% chord is manufacturable"


def _get_te_message(te_angle: float, spec: ManufacturingSpec) -> str:
    if te_angle < spec.min_trailing_edge_angle:
        return f"❌ TE angle {te_angle:.1f}° too sharp - manufacturing difficulty"
    elif te_angle > spec.max_trailing_edge_angle:
        return f"⚠️ TE angle {te_angle:.1f}° blunt - may increase base drag"
    else:
        return f"✅ TE angle {te_angle:.1f}° within manufacturing limits"


def _get_structural_message(score: float) -> str:
    if score >= 0.9:
        return "✅ Excellent structural feasibility"
    elif score >= 0.7:
        return "✅ Good structural feasibility"
    elif score >= 0.6:
        return "⚠️ Marginal structural feasibility - review required"
    else:
        return "❌ Poor structural feasibility - redesign recommended"


def _get_cnc_message(score: float) -> str:
    if score >= 0.9:
        return "✅ Excellent CNC machinability"
    elif score >= 0.8:
        return "✅ Good CNC machinability"
    elif score >= 0.7:
        return "⚠️ Acceptable CNC machinability"
    else:
        return "❌ Poor CNC machinability - complex tooling required"


def get_manufacturing_penalty(m: float, p: float, t: float) -> float:
    """
    Calculate penalty for non-manufacturable designs.
    
    Returns value between 0 (fully manufacturable) and 10 (impossible).
    Used in RL reward function.
    """
    is_valid, results = check_manufacturability(m, p, t)
    
    if is_valid:
        return 0.0
    
    penalty = 0.0
    
    for name, check in results.items():
        if not check["passed"]:
            # Weight by severity
            if name in ["thickness_ratio", "structural"]:
                penalty += 3.0  # Critical
            elif name in ["max_camber", "camber_position"]:
                penalty += 2.0  # Important
            else:
                penalty += 1.0  # Minor
    
    return min(penalty, 10.0)


def get_manufacturing_summary(m: float, p: float, t: float) -> str:
    """Get a human-readable manufacturing summary."""
    is_valid, results = check_manufacturability(m, p, t)
    
    lines = ["=" * 50]
    lines.append("MANUFACTURING FEASIBILITY REPORT")
    lines.append("=" * 50)
    lines.append(f"Airfoil Parameters: m={m:.4f}, p={p:.4f}, t={t:.4f}")
    lines.append("-" * 50)
    
    passed = 0
    failed = 0
    
    for name, check in results.items():
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"{name:20s}: {status:4s} - {check['message']}")
        if check["passed"]:
            passed += 1
        else:
            failed += 1
    
    lines.append("-" * 50)
    lines.append(f"OVERALL: {'✅ MANUFACTURABLE' if is_valid else '❌ NOT MANUFACTURABLE'}")
    lines.append(f"Checks passed: {passed}/{passed + failed}")
    lines.append("=" * 50)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test manufacturing constraints
    print("\n" + "=" * 60)
    print("Testing Manufacturing Constraints Module")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        (0.02, 0.4, 0.12, "NACA 2412 (baseline)"),
        (0.04, 0.4, 0.12, "High camber"),
        (0.02, 0.4, 0.08, "Very thin"),
        (0.02, 0.4, 0.16, "Thick"),
        (0.02, 0.2, 0.12, "Forward camber"),
        (0.02, 0.6, 0.12, "Aft camber"),
    ]
    
    for m, p, t, name in test_cases:
        print(f"\n{name}:")
        print(get_manufacturing_summary(m, p, t))
