"""
Advanced Visualizations Module
Professional-grade plots for airfoil analysis and presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')

def plot_airfoil_comparison(
    baseline_coords: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    optimized_coords: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline_label: str = "Baseline NACA 2412",
    optimized_label: str = "RL-Optimized"
) -> plt.Figure:
    """Plot baseline vs optimized airfoil comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    xu_b, yu_b, xl_b, yl_b = baseline_coords
    xu_o, yu_o, xl_o, yl_o = optimized_coords
    
    ax.fill_between(xu_b, yu_b, yl_b, alpha=0.3, color='blue', label=baseline_label)
    ax.fill_between(xu_o, yu_o, yl_o, alpha=0.3, color='green', label=optimized_label)
    ax.plot(xu_b, yu_b, 'b-', linewidth=2)
    ax.plot(xl_b, yl_b, 'b-', linewidth=2)
    ax.plot(xu_o, yu_o, 'g-', linewidth=2)
    ax.plot(xl_o, yl_o, 'g-', linewidth=2)
    
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel('y/c', fontsize=12)
    ax.set_title('Airfoil Geometry Comparison', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_polar_comparison(
    baseline_polar: dict,
    optimized_polar: dict
) -> plt.Figure:
    """Plot Cl-Cd polars for baseline and optimized."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Cl vs alpha
    axes[0].plot(baseline_polar['alpha'], baseline_polar['Cl'], 'b-o', label='Baseline', linewidth=2)
    axes[0].plot(optimized_polar['alpha'], optimized_polar['Cl'], 'g-s', label='Optimized', linewidth=2)
    axes[0].set_xlabel('Angle of Attack (°)', fontsize=11)
    axes[0].set_ylabel('Lift Coefficient Cl', fontsize=11)
    axes[0].set_title('Lift Curve', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cd vs alpha
    axes[1].plot(baseline_polar['alpha'], baseline_polar['Cd'], 'b-o', label='Baseline', linewidth=2)
    axes[1].plot(optimized_polar['alpha'], optimized_polar['Cd'], 'g-s', label='Optimized', linewidth=2)
    axes[1].set_xlabel('Angle of Attack (°)', fontsize=11)
    axes[1].set_ylabel('Drag Coefficient Cd', fontsize=11)
    axes[1].set_title('Drag Curve', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # L/D vs alpha
    axes[2].plot(baseline_polar['alpha'], baseline_polar['L/D'], 'b-o', label='Baseline', linewidth=2)
    axes[2].plot(optimized_polar['alpha'], optimized_polar['L/D'], 'g-s', label='Optimized', linewidth=2)
    axes[2].set_xlabel('Angle of Attack (°)', fontsize=11)
    axes[2].set_ylabel('Lift-to-Drag Ratio L/D', fontsize=11)
    axes[2].set_title('Efficiency Curve', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_pressure_distribution(
    coords: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    m: float, p: float, t: float, alpha: float = 4.0
) -> plt.Figure:
    """Plot estimated pressure coefficient distribution."""
    xu, yu, xl, yl = coords
    
    # Simplified thin airfoil theory for Cp
    theta = np.arccos(1 - 2 * xu)
    cp_upper = -2 * (yu / np.maximum(yu.max(), 0.01)) * (1 + 0.5 * np.radians(alpha))
    cp_lower = 2 * (abs(yl) / np.maximum(abs(yl).max(), 0.01)) * (1 - 0.3 * np.radians(alpha))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xu, cp_upper, 'r-', linewidth=2, label='Upper Surface')
    ax.plot(xl, cp_lower, 'b-', linewidth=2, label='Lower Surface')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel('Pressure Coefficient Cp', fontsize=12)
    ax.set_title(f'Pressure Distribution (α = {alpha}°)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Aerodynamic convention
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_training_history(
    rewards: List[float],
    m_history: List[float] = None,
    p_history: List[float] = None,
    t_history: List[float] = None
) -> plt.Figure:
    """Plot RL training progress."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reward curve
    episodes = range(len(rewards))
    axes[0].plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Reward')
    
    # Moving average
    window = min(50, len(rewards) // 10 + 1)
    ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, len(rewards)), ma, 'b-', linewidth=2, label=f'Moving Avg ({window})')
    
    axes[0].set_xlabel('Episode', fontsize=11)
    axes[0].set_ylabel('Reward (L/D)', fontsize=11)
    axes[0].set_title('RL Training Progress', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Parameter evolution
    if m_history and p_history and t_history:
        axes[1].plot(m_history, label='Max Camber (m)', linewidth=2)
        axes[1].plot(p_history, label='Camber Position (p)', linewidth=2)
        axes[1].plot(t_history, label='Thickness (t)', linewidth=2)
        axes[1].set_xlabel('Episode', fontsize=11)
        axes[1].set_ylabel('Parameter Value', fontsize=11)
        axes[1].set_title('Parameter Evolution', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_wind_tunnel_validation(wt_results: dict) -> plt.Figure:
    """Plot CFD vs Wind Tunnel comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    alpha = wt_results['alpha']
    
    # Cl comparison
    axes[0].errorbar(alpha, wt_results['wt_cl'], yerr=wt_results['cl_err'], 
                    fmt='o', color='red', capsize=3, label='Wind Tunnel')
    axes[0].plot(alpha, wt_results['cfd_cl'], 'b--', linewidth=2, label='CFD (XFOIL)')
    axes[0].set_xlabel('Angle of Attack (°)', fontsize=11)
    axes[0].set_ylabel('Lift Coefficient Cl', fontsize=11)
    axes[0].set_title('Lift Coefficient Validation', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cd comparison
    axes[1].errorbar(alpha, wt_results['wt_cd'], yerr=wt_results['cd_err'],
                    fmt='o', color='red', capsize=3, label='Wind Tunnel')
    axes[1].plot(alpha, wt_results['cfd_cd'], 'b--', linewidth=2, label='CFD (XFOIL)')
    axes[1].set_xlabel('Angle of Attack (°)', fontsize=11)
    axes[1].set_ylabel('Drag Coefficient Cd', fontsize=11)
    axes[1].set_title('Drag Coefficient Validation', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_manufacturing_feasibility(
    m: float, p: float, t: float, is_valid: bool
) -> plt.Figure:
    """Plot manufacturing feasibility status."""
    # Simplified visualization since validation logic moved
    fig, ax = plt.subplots(figsize=(8, 2))
    color = 'green' if is_valid else 'red'
    text = "Valid" if is_valid else "Invalid"
    
    ax.barh([0], [1], color=color, alpha=0.6)
    ax.text(0.5, 0, f"Manufacturing Status: {text}", 
           ha='center', va='center', color='white', fontweight='bold', fontsize=14)
    ax.set_axis_off()
    return fig

if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    try:
        from src.aerodynamics.airfoil_gen import naca4
    except ImportError:
        # Fallback for direct execution
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from src.aerodynamics.airfoil_gen import naca4

    xu, yu, xl, yl = naca4(0.02, 0.4, 0.12)
    fig = plot_pressure_distribution((xu, yu, xl, yl), 0.02, 0.4, 0.12)
    plt.savefig('test_pressure.png', dpi=150)
    print("Test plot saved!")
