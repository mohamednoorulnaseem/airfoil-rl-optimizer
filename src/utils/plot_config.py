"""
Publication-Quality Plot Configuration

This module provides matplotlib configuration for conference/journal-quality plots.
Use before generating figures for papers, presentations, or portfolio.

Targets:
- AIAA Journal format
- IEEE conference papers
- Stanford ADL style
- 300 DPI publication standards

Author: Mohamed Noorul Naseem
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np


def configure_publication_plots(style='aiaa'):
    """
    Configure matplotlib for publication-quality plots.
    
    Args:
        style: 'aiaa', 'ieee', 'stanford', or 'nature'
    
    Usage:
        from src.utils.plot_config import configure_publication_plots
        configure_publication_plots('aiaa')
        # Now all plots will use publication settings
    """
    
    # Font settings (LaTeX-style)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 13
    
    # Use LaTeX rendering if available
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    except:
        plt.rcParams['text.usetex'] = False
    
    # Figure size (inches) - standard two-column format
    if style == 'aiaa':
        plt.rcParams['figure.figsize'] = (6.5, 4.5)  # AIAA two-column
    elif style == 'ieee':
        plt.rcParams['figure.figsize'] = (3.5, 2.5)  # IEEE single-column
    elif style == 'stanford':
        plt.rcParams['figure.figsize'] = (7, 5)      # Stanford ADL style
    else:  # nature, general
        plt.rcParams['figure.figsize'] = (6, 4)
    
    # DPI settings
    plt.rcParams['figure.dpi'] = 100        # Screen display
    plt.rcParams['savefig.dpi'] = 300       # Publication quality
    plt.rcParams['savefig.bbox'] = 'tight'  # No whitespace clipping
    plt.rcParams['savefig.pad_inches'] = 0.05
    
    # Line styles
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['axes.linewidth'] = 0.8
    
    # Grid
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.3
    
    # Legend
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.edgecolor'] = 'black'
    
    # Color cycle (colorblind-friendly)
    colors = [
        '#0173B2',  # Blue
        '#DE8F05',  # Orange
        '#029E73',  # Green
        '#CC78BC',  # Purple
        '#CA9161',  # Brown
        '#949494',  # Gray
        '#ECE133',  # Yellow
        '#56B4E9'   # Sky blue
    ]
    plt.rcParams['axes.prop_cycle'] = cycler('color', colors)
    
    # Remove top and right spines (cleaner look)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Tight layout by default
    plt.rcParams['figure.autolayout'] = True
    
    print(f"✅ Publication-quality plot settings configured ({style} style)")
    print(f"   Figure size: {plt.rcParams['figure.figsize']}")
    print(f"   Save DPI: {plt.rcParams['savefig.dpi']}")
    print(f"   Font: {plt.rcParams['font.family']}, {plt.rcParams['font.size']}pt")


def create_multi_panel_figure(nrows=2, ncols=2, figsize=None, style='aiaa'):
    """
    Create multi-panel figure with consistent styling.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size (inches), auto if None
        style: Publication style
    
    Returns:
        fig, axes: Figure and axes array
    
    Example:
        fig, axes = create_multi_panel_figure(2, 2)
        axes[0, 0].plot(x, y1)
        axes[0, 1].plot(x, y2)
        fig.savefig('results.pdf')
    """
    configure_publication_plots(style)
    
    if figsize is None:
        # Auto-size based on panels
        if style == 'aiaa':
            figsize = (6.5, 2.5 * nrows)
        elif style == 'ieee':
            figsize = (3.5, 2.0 * nrows)
        else:
            figsize = (4 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Add panel labels (a), (b), (c), etc.
    if nrows * ncols > 1:
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        for idx, ax in enumerate(axes_flat):
            label = chr(97 + idx)  # 'a', 'b', 'c', ...
            ax.text(
                -0.1, 1.05, f'({label})',
                transform=ax.transAxes,
                fontsize=11,
                fontweight='bold',
                va='top'
            )
    
    return fig, axes


def save_publication_figure(fig, filename, formats=['pdf', 'png', 'eps']):
    """
    Save figure in multiple formats for publication.
    
    Args:
        fig: Matplotlib figure
        filename: Base filename (without extension)
        formats: List of formats to save
    
    Example:
        save_publication_figure(fig, 'results/airfoil_comparison')
        # Saves: airfoil_comparison.pdf, .png, .eps
    """
    import os
    
    # Create directory if needed
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    for fmt in formats:
        output_path = f"{filename}.{fmt}"
        fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
        
        # Get file size
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"✅ Saved: {output_path} ({size_mb:.2f} MB)")


def plot_airfoil_comparison(baseline_coords, optimized_coords, 
                           baseline_name='NACA 2412', 
                           optimized_name='RL-Optimized'):
    """
    Create publication-quality airfoil geometry comparison.
    
    Returns:
        fig: Figure object
    """
    configure_publication_plots('aiaa')
    
    fig, ax = plt.subplots(figsize=(6.5, 3))
    
    # Plot airfoils
    ax.plot(baseline_coords[:, 0], baseline_coords[:, 1], 
            'k--', linewidth=1.5, label=baseline_name)
    ax.plot(optimized_coords[:, 0], optimized_coords[:, 1], 
            'r-', linewidth=2, label=optimized_name)
    
    # Formatting
    ax.set_xlabel('$x/c$')
    ax.set_ylabel('$y/c$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('Airfoil Geometry Comparison')
    
    # Axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 0.15)
    
    return fig


def plot_polar_comparison(alpha_baseline, cl_baseline, cd_baseline,
                         alpha_optimized, cl_optimized, cd_optimized,
                         baseline_name='NACA 2412',
                         optimized_name='RL-Optimized'):
    """
    Create publication-quality polar curve comparison.
    
    Returns:
        fig: Figure object
    """
    configure_publication_plots('aiaa')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
    
    # Cl vs alpha
    ax1.plot(alpha_baseline, cl_baseline, 'ko-', 
             linewidth=1.5, markersize=4, label=baseline_name)
    ax1.plot(alpha_optimized, cl_optimized, 'rs-', 
             linewidth=2, markersize=5, label=optimized_name)
    ax1.set_xlabel(r'Angle of Attack $\alpha$ (deg)')
    ax1.set_ylabel(r'Lift Coefficient $C_l$')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('(a) Lift Curve')
    
    # Cl vs Cd (drag polar)
    ax2.plot(cd_baseline, cl_baseline, 'ko-', 
             linewidth=1.5, markersize=4, label=baseline_name)
    ax2.plot(cd_optimized, cl_optimized, 'rs-', 
             linewidth=2, markersize=5, label=optimized_name)
    ax2.set_xlabel(r'Drag Coefficient $C_d$')
    ax2.set_ylabel(r'Lift Coefficient $C_l$')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('(b) Drag Polar')
    
    plt.tight_layout()
    
    return fig


def plot_optimization_history(iterations, rewards, ld_ratios):
    """
    Create publication-quality optimization convergence plot.
    
    Returns:
        fig: Figure object
    """
    configure_publication_plots('aiaa')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5))
    
    # Reward history
    ax1.plot(iterations, rewards, 'b-', linewidth=1.5)
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Mean Reward')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('(a) RL Training Convergence')
    
    # L/D history
    ax2.plot(iterations, ld_ratios, 'r-', linewidth=1.5)
    ax2.axhline(y=56.7, color='k', linestyle='--', linewidth=1, label='Baseline NACA 2412')
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Lift-to-Drag Ratio $L/D$')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('(b) Aerodynamic Performance')
    
    plt.tight_layout()
    
    return fig


def plot_aircraft_benchmark(aircraft_names, ld_baseline, ld_optimized):
    """
    Create publication-quality aircraft comparison bar chart.
    
    Returns:
        fig: Figure object
    """
    configure_publication_plots('aiaa')
    
    fig, ax = plt.subplots(figsize=(6.5, 4))
    
    x = np.arange(len(aircraft_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ld_baseline, width, 
                   label='Baseline', color='#0173B2')
    bars2 = ax.bar(x + width/2, ld_optimized, width, 
                   label='RL-Optimized', color='#DE8F05')
    
    # Add improvement percentages
    for i, (b, o) in enumerate(zip(ld_baseline, ld_optimized)):
        if o > 0:
            improvement = ((o - b) / b) * 100
            ax.text(i, max(b, o) + 0.5, f'+{improvement:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Aircraft')
    ax.set_ylabel('Cruise Lift-to-Drag Ratio $L/D$')
    ax.set_title('Aerodynamic Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(aircraft_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig


# Quick usage example
if __name__ == '__main__':
    # Test configuration
    configure_publication_plots('aiaa')
    
    # Example plot
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.sin(x) * 1.2
    
    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Baseline')
    ax.plot(x, y2, label='Optimized')
    ax.set_xlabel('$x$ (units)')
    ax.set_ylabel('$y$ (units)')
    ax.set_title('Example Publication Plot')
    ax.legend()
    ax.grid(True)
    
    # Save in multiple formats
    save_publication_figure(fig, 'test_plot', formats=['pdf', 'png'])
    
    print("\n✅ Test plot created successfully!")
    print("   Output: test_plot.pdf, test_plot.png")
