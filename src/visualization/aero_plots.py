"""
Visualization tools for aerodynamic analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def plot_airfoil_geometry(coords, title="Airfoil Geometry", save_path=None):
    """
    Plot the airfoil geometry.
    
    Args:
        coords (np.ndarray): N x 2 array of x, y coordinates
        title (str): Plot title
        save_path (str/Path): Path to save the figure
    """
    plt.figure(figsize=(10, 3))
    plt.plot(coords[:,0], coords[:,1], 'b-', linewidth=2, label='Airfoil Surface')
    plt.plot(coords[:,0], coords[:,1], 'bo', markersize=2, alpha=0.5)
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_cp_distribution(x, cp, title="Cp Distribution", save_path=None):
    """
    Plot the pressure coefficient distribution.
    
    Args:
        x (np.ndarray): x-coordinates
        cp (np.ndarray): Cp values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, cp, 'r-', linewidth=2)
    plt.gca().invert_yaxis()  # Cp is typically plotted inverted
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('x/c')
    plt.ylabel('Cp')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_polars(alphas, cls, cds, title="Polars", save_path=None):
    """
    Plot CL-alpha and Polar (CL-CD) curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # CL-alpha
    ax1.plot(alphas, cls, 'b-o')
    ax1.set_xlabel('Alpha (deg)')
    ax1.set_ylabel('Cl')
    ax1.grid(True)
    ax1.set_title('Lift Curve')
    
    # Drag Polar
    ax2.plot(cds, cls, 'r-o')
    ax2.set_xlabel('Cd')
    ax2.set_ylabel('Cl')
    ax2.grid(True)
    ax2.set_title('Drag Polar')
    
    plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
         plt.close()
    else:
        plt.show()
