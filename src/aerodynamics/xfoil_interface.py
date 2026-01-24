"""
XFOIL CFD Interface
Real aerodynamic analysis for airfoils
"""

import subprocess
import os
import numpy as np
from pathlib import Path
import re

class XFOILRunner:
    """
    Python wrapper for XFOIL CFD solver
    """
    
    def __init__(self, reynolds=1e6, mach=0.0, n_iter=100):
        """
        Args:
            reynolds: Reynolds number (1e6 = typical small aircraft)
            mach: Mach number (0.0-0.3 for low speed)
            n_iter: Max iterations for convergence
        """
        self.reynolds = reynolds
        self.mach = mach
        self.n_iter = n_iter
        self.temp_dir = Path("./temp_xfoil")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Find XFOIL executable
        self.xfoil_path = self._find_xfoil()
    
    def _find_xfoil(self):
        """Find XFOIL executable in various locations"""
        # Check common locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "scripts" / "xfoil.exe",  # scripts folder
            Path(__file__).parent.parent.parent / "xfoil.exe",  # project root
            Path("scripts/xfoil.exe"),  # relative scripts
            Path("xfoil.exe"),  # current directory
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path.absolute())
        
        # Fall back to system PATH
        return "xfoil"
        
    def analyze_airfoil(self, airfoil_coords, alpha_range=None):
        """
        Run XFOIL analysis for given airfoil
        
        Args:
            airfoil_coords: numpy array of shape (N, 2) with x, y coordinates
            alpha_range: list of angles of attack (degrees)
            
        Returns:
            dict with cl, cd, cm for each alpha
        """
        if alpha_range is None:
            alpha_range = [0.0, 2.0, 4.0, 6.0, 8.0]
        
        # Write airfoil coordinates to file
        airfoil_file = self.temp_dir / "airfoil.dat"
        self._write_airfoil(airfoil_coords, airfoil_file)
        
        # Create XFOIL command file
        polar_file = self.temp_dir / "polar.txt"
        commands = self._create_commands(airfoil_file, polar_file, alpha_range)
        
        # Run XFOIL
        success = self._run_xfoil(commands)
        
        if not success:
            return None
        
        # Parse results
        results = self._parse_polar(polar_file)
        
        return results
    
    def _write_airfoil(self, coords, filename):
        """Write airfoil coordinates to file"""
        with open(filename, 'w') as f:
            f.write(f"Generated Airfoil\n")
            for x, y in coords:
                f.write(f"  {x:10.6f}  {y:10.6f}\n")
    
    def _create_commands(self, airfoil_file, polar_file, alpha_range):
        """
        Create XFOIL command sequence
        """
        commands = f"""
LOAD {airfoil_file}

PANE

OPER
VISC {self.reynolds}
MACH {self.mach}
ITER {self.n_iter}

PACC
{polar_file}


"""
        
        # Add each angle of attack
        for alpha in alpha_range:
            commands += f"ALFA {alpha}\n"
        
        commands += "\n\nQUIT\n"
        
        return commands
    
    def _run_xfoil(self, commands):
        """
        Execute XFOIL with given commands
        """
        try:
            process = subprocess.run(
                [self.xfoil_path],
                input=commands.encode(),
                capture_output=True,
                timeout=30,
                cwd=str(self.temp_dir)
            )
            
            # Check if polar file was created
            polar_file = self.temp_dir / "polar.txt"
            return polar_file.exists()
            
        except subprocess.TimeoutExpired:
            print("XFOIL timeout - airfoil may not have converged")
            return False
        except Exception as e:
            print(f"XFOIL error: {e}")
            return False
    
    def _parse_polar(self, polar_file):
        """
        Parse XFOIL polar output file
        
        Returns:
            list of dicts with alpha, cl, cd, cm, etc.
        """
        if not polar_file.exists():
            return []
        
        results = []
        
        with open(polar_file, 'r') as f:
            lines = f.readlines()
        
        # Find data section (starts after header)
        data_started = False
        for line in lines:
            # Skip header lines
            if '---' in line:
                data_started = True
                continue
            
            if not data_started:
                continue
            
            # Parse data line
            # Format: alpha  CL  CD  CDp  CM  Top_Xtr  Bot_Xtr
            parts = line.split()
            
            if len(parts) >= 7:
                try:
                    results.append({
                        'alpha': float(parts[0]),
                        'cl': float(parts[1]),
                        'cd': float(parts[2]),
                        'cdp': float(parts[3]),    # Pressure drag
                        'cm': float(parts[4]),     # Pitching moment
                        'top_xtr': float(parts[5]), # Top transition
                        'bot_xtr': float(parts[6])  # Bottom transition
                    })
                except ValueError:
                    continue
        
        return results
    
    def get_ld_max(self, airfoil_coords):
        """
        Find maximum L/D ratio
        """
        results = self.analyze_airfoil(airfoil_coords)
        
        if not results:
            return None
        
        # Calculate L/D for each point
        ld_ratios = [(r['cl'] / r['cd'], r['alpha']) for r in results if r['cd'] > 0]
        
        if not ld_ratios:
            return None
        
        max_ld, alpha_for_max_ld = max(ld_ratios, key=lambda x: x[0])
        
        return {
            'ld_max': max_ld,
            'alpha_for_ld_max': alpha_for_max_ld,
            'all_results': results
        }
    
    def cleanup(self):
        """Remove temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


# Convenience function
def run_xfoil_quick(airfoil_coords, alpha=4.0, reynolds=1e6):
    """
    Quick XFOIL analysis at single angle of attack
    
    Returns: cl, cd, ld
    """
    xfoil = XFOILRunner(reynolds=reynolds)
    results = xfoil.analyze_airfoil(airfoil_coords, alpha_range=[alpha])
    xfoil.cleanup()
    
    if results and len(results) > 0:
        r = results[0]
        return r['cl'], r['cd'], r['cl']/r['cd']
    else:
        return None, None, None
