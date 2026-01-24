"""
Updated Airfoil Environment with XFOIL validation
"""

import gymnasium as gym
import numpy as np
from src.aerodynamics.xfoil_interface import XFOILRunner
from src.aerodynamics.airfoil_gen import generate_naca_4digit

class AirfoilEnvXFOIL(gym.Env):
    """
    RL environment using REAL CFD (XFOIL) instead of surrogate
    """
    
    def __init__(self, use_xfoil=True):
        super().__init__()
        
        self.use_xfoil = use_xfoil
        
        # XFOIL runner
        if self.use_xfoil:
            self.xfoil = XFOILRunner(reynolds=1e6, mach=0.0)
        
        # Action space: delta changes to m, p, t
        self.action_space = gym.spaces.Box(
            low=np.array([-0.005, -0.05, -0.01]),
            high=np.array([0.005, 0.05, 0.01]),
            dtype=np.float32
        )
        
        # State space: m, p, t + performance metrics
        self.observation_space = gym.spaces.Box(
            low=np.array([0.00, 0.10, 0.11, -2.0, 0.0]),
            high=np.array([0.06, 0.70, 0.18, 2.0, 0.1]),
            dtype=np.float32
        )
        
        # Baseline for comparison
        self.baseline_params = [0.02, 0.4, 0.12]  # NACA 2412
        self.baseline_ld = None
        self.current_params = self.baseline_params.copy()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start from baseline airfoil
        self.current_params = self.baseline_params.copy()
        
        # Calculate baseline performance (once)
        if self.baseline_ld is None:
            self.baseline_ld = self._evaluate_airfoil(self.baseline_params)
        
        # Get current performance
        current_ld = self._evaluate_airfoil(self.current_params)
        
        obs = np.array(self.current_params + [current_ld, 0.0], dtype=np.float32)
        
        return obs, {}
    
    def step(self, action):
        """
        Apply action and return new state
        """
        # Update parameters
        m, p, t = self.current_params
        dm, dp, dt = action
        
        m_new = np.clip(m + dm, 0.00, 0.06)
        p_new = np.clip(p + dp, 0.10, 0.70)
        t_new = np.clip(t + dt, 0.11, 0.18)
        
        self.current_params = [m_new, p_new, t_new]
        
        # Evaluate with XFOIL
        current_ld = self._evaluate_airfoil(self.current_params)
        
        # Calculate reward
        reward = self._calculate_reward(current_ld)
        
        # Observation
        obs = np.array(
            self.current_params + [current_ld, 0.0],
            dtype=np.float32
        )
        
        # Episode ends after 100 steps (handled by wrapper usually, but here we can rely on caller)
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _evaluate_airfoil(self, params):
        """
        Evaluate airfoil performance using XFOIL
        """
        m, p, t = params
        coords = generate_naca_4digit(m, p, t, n_points=100)
        
        if self.use_xfoil:
            # Real CFD with XFOIL
            results = self.xfoil.analyze_airfoil(
                coords,
                alpha_range=[0.0, 4.0, 8.0]  # Multiple angles
            )
            
            if results and len(results) >= 3:
                # Calculate mean L/D across angles
                ld_values = [r['cl']/r['cd'] for r in results if r['cd'] > 0]
                mean_ld = np.mean(ld_values) if ld_values else 0.0
                return mean_ld
            else:
                # XFOIL failed (bad airfoil geometry)
                return 0.0
        else:

            # Use Physics-Informed Neural Network Surrogate
            if not hasattr(self, 'pinn'):
                from src.aerodynamics.surrogate_model import get_pretrained_pinn
                self.pinn = get_pretrained_pinn()
            
            # Predict L/D
            # Assume 4 degrees AoA for optimization target
            _, _, ld = self.pinn.predict(m, p, t, alpha=4.0)
            return ld

    def _calculate_reward(self, current_ld):
        """
        Reward = improvement over baseline
        """
        if current_ld <= 0 or self.baseline_ld is None or self.baseline_ld <= 0:
            if current_ld <= 0:
                pass # Already handled 0.0 return
            return -10.0  # Heavy penalty for invalid airfoil
        
        # Improvement percentage
        improvement = ((current_ld - self.baseline_ld) / self.baseline_ld) * 100
        
        # Reward = improvement
        reward = improvement
        
        return reward
