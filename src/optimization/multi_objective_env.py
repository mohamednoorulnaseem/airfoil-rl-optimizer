"""
Multi-Objective Reinforcement Learning Environment

Pareto-optimal optimization for:
1. Maximize L/D (cruise efficiency)
2. Maximize Cl_max (takeoff performance)
3. Minimize pitching moment (stability)
4. Maximize manufacturing score (buildability)

Based on Stanford ADL multi-objective aerospace optimization research.

Author: Mohamed Noorul Naseem
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from aero_eval import aero_score_multi
from manufacturing_constraints import check_manufacturability, get_manufacturing_penalty


@dataclass
class ObjectiveWeights:
    """Weights for multi-objective optimization."""
    ld_ratio: float = 0.40      # Cruise efficiency
    cl_max: float = 0.25        # High-lift capability
    stability: float = 0.20    # Pitching moment
    manufacturing: float = 0.15  # Buildability


class MultiObjectiveAirfoilEnv(gym.Env):
    """
    Multi-objective RL environment for Pareto-optimal airfoil design.
    
    State Space (9D):
        [m, p, t, Cl_cruise, Cd_cruise, Cl_max, Cm, L/D, mfg_score]
    
    Action Space:
        Continuous delta changes to [m, p, t]
    
    Reward:
        Weighted sum of normalized objectives, dynamically adjusted
        to encourage Pareto frontier exploration.
    
    Features:
        - Multi-angle evaluation (cruise + high-lift conditions)
        - Pareto dominance tracking
        - Adaptive objective weighting
        - Manufacturing feasibility constraints
    """

    metadata = {"render_modes": []}

    def __init__(
        self, 
        weights: ObjectiveWeights = None,
        adaptive_weights: bool = True,
        pareto_tracking: bool = True
    ):
        super().__init__()
        
        self.weights = weights or ObjectiveWeights()
        self.adaptive_weights = adaptive_weights
        self.pareto_tracking = pareto_tracking
        
        # Parameter bounds
        self.param_low = np.array([0.0, 0.1, 0.10], dtype=np.float32)
        self.param_high = np.array([0.05, 0.65, 0.20], dtype=np.float32)
        
        # Action space: small continuous changes
        self.action_space = spaces.Box(
            low=np.array([-0.004, -0.04, -0.008], dtype=np.float32),
            high=np.array([0.004, 0.04, 0.008], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Observation space: 9D state
        obs_low = np.array([0.0, 0.1, 0.08, -0.5, 0.0, 0.0, -0.5, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([0.06, 0.8, 0.25, 2.0, 0.1, 2.5, 0.5, 100.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Flight conditions
        self.cruise_alphas = np.array([0.0, 4.0, 8.0], dtype=np.float32)
        self.high_lift_alphas = np.array([8.0, 10.0, 12.0], dtype=np.float32)
        
        # Baseline (NACA 2412)
        self.baseline = np.array([0.02, 0.4, 0.12], dtype=np.float32)
        
        # Episode tracking
        self.max_steps = 40
        self.current_step = 0
        self.params = None
        
        # Pareto front tracking
        self.pareto_front: List[Dict] = []
        self.episode_history: List[Dict] = []
        
        # Best values per objective (for normalization)
        self.best_ld = 30.0
        self.best_cl_max = 1.2
        self.best_stability = 0.0
        self.best_manufacturing = 1.0

    def _evaluate_objectives(self) -> Dict[str, float]:
        """Evaluate all optimization objectives."""
        m, p, t = self.params
        
        # Cruise performance (0°, 4°, 8°)
        cls_cruise, cds_cruise, lds_cruise = aero_score_multi(m, p, t, alphas=self.cruise_alphas)
        
        # High-lift performance (8°, 10°, 12°)
        cls_high, cds_high, lds_high = aero_score_multi(m, p, t, alphas=self.high_lift_alphas)
        
        # Pitching moment estimate (simplified)
        cm = -0.025 - 0.1 * m - 0.05 * (p - 0.4)
        
        # Manufacturing score
        is_valid, results = check_manufacturability(m, p, t)
        mfg_score = 1.0 if is_valid else 0.5 - 0.1 * sum(1 for r in results.values() if not r['passed'])
        mfg_score = max(0.0, mfg_score)
        
        return {
            'cl_cruise': cls_cruise[1],  # At 4°
            'cd_cruise': cds_cruise[1],
            'ld_mean': float(np.mean(lds_cruise)),
            'cl_max': max(cls_high),
            'cm': cm,
            'manufacturing_score': mfg_score,
            'is_manufacturable': is_valid,
        }

    def _get_observation(self, objectives: Dict) -> np.ndarray:
        """Build observation from objectives."""
        m, p, t = self.params
        
        obs = np.array([
            m, p, t,
            objectives['cl_cruise'],
            objectives['cd_cruise'],
            objectives['cl_max'],
            objectives['cm'],
            objectives['ld_mean'],
            objectives['manufacturing_score']
        ], dtype=np.float32)
        
        return obs

    def _calculate_reward(self, objectives: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-objective reward with Pareto optimization.
        
        Uses weighted scalarization with adaptive weights:
        R = Σ wi * normalize(objective_i)
        """
        # Normalize objectives to [0, 1] range
        ld_normalized = min(objectives['ld_mean'] / self.best_ld, 1.5)
        cl_max_normalized = min(objectives['cl_max'] / self.best_cl_max, 1.5)
        stability_normalized = 1.0 - min(abs(objectives['cm']) / 0.1, 1.0)
        mfg_normalized = objectives['manufacturing_score']
        
        # Weighted sum
        w = self.weights
        reward = (
            w.ld_ratio * ld_normalized +
            w.cl_max * cl_max_normalized +
            w.stability * stability_normalized +
            w.manufacturing * mfg_normalized
        )
        
        # Pareto improvement bonus
        if self.pareto_tracking and self._is_pareto_improvement(objectives):
            reward += 1.0
        
        # Update best values
        self.best_ld = max(self.best_ld, objectives['ld_mean'])
        self.best_cl_max = max(self.best_cl_max, objectives['cl_max'])
        
        components = {
            'ld_component': w.ld_ratio * ld_normalized,
            'cl_max_component': w.cl_max * cl_max_normalized,
            'stability_component': w.stability * stability_normalized,
            'manufacturing_component': w.manufacturing * mfg_normalized,
        }
        
        return float(reward), components

    def _is_pareto_improvement(self, objectives: Dict) -> bool:
        """Check if current solution is Pareto-dominant."""
        if not self.pareto_front:
            return True
        
        current = np.array([
            objectives['ld_mean'],
            objectives['cl_max'],
            -abs(objectives['cm']),  # Negate (minimize magnitude)
            objectives['manufacturing_score']
        ])
        
        for point in self.pareto_front:
            existing = np.array([
                point['ld_mean'],
                point['cl_max'],
                -abs(point['cm']),
                point['manufacturing_score']
            ])
            
            # Check if current dominates existing
            if np.all(current >= existing) and np.any(current > existing):
                return True
        
        return False

    def _update_pareto_front(self, objectives: Dict):
        """Update Pareto front with new solution."""
        self.pareto_front.append(objectives.copy())
        
        # Remove dominated solutions
        new_front = []
        for point in self.pareto_front:
            dominated = False
            for other in self.pareto_front:
                if point == other:
                    continue
                # Check if other dominates point
                if (other['ld_mean'] >= point['ld_mean'] and
                    other['cl_max'] >= point['cl_max'] and
                    abs(other['cm']) <= abs(point['cm']) and
                    other['manufacturing_score'] >= point['manufacturing_score']):
                    if (other['ld_mean'] > point['ld_mean'] or
                        other['cl_max'] > point['cl_max'] or
                        abs(other['cm']) < abs(point['cm']) or
                        other['manufacturing_score'] > point['manufacturing_score']):
                        dominated = True
                        break
            if not dominated:
                new_front.append(point)
        
        self.pareto_front = new_front[-50:]  # Keep last 50

    def reset(self, *, seed=None, options=None):
        """Reset to baseline state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_history = []
        
        # Start from baseline
        self.params = self.baseline.copy()
        
        objectives = self._evaluate_objectives()
        obs = self._get_observation(objectives)
        
        return obs, {'objectives': objectives}

    def step(self, action):
        """Execute one optimization step."""
        self.current_step += 1
        
        # Apply action
        self.params = self.params + action
        self.params = np.clip(self.params, self.param_low, self.param_high)
        
        # Evaluate
        objectives = self._evaluate_objectives()
        obs = self._get_observation(objectives)
        reward, reward_components = self._calculate_reward(objectives)
        
        # Update Pareto front
        if self.pareto_tracking:
            self._update_pareto_front(objectives)
        
        # Track history
        self.episode_history.append({
            'step': self.current_step,
            'params': self.params.tolist(),
            'objectives': objectives,
            'reward': reward
        })
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            'objectives': objectives,
            'reward_components': reward_components,
            'pareto_front_size': len(self.pareto_front),
            'is_pareto_optimal': self._is_pareto_improvement(objectives),
        }
        
        return obs, reward, terminated, truncated, info

    def get_pareto_front(self) -> List[Dict]:
        """Get current Pareto front solutions."""
        return self.pareto_front.copy()

    def render(self):
        pass


class WeightedSumEnv(MultiObjectiveAirfoilEnv):
    """
    Variant with user-specified fixed weights.
    Useful for generating specific trade-off solutions.
    """
    
    def __init__(self, **kwargs):
        super().__init__(adaptive_weights=False, **kwargs)


class HierarchicalEnv(MultiObjectiveAirfoilEnv):
    """
    Hierarchical objective optimization.
    
    Priority:
    1. Manufacturing feasibility (constraint)
    2. L/D optimization (primary)
    3. Cl_max (secondary)
    4. Stability (tertiary)
    """
    
    def _calculate_reward(self, objectives: Dict) -> Tuple[float, Dict]:
        # Manufacturing is a hard constraint
        if not objectives['is_manufacturable']:
            return -5.0, {'penalty': 'manufacturing_violation'}
        
        # Hierarchical reward
        reward = 0.0
        
        # Primary: L/D
        reward += 0.6 * objectives['ld_mean'] / 50.0
        
        # Secondary: Cl_max (only if L/D threshold met)
        if objectives['ld_mean'] > 25:
            reward += 0.3 * objectives['cl_max'] / 1.5
        
        # Tertiary: Stability (only if above secondary threshold)
        if objectives['cl_max'] > 1.0:
            reward += 0.1 * (1.0 - abs(objectives['cm']) * 10)
        
        return reward, {}


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Objective Airfoil Environment")
    print("=" * 60)
    
    env = MultiObjectiveAirfoilEnv()
    obs, info = env.reset()
    
    print(f"\nInitial State:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  L/D: {info['objectives']['ld_mean']:.1f}")
    print(f"  Cl_max: {info['objectives']['cl_max']:.3f}")
    print(f"  Cm: {info['objectives']['cm']:.4f}")
    print(f"  Manufacturing: {info['objectives']['manufacturing_score']:.2f}")
    
    # Run episode
    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
    
    print(f"\nAfter 10 steps:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Pareto front size: {len(env.get_pareto_front())}")
    print(f"  Final L/D: {info['objectives']['ld_mean']:.1f}")
