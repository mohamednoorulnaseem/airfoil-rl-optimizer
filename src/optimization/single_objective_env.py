"""
Airfoil RL Environment - Production Grade (Single Objective Version)

Custom Gymnasium environment for optimizing NACA-like airfoil parameters
using reinforcement learning. Now includes:
- Multi-angle aerodynamic evaluation
- Manufacturing constraints in reward function
- Improved observation space

Author: Mohamed Noorul Naseem
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.aerodynamics.legacy_eval import aero_score, aero_score_multi
from src.validation.manufacturing import get_manufacturing_penalty, check_manufacturability

class AirfoilEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, include_manufacturing: bool = True, strict_mode: bool = False):
        super().__init__()
        self.include_manufacturing = include_manufacturing
        self.strict_mode = strict_mode
        self.param_low = np.array([0.0, 0.1, 0.10], dtype=np.float32)
        self.param_high = np.array([0.04, 0.7, 0.20], dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-0.005, -0.05, -0.01], dtype=np.float32),
            high=np.array([0.005, 0.05, 0.01], dtype=np.float32),
            dtype=np.float32,
        )
        obs_low = np.array([0.0, 0.1, 0.08, -0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([0.06, 0.8, 0.25, 2.0, 0.1, 100.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.max_steps = 30
        self.current_step = 0
        self.params = None
        self.baseline = np.array([0.02, 0.4, 0.12], dtype=np.float32)
        self.alphas = np.array([0.0, 4.0, 8.0], dtype=np.float32)
        self.best_ld = 0.0
        self.episode_rewards = []

    def _get_observation(self) -> np.ndarray:
        m, p, t = self.params
        cls, cds, lds = aero_score_multi(m, p, t, alphas=self.alphas)
        mid_idx = len(self.alphas) // 2
        Cl_mid = cls[mid_idx]
        Cd_mid = cds[mid_idx]
        ld_mean = float(np.mean(lds))
        is_valid, _ = check_manufacturability(m, p, t)
        mfg_score = 1.0 if is_valid else 0.5
        obs = np.array([m, p, t, Cl_mid, Cd_mid, ld_mean, mfg_score], dtype=np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.best_ld = 0.0
        self.episode_rewards = []
        self.params = self.baseline.copy()
        obs = self._get_observation()
        info = {"reset": True}
        return obs, info

    def step(self, action):
        self.current_step += 1
        self.params = self.params + action
        self.params = np.clip(self.params, self.param_low, self.param_high)
        m, p, t = self.params
        cls, cds, lds = aero_score_multi(m, p, t, alphas=self.alphas)
        mid_idx = len(self.alphas) // 2
        Cl_mid = cls[mid_idx]
        Cd_mid = cds[mid_idx]
        ld_mean = float(np.mean(lds))
        reward = ld_mean
        reward += 0.5 * Cl_mid
        deviation = float(np.sum((self.params - self.baseline) ** 2))
        reward -= 0.05 * deviation
        if self.include_manufacturing:
            mfg_penalty = get_manufacturing_penalty(m, p, t)
            reward -= 0.5 * mfg_penalty
        if ld_mean > self.best_ld:
            reward += 2.0 * (ld_mean - self.best_ld)
            self.best_ld = ld_mean
        self.episode_rewards.append(reward)
        obs = self._get_observation()
        terminated = False
        if self.strict_mode:
            is_valid, _ = check_manufacturability(m, p, t)
            if not is_valid:
                terminated = True
                reward -= 10.0
        truncated = self.current_step >= self.max_steps
        info = {
            "Cl": Cl_mid, "Cd": Cd_mid, "L/D": ld_mean, "alphas": self.alphas.tolist(),
            "L/D_all": lds, "params": self.params.tolist(), "step": self.current_step,
            "best_ld": self.best_ld, "manufacturing_valid": check_manufacturability(m, p, t)[0],
        }
        return obs, reward, terminated, truncated, info

    def render(self): pass
    
    def get_episode_summary(self) -> dict:
        return {
            "total_reward": sum(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "best_ld": self.best_ld,
            "final_params": self.params.tolist(),
            "manufacturing_valid": check_manufacturability(*self.params)[0],
            "steps": self.current_step,
        }
