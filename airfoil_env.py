import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aero_eval import aero_score, aero_score_multi  # use both single & multi angle


class AirfoilEnv(gym.Env):
    """
    RL environment for optimizing NACA-like airfoil parameters [m, p, t].
    State:  [m, p, t, Cl_mid, Cd_mid]  (mid-angle, e.g. 4 deg)
    Action: delta changes to [m, p, t]
    Reward: mean Lift-to-drag ratio L/D across several angles of attack,
            plus a lift bonus at mid AoA.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Bounds for parameters [m, p, t]
        #   m in [0.0, 0.04]
        #   p in [0.1, 0.7]
        #   t in [0.11, 0.18]  (no super-thin airfoils)
        self.param_low = np.array([0.0, 0.1, 0.11], dtype=np.float32)
        self.param_high = np.array([0.04, 0.7, 0.18], dtype=np.float32)

        # Action: small changes in [m, p, t]
        self.action_space = spaces.Box(
            low=np.array([-0.005, -0.05, -0.01], dtype=np.float32),
            high=np.array([0.005, 0.05, 0.01], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: [m, p, t, Cl_mid, Cd_mid]
        obs_low = np.concatenate(
            [self.param_low, np.array([0.0, 0.0], dtype=np.float32)]
        )
        obs_high = np.concatenate(
            [self.param_high, np.array([2.0, 0.5], dtype=np.float32)]
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self.max_steps = 30
        self.current_step = 0
        self.params = None  # [m, p, t]

        # Angles of attack (deg) to optimize over
        self.alphas = np.array([0.0, 4.0, 8.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # baseline like NACA 2412
        self.params = np.array([0.02, 0.4, 0.12], dtype=np.float32)

        # Use mid angle (4 deg) for state features
        Cl_mid, Cd_mid = aero_score(*self.params)
        obs = np.concatenate(
            [self.params, np.array([Cl_mid, Cd_mid], dtype=np.float32)]
        )
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1

        # update params with action and clip
        self.params = self.params + action
        self.params = np.clip(self.params, self.param_low, self.param_high)

        m, p, t = self.params

        # Multi-angle evaluation: 0, 4, 8 degrees
        cls, cds, lds = aero_score_multi(m, p, t, alphas=self.alphas)

        # Use mid angle (4 deg) for observation features
        mid_idx = len(self.alphas) // 2
        Cl_mid = cls[mid_idx]
        Cd_mid = cds[mid_idx]

        # Reward = mean L/D + lift bonus (prevents collapse into thin airfoils)
        ld_mean = float(np.mean(lds))
        reward = ld_mean + 0.5 * Cl_mid  # encourages decent lift at mid AoA

        # Softer penalty for moving far from baseline
        baseline = np.array([0.02, 0.4, 0.12], dtype=np.float32)
        reward -= 0.05 * float(np.sum((self.params - baseline) ** 2))

        # observation uses mid-angle Cl, Cd
        obs = np.concatenate(
            [self.params, np.array([Cl_mid, Cd_mid], dtype=np.float32)]
        )

        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {
            "Cl": Cl_mid,
            "Cd": Cd_mid,
            "L/D": ld_mean,
            "alphas": self.alphas,
            "L/D_all": lds,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass
