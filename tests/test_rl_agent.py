"""
Tests for RL Agent
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_env_init():
    """Test environment initialization."""
    from airfoil_env import AirfoilEnv
    
    env = AirfoilEnv()
    obs, info = env.reset()
    
    assert obs.shape == env.observation_space.shape
    assert env.params is not None


def test_env_step():
    """Test environment step."""
    from airfoil_env import AirfoilEnv
    
    env = AirfoilEnv()
    obs, info = env.reset()
    
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    
    assert obs2.shape == obs.shape
    assert isinstance(reward, (int, float))
    assert 'L/D' in info


def test_env_bounds():
    """Test that parameters stay in bounds."""
    from airfoil_env import AirfoilEnv
    
    env = AirfoilEnv()
    env.reset()
    
    # Push parameters to extremes
    for _ in range(100):
        action = np.array([0.01, 0.1, 0.02])  # Max action
        env.step(action)
    
    m, p, t = env.params
    assert env.param_low[0] <= m <= env.param_high[0]
    assert env.param_low[1] <= p <= env.param_high[1]
    assert env.param_low[2] <= t <= env.param_high[2]


def test_manufacturing_penalty():
    """Test manufacturing constraints in reward."""
    from airfoil_env import AirfoilEnv
    
    env = AirfoilEnv(include_manufacturing=True)
    env.reset()
    
    # Force parameters to manufacturing violation
    env.params = np.array([0.06, 0.1, 0.08])  # Extreme values
    obs, reward, _, _, info = env.step(np.zeros(3))
    
    # Should have manufacturing-related info
    assert 'manufacturing_valid' in info


def test_episode_completion():
    """Test episode terminates correctly."""
    from airfoil_env import AirfoilEnv
    
    env = AirfoilEnv()
    env.reset()
    
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
        if steps > 100:
            break
    
    assert steps <= env.max_steps + 1


def test_model_loading():
    """Test loading pretrained model."""
    from airfoil_env import AirfoilEnv
    from stable_baselines3 import PPO
    import os
    
    model_path = "models/ppo_airfoil_fake.zip"
    if os.path.exists(model_path):
        env = AirfoilEnv()
        model = PPO.load(model_path)
        
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        assert action.shape == env.action_space.shape


if __name__ == "__main__":
    print("Running RL Agent tests...")
    test_env_init()
    print("✓ Environment init")
    test_env_step()
    print("✓ Environment step")
    test_env_bounds()
    print("✓ Parameter bounds")
    test_manufacturing_penalty()
    print("✓ Manufacturing penalty")
    test_episode_completion()
    print("✓ Episode completion")
    test_model_loading()
    print("✓ Model loading")
    print("\nAll tests passed!")
