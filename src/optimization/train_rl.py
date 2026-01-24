"""
Training Script for Airfoil RL Optimizer

Uses the Stanford-level production PPO agent to optmize 
airfoil geometry for multi-objective performance.
"""

import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.optimization.rl_agent import AirfoilRLAgent
from src.optimization.airfoil_env import AirfoilEnvXFOIL as AirfoilEnv

def main():
    # Ensure models dir exists
    os.makedirs("models", exist_ok=True)
    
    print("Initializing Multi-Objective Environment...")
    # Use Surrogate Model for training (60x faster)
    env = AirfoilEnv(use_xfoil=False)
    
    print("Creating PPO Agent...")
    # Initialize agent wrapper
    agent = AirfoilRLAgent(
        env,
        model_path="models/ppo_airfoil_final.zip", # Will load if exists
        verbose=1
    )
    
    # Train
    print("Starting Training...")
    # Using 50,000 steps for production-level training
    results = agent.train(total_timesteps=50000, save_path="models/ppo_airfoil_final.zip")
    
    print(f"Training Complete!")
    print(f"Mean Reward: {results['mean_reward']:.2f}")
    
    # Validation Run
    print("\nRunning Validation Episode...")
    val_result = agent.optimize()
    
    print(f"Best L/D Achieved: {val_result['best_ld']:.2f}")
    if val_result['best_params'] is not None:
        m, p, t = val_result['best_params']
        print(f"Best Parameters: Camber={m:.3f}, Pos={p:.2f}, Thickness={t:.3f}")

if __name__ == "__main__":
    main()
