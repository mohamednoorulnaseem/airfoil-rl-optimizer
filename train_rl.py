"""
Training Script for Airfoil RL Optimizer

Uses the Stanford-level production PPO agent to optmize 
airfoil geometry for multi-objective performance.
"""

import os
from src.optimization.rl_agent import AirfoilRLAgent
try:
    from src.optimization.multi_objective_env import MultiObjectiveAirfoilEnv as AirfoilEnv
except ImportError:
    # Fallback/Compat if relative import fails or file missing
    import sys
    sys.path.append(os.path.dirname(__file__))
    try:
        from src.optimization.multi_objective_env import MultiObjectiveAirfoilEnv as AirfoilEnv
    except ImportError:    
        from airfoil_env import AirfoilEnv

def main():
    # Ensure models dir exists
    os.makedirs("models", exist_ok=True)
    
    print("Initializing Multi-Objective Environment...")
    env = AirfoilEnv()
    
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
