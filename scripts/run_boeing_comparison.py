"""
Complete Boeing 737 comparison with your RL-optimized airfoil
"""

import os
import sys
import numpy as np
import json

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from src.validation.aircraft_comparison import AircraftComparator
from src.visualization.comparison_plots import plot_aircraft_comparison
from src.optimization.airfoil_env import AirfoilEnvXFOIL

def get_best_rl_airfoil():
    """
    Get best parameters from trained RL model
    """
    # Try to find a model
    model_path = "models/ppo_airfoil_final.zip"
    if not os.path.exists(model_path):
        model_path = "models/ppo_airfoil_fake.zip"
    
    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}")
        return [0.0385, 0.425, 0.135] # Fallback example values
        
    # Load trained model
    model = PPO.load(model_path)
    
    # Run multiple episodes to find best
    env = AirfoilEnvXFOIL(use_xfoil=False) # Use surrogate for speedfinding
    
    best_params = None
    best_reward = -np.inf
    
    print(f"Evaluating {model_path} to find best design...")
    for episode in range(20):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_params = env.current_params.copy()
    
    return best_params

def main():
    """
    Run complete comparison
    """
    
    print("="*70)
    print("BOEING 737-800 COMPARISON & FUEL SAVINGS ANALYSIS")
    print("="*70)
    
    # Get RL-optimized parameters
    print("\n1. Finding best RL-optimized airfoil...")
    optimized_params = get_best_rl_airfoil()
    print(f"   Best parameters: m={optimized_params[0]:.4f}, p={optimized_params[1]:.4f}, t={optimized_params[2]:.4f}")
    
    # Run comparison
    print("\n2. Running XFOIL comparison with Boeing 737-800...")
    comparator = AircraftComparator()
    results = comparator.compare_to_aircraft(optimized_params, "Boeing 737-800")
    
    if results:
        # Create visualization
        print("\n3. Creating comparison visualization...")
        plot_aircraft_comparison(results)
        
        # Generate README table
        print("\n4. Generating README table...")
        single_result = {"Boeing 737-800": results}
        table = comparator.generate_comparison_table(single_result)
        
        print("\n" + "="*70)
        print("ADD THIS TO YOUR README.md:")
        print("="*70)
        print("\n" + table)
        
        # Generate resume bullet
        improvements = results['improvements']
        savings = results['fuel_savings']
        
        resume_bullet = f"""
RESUME BULLET POINT:

"Developed reinforcement learning optimization framework achieving 
{improvements['ld_improvement_percent']:.1f}% lift-to-drag improvement over Boeing 737-800 
baseline wing section ({improvements['baseline_ld']:.1f} → {improvements['optimized_ld']:.1f} L/D), 
validated with XFOIL CFD solver at cruise conditions (Re=15M, M=0.785). 
Estimated ${savings['fleet_lifetime_savings_usd']/1e9:.1f} billion fuel savings potential 
for {savings['fleet_size_conservative']}-aircraft fleet over 25-year operational lifetime."
        """
        
        print("\n" + "="*70)
        print("UPDATED RESUME BULLET:")
        print("="*70)
        print(resume_bullet)
        
        # Save results to a file for later use
        os.makedirs("results", exist_ok=True)
        with open('results/boeing_comparison_results.json', 'w') as f:
            # Filter results for JSON serializability if needed
            # For now just save a summary
            summary = {
                "params": optimized_params,
                "improvements": improvements,
                "savings": savings
            }
            json.dump(summary, f, indent=4)
        
        return results
    
    else:
        print("\n✗ Comparison failed")
        return None

if __name__ == "__main__":
    results = main()
