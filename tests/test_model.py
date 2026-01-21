"""
Test Trained RL Model

Loads and tests the trained PPO model for airfoil optimization.
"""

import numpy as np
from src.optimization.rl_agent import AirfoilRLAgent
from src.optimization.multi_objective_env import MultiObjectiveAirfoilEnv

print("="*60)
print("🎯 Testing Trained RL Model")
print("="*60)

# Load environment and agent
print("\n1. Loading environment and trained agent...")
env = MultiObjectiveAirfoilEnv()
agent = AirfoilRLAgent(env, model_path="models/ppo_airfoil_final.zip", verbose=0)
print("✅ Model loaded successfully")

# Run optimization episode
print("\n2. Running optimization episode...")
result = agent.optimize(max_steps=50)
print(f"✅ Optimization complete!")
print(f"   Best L/D: {result['best_ld']:.2f}")
if result['best_params'] is not None:
    m, p, t = result['best_params']
    print(f"   Best Parameters: Camber={m:.4f}, Position={p:.3f}, Thickness={t:.4f}")

# Test multiple episodes
print("\n3. Running 5 test episodes...")
ld_scores = []
for i in range(5):
    result = agent.optimize(max_steps=40)
    ld_scores.append(result['best_ld'])
    print(f"   Episode {i+1}: L/D={result['best_ld']:.2f}")

print(f"\n✅ Average L/D: {np.mean(ld_scores):.2f}")
print(f"   Best L/D: {np.max(ld_scores):.2f}")
print(f"   Std Dev: {np.std(ld_scores):.2f}")

print("\n" + "="*60)
print("✅ Model Testing Complete!")
print("="*60)
