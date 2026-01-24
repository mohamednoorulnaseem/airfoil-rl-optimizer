"""
Validate RL-optimized airfoil using XFOIL CFD
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from stable_baselines3 import PPO
from src.aerodynamics.xfoil_interface import XFOILRunner
from src.aerodynamics.airfoil_gen import generate_naca_4digit
from src.optimization.airfoil_env import AirfoilEnvXFOIL

def validate_rl_optimization():
    """
    Compare baseline vs RL-optimized airfoil using XFOIL
    """
    
    print("="*60)
    print("VALIDATION: Baseline vs RL-Optimized Airfoil (XFOIL CFD)")
    print("="*60)
    
    model_path = "models/ppo_airfoil_fake.zip"
    if not os.path.exists(model_path):
        # Try to find any .zip model in models/
        import glob
        models = glob.glob("models/*.zip")
        if models:
            model_path = models[0]
            print(f"Default model not found, using {model_path} instead.")
        else:
            print("No trained model found. Skipping RL optimization part.")
            model_path = None
    
    # Baseline NACA 2412
    baseline_params = [0.02, 0.4, 0.12]
    
    # Initialize XFOIL
    try:
        xfoil = XFOILRunner(reynolds=1e6, mach=0.0)
    except Exception as e:
        print(f"Failed to initialize XFOIL: {e}")
        return

    # Analyze baseline
    print("\n1. Analyzing BASELINE (NACA 2412)...")
    try:
        baseline_coords = generate_naca_4digit(*baseline_params, n_points=100)
        baseline_results = xfoil.analyze_airfoil(
            baseline_coords,
            alpha_range=[0, 2, 4, 6, 8]
        )
    except Exception as e:
        print(f"Error analyzing baseline: {e}")
        baseline_results = None

    if not baseline_results:
        print("Baseline analysis failed. Check XFOIL installation.")
        return

    # Get RL-optimized parameters
    best_params = baseline_params # Default to baseline if no model
    optimized_results = None

    if model_path:
        try:
            # Load trained RL model
            model = PPO.load(model_path)
            
            # (Run multiple episodes and pick best)
            print("\nSearching for optimized parameters...")
            best_params, best_ld = find_best_rl_params(model)
            
            # Analyze optimized
            print("\n2. Analyzing RL-OPTIMIZED airfoil...")
            optimized_coords = generate_naca_4digit(*best_params, n_points=100)
            optimized_results = xfoil.analyze_airfoil(
                optimized_coords,
                alpha_range=[0, 2, 4, 6, 8]
            )
        except Exception as e:
           print(f"Error evaluating RL model: {e}")
    else:
        print("\nSkipping RL optimization (no model). Comparing baseline only.")

    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    print(f"\nBaseline (NACA 2412):")
    print(f"  Parameters: m={baseline_params[0]:.3f}, p={baseline_params[1]:.3f}, t={baseline_params[2]:.3f}")
    
    if optimized_results:
        print(f"\nRL-Optimized:")
        print(f"  Parameters: m={best_params[0]:.3f}, p={best_params[1]:.3f}, t={best_params[2]:.3f}")
        
        # Performance at α=4° (typical cruise)
        baseline_4deg = next((r for r in baseline_results if abs(r['alpha'] - 4.0) < 0.1), None)
        optimized_4deg = next((r for r in optimized_results if abs(r['alpha'] - 4.0) < 0.1), None)
        
        if baseline_4deg and optimized_4deg:
            print(f"\nPerformance at α=4° (Cruise Condition):")
            print(f"  Baseline  - Cl: {baseline_4deg['cl']:.4f}, Cd: {baseline_4deg['cd']:.6f}, L/D: {baseline_4deg['cl']/baseline_4deg['cd']:.2f}")
            print(f"  Optimized - Cl: {optimized_4deg['cl']:.4f}, Cd: {optimized_4deg['cd']:.6f}, L/D: {optimized_4deg['cl']/optimized_4deg['cd']:.2f}")
            
            # Calculate improvements
            cl_improvement = ((optimized_4deg['cl'] - baseline_4deg['cl']) / baseline_4deg['cl']) * 100
            cd_reduction = ((baseline_4deg['cd'] - optimized_4deg['cd']) / baseline_4deg['cd']) * 100
            ld_improvement = ((optimized_4deg['cl']/optimized_4deg['cd'] - baseline_4deg['cl']/baseline_4deg['cd']) / 
                              (baseline_4deg['cl']/baseline_4deg['cd'])) * 100
            
            print(f"\n  Improvements:")
            print(f"    Cl improvement: {cl_improvement:+.1f}%")
            print(f"    Cd reduction:   {cd_reduction:+.1f}%")
            print(f"    L/D improvement: {ld_improvement:+.1f}%")
        
        # Plot comparison
        try:
            plot_comparison(baseline_results, optimized_results, baseline_coords, optimized_coords)
        except Exception as e:
             print(f"Plotting failed: {e}")
    else:
        print("No optimized results to compare.")

    xfoil.cleanup()
    
    return {
        'baseline_params': baseline_params,
        'optimized_params': best_params,
        'baseline_results': baseline_results,
        'optimized_results': optimized_results
    }

def find_best_rl_params(model, n_episodes=10):
    """Run RL agent multiple times, return best parameters"""
    try:
        env = AirfoilEnvXFOIL(use_xfoil=False)  # Use surrogate for speed finding params, then validate with XFOIL
    except Exception as e:
        print(f"Error creating environment: {e}")
        return [0.02, 0.4, 0.12], 0

    best_params = None
    best_ld = 0
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        # Get final params
        current_params = env.current_params
        current_ld = obs[3]  # L/D from observation
        
        if current_ld > best_ld:
            best_ld = current_ld
            best_params = current_params
            
    if best_params is None:
         best_params = [0.02, 0.4, 0.12]
    
    return best_params, best_ld

def plot_comparison(baseline_results, optimized_results, baseline_coords, optimized_coords):
    """Create comparison plots"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Airfoil geometry
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(baseline_coords[:, 0], baseline_coords[:, 1], 'b-', linewidth=2, label='Baseline')
    ax1.plot(optimized_coords[:, 0], optimized_coords[:, 1], 'r-', linewidth=2, label='Optimized')
    ax1.set_xlabel('x/c')
    ax1.set_ylabel('y/c')
    ax1.set_title('Airfoil Geometry Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Cl vs alpha
    ax2 = plt.subplot(2, 3, 2)
    baseline_alphas = [r['alpha'] for r in baseline_results]
    baseline_cls = [r['cl'] for r in baseline_results]
    optimized_alphas = [r['alpha'] for r in optimized_results]
    optimized_cls = [r['cl'] for r in optimized_results]
    
    ax2.plot(baseline_alphas, baseline_cls, 'bo-', linewidth=2, label='Baseline')
    ax2.plot(optimized_alphas, optimized_cls, 'ro-', linewidth=2, label='Optimized')
    ax2.set_xlabel('Angle of Attack (deg)')
    ax2.set_ylabel('Lift Coefficient (Cl)')
    ax2.set_title('Lift Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Polar (Cl vs Cd)
    ax3 = plt.subplot(2, 3, 3)
    baseline_cds = [r['cd'] for r in baseline_results]
    optimized_cds = [r['cd'] for r in optimized_results]
    
    ax3.plot(baseline_cds, baseline_cls, 'bo-', linewidth=2, label='Baseline')
    ax3.plot(optimized_cds, optimized_cls, 'ro-', linewidth=2, label='Optimized')
    ax3.set_xlabel('Drag Coefficient (Cd)')
    ax3.set_ylabel('Lift Coefficient (Cl)')
    ax3.set_title('Polar Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. L/D vs alpha
    ax4 = plt.subplot(2, 3, 4)
    baseline_lds = [r['cl']/r['cd'] for r in baseline_results]
    optimized_lds = [r['cl']/r['cd'] for r in optimized_results]
    
    ax4.plot(baseline_alphas, baseline_lds, 'bo-', linewidth=2, label='Baseline')
    ax4.plot(optimized_alphas, optimized_lds, 'ro-', linewidth=2, label='Optimized')
    ax4.set_xlabel('Angle of Attack (deg)')
    ax4.set_ylabel('L/D Ratio')
    ax4.set_title('Aerodynamic Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Drag comparison
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(baseline_alphas, baseline_cds, 'bo-', linewidth=2, label='Baseline')
    ax5.plot(optimized_alphas, optimized_cds, 'ro-', linewidth=2, label='Optimized')
    ax5.set_xlabel('Angle of Attack (deg)')
    ax5.set_ylabel('Drag Coefficient (Cd)')
    ax5.set_title('Drag Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Improvement bar chart
    ax6 = plt.subplot(2, 3, 6)
    # Calculate improvements at α=4°
    baseline_4 = next((r for r in baseline_results if abs(r['alpha'] - 4.0) < 0.1), None)
    optimized_4 = next((r for r in optimized_results if abs(r['alpha'] - 4.0) < 0.1), None)
    
    if baseline_4 and optimized_4:
        improvements = [
            ((optimized_4['cl'] - baseline_4['cl']) / baseline_4['cl']) * 100,
            ((baseline_4['cd'] - optimized_4['cd']) / baseline_4['cd']) * 100,
            ((optimized_4['cl']/optimized_4['cd'] - baseline_4['cl']/baseline_4['cd']) / 
             (baseline_4['cl']/baseline_4['cd'])) * 100
        ]
        
        colors = ['green' if x > 0 else 'red' for x in improvements]
        ax6.bar(['Cl\nImprovement', 'Cd\nReduction', 'L/D\nImprovement'], 
                improvements, color=colors, alpha=0.7)
        ax6.set_ylabel('Improvement (%)')
        ax6.set_title('Performance Gains at α=4°')
        ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/figures/xfoil_validation_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison plot: xfoil_validation_comparison.png")
    # plt.show()

if __name__ == "__main__":
    results = validate_rl_optimization()
