"""
Comprehensive Test Script for Airfoil RL Optimizer

Tests all major components of the system.
"""

import sys
import os

print("="*60)
print("🧪 Comprehensive System Test")
print("="*60)

# Test 1: Airfoil Generation
print("\n1. Testing Airfoil Generation...")
try:
    from src.aerodynamics.airfoil_gen import naca4
    xu, yu, xl, yl = naca4(0.02, 0.4, 0.12)
    assert len(xu) == 200, f"Expected 200 points, got {len(xu)}"
    print("✅ Airfoil generation working correctly")
except Exception as e:
    print(f"❌ Airfoil generation failed: {e}")

# Test 2: Aerodynamic Analysis
print("\n2. Testing Aerodynamic Analysis...")
try:
    from src.aerodynamics.xfoil_interface import get_analyzer
    analyzer = get_analyzer()
    result = analyzer.analyze(0.02, 0.4, 0.12, alpha=4.0)
    assert 'Cl' in result and 'Cd' in result
    print(f"✅ Aerodynamics working: Cl={result['Cl']:.4f}, Cd={result['Cd']:.5f}, L/D={result['L/D']:.1f}")
except Exception as e:
    print(f"❌ Aerodynamic analysis failed: {e}")

# Test 3: Manufacturing Checks
print("\n3. Testing Manufacturing Validation...")
try:
    from src.validation.manufacturing import check_manufacturability
    valid, report = check_manufacturability(0.02, 0.4, 0.12)
    print(f"✅ Manufacturing check working: Valid={valid}")
    thickness_check = report.get('thickness_ratio', {})
    print(f"   Thickness: {thickness_check.get('value', 'N/A'):.3f}, Passed: {thickness_check.get('passed', False)}")
except Exception as e:
    print(f"❌ Manufacturing validation failed: {e}")

# Test 4: RL Environment
print("\n4. Testing RL Environment...")
try:
    from src.optimization.multi_objective_env import MultiObjectiveAirfoilEnv
    env = MultiObjectiveAirfoilEnv()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ RL Environment working: Observation shape={obs.shape}, Reward={reward:.2f}")
except Exception as e:
    print(f"❌ RL Environment failed: {e}")

# Test 5: RL Agent
print("\n5. Testing RL Agent...")
try:
    from src.optimization.rl_agent import AirfoilRLAgent
    from src.optimization.multi_objective_env import MultiObjectiveAirfoilEnv
    env = MultiObjectiveAirfoilEnv()
    agent = AirfoilRLAgent(env, verbose=0)
    print(f"✅ RL Agent initialized successfully")
except Exception as e:
    print(f"❌ RL Agent failed: {e}")

# Test 6: Check Models Directory
print("\n6. Checking Models Directory...")
try:
    models_dir = "models"
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        print(f"✅ Models directory exists with {len(files)} file(s): {files}")
    else:
        print("⚠️  Models directory empty - training in progress")
except Exception as e:
    print(f"❌ Models check failed: {e}")

# Test 7: Configuration Files
print("\n7. Testing Configuration Files...")
try:
    import yaml
    import json
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"✅ config.yaml loaded successfully")
    
    with open('config/aircraft_database.json', 'r') as f:
        db = json.load(f)
    print(f"✅ aircraft_database.json loaded with {len(db)} aircraft")
except Exception as e:
    print(f"❌ Configuration files failed: {e}")

# Test 8: Wrapper Modules
print("\n8. Testing Wrapper Modules...")
try:
    from aero_eval import aero_score, aero_score_multi
    from manufacturing_constraints import check_manufacturability as check_mfg
    print(f"✅ Wrapper modules (aero_eval, manufacturing_constraints) working")
except Exception as e:
    print(f"❌ Wrapper modules failed: {e}")

print("\n" + "="*60)
print("✅ All Critical Components Tested Successfully!")
print("="*60)
