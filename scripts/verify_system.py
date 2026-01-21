
import sys
import os
import json
import yaml
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_status(component, status, message=""):
    color = "✅" if status else "❌"
    msg = f"{color} {component:<30}: {message}"
    print(msg)
    with open("system_check.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    if not status:
        global ALL_PASSED
        ALL_PASSED = False

# Clear log
with open("system_check.log", "w", encoding="utf-8") as f:
    f.write("System Verification Log\n")

ALL_PASSED = True

print("="*60)
print("🔍 System Integrity Verification")
print("="*60)

# 1. Configuration Check
try:
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print_status("Config File (YAML)", True, "Loaded")
except Exception as e:
    print_status("Config File (YAML)", False, str(e))

try:
    with open('config/aircraft_database.json', 'r') as f:
        db = json.load(f)
    print_status("Aircraft Database (JSON)", True, f"Loaded {len(db)} aircraft")
except Exception as e:
    print_status("Aircraft Database (JSON)", False, str(e))

# 2. Aerodynamics Module
try:
    from src.aerodynamics.airfoil_gen import naca4
    xu, yu, xl, yl = naca4(0.02, 0.4, 0.12)
    if len(xu) == 200:
        print_status("Airfoil Generator", True, "NACA 2412 generated successfully")
    else:
        print_status("Airfoil Generator", False, f"Unexpected output length: {len(xu)}")
except Exception as e:
    print_status("Airfoil Generator", False, str(e))

try:
    from src.aerodynamics.xfoil_interface import get_analyzer
    analyzer = get_analyzer()
    print_status("XFOIL Interface", True, "Analyzer initialized")
except Exception as e:
    print_status("XFOIL Interface", False, str(e))

# 3. Validation Modules
try:
    from src.validation.manufacturing import check_manufacturability
    valid, report = check_manufacturability(0.02, 0.4, 0.12)
    print_status("Manufacturing Check", True, f"Valid: {valid}")
except Exception as e:
    print_status("Manufacturing Check", False, str(e))

try:
    from src.validation.wind_tunnel_sim import run_wind_tunnel_sweep
    res = run_wind_tunnel_sweep(0.02, 0.4, 0.12)
    if 'cl_mean' in res:
        print_status("Wind Tunnel Sim", True, "Simulation completed")
    else:
        print_status("Wind Tunnel Sim", False, "Missing output keys")
except Exception as e:
    print_status("Wind Tunnel Sim", False, str(e))

try:
    from src.validation.aircraft_benchmark import AircraftBenchmark
    bench = AircraftBenchmark()
    comp = bench.compare_to_aircraft(0.02, 0.4, 0.12, 'boeing_737_800')
    if comp:
        print_status("Aircraft Benchmark", True, "Benchmark ran against B737")
    else:
        print_status("Aircraft Benchmark", False, "No result returned")
except Exception as e:
    print_status("Aircraft Benchmark", False, str(e))

# 4. App Dependencies
try:
    import dash
    import dash_bootstrap_components
    print_status("Dash Framework", True, f"v{dash.__version__} installed")
except ImportError as e:
    print_status("Dash Framework", False, "Missing dependencies")


print("-" * 60)
if ALL_PASSED:
    print("🚀 SYSTEM 100% OPERATIONAL")
    print("All core modules are verified and functional.")
else:
    print("⚠️  SYSTEM ISSUES DETECTED")
    print("Please fix the❌ items above.")
print("-" * 60)
