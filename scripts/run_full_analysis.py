"""
Run complete analysis pipeline
Executes all validation and comparison steps in sequence.
"""

import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_step(description, command, cwd=None):
    """Run a pipeline step"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}\n")
    
    if cwd is None:
        cwd = Path(__file__).parent.parent
    
    result = subprocess.run(command, shell=True, cwd=str(cwd))
    if result.returncode != 0:
        print(f"\n✗ Step failed: {description}")
        return False
    print(f"\n✓ Step completed: {description}")
    return True

def main():
    """Run full analysis pipeline"""
    
    print("="*60)
    print("AIRFOIL RL OPTIMIZER - FULL ANALYSIS PIPELINE")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    
    # Ensure results directories exist
    (project_root / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (project_root / "results" / "tables").mkdir(parents=True, exist_ok=True)
    
    steps = [
        ("1. Test XFOIL Installation", "python tests/test_xfoil.py"),
        ("2. Validate RL with XFOIL", "python src/validation/validate_rl_with_xfoil.py"),
        ("3. Boeing 737 Comparison", "python run_boeing_comparison.py"),
    ]
    
    failed = []
    for desc, cmd in steps:
        if not run_step(desc, cmd, project_root):
            failed.append(desc)
    
    print("\n" + "="*60)
    if failed:
        print("⚠ ANALYSIS COMPLETED WITH WARNINGS")
        print(f"  Failed steps: {len(failed)}")
        for f in failed:
            print(f"    - {f}")
    else:
        print("✓ COMPLETE ANALYSIS FINISHED!")
    print("="*60)
    print("\nResults saved to:")
    print("  - results/figures/")
    print("  - results/tables/")

if __name__ == "__main__":
    main()
