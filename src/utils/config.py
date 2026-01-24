"""
Configuration settings for Airfoil RL Optimizer
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# XFOIL Settings
XFOIL_TIMEOUT = 10  # seconds
XFOIL_MAX_ITER = 100

# RL Settings
RL_ALGORITHM = "PPO"
RL_TIMESTEPS = 100000

# Plotting
PLOT_STYLE = "seaborn-v0_8-whitegrid"
