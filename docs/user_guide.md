# User Guide - Airfoil RL Optimizer

## Overview

This project provides a production-grade reinforcement learning framework for optimizing NACA airfoil geometry. It combines PPO agents with CFD validation to achieve meaningful aerodynamic improvements.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mohamednoorulnaseem/airfoil-rl-optimizer.git
cd airfoil-rl-optimizer

# Create environment
conda create -n airfoil_rl python=3.10 -y
conda activate airfoil_rl

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

### Train a New Model

```bash
python train_rl.py
```

## Features

### 1. CFD Analysis Tab

- Analyze any NACA airfoil configuration
- View Cl, Cd, L/D across angle of attack sweep
- Pressure distribution visualization

### 2. RL Optimization Tab

- Run PPO agent to find optimal parameters
- Compare baseline vs optimized performance
- View improvement percentages

### 3. Manufacturing Tab

- Check if design is manufacturable
- View constraint violations
- Industry-standard specifications

### 4. Validation Tab

- Wind tunnel simulation
- CFD-to-experiment comparison
- Uncertainty quantification

### 5. Export Tab

- Download airfoil coordinates (.dat)
- Export to MATLAB (.mat)
- CAD-compatible formats

## Configuration

Edit `config/config.yaml` to customize:

```yaml
rl:
  total_timesteps: 100000
  learning_rate: 0.0003

aerodynamics:
  primary_solver: xfoil
  reynolds: 1.0e6
```

## Project Structure

```
airfoil-rl-optimizer/
├── app.py                 # Streamlit dashboard
├── src/                   # Core modules
│   ├── aerodynamics/      # CFD solvers
│   ├── optimization/      # RL agents
│   └── validation/        # Benchmarking
├── config/                # Configuration files
├── tests/                 # Test suites
└── docs/                  # Documentation
```

## API Reference

### Aerodynamic Analysis

```python
from xfoil_integration import xfoil_analysis

cl, cd, ld = xfoil_analysis(m=0.02, p=0.4, t=0.12, alpha=4.0)
```

### RL Optimization

```python
from src.optimization.rl_agent import AirfoilRLAgent
from airfoil_env import AirfoilEnv

env = AirfoilEnv()
agent = AirfoilRLAgent(env, model_path="models/ppo_airfoil_fake.zip")
result = agent.optimize()
print(f"Best L/D: {result['best_ld']}")
```

### Manufacturing Check

```python
from manufacturing_constraints import check_manufacturability

is_valid, results = check_manufacturability(m=0.02, p=0.4, t=0.12)
```

## Troubleshooting

### XFOIL not found

Install XFOIL separately:

- Windows: Download from MIT website
- Linux: `sudo apt-get install xfoil`
- Mac: `brew install xfoil`

### Model not loading

Ensure you have the trained model in `models/` directory.

### Slow performance

Reduce `total_timesteps` or use the PINN surrogate.

## Contributing

See CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file.
