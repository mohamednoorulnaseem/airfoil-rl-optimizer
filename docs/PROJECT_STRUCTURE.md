# Project Structure

This document describes the professional organization of the Airfoil RL Optimizer codebase.

## Directory Tree

```
airfoil-rl-optimizer/
├── README.md                          # Main documentation with badges and metrics
├── LICENSE                            # MIT License
├── setup.py                          # Package installation configuration
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Comprehensive ignore patterns
│
├── app.py                            # Web interface (Gradio)
├── train_rl.py                       # Main training script
├── aero_eval.py                      # Aerodynamic evaluation wrapper
├── manufacturing_constraints.py       # Manufacturing constraints wrapper
│
├── config/                           # Configuration files
│   ├── aircraft_database.json        # Aircraft performance database
│   ├── cfd_config.yaml              # CFD simulation settings
│   ├── config.yaml                  # Main configuration
│   └── rl_config.yaml               # RL training parameters
│
├── src/                              # Source code
│   ├── aerodynamics/                # Aerodynamic analysis modules
│   │   ├── xfoil_interface.py       # XFOIL integration
│   │   ├── su2_interface.py         # SU2 integration
│   │   ├── pinn_surrogate.py        # Physics-informed neural network
│   │   ├── airfoil_gen.py           # Airfoil geometry generation
│   │   └── aero_coefficients.py     # Coefficient calculations
│   │
│   ├── optimization/                # Optimization algorithms
│   │   ├── rl_agent.py              # PPO reinforcement learning agent
│   │   ├── multi_objective_env.py   # Multi-objective RL environment
│   │   ├── multi_objective.py       # Multi-objective optimization
│   │   └── single_objective_env.py  # Single-objective RL environment
│   │
│   ├── validation/                  # Validation and benchmarking
│   │   ├── aircraft_benchmark.py    # Aircraft performance validation
│   │   ├── manufacturing.py         # Manufacturing constraints
│   │   ├── uncertainty.py           # Uncertainty quantification
│   │   └── wind_tunnel_sim.py       # Wind tunnel simulation
│   │
│   └── utils/                       # Utility functions
│       ├── visualizations.py        # Plotting and visualization
│       ├── plot_config.py           # Publication-quality plot configuration
│       └── export_tools.py          # Data export utilities
│
├── tests/                           # Unit and integration tests
│   ├── test_xfoil.py               # XFOIL interface tests
│   ├── test_rl_agent.py            # RL agent tests
│   ├── test_manufacturing.py       # Manufacturing constraint tests
│   ├── test_model.py               # Model validation tests
│   └── test_system.py              # System integration tests
│
├── notebooks/                       # Jupyter notebooks for analysis
│   ├── 01_xfoil_validation.ipynb   # XFOIL validation analysis
│   ├── 02_rl_training.ipynb        # RL training experiments
│   ├── 03_aircraft_comparison.ipynb # Aircraft performance comparison
│   └── 04_sensitivity_analysis.ipynb # Sensitivity analysis
│
├── docs/                            # Documentation
│   ├── README.md                    # Documentation overview
│   ├── technical_report.md          # Technical specification
│   ├── technical_summary.md         # 17K-word comprehensive technical report
│   ├── technical_paper.tex          # AIAA LaTeX paper template
│   ├── user_guide.md               # User guide
│   │
│   ├── guides/                     # User-facing guides
│   │   ├── QUICKSTART.md           # Quick start guide
│   │   ├── VALIDATION.md           # Validation checklist and scorecard
│   │   └── INTERVIEW_PREP.md       # Interview preparation guide
│   │
│   └── development/                # Development documentation
│       ├── PROJECT_STATUS_ULTIMATE.md # Feature analysis vs requirements
│       ├── IMPLEMENTATION_COMPLETE.md # Implementation summary
│       └── QUICK_ACTION_GUIDE.md      # Quick action items
│
├── scripts/                         # Utility scripts
│   ├── verify_system.py            # System verification script
│   └── compare_multi.py            # Multi-objective comparison script
│
├── results/                         # Output directory
│   ├── figures/                    # Generated plots and figures
│   ├── tables/                     # Generated data tables
│   └── reports/                    # Generated reports
│
├── models/                          # Trained model checkpoints
│   └── (*.zip, *.pth files stored here)
│
├── data/                            # Data directory
│   ├── benchmarks/                 # Benchmark datasets
│   ├── training/                   # Training data
│   └── validation/                 # Validation data
│
└── archive/                         # Archived/deprecated files
    ├── project_status.py           # Old status script
    └── system_check.log            # Old check logs
```

## Key Files

### Root Level (Essential Only)

- **README.md**: Main project documentation with badges, features, and usage
- **app.py**: Web interface for interactive airfoil optimization
- **train_rl.py**: Main training script for RL agent
- **setup.py**: Package installation and dependency management
- **requirements.txt**: Python package dependencies

### Source Code (`src/`)

All production code is organized by function:

- `aerodynamics/`: CFD integration (XFOIL, SU2, PINN)
- `optimization/`: RL algorithms and environments
- `validation/`: Validation and benchmarking tools
- `utils/`: Shared utilities and visualization

### Documentation (`docs/`)

- **User Guides** (`guides/`): Quick start, validation checklist, interview prep
- **Development Docs** (`development/`): Status reports, implementation summaries
- **Technical Reports**: Comprehensive 17K-word summary, AIAA paper template

### Testing (`tests/`)

- Unit tests for core components
- Integration tests for system validation
- Manufacturing constraint tests

### Configuration (`config/`)

- YAML files for CFD, RL, and system configuration
- JSON aircraft database

## File Organization Principles

1. **Clean Root Directory**: Only essential files in root (README, main scripts, setup files)
2. **Organized by Function**: Code organized by purpose (aerodynamics, optimization, validation)
3. **Separate Documentation**: User guides separate from development docs
4. **Testing Consolidation**: All tests in `tests/` directory
5. **Results Isolation**: Generated outputs in `results/` (ignored by git)
6. **Model Storage**: Trained models in `models/` (large files ignored)
7. **Archive for Old Files**: Deprecated files moved to `archive/`

## Ignored Files (`.gitignore`)

### Python

- `__pycache__/`, `*.pyc`, `*.pyo`
- `.pytest_cache/`, `.mypy_cache/`
- Virtual environments (`.venv/`, `venv/`)

### Project-Specific

- Training results: `results/**/*.png`, `*.log`
- Trained models: `models/*.zip`, `models/*.pth`
- CFD files: `*.su2`, `*.mesh`, `*.cgns`
- Archive directory: `archive/`
- Temporary files: `*.bak`, `*~`, `.DS_Store`

### IDE/Editors

- VSCode: `.vscode/settings.json`
- JetBrains: `.idea/`
- Cursor: `.cursorignore`

## Professional Standards

This structure follows industry best practices:

- ✅ Clean, organized root directory
- ✅ Logical code organization by function
- ✅ Comprehensive documentation
- ✅ Testing infrastructure
- ✅ Configuration management
- ✅ Results/outputs separated
- ✅ Professional .gitignore
- ✅ No cache or temporary files in version control

## Quick Navigation

| Need                    | Location                                                                        |
| ----------------------- | ------------------------------------------------------------------------------- |
| Start using the project | [README.md](README.md) → [docs/guides/QUICKSTART.md](docs/guides/QUICKSTART.md) |
| Technical details       | [docs/technical_summary.md](docs/technical_summary.md)                          |
| Interview preparation   | [docs/guides/INTERVIEW_PREP.md](docs/guides/INTERVIEW_PREP.md)                  |
| Run training            | `python train_rl.py`                                                            |
| Launch web app          | `python app.py`                                                                 |
| Run tests               | `pytest tests/`                                                                 |
| Validate project        | [docs/guides/VALIDATION.md](docs/guides/VALIDATION.md)                          |

---

**Last Updated**: Directory reorganization completed for professional GitHub presentation.
