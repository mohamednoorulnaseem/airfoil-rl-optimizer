# Training Data

This directory contains CFD training data for the PINN surrogate model.

## Structure

- `xfoil_runs/` - Raw XFOIL output files
- `polar_data/` - Processed polar data (Cl, Cd, Cm vs alpha)
- `surface_data/` - Surface pressure distributions

## Generating Data

Run the data generation script:

```bash
python scripts/generate_training_data.py
```

This will:

1. Sample 500 random NACA configurations
2. Run XFOIL analysis at multiple angles
3. Store results in JSON format
