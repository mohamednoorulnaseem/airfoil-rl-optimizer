# Usage Guide

## 🚀 Quick Start

1. **Activate Environment**

   ```bash
   source venv/bin/activate
   ```

2. **Run Dashboard**
   ```bash
   python dash_app.py
   ```
   Open http://127.0.0.1:8050 in your browser.

## 🧪 Running Experiments

### 1. Optimize an Airfoil

To run a single optimization loop:

```bash
python src/optimization/train_rl.py
```

### 2. Validate with XFOIL

To validate the surrogate model against real XFOIL physics:

```bash
python src/validation/validate_rl_with_xfoil.py
```

### 3. Compare with Boeing 737

To compare your optimized airfoil against the standard Boeing 737 airfoil:

```bash
python scripts/run_boeing_comparison.py
```

## 📊 Interpreting Results

Results are saved in `results/`.

- **Figures**: Visualizations of pressure distributions and polars.
- **Tables**: CSV files comparing lift, drag, and L/D ratios.
