# API Reference

## Core Modules

### `src.aerodynamics`

#### `XFOILRunner`

Python wrapper for XFOIL CFD solver.

```python
from src.aerodynamics import XFOILRunner

runner = XFOILRunner(reynolds=1e6, mach=0.0)
results = runner.analyze_airfoil(coords, alpha_range=[0, 2, 4, 6, 8])
```

**Parameters:**

- `reynolds` (float): Reynolds number (default: 1e6)
- `mach` (float): Mach number (default: 0.0)
- `n_iter` (int): Max iterations (default: 100)

**Methods:**

- `analyze_airfoil(coords, alpha_range)`: Run XFOIL analysis
- `get_ld_max(coords)`: Find maximum L/D ratio
- `cleanup()`: Remove temporary files

---

#### `generate_naca_4digit`

Generate NACA 4-digit airfoil coordinates.

```python
from src.aerodynamics import generate_naca_4digit

coords = generate_naca_4digit(m=0.02, p=0.4, t=0.12, n_points=100)
```

**Parameters:**

- `m` (float): Max camber (0.00-0.06)
- `p` (float): Camber position (0.1-0.7)
- `t` (float): Thickness ratio (0.08-0.20)
- `n_points` (int): Number of points (default: 100)

**Returns:** `numpy.ndarray` shape (n_points, 2)

---

### `src.optimization`

#### `AirfoilEnvXFOIL`

Gymnasium environment for RL training.

```python
from src.optimization import AirfoilEnvXFOIL

env = AirfoilEnvXFOIL(use_xfoil=True)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

**Action Space:** Box(3) - delta changes to [m, p, t]

**Observation Space:** Box(6) - [m, p, t, cl, cd, ld]

---

### `src.validation`

#### `AircraftComparator`

Compare optimized airfoil to real aircraft.

```python
from src.validation import AircraftComparator

comparator = AircraftComparator()
result = comparator.compare_to_aircraft([m, p, t], "Boeing 737-800")
```

**Returns:**

```python
{
    'improvements': {
        'ld_improvement_percent': 40.8,
        'cd_reduction_percent': 28.9,
        'cl_change_percent': 3.2
    },
    'fuel_savings': {
        'annual_cost_savings_usd': 1_200_000,
        'fleet_lifetime_savings_usd': 1_600_000_000
    }
}
```

---

## Dash Application

### Running the Dashboard

```python
python dash_app.py
```

Access at: http://localhost:8050

### Available Endpoints

| Endpoint        | Description              |
| --------------- | ------------------------ |
| `/`             | Main dashboard           |
| `/aerodynamics` | Aerodynamic analysis tab |
| `/comparison`   | Aircraft comparison tab  |
| `/metrics`      | Performance metrics tab  |

---

## Configuration

### Environment Variables

| Variable     | Description              | Default                  |
| ------------ | ------------------------ | ------------------------ |
| `XFOIL_PATH` | Path to XFOIL executable | System PATH              |
| `MODEL_PATH` | Path to trained RL model | `models/ppo_airfoil.zip` |
| `DEBUG`      | Enable debug mode        | `False`                  |

### Config File

```python
# src/utils/config.py

DEFAULT_REYNOLDS = 1e6
DEFAULT_MACH = 0.0
MODEL_PATH = "models/ppo_airfoil_final.zip"
```
