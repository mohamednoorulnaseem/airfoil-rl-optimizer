# Installation Guide

## Prerequisites

- Python 3.10+
- XFOIL (for CFD validation)
- Git

## Quick Install

### 1. Clone Repository

```bash
git clone https://github.com/mohamednoorulnaseem/airfoil-rl-optimizer.git
cd airfoil-rl-optimizer
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install XFOIL

**Windows:**

- Download XFOIL from [official site](https://web.mit.edu/drela/Public/web/xfoil/)
- Add to PATH or place `xfoil.exe` in project root

**Linux:**

```bash
sudo apt-get install xfoil
```

**macOS:**

```bash
brew install xfoil
```

### 5. Verify Installation

```bash
python test_xfoil.py
```

## Docker Installation (Alternative)

```bash
docker build -t airfoil-optimizer .
docker run -p 8050:8050 airfoil-optimizer
```

## Troubleshooting

### XFOIL not found

- Ensure XFOIL is in your PATH
- Or place `xfoil.exe` in project root

### Module not found errors

- Ensure you activated the virtual environment
- Run `pip install -e .` for development install

### PyTorch issues

- For GPU support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
