#!/bin/bash

echo "Setting up Airfoil RL Optimizer environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install XFOIL
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y xfoil
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install xfoil
fi

# Create necessary directories
mkdir -p data/validation_data
mkdir -p models/training_logs
mkdir -p results/figures
mkdir -p results/tables
mkdir -p temp_xfoil

echo "✓ Environment setup complete!"
echo "Activate with: source venv/bin/activate"
