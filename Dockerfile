# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (including git for pip install and XFOIL dependencies if needed)
# Note: XFOIL in docker might be tricky, we'll stick to the surrogate or python-only parts for now
# or install basic build tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements/setup files
COPY setup.py setup.py
COPY README.md README.md
# We need to create a dummy wrapper to install deps if we don't have a requirements.txt generated from setup.py
# But we can just install .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy the rest of the application
COPY . .

# Expose ports for Dash (8050) and Jupyter (8888)
EXPOSE 8050 8888

# Default command
CMD ["python", "app.py"]
