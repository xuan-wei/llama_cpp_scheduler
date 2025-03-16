#!/bin/bash

# Get the full path to Python
PYTHON_PATH=$(which python)

# Check if Python was found
if [ -z "$PYTHON_PATH" ]; then
    echo "Error: Python not found in PATH"
    exit 1
fi

# Run the scheduler with sudo and the full Python path
sudo "$PYTHON_PATH" scheduler.py
