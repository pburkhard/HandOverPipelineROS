#!/bin/bash
# This script sets up the conda environment for the pipeline package

# Remove conda environment (if it already exists)
if conda env list | grep -q "pipeline"; then
    echo "Removing existing conda environment 'pipeline'..."
    conda remove -n pipeline --all -y
fi
# Create a new conda environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda env create -n pipeline -f ${SCRIPT_DIR}/environment.yml