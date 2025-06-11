#!/bin/bash
# This script sets up the conda environment for the transform_estimator package

# Remove conda environment (if it already exists)
if conda env list | grep -q "transform_estimator"; then
    echo "Removing existing conda environment 'transform_estimator'..."
    conda remove -n transform_estimator --all -y
fi
# Create a new conda environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda env create -n transform_estimator -f ${SCRIPT_DIR}/environment.yml