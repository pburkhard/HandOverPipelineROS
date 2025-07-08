#!/bin/bash
# This script sets up the conda environment for the motion_executor package

# Remove conda environment (if it already exists)
if conda env list | grep -q "motion_executor"; then
    echo "Removing existing conda environment 'motion_executor'..."
    conda remove -n motion_executor --all -y
fi
# Create a new conda environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda env create -n motion_executor -f ${SCRIPT_DIR}/environment.yml