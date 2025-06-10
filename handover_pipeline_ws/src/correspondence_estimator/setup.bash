#!/bin/bash
# This script sets up the conda environment for the correspondence_estimator package

# Remove conda environment (if it already exists)
if conda env list | grep -q "correspondence_estimator"; then
    echo "Removing existing conda environment 'correspondence_estimator'..."
    conda remove -n correspondence_estimator --all -y
fi
# Create a new conda environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda env create -n correspondence_estimator -f ${SCRIPT_DIR}/environment.yml

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate correspondence_estimator