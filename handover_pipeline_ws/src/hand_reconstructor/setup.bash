#!/bin/bash
# This script sets up the conda environment for the hand_reconstructor package

# Remove conda environment (if it already exists)
if conda env list | grep -q "hand_reconstructor"; then
    echo "Removing existing conda environment 'hand_reconstructor'..."
    conda remove -n hand_reconstructor --all -y
fi
# Create a new conda environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda env create -n hand_reconstructor -f ${SCRIPT_DIR}/environment.yml

# Install third-party dependencies in editable mode
cd "${SCRIPT_DIR}"
conda run -n hand_reconstructor pip install -e "${SCRIPT_DIR}/third-party/hamer[all]"
conda run -n hand_reconstructor pip install -e "${SCRIPT_DIR}/third-party/hamer/third-party/ViTPose"