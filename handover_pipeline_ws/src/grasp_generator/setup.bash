#!/bin/bash
# This script sets up the conda environment for the grasp_generator package

# Remove conda environment (if it already exists)
if conda env list | grep -q "grasp_generator"; then
    echo "Removing existing conda environment 'grasp_generator'..."
    conda remove -n grasp_generator --all -y
fi
# Create a new conda environment
conda env create -n grasp_generator -f environment.yml