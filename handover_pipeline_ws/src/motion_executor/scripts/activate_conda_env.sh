#!/bin/bash
# Wrapper to activate a given conda environment and run a command

CONDA_PATH="${CONDA_PATH:-/home/franka/miniconda3/}"

# Source conda environment setup
source "$CONDA_PATH/etc/profile.d/conda.sh"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <env_name> <command> [args...]"
    exit 1
fi

ENV_NAME="$1"
shift

conda activate "$ENV_NAME"
exec "$@"
