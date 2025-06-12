#!/bin/bash
# This script sets up the conda environments for the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse argument
MODE="${1:-base}"  # Default to 'base' if no argument is given

# Set up the base environment such that commands like 'catkin_make' can be run
conda env update -n base -f ${SCRIPT_DIR}/environment.yml

# build the catkin workspace
cd ${SCRIPT_DIR}
catkin_make

if [[ "$MODE" == "all" ]]; then
    # Set up all package-specific environments
    for pkg_dir in "${SCRIPT_DIR}/src"/*/; do
        if [ -f "${pkg_dir}setup.bash" ]; then
            echo "Sourcing ${pkg_dir}setup.bash"
            source "${pkg_dir}setup.bash"
        fi
    done
elif [[ "$MODE" == "base" ]]; then
    echo "Only base environment set up."
else
    echo "Unknown argument: $MODE"
    echo "Usage: source setup.bash [all|base]"
    return 1 2>/dev/null || exit 1
fi