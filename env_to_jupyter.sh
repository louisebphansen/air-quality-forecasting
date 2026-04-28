#! /bin/bash

# Get the directory of the Bash script
script_dir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

# Check if the virtual environment exists
if [ ! -d "$script_dir/.venv" ]; then
    echo "Virtual environment not found in $script_dir/.venv. Please create the virtual environment first."
    exit 1
fi

# Activate the virtual environement
source "$script_dir/.venv/bin/activate"

# install kernel so env can be used in jupyter notebooks
python -m ipykernel install --user --name=air_quality_forecasting