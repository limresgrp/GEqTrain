#!/bin/bash

# Check if Python version is >= 3.8
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l) -eq 1 ]]; then
    echo "Python 3.10 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Function to display the menu and get user input
choose_cuda_version() {
    echo "Please choose the CUDA version to install:"
    echo "1) cpu"
    echo "2) cu118"
    echo "3) cu124"
    echo "4) cu126"
    read -p "Enter your choice (1-4): " choice

    case $choice in
        1) CUDA_VERSION="cpu" ;;
        2) CUDA_VERSION="cu118" ;;
        3) CUDA_VERSION="cu124" ;;
        4) CUDA_VERSION="cu126" ;;
        *) echo "Invalid choice. Defaulting to cu124."; CUDA_VERSION="cu124" ;;
    esac
}

# Function to check if a Python package is installed and echo the output
is_package_installed() {
    output=$(python3 -c "import $1" 2>&1)
    if [ $? -eq 0 ]; then
        echo "$1 is installed."
        return 0
    else
        echo "$1 is not installed. Error: $output"
        return 1
    fi
}

# Check if torch and torch-scatter are already installed
if is_package_installed torch && is_package_installed torch_scatter; then
    echo "torch and torch-scatter are already installed. Skipping CUDA version selection."
else
    choose_cuda_version

    # Install torch and torch-scatter with the selected CUDA version
    pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/$CUDA_VERSION
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    pip3 install torch-scatter -f https://data.pyg.org/whl/torch-$TORCH_VERSION.html
fi

# Install the current package
echo "Installing the package in editable mode..."
pip3 install -e .

echo "Installation complete!"