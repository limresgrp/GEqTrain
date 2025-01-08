#!/bin/bash

# Function to display the menu and get user input
choose_cuda_version() {
    echo "Select your CUDA version:"
    echo "1) CPU only (no CUDA)"
    echo "2) CUDA 11.7 (cu117)"
    echo "3) CUDA 11.8 (cu118)"
    echo "4) CUDA 12.1 (cu121)"
    echo "5) CUDA 12.4 (cu124)"
    echo "6) pytorch 2.5 CUDA 11.8 (cu118)"

    echo
    read -p "Enter the number corresponding to your CUDA version: " choice

    case $choice in
        1)
            CUDA_VERSION="cpu"
            TORCH_VERSION="2.4.0"
            TORCH_URL="https://download.pytorch.org/whl/cpu"
            SCATTER_URL="https://data.pyg.org/whl/torch-2.4.0+cpu.html"
            ;;
        2)
            CUDA_VERSION="cu117"
            TORCH_VERSION="2.0.1"
            TORCH_URL="https://download.pytorch.org/whl/cu117"
            SCATTER_URL="https://data.pyg.org/whl/torch-2.0.1+cu117.html"
            ;;
        3)
            CUDA_VERSION="cu118"
            TORCH_VERSION="2.4.0"
            TORCH_URL="https://download.pytorch.org/whl/cu118"
            SCATTER_URL="https://data.pyg.org/whl/torch-2.4.0+cu118.html"
            ;;
        4)
            CUDA_VERSION="cu121"
            TORCH_VERSION="2.4.0"
            TORCH_URL="https://download.pytorch.org/whl/cu121"
            SCATTER_URL="https://data.pyg.org/whl/torch-2.4.0+cu121.html"
            ;;
        5)
            CUDA_VERSION="cu124"
            TORCH_VERSION="2.4.0"
            TORCH_URL="https://download.pytorch.org/whl/cu124"
            SCATTER_URL="https://data.pyg.org/whl/torch-2.4.0+cu124.html"
            ;;
        6)
            CUDA_VERSION="cu118"
            TORCH_VERSION="2.5.1"
            TORCH_URL="https://download.pytorch.org/whl/cu118"
            SCATTER_URL="https://data.pyg.org/whl/torch-2.5.1+cu118.html"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
}

# Get the user's CUDA version
choose_cuda_version

# Install PyTorch with the selected CUDA version
echo "Installing PyTorch $TORCH_VERSION with $CUDA_VERSION support..."
pip3 install torch==$TORCH_VERSION --index-url $TORCH_URL

# Install torch-scatter based on PyTorch and CUDA version
echo "Installing torch-scatter for PyTorch $PYTORCH_VERSION with $CUDA_VERSION support..."
pip3 install torch-scatter -f $SCATTER_URL

# Install the current package
echo "Installing the package in editable mode..."
pip3 install -e .

echo "Installation complete!"