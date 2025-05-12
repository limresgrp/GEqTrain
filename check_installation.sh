#!/bin/bash

# Check if torch is installed
torch_installed=$(python3 -c "import importlib.util; print(importlib.util.find_spec('torch') is not None)")

# Check if torch_scatter is installed
torch_scatter_installed=$(python3 -c "import importlib.util; print(importlib.util.find_spec('torch_scatter') is not None)")

# Check if CUDA is supported
cuda_supported=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

# Echo results
if [ "$torch_installed" == "True" ]; then
    echo "Torch is installed."
else
    echo "Torch is NOT installed."
fi

if [ "$torch_scatter_installed" == "True" ]; then
    echo "Torch_scatter is installed."
else
    echo "Torch_scatter is NOT installed."
fi

if [ "$cuda_supported" == "True" ]; then
    echo "CUDA is supported."
else
    echo "CUDA is NOT supported."
fi