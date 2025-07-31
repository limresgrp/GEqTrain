# GEqTrain

GEqTrain is a framework to dynamically build, train, and deploy E(3)-equivariant graph-based models using PyTorch and e3nn.

## Installation

GEqTrain requires PyTorch. Since PyTorch must be installed for your specific hardware (CPU or a particular CUDA version), you need to install it manually before installing GEqTrain.

### For Users (from PyPI)

1.  **Install PyTorch**: Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and run the installation command that matches your system. For example, for CUDA 12.1:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

2.  **Install GEqTrain**: Once PyTorch is installed, you can install this package from PyPI:
    ```bash
    pip install geqtrain
    ```

### For Developers (from source)

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/limresgrp/GEqTrain.git](https://github.com/limresgrp/GEqTrain.git)
    cd GEqTrain
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install PyTorch**: Follow step 1 from the user installation guide above.

4.  **Install GEqTrain in editable mode**:
    ```bash
    pip install -e .
    ```

***

## Quick Start

After installation, you can use the command-line scripts provided by GEqTrain. For example, to start a training run with a configuration file:

```bash
geqtrain-train --config path/to/your/config.yaml -d cuda:0