# GEqTrain

GEqTrain is a framework to dynamically build, train, and deploy E(3)-equivariant graph-based models using PyTorch and e3nn.
This guide provides instructions for installation, usage on local machines, and deployment on High-Performance Computing (HPC) clusters.

## 🚀 Installation

GEqTrain requires PyTorch to be installed first, as the correct version depends on your specific hardware (CPU or a particular CUDA version).
**For Users**
1. **Install PyTorch**: Visit the official PyTorch website and run the installation command that matches your system. For example, for CUDA 12.1:
    ```bash
    pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
2. **Install GEqTrain**: Once PyTorch is installed, you can install the package from PyPI:
    ```bash
    pip install geqtrain
    ```

**For Developers**
If you want to contribute to the project, follow these steps to set up an editable installation.

1. **Clone the Repository**:

    ```bash
    git clone [https://github.com/limresgrp/GEqTrain.git](https://github.com/limresgrp/GEqTrain.git)
    cd GEqTrain
    ```

2. **Create a Virtual Environment**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. **Install PyTorch**: Follow step 1 from the user installation guide above.

4. **Install GEqTrain in Editable Mode**: This allows your changes to the source code to be immediately reflected.

    ```bash
    pip install -e .
    ```

## ▶️ Usage
You can run training jobs using the geqtrain-train command-line script.

**Single-GPU Training**
To start a training run with a configuration file on a single GPU:
    ```bash
    geqtrain-train path/to/your/config.yaml --device cuda:0
    ```

**Multi-GPU Distributed Training**
For larger models, you can train in parallel across multiple GPUs on a single machine using torchrun.

1. **(Optional) Select GPUs**: You can control which GPUs are visible to your script using the CUDA_VISIBLE_DEVICES environment variable. For example, to use only GPUs 0 and 2:
    
    ```bash
    export CUDA_VISIBLE_DEVICES=0,2
    ```

2. **Launch the Training**: Use torchrun and set `--nproc_per_node` to the number of GPUs you want to use. The `--ddp` flag enables distributed mode.
    
    ```bash
    # This will launch 2 processes, one for each visible GPU (0 and 2)
    torchrun --nproc_per_node=2 geqtrain/scripts/train.py --ddp path/to/your/config.yaml
    ```

# 💻 Running on an HPC Cluster (SLURM)
Running on a supercomputer typically involves three main steps: loading a pre-configured software environment, installing GEqTrain into a personal Python virtual environment, and submitting your job to the workload manager (e.g., SLURM).

## Step 1: Set Up the Environment
While specific commands vary between clusters, the general process is similar. Here, we use a CSCS-like system with the uenv environment manager as a representative example.

1. **Find a Suitable Environment**:
First, search for a pre-built environment that includes a recent version of PyTorch and CUDA.

    ```bash
    # Example: Find PyTorch images available on the system
    uenv image find pytorch
    ```

2. **Pull the Image**:
If the desired image is not available locally, pull it from the registry.

    ```bash
    # Example: Pull a specific PyTorch version
    uenv image pull pytorch/v2.6.0:v1
    ```

3. **List Local Images**:
You can check all the environments you have downloaded.

    ```bash
    # Show all downloaded and available uenvs
    uenv image ls
    ```

4. **Load the Environment**:
Start an interactive session within the chosen containerized environment. This gives you access to its pre-installed software, like PyTorch.

    ```bash
    # Example: Start a shell in the 'pytorch/v2.6.0:v1' environment
    uenv run --view=default pytorch/v2.6.0:v1 -- bash
    ```

5. **Create a Local Python Virtual Environment**:
To keep your packages isolated, create a virtual environment. Using `--system-site-packages` allows your new environment to inherit packages (like PyTorch) from the base HPC environment.

    ```bash
    python -m venv --system-site-packages ./geqtrain-venv
    source ./geqtrain-venv/bin/activate
    ```

6. **Install GEqTrain**:
Install the package and all its dependencies.

    ```bash
    pip install -e .
    ```

## Step 2: Submit a Training Job with SLURM
Create a submission script (e.g., train.sbatch) to define the resources you need and the commands to run.
```
#!/bin/bash
#SBATCH --job-name=geqtrain-ddp
#SBATCH --nodes=1                    # Number of compute nodes
#SBATCH --ntasks=4                   # Total number of processes (e.g., for 4 GPUs)
#SBATCH --gpus-per-task=1            # Assign one GPU to each process
#SBATCH --cpus-per-task=8            # Assign 8 CPU cores to each process
#SBATCH --mem=64GB                   # Memory per node
#SBATCH --time=08:00:00              # Max walltime
#SBATCH --partition=<your_partition> # Cluster-specific partition (e.g., debug, normal)
#SBATCH --account=<your_account>     # Your project account

# Navigate to your project directory
cd <path_to_your_project>/GEqTrain

# Get the config file path from the command line argument
CONFIG=$1
if [ -z "$CONFIG" ]; then
  echo "Error: Missing CONFIG file argument" >&2
  exit 1
fi

# Set MASTER_ADDR to the first node in the job allocation
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500 # A free port for communication

# Launch the distributed training job using srun
# This command runs your code inside the previously set up virtual environment
srun bash -c "
  source <path_to_your_project>/GEqTrain/geqtrain-venv/bin/activate && \
  geqtrain-train --ddp --master-addr \$MASTER_ADDR --master-port \$MASTER_PORT \$CONFIG
"

echo "Job finished"
```

1. **Submit your job to the scheduler**:

    ```bash
    sbatch train.sbatch path/to/your/config.yaml
    ```

## Step 3: Manage Your SLURM Jobs
Here are some common commands for checking on your jobs:
```bash
    # List your jobs:
    squeue -u $USER
    # Cancel a job:
    scancel <job_id>
    # View detailed job information:
    scontrol show job <job_id>
```