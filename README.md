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
    git clone https://github.com/limresgrp/GEqTrain.git
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

-----

## Quick Start

After installation, you can use the command-line scripts provided by GEqTrain. For example, to start a training run with a configuration file on a single GPU:

```bash
geqtrain-train path/to/your/config.yaml --device cuda:0
```

## Distributed Training (Multi-GPU)

For larger models and datasets, you can train in parallel across multiple GPUs using PyTorch's Distributed Data Parallel (DDP).

### Using `torchrun` on a Single Machine

`torchrun` is the standard tool for launching distributed training on a single multi-GPU machine.

1.  **(Optional) Select specific GPUs:** You can control which GPUs are visible to your script using the `CUDA_VISIBLE_DEVICES` environment variable. For example, to only use GPUs 0 and 2:

    ```bash
    export CUDA_VISIBLE_DEVICES=0,2
    ```

2.  **(Optional) Set NCCL Workaround:** On some systems, low-level hardware or driver conflicts can cause hangs. If you experience this, setting the following variable forces a more robust communication path:

    ```bash
    export NCCL_P2P_DISABLE=1
    ```

3.  **Launch the training:** Use `torchrun` and set `--nproc_per_node` to the number of GPUs you want to use. The `--ddp` flag enables distributed mode in the script.

    ```bash
    # This example will launch 2 processes, one for each visible GPU (0 and 2)
    torchrun --nproc_per_node=2 geqtrain/scripts/train.py --ddp path/to/your/config.yaml
    ```

### Using SLURM on an HPC Cluster

On a cluster managed by the SLURM workload manager, you can submit your training job using an `sbatch` script.

#### Step 1: Create a submission script (`train.sbatch`)

Create a file named `train.sbatch` with the following content. You will need to replace the placeholder values (like `<your_account>`) with the specific details of your cluster.

```bash
#!/bin/bash
#SBATCH --job-name=geqtrain-ddp
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # Request 4 tasks (one for each GPU process)
#SBATCH --gpus-per-task=1          # Assign one GPU to each task
#SBATCH --cpus-per-task=8          # Assign CPUs for each task
#SBATCH --mem=<memory_amount>      # e.g., 64GB
#SBATCH --time=<hh:mm:ss>          # e.g., 08:00:00
#SBATCH --partition=<your_partition> # e.g., normal or debug
#SBATCH --account=<your_account>

# Navigate to your project directory
cd <path_to_your_project>/GEqTrain

# Get the config file path from the command line
CONFIG=$1
if [ -z "$CONFIG" ];
then
  echo "Missing CONFIG file argument"
  exit 1
fi

# Set MASTER_ADDR and MASTER_PORT for process communication
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# Set Wandb api key if you need to log training on W&B
export WANDB_API_KEY=myapikey

# srun will launch 4 independent copies of the following command.
# You may need to adapt this line to your cluster's module or environment system.
srun bash -c "
  source <path_to_your_project>/GEqTrain/.venv/bin/activate && \
  geqtrain-train --ddp --master-addr \$MASTER_ADDR --master-port \$MASTER_PORT \$CONFIG
"

echo "Job finished"
```

#### Step 2: Submit the Job

Submit your script to the SLURM scheduler, passing your configuration file as an argument.

```bash
sbatch train.sbatch path/to/your/config.yaml
```