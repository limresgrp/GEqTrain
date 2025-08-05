#!/bin/bash
#SBATCH --job-name=geqtrain-ddp
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # Request 4 tasks (one for each GPU process) 
#SBATCH --gpus-per-task=1          # Assign one GPU to each task 
#SBATCH --cpus-per-task=8          # Assign CPUs for each task 
#SBATCH --mem=64GB
#SBATCH --time=00:05:00
#SBATCH --partition=debug
#SBATCH --account=lp40

cd $HOME/GEqTrain

CONFIG=$1
if [ -z "$CONFIG" ];
then
  echo "Missing CONFIG"
  exit 1
fi

# Get the hostname of the first node in the job allocation.
# All processes will use this as the master address.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Set a port for the master process. A fixed port is fine for single-node jobs.
export MASTER_PORT=29500

# for debugging
# export CUDA_LAUNCH_BLOCKING=1

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# srun will launch 4 independent copies of the following command.
# Each process will have its environment set up correctly by sourcing the activate scripts.
srun --uenv=pytorch/v2.6.0:v1 --view=default -- bash -c "
  source $HOME/GEqTrain/geqtrain-venv/bin/activate && \
  geqtrain-train --ddp --master-addr \$MASTER_ADDR --master-port \$MASTER_PORT $CONFIG
"

echo "Job finished"

# For debugging, having access to a bash on the node which is running the training
# salloc -N 1 -t 15 --partition debug
# srun --ntasks=4 --gpus-per-task=1 --uenv=pytorch/v2.6.0:v1 --view=default geqtrain-train --ddp --master-addr localhost --master-port 29500 /users/dangiole/GEqTrain/config/xxMD-DFT/xxMD.DFT.azo.goten.yaml &> log.txt &